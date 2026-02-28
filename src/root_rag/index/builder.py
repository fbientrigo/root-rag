"""Index builder: convert chunks to persisted JSONL and FTS5 formats."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from root_rag.index.fts import build_fts_index, check_fts5_available
from root_rag.index.schemas import IndexManifest
from root_rag.parser.chunks import chunk_corpus

logger = logging.getLogger(__name__)


def build_index(
    manifest,
    output_dir: Path,
    window_lines: int = 80,
    overlap_lines: int = 10,
) -> Dict[str, any]:
    """Build index from manifest: chunk corpus and write JSONL.
    
    Args:
        manifest: Manifest object with root_ref, resolved_commit, local_path
        output_dir: Directory where chunks.jsonl will be written
        window_lines: Lines per window (default 80)
        overlap_lines: Lines of overlap (default 10)
    
    Returns:
        Metadata dict: {
            "chunk_count": int,
            "file_count": int,
            "root_ref": str,
            "resolved_commit": str,
            "chunks_path": str,
            "status": "success" | "partial" | "failed"
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate corpus ID for subdirectory
    corpus_id = f"{manifest.root_ref}__{manifest.resolved_commit[:12]}"
    corpus_output_dir = output_dir / corpus_id
    corpus_output_dir.mkdir(parents=True, exist_ok=True)
    
    chunks_file = corpus_output_dir / "chunks.jsonl"
    
    logger.info(f"Building index from manifest at {manifest.local_path}")
    
    # Chunk the corpus
    repo_root = Path(manifest.local_path)
    chunks = chunk_corpus(
        manifest=manifest,
        repo_root=repo_root,
        window_lines=window_lines,
        overlap_lines=overlap_lines,
    )
    
    if not chunks:
        logger.warning("No chunks generated from corpus")
        return {
            "chunk_count": 0,
            "file_count": 0,
            "root_ref": manifest.root_ref,
            "resolved_commit": manifest.resolved_commit,
            "chunks_path": str(chunks_file),
            "status": "failed",
            "error": "no_chunks",
        }
    
    # Write chunks to JSONL
    try:
        with open(chunks_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.to_jsonl_line())
        
        logger.info(f"Wrote {len(chunks)} chunks to {chunks_file}")
    except Exception as e:
        logger.error(f"Failed to write chunks file: {e}")
        return {
            "chunk_count": len(chunks),
            "file_count": len(set(c.file_path for c in chunks)),
            "root_ref": manifest.root_ref,
            "resolved_commit": manifest.resolved_commit,
            "chunks_path": str(chunks_file),
            "status": "failed",
            "error": str(e),
        }
    
    # Calculate file count (unique file paths)
    file_count = len(set(chunk.file_path for chunk in chunks))
    
    return {
        "chunk_count": len(chunks),
        "file_count": file_count,
        "root_ref": manifest.root_ref,
        "resolved_commit": manifest.resolved_commit,
        "chunks_path": str(chunks_file),
        "status": "success",
    }


def build_full_index(
    manifest,
    output_dir: Path,
    window_lines: int = 80,
    overlap_lines: int = 10,
) -> Dict[str, any]:
    """Build full index: chunk corpus → JSONL + FTS5 + IndexManifest.
    
    Complete indexing pipeline:
    1. Chunk the corpus using sliding window
    2. Write chunks to JSONL
    3. Build FTS5 fulltext search index
    4. Create and persist IndexManifest
    
    Args:
        manifest: Manifest object with root_ref, resolved_commit, local_path, repo_url
        output_dir: Directory where indexes/<index_id>/ will be created
        window_lines: Lines per window (default 80)
        overlap_lines: Lines of overlap (default 10)
    
    Returns:
        Dict with keys:
            - index_id: Unique index identifier
            - corpus_id: Deterministic corpus identifier
            - chunk_count: Total chunks indexed
            - file_count: Unique source files
            - chunks_path: Path to chunks.jsonl
            - fts_db_path: Path to fts.sqlite
            - index_manifest_path: Path to index_manifest.json
            - retrieval_modes: ["lexical"]
            - created_at: ISO8601 timestamp
            - status: "success" | "failed" | "partial"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate deterministic corpus_id: {root_ref}__{commit[:12]}
    corpus_id = IndexManifest.compute_corpus_id(manifest.root_ref, manifest.resolved_commit)
    
    # Chunk the corpus
    logger.info(f"Building full index for {corpus_id}")
    repo_root = Path(manifest.local_path)
    chunks = chunk_corpus(
        manifest=manifest,
        repo_root=repo_root,
        window_lines=window_lines,
        overlap_lines=overlap_lines,
    )
    
    if not chunks:
        logger.error("No chunks generated from corpus")
        return {
            "status": "failed",
            "error": "no_chunks",
        }
    
    # Write chunks to JSONL under data/processed/chunks/
    chunks_dir = Path("data/processed/chunks") / corpus_id
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks_file = chunks_dir / "chunks.jsonl"
    
    try:
        with open(chunks_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.to_jsonl_line())
        logger.info(f"Wrote {len(chunks)} chunks to {chunks_file}")
    except Exception as e:
        logger.error(f"Failed to write chunks file: {e}")
        return {
            "status": "failed",
            "error": f"chunks_write_failed:{e}",
        }
    
    # Create FTS5 index
    if not check_fts5_available():
        logger.error("FTS5 not available on this system")
        return {
            "status": "failed",
            "error": "fts5_unavailable",
        }
    
    # Generate timestamp-based index_id (unique, sortable)
    created_at = datetime.utcnow().isoformat()
    if not created_at.endswith("Z"):
        created_at += "Z"
    
    index_id = IndexManifest.compute_index_id(corpus_id, created_at)
    
    # Create index directory
    index_dir = output_dir / index_id
    index_dir.mkdir(parents=True, exist_ok=True)
    
    fts_db_path = index_dir / "fts.sqlite"
    fts_stats = build_fts_index(fts_db_path, chunks)
    
    if fts_stats.get("status") != "success":
        logger.error(f"FTS5 index build failed: {fts_stats.get('status')}")
        return {
            "status": "failed",
            "error": "fts_build_failed",
        }
    
    # Create IndexManifest
    file_count = len(set(chunk.file_path for chunk in chunks))
    
    index_manifest = IndexManifest(
        index_id=index_id,
        corpus_id=corpus_id,
        root_ref=manifest.root_ref,
        resolved_commit=manifest.resolved_commit,
        corpus_url=manifest.repo_url,
        chunks_path=str(chunks_file),
        fts_db_path=str(fts_db_path),
        chunk_count=len(chunks),
        file_count=file_count,
        retrieval_modes=["lexical"],
        created_at=created_at,
        tool_version="0.1.0",
    )
    
    # Persist IndexManifest
    manifest_file = index_dir / "index_manifest.json"
    try:
        index_manifest.save(manifest_file)
        logger.info(f"Saved index manifest to {manifest_file}")
    except Exception as e:
        logger.error(f"Failed to save index manifest: {e}")
        return {
            "status": "failed",
            "error": f"manifest_write_failed:{e}",
        }
    
    logger.info(
        f"Index ready: {index_id}, "
        f"retrieval_modes=['lexical'], "
        f"chunks={len(chunks)}, "
        f"files={file_count}"
    )
    
    return {
        "index_id": index_id,
        "corpus_id": corpus_id,
        "chunk_count": len(chunks),
        "file_count": file_count,
        "chunks_path": str(chunks_file),
        "fts_db_path": str(fts_db_path),
        "index_manifest_path": str(manifest_file),
        "retrieval_modes": ["lexical"],
        "created_at": created_at,
        "status": "success",
    }
