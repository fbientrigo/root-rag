"""Index builder: convert chunks to persisted JSONL format."""
import logging
from pathlib import Path
from typing import Dict

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
