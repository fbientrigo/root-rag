#!/usr/bin/env python3
"""
Index FairShip codebase for RAG retrieval.

This script creates an index of the FairShip source code (not ROOT),
enabling queries like:
- "How does FairShip implement detector hits?"
- "Show me FairShip's TGeoManager usage"
- "Where does FairShip construct geometry?"
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from root_rag.corpus.manifest import Manifest
from root_rag.index.builder import build_index
from root_rag.index.fts import build_fts_index, check_fts5_available
from root_rag.index.schemas import IndexManifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def get_fairship_commit(fairship_path: Path) -> str:
    """Get current HEAD commit of FairShip repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=fairship_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get FairShip commit: {e}")


def get_fairship_branch(fairship_path: Path) -> str:
    """Get current branch/tag of FairShip repo."""
    try:
        # Try branch first
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=fairship_path,
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()
        if branch != "HEAD":
            return branch
        
        # If detached HEAD, try to find tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=fairship_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        
        # Fall back to commit SHA
        return "HEAD"
    except subprocess.CalledProcessError:
        return "HEAD"


def create_fairship_manifest(fairship_path: Path) -> Manifest:
    """Create a manifest for FairShip corpus."""
    fairship_path = fairship_path.resolve()
    
    if not fairship_path.is_dir():
        raise ValueError(f"FairShip path does not exist: {fairship_path}")
    
    commit = get_fairship_commit(fairship_path)
    ref = get_fairship_branch(fairship_path)
    
    logger.info(f"FairShip at {ref} (commit: {commit[:12]})")
    
    manifest = Manifest(
        repo_url="https://github.com/ShipSoft/FairShip.git",
        root_ref=ref,
        resolved_commit=commit,
        local_path=str(fairship_path),
        fetched_at=datetime.now(timezone.utc).isoformat(),
        dirty=False,
        tool_version="0.1.0",
    )
    
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Index FairShip codebase for ROOT-RAG retrieval"
    )
    parser.add_argument(
        "--fairship-path",
        type=Path,
        required=True,
        help="Path to FairShip repository",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/indexes_fairship"),
        help="Output directory for index (default: data/indexes_fairship)",
    )
    parser.add_argument(
        "--window-lines",
        type=int,
        default=80,
        help="Lines per chunk window (default: 80)",
    )
    parser.add_argument(
        "--overlap-lines",
        type=int,
        default=10,
        help="Lines of overlap between windows (default: 10)",
    )
    
    args = parser.parse_args()
    
    # Check FTS5 availability
    if not check_fts5_available():
        logger.error("SQLite FTS5 extension not available!")
        sys.exit(1)
    
    # Create manifest
    logger.info("Creating FairShip corpus manifest...")
    manifest = create_fairship_manifest(args.fairship_path)
    
    # Build index
    logger.info("Building FairShip index...")
    logger.info(f"  Repo: {manifest.repo_url}")
    logger.info(f"  Ref: {manifest.root_ref}")
    logger.info(f"  Commit: {manifest.resolved_commit[:12]}")
    logger.info(f"  Path: {manifest.local_path}")
    
    # Generate index ID: fairship__{ref}__{commit[:12]}__{timestamp}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "+0000Z"
    index_id = f"fairship__{manifest.root_ref}__{manifest.resolved_commit[:12]}__{timestamp}"
    index_dir = args.output_dir / index_id
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Build chunks
    logger.info("Chunking FairShip source files...")
    result = build_index(
        manifest=manifest,
        output_dir=Path("data/processed/chunks"),
        window_lines=args.window_lines,
        overlap_lines=args.overlap_lines,
        seed_corpus_config=None,  # No filtering - index everything
    )
    
    if result["status"] != "success":
        logger.error(f"Index build failed: {result.get('error', 'unknown')}")
        sys.exit(1)
    
    logger.info(f"Generated {result['chunk_count']} chunks from {result['file_count']} files")
    
    # Build FTS5 index
    logger.info("Building FTS5 search index...")
    chunks_file = Path(result["chunks_path"])
    fts_db = index_dir / "fts.sqlite"
    
    # Load chunks from JSONL
    from root_rag.index.schemas import Chunk
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk.model_validate_json(line))
    
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
    
    fts_result = build_fts_index(
        db_path=fts_db,
        chunks=chunks,
    )
    
    logger.info(f"FTS5 index ready: {fts_db}")
    
    # Create index manifest
    index_manifest = IndexManifest(
        index_id=index_id,
        corpus_id=f"fairship__{manifest.root_ref}__{manifest.resolved_commit[:12]}",
        corpus_url=manifest.repo_url,
        root_ref=manifest.root_ref,
        resolved_commit=manifest.resolved_commit,
        created_at=datetime.now(timezone.utc).isoformat(),
        chunk_count=result["chunk_count"],
        file_count=result["file_count"],
        retrieval_modes=["lexical"],
        fts_db_path=str(fts_db.relative_to(index_dir)),
        chunks_path=str(chunks_file.relative_to(index_dir.parent.parent)),
        corpus_config="FairShip codebase (all C++ files)",
        description=f"FairShip codebase index for ref={manifest.root_ref}",
    )
    
    manifest_path = index_dir / "index_manifest.json"
    index_manifest.save(manifest_path)
    logger.info(f"Saved index manifest to {manifest_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("[OK] FairShip index created!")
    print("="*60)
    print(f"  Index ID: {index_id}")
    print(f"  Ref: {manifest.root_ref}")
    print(f"  Commit: {manifest.resolved_commit[:12]}")
    print(f"  Chunks: {result['chunk_count']}")
    print(f"  Files: {result['file_count']}")
    print(f"  FTS DB: {fts_db}")
    print(f"  Manifest: {manifest_path}")
    print("="*60)


if __name__ == "__main__":
    main()
