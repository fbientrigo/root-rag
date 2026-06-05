#!/usr/bin/env python3
"""
Index local project documentation for RAG retrieval.

This script creates an index of the root-rag project documentation,
including AGENTS.md, boulder.json, docs/, reports/, etc.
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
from root_rag.index.builder import build_full_index
from root_rag.index.fts import check_fts5_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def get_git_info(project_path: Path):
    """Get current HEAD commit and branch of the local repo."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        
        return commit, branch
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to get git info: {e}")
        return "unknown", "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Index root-rag project documentation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/indexes_project_docs"),
        help="Output directory for index (default: data/indexes_project_docs)",
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
    
    project_root = Path(__file__).parent.parent.resolve()
    commit, branch = get_git_info(project_root)
    
    logger.info(f"Indexing project documentation at {project_root}")
    logger.info(f"Git: {branch} ({commit[:12]})")
    
    # Create manifest
    manifest = Manifest(
        repo_url="local-project-docs",
        root_ref=branch,
        resolved_commit=commit,
        local_path=str(project_root),
        fetched_at=datetime.now(timezone.utc).isoformat(),
        dirty=True,
        tool_version="0.1.0",
    )
    
    # Build index
    result = build_full_index(
        manifest=manifest,
        output_dir=args.output_dir,
        window_lines=args.window_lines,
        overlap_lines=args.overlap_lines,
        discovery_profile="project_docs",
    )
    
    if result["status"] != "success":
        logger.error(f"Index build failed: {result.get('error', 'unknown')}")
        sys.exit(1)
    
    # Print summary
    print("\n" + "="*60)
    print("[OK] Project documentation index created!")
    print("="*60)
    print(f"  Index ID: {result['index_id']}")
    print(f"  Chunks: {result['chunk_count']}")
    print(f"  Files: {result['file_count']}")
    print(f"  FTS DB: {result['fts_db_path']}")
    print(f"  Manifest: {result['index_manifest_path']}")
    print("="*60)
    print("\nUsage:")
    print(f"  root-rag ask \"query\" --profile project_docs --root-ref {branch}")


if __name__ == "__main__":
    main()
