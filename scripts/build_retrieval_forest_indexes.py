#!/usr/bin/env python3
"""Build experimental indexes for the retrieval forest."""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from root_rag.corpus.manifest import Manifest
from root_rag.index.builder import build_index
from root_rag.index.fts import build_fts_index, check_fts5_available
from root_rag.index.schemas import IndexManifest, Chunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def get_fairship_commit(fairship_path: Path) -> str:
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=fairship_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"Failed to get commit: {e}")
        return "unknown"

def main():
    fairship_path = Path("../FairShip").resolve()
    if not fairship_path.is_dir():
        logger.error(f"FairShip path not found: {fairship_path}")
        sys.exit(1)

    config_path = Path("configs/retrieval_forest_profiles.json")
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    if not check_fts5_available():
        logger.error("SQLite FTS5 extension not available!")
        sys.exit(1)

    # We need a manifest to pass to build_index
    commit = get_fairship_commit(fairship_path)
    manifest = Manifest(
        repo_url="https://github.com/ShipSoft/FairShip.git",
        root_ref="master",
        resolved_commit=commit,
        local_path=str(fairship_path),
        fetched_at=datetime.now(timezone.utc).isoformat(),
        dirty=False,
        tool_version="0.1.0",
    )

    for profile in config["profiles"]:
        profile_id = profile["profile_id"]
        if profile_id == "baseline_current":
            logger.info(f"Skipping baseline profile: {profile_id}")
            continue

        logger.info(f"Building index for profile: {profile_id}")
        
        index_dir = Path(profile["index_output_path"])
        index_dir.mkdir(parents=True, exist_ok=True)

        window_lines = profile["chunk_size_lines"]
        overlap_lines = profile["overlap_lines"]

        # Build chunks
        # Note: build_index writes to a subdir based on commit
        # We want to control the output path more directly or at least know where it goes.
        # Looking at builder.py:
        # corpus_id = f"{manifest.root_ref}__{manifest.resolved_commit[:12]}"
        # corpus_output_dir = output_dir / corpus_id
        
        temp_chunks_dir = Path("data/processed/chunks_forest") / profile_id
        temp_chunks_dir.mkdir(parents=True, exist_ok=True)

        result = build_index(
            manifest=manifest,
            output_dir=temp_chunks_dir,
            window_lines=window_lines,
            overlap_lines=overlap_lines,
            discovery_profile="fairship_workflow",
        )

        if result["status"] != "success":
            logger.error(f"Chunking failed for {profile_id}: {result.get('error')}")
            continue

        chunks_file = Path(result["chunks_path"])
        fts_db = index_dir / "fts.sqlite"

        # Load chunks
        chunks = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(Chunk.model_validate_json(line))

        logger.info(f"Loaded {len(chunks)} chunks for {profile_id}")

        # Build FTS
        build_fts_index(db_path=fts_db, chunks=chunks)
        logger.info(f"FTS5 index ready for {profile_id}: {fts_db}")

        # Save index manifest
        index_manifest = IndexManifest(
            index_id=f"fairship_forest_{profile_id}",
            corpus_id=f"fairship__{manifest.root_ref}__{manifest.resolved_commit[:12]}",
            corpus_url=manifest.repo_url,
            root_ref=manifest.root_ref,
            resolved_commit=manifest.resolved_commit,
            created_at=datetime.now(timezone.utc).isoformat(),
            chunk_count=result["chunk_count"],
            file_count=result["file_count"],
            retrieval_modes=["lexical"],
            fts_db_path="fts.sqlite",
            chunks_path=str(chunks_file.absolute()), # Using absolute for forest experiments
            corpus_config=f"FairShip forest profile: {profile_id}",
            description=f"Experimental retrieval forest index for {profile_id}",
        )
        index_manifest.save(index_dir / "index_manifest.json")
        logger.info(f"Saved manifest for {profile_id}")

if __name__ == "__main__":
    main()
