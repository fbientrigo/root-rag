#!/usr/bin/env python3
"""
Set up the ROOT v6-36-08 corpus and build the seed/Tier-1/SOFIE indexes.

This reproduces the gitignored `data/raw/corpora`, `data/indexes`,
`data/indexes_tier1`, and `data/indexes_sofie` artifacts required by the
integration tests in tests/test_golden_queries.py, tests/test_cross_index.py,
and tests/test_sofie_indexing.py.

A full clone of root-project/root is multi-gigabyte, so this performs a
partial/sparse/shallow clone (blob:none filter, depth=1, sparse-checkout)
limited to the directories referenced by configs/seed_corpus_root_636.yaml,
configs/tier1_corpus_root_636.yaml, and configs/sofie_corpus_root_636.yaml.

Usage:
    python scripts/setup_root_corpus.py
    python scripts/setup_root_corpus.py --force-refresh
"""

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from root_rag.corpus.fetcher import resolve_git_ref
from root_rag.corpus.manifest import Manifest
from root_rag.index.builder import build_full_index
from root_rag.index.fts import check_fts5_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/root-project/root.git"
ROOT_REF = "v6-36-08"

# Top-level directories referenced by configs/{seed,tier1,sofie}_corpus_root_636.yaml.
SPARSE_CHECKOUT_PATTERNS = [
    "/*",
    "!/*/",
    "/core/",
    "/eg/",
    "/geom/",
    "/hist/",
    "/io/",
    "/math/",
    "/montecarlo/",
    "/physics/",
    "/tmva/",
    "!/tmva/*/",
    "/tmva/sofie/",
    "/tree/",
]

# (seed_corpus_config, output_dir)
INDEX_BUILDS = [
    (Path("configs/seed_corpus_root_636.yaml"), Path("data/indexes")),
    (Path("configs/tier1_corpus_root_636.yaml"), Path("data/indexes_tier1")),
    (Path("configs/sofie_corpus_root_636.yaml"), Path("data/indexes_sofie")),
]


def run(cmd, cwd=None) -> None:
    logger.info("+ %s", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def sparse_clone(repo_path: Path) -> None:
    run([
        "git", "clone", "--quiet", "--filter=blob:none", "--sparse",
        "--no-checkout", "--depth=1", "--branch", ROOT_REF, REPO_URL, str(repo_path),
    ])
    run(["git", "sparse-checkout", "set", "--no-cone", *SPARSE_CHECKOUT_PATTERNS], cwd=repo_path)
    run(["git", "checkout", "--quiet", ROOT_REF], cwd=repo_path)


def fetch_root_corpus(cache_dir: Path, force_refresh: bool) -> Manifest:
    """Sparse-clone ROOT @ v6-36-08 and write a corpus manifest.

    Mirrors root_rag.corpus.fetcher.fetch_corpus's corpus_id/manifest
    conventions (including resolving the annotated tag object SHA) so the
    resulting cache directory matches what `root-rag index` expects.
    """
    resolved_commit = resolve_git_ref(REPO_URL, ROOT_REF)
    corpus_id = f"root-project__root__{resolved_commit[:12]}"
    corpus_dir = cache_dir / corpus_id
    repo_path = corpus_dir / "repo"
    manifest_path = corpus_dir / "manifest.json"

    if repo_path.exists() and manifest_path.exists() and not force_refresh:
        logger.info(f"Using cached ROOT corpus: {corpus_id}")
        return Manifest.load(manifest_path)

    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)

    logger.info(f"Sparse-cloning ROOT {ROOT_REF} into {repo_path}")
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    sparse_clone(repo_path)

    manifest = Manifest(
        repo_url=REPO_URL,
        root_ref=ROOT_REF,
        resolved_commit=resolved_commit,
        local_path=str(repo_path.resolve()),
        fetched_at=datetime.now(timezone.utc).isoformat(),
        dirty=False,
        tool_version="0.0.1",
    )
    manifest.save(manifest_path)
    logger.info(f"Wrote manifest: {manifest_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/corpora"),
        help="Cache directory for the ROOT corpus (default: data/raw/corpora)",
    )
    parser.add_argument("--window-lines", type=int, default=80, help="Lines per chunk window")
    parser.add_argument("--overlap-lines", type=int, default=10, help="Lines of overlap between windows")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-clone the ROOT corpus even if already cached",
    )
    args = parser.parse_args()

    if not check_fts5_available():
        logger.error("SQLite FTS5 extension not available!")
        return 8

    manifest = fetch_root_corpus(args.cache_dir, args.force_refresh)
    logger.info(f"ROOT corpus ready at {manifest.local_path} (commit {manifest.resolved_commit[:12]})")

    for seed_corpus_config, output_dir in INDEX_BUILDS:
        logger.info(f"Building index from {seed_corpus_config} -> {output_dir}")
        result = build_full_index(
            manifest=manifest,
            output_dir=output_dir,
            window_lines=args.window_lines,
            overlap_lines=args.overlap_lines,
            seed_corpus_config=seed_corpus_config,
        )
        if result.get("status") != "success":
            logger.error(f"Index build failed for {output_dir}: {result.get('error')}")
            return 1
        logger.info(
            f"  [OK] {result['chunk_count']} chunks from {result['file_count']} files "
            f"-> {result['index_manifest_path']}"
        )

    print("\n" + "=" * 60)
    print("[OK] ROOT v6-36-08 corpus and indexes ready")
    print("=" * 60)
    for _, output_dir in INDEX_BUILDS:
        print(f"  {output_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
