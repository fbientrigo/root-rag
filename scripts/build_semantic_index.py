#!/usr/bin/env python3
"""Build S1 semantic artifacts from a chunk/corpus JSONL file."""

from __future__ import annotations

import argparse
from pathlib import Path

from root_rag.retrieval.s1_semantic import (
    SentenceTransformerLocalEmbedder,
    build_semantic_index_artifacts,
    load_corpus_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Chunk/corpus JSONL path to embed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for semantic artifacts.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Local sentence-transformers model name.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding inference device (default: cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--corpus-source-id",
        default=None,
        help="Optional reproducibility identifier for the embedded corpus.",
    )
    args = parser.parse_args()

    corpus_rows = load_corpus_rows(args.corpus)
    embedder = SentenceTransformerLocalEmbedder(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    manifest = build_semantic_index_artifacts(
        corpus_rows=corpus_rows,
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        embedder=embedder,
        corpus_source_identifier=args.corpus_source_id or args.corpus.stem,
    )
    print(f"[OK] Semantic manifest: {args.output_dir / 'semantic_manifest.json'}")
    print(f"Model: {manifest.model_name}")
    print(f"Dimension: {manifest.embedding_dimension}")
    print(f"Rows: {manifest.row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
