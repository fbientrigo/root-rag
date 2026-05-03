#!/usr/bin/env python3
"""Build S1 semantic artifacts from a chunk/corpus JSONL file."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from math import ceil
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path

from root_rag.retrieval.s1_semantic import (
    SemanticIndexManifest,
    SentenceTransformerLocalEmbedder,
    _file_sha256,
    _import_faiss,
    _import_numpy,
    build_semantic_record,
    load_corpus_rows,
    normalize_vectors,
)


def _log(message: str) -> None:
    print(message, flush=True)


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

    start_time = perf_counter()
    output_dir = args.output_dir
    batch_size = max(1, args.batch_size)

    _log("[build] start")
    _log(f"[build] corpus: {args.corpus}")
    _log(f"[build] model: {args.model_name} on {args.device}")
    _log(f"[build] output_dir: {output_dir}")

    _log(f"[1/4] Loading corpus rows from {args.corpus}")
    corpus_rows = load_corpus_rows(args.corpus)
    _log(f"[build] canonical chunks: {len(corpus_rows)}")
    _log(f"[2/4] Loading embedding model {args.model_name} on {args.device}")
    embedder = SentenceTransformerLocalEmbedder(
        model_name=args.model_name,
        device=args.device,
        batch_size=batch_size,
    )
    np = _import_numpy()
    faiss = _import_faiss()

    _log(f"[3/4] Building semantic records in {output_dir}")
    ordered_rows = []
    total_rows = len(corpus_rows)
    manifest_step = max(1, ceil(total_rows / 10))
    for idx, row in enumerate(corpus_rows, start=1):
        ordered_rows.append(build_semantic_record(row))
        if idx == 1 or idx == total_rows or idx % manifest_step == 0:
            _log(f"[build] manifest {idx}/{total_rows}")
    ordered_rows.sort(key=lambda row: row["chunk_id"])

    _log(f"[build] embedding batches: {ceil(total_rows / batch_size)} @ batch_size={batch_size}")
    texts = [row["semantic_text"] for row in ordered_rows]
    embedded_batches = []
    for batch_idx, batch_start in enumerate(range(0, total_rows, batch_size), start=1):
        batch_texts = texts[batch_start : batch_start + batch_size]
        _log(f"[build] embed batch {batch_idx}/{ceil(total_rows / batch_size)} ({len(batch_texts)} rows)")
        embedded_batches.append(embedder.embed(batch_texts))
    vectors = np.vstack(embedded_batches) if embedded_batches else np.zeros((0, embedder.embedding_dimension()), dtype=np.float32)
    vectors = normalize_vectors(np.asarray(vectors, dtype=np.float32))
    if len(ordered_rows) != int(vectors.shape[0]):
        raise ValueError("row/vector count mismatch while building semantic index")
    vector_dim = int(vectors.shape[1]) if vectors.size else embedder.embedding_dimension()

    _log("[4/4] Saving semantic artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = output_dir / "vectors.npy"
    index_path = output_dir / "index.faiss"
    records_path = output_dir / "semantic_manifest.jsonl"
    manifest_path = output_dir / "semantic_manifest.json"

    _log(f"[build] write vectors -> {vectors_path}")
    np.save(vectors_path, vectors)
    _log(f"[build] write index -> {index_path}")
    faiss_index = faiss.IndexFlatIP(vector_dim)
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, str(index_path))
    _log(f"[build] write manifest rows -> {records_path}")
    with records_path.open("w", encoding="utf-8") as handle:
        for row in ordered_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = SemanticIndexManifest(
        schema_version="1.0.0",
        model_name=embedder.model_name,
        embedding_dimension=vector_dim,
        normalization="l2",
        corpus_source_identifier=args.corpus_source_id or args.corpus.stem,
        corpus_path=str(Path(args.corpus)),
        corpus_sha256=_file_sha256(Path(args.corpus)),
        row_count=len(ordered_rows),
        faiss_index_type="IndexFlatIP",
        index_path=str(index_path),
        records_path=str(records_path),
        vectors_path=str(vectors_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        build_backend="sentence_transformers_local",
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )
    _log(f"[build] write manifest json -> {manifest_path}")
    manifest.save(manifest_path)
    elapsed = perf_counter() - start_time
    _log(
        "[build] done | rows={rows} | vector_shape={shape} | files={files} | elapsed={elapsed:.2f}s".format(
            rows=manifest.row_count,
            shape=list(vectors.shape),
            files=", ".join(
                str(path)
                for path in (manifest_path, records_path, index_path, vectors_path)
            ),
            elapsed=elapsed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
