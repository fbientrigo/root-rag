#!/usr/bin/env python3
"""Run semantic retrieval V1 benchmark on frozen semantic slice."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path

from root_rag.evaluation.semantic_v1 import (
    compare_mode_runs,
    evaluate_mode,
    load_corpus,
    load_qrels,
    load_queries,
    _pick_recommendation,
    render_semantic_v1_markdown,
)
from root_rag.retrieval.s1_semantic import SemanticIndexManifest, SentenceTransformerLocalEmbedder


def _log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/corpus.jsonl"),
        help="Frozen canonical corpus JSONL path.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("configs/benchmark_queries_semantic.json"),
        help="Frozen semantic benchmark query set.",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("configs/benchmark_qrels_semantic.jsonl"),
        help="Frozen semantic benchmark qrels.",
    )
    parser.add_argument(
        "--semantic-manifest",
        type=Path,
        required=True,
        help="Semantic index manifest JSON path.",
    )
    parser.add_argument(
        "--semantic-model",
        default="intfloat/e5-base-v2",
        help="Sentence-transformers model used for semantic retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Evaluation cutoff.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Combined V1 benchmark JSON output.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        required=True,
        help="Combined V1 benchmark markdown report.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Optional prior benchmark JSON for delta reporting.",
    )
    parser.add_argument(
        "--exact-change",
        action="append",
        default=[],
        help="Exact change note to capture in the markdown report. Repeatable.",
    )
    parser.add_argument(
        "--report-title",
        default="Semantic Retrieval V1 Results",
        help="Markdown report title.",
    )
    args = parser.parse_args()

    start_time = perf_counter()
    _log("[run] start")
    _log(f"[run] semantic_manifest: {args.semantic_manifest}")
    _log(f"[run] semantic_model: {args.semantic_model}")
    _log(f"[run] output_json: {args.output_json}")
    _log(f"[run] output_md: {args.output_md}")

    _log("[1/4] Load manifest")
    semantic_manifest = SemanticIndexManifest.load(args.semantic_manifest)
    _log("[2/4] Load model")
    shared_embedder = SentenceTransformerLocalEmbedder(
        model_name=args.semantic_model,
        device="cpu",
        batch_size=16,
        local_files_only=True,
    )
    build_command = (
        f"{sys.executable} scripts/build_semantic_index.py --corpus {semantic_manifest.corpus_path} "
        f"--output-dir {args.semantic_manifest.parent} --model-name {semantic_manifest.model_name}"
    )
    benchmark_command = " ".join([sys.executable, *sys.argv])

    baseline_results = None
    if args.baseline_json is not None and args.baseline_json.exists():
        baseline_results = json.loads(args.baseline_json.read_text(encoding="utf-8"))

    _log("[3/4] Load corpus and qrels")
    corpus_rows = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels_map = load_qrels(args.qrels)

    _log("[3/4] Run bm25_only")
    bm25_only = evaluate_mode(
        mode="bm25_only",
        corpus_rows=corpus_rows,
        corpus_path=args.corpus,
        queries=queries,
        qrels_map=qrels_map,
        top_k=args.top_k,
        semantic_manifest_path=args.semantic_manifest,
        semantic_model_name=args.semantic_model,
        semantic_embedder=shared_embedder,
    )
    _log("[run] bm25_only done")

    _log("[3/4] Run semantic_only")
    semantic_only = evaluate_mode(
        mode="semantic_only",
        corpus_rows=corpus_rows,
        corpus_path=args.corpus,
        queries=queries,
        qrels_map=qrels_map,
        top_k=args.top_k,
        semantic_manifest_path=args.semantic_manifest,
        semantic_model_name=args.semantic_model,
        semantic_embedder=shared_embedder,
    )
    _log("[run] semantic_only done")

    _log("[3/4] Run hybrid")
    hybrid = evaluate_mode(
        mode="hybrid",
        corpus_rows=corpus_rows,
        corpus_path=args.corpus,
        queries=queries,
        qrels_map=qrels_map,
        top_k=args.top_k,
        semantic_manifest_path=args.semantic_manifest,
        semantic_model_name=args.semantic_model,
        semantic_embedder=shared_embedder,
    )
    _log("[run] hybrid done")

    semantic_manifest = SemanticIndexManifest.load(args.semantic_manifest)
    results = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "top_k": args.top_k,
            "corpus_path": str(args.corpus),
            "queries_path": str(args.queries),
            "qrels_path": str(args.qrels),
            "semantic_manifest_path": str(args.semantic_manifest),
            "semantic_manifest_jsonl_path": semantic_manifest.records_path,
            "semantic_index_path": semantic_manifest.index_path,
            "semantic_vectors_path": semantic_manifest.vectors_path,
            "semantic_model_name": args.semantic_model,
            "commands_run": [build_command, benchmark_command],
            "report_title": args.report_title,
            "exact_changes": list(args.exact_change),
        },
        "modes": {
            "bm25_only": bm25_only,
            "semantic_only": semantic_only,
            "hybrid": hybrid,
        },
        "comparisons": {
            "semantic_only_vs_bm25_only": compare_mode_runs(bm25_only, semantic_only),
            "hybrid_vs_bm25_only": compare_mode_runs(bm25_only, hybrid),
        },
    }
    results["recommendation"] = _pick_recommendation(results)
    if baseline_results is not None:
        results["metadata"]["baseline_json"] = str(args.baseline_json)
        results["baseline_comparison"] = baseline_results

    _log("[4/4] Write outputs")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.output_md.write_text(render_semantic_v1_markdown(results), encoding="utf-8")
    elapsed = perf_counter() - start_time
    _log(
        "[run] done | outputs={outputs} | elapsed={elapsed:.2f}s".format(
            outputs=", ".join([str(args.output_json), str(args.output_md)]),
            elapsed=elapsed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
