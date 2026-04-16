#!/usr/bin/env python3
"""Generate query-level retrieval competition diagnostics without changing behavior."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from root_rag.evaluation.competition_diagnostics import (
    DIAGNOSIS_LABELS,
    analyze_split_gold_same_file,
    assign_diagnosis_label,
    extract_competitors_above_gold,
)
from root_rag.evaluation.semantic_v1 import load_corpus, load_qrels, load_queries
from root_rag.retrieval.backends import build_retrieval_backend
from root_rag.retrieval.pipeline import RetrievalPipeline
from root_rag.retrieval.transformers import build_query_transformer


def _load_corpus_by_id(rows: List[dict]) -> Dict[str, dict]:
    return {row["chunk_id"]: row for row in rows}


def _load_query_map(queries_path: Path) -> Dict[str, dict]:
    queries = load_queries(queries_path)
    return {
        row.query_id: {
            "id": row.query_id,
            "query": row.query,
            "query_class": row.query_class,
        }
        for row in queries
    }


def _render_markdown(report: dict) -> str:
    lines = [
        f"# Query Competition Diagnostic: {report['query_id']}",
        "",
        f"- Query: `{report['query_text']}`",
        f"- Class: `{report['query_class']}`",
        f"- Gold chunks: {', '.join(report['gold_chunk_ids'])}",
        f"- Diagnosis label: `{report['diagnosis_label']}`",
        "",
        "## Gold Split Signal",
        "",
        f"- split_gold_same_file: `{report['split_gold_same_file']['is_split_gold_same_file']}`",
    ]
    repeated = report["split_gold_same_file"]["repeated_gold_file_counts"]
    if repeated:
        lines.append(f"- repeated_gold_file_counts: `{json.dumps(repeated, sort_keys=True)}`")
    else:
        lines.append("- repeated_gold_file_counts: `{}`")

    lines.extend(["", "## Mode Diagnostics", ""])
    for mode_name in ("bm25", "semantic", "hybrid"):
        mode = report["modes"][mode_name]
        lines.extend([f"### {mode_name}", ""])
        if not mode["present"]:
            lines.append("- not present")
            lines.append("")
            continue
        lines.append(f"- backend: `{mode['backend']}`")
        lines.append(f"- best_gold_rank: `{mode['best_gold_rank']}`")
        lines.append(f"- best_gold_score: `{mode['best_gold_score']}`")
        lines.append(f"- top_minus_best_gold: `{mode['top_minus_best_gold']}`")
        lines.append(f"- gold_rank_positions: `{json.dumps(mode['gold_rank_positions'], sort_keys=True)}`")
        lines.append("- gold presence detail:")
        for gold_chunk_id in report["gold_chunk_ids"]:
            presence = mode.get("gold_presence", {}).get(gold_chunk_id, {})
            lines.append(
                "  - "
                f"{gold_chunk_id}: present={presence.get('present')}, "
                f"rank={presence.get('rank')}, score={presence.get('score')}, "
                f"prev={presence.get('prev_competitor_chunk_id')}, "
                f"gap_to_prev={presence.get('gap_to_prev_competitor')}"
            )
        lines.append("- top competitors above gold:")
        competitors = mode["competitors_above_gold"]
        if not competitors:
            lines.append("  - (none)")
        else:
            for comp in competitors:
                lines.append(
                    "  - "
                    f"{comp['chunk_id']} "
                    f"(rank={comp['rank']}, score={comp['score']:.6f}, "
                    f"same_file_as_gold={comp['same_file_as_any_gold']}, "
                    f"delta_vs_best_gold={comp['delta_vs_best_gold_score']})"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-id", default="br006", help="Benchmark query id to diagnose.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/corpus.jsonl"),
        help="Canonical corpus jsonl path.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("configs/benchmark_queries_semantic.json"),
        help="Benchmark queries path.",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("configs/benchmark_qrels_semantic.jsonl"),
        help="Benchmark qrels path.",
    )
    parser.add_argument(
        "--semantic-manifest",
        type=Path,
        default=Path("artifacts/semantic_v1_2/semantic_manifest.json"),
        help="Semantic manifest path. If missing, semantic/hybrid diagnostics are omitted.",
    )
    parser.add_argument(
        "--semantic-model",
        default="intfloat/e5-base-v2",
        help="Semantic model for semantic/hybrid backends.",
    )
    parser.add_argument(
        "--query-mode",
        default="baseline",
        choices=["baseline", "lexnorm"],
        help="Query transformer mode; default baseline to match benchmark.",
    )
    parser.add_argument("--search-depth", type=int, default=50, help="Search depth per mode.")
    parser.add_argument("--max-competitors", type=int, default=5, help="Max competitors per mode.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/diagnostics/br006_competition_diagnostic.json"),
        help="Output JSON artifact path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/diagnostics/br006_competition_diagnostic.md"),
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    corpus_rows = load_corpus(args.corpus)
    corpus_by_id = _load_corpus_by_id(corpus_rows)
    query_map = _load_query_map(args.queries)
    qrels_map = load_qrels(args.qrels)

    if args.query_id not in query_map:
        raise SystemExit(f"query_id not found in queries: {args.query_id}")
    if args.query_id not in qrels_map:
        raise SystemExit(f"query_id not found in qrels: {args.query_id}")

    query_entry = query_map[args.query_id]
    gold_chunk_ids = sorted(qrels_map[args.query_id].keys())
    split_info = analyze_split_gold_same_file(gold_chunk_ids=gold_chunk_ids, corpus_by_id=corpus_by_id)
    query_transformer = build_query_transformer(args.query_mode)

    modes: Dict[str, dict] = {
        "bm25": {"present": True, "backend": "lexical_bm25_memory"},
        "semantic": {"present": False, "backend": None},
        "hybrid": {"present": False, "backend": None},
    }

    bm25_backend = build_retrieval_backend(
        "bm25_only",
        corpus_rows=corpus_rows,
        corpus_artifact_path=args.corpus,
        k1=1.5,
        b=0.75,
        dense_dim=512,
    )
    bm25_pipeline = RetrievalPipeline(backend=bm25_backend, query_transformer=query_transformer)
    bm25_results = bm25_pipeline.search(query_entry["query"], top_k=args.search_depth)
    modes["bm25"].update(
        extract_competitors_above_gold(
            mode_results=bm25_results,
            gold_chunk_ids=gold_chunk_ids,
            corpus_by_id=corpus_by_id,
            max_competitors=args.max_competitors,
        )
    )

    if args.semantic_manifest.exists():
        semantic_backend = build_retrieval_backend(
            "semantic_only",
            corpus_rows=corpus_rows,
            corpus_artifact_path=args.corpus,
            semantic_manifest_path=args.semantic_manifest,
            semantic_model_name=args.semantic_model,
            k1=1.5,
            b=0.75,
            dense_dim=512,
        )
        semantic_pipeline = RetrievalPipeline(backend=semantic_backend, query_transformer=query_transformer)
        semantic_results = semantic_pipeline.search(query_entry["query"], top_k=args.search_depth)
        modes["semantic"] = {
            "present": True,
            "backend": semantic_backend.backend_id,
            **extract_competitors_above_gold(
                mode_results=semantic_results,
                gold_chunk_ids=gold_chunk_ids,
                corpus_by_id=corpus_by_id,
                max_competitors=args.max_competitors,
            ),
        }

        hybrid_backend = build_retrieval_backend(
            "hybrid",
            corpus_rows=corpus_rows,
            corpus_artifact_path=args.corpus,
            semantic_manifest_path=args.semantic_manifest,
            semantic_model_name=args.semantic_model,
            k1=1.5,
            b=0.75,
            dense_dim=512,
        )
        hybrid_pipeline = RetrievalPipeline(backend=hybrid_backend, query_transformer=query_transformer)
        hybrid_results = hybrid_pipeline.search(query_entry["query"], top_k=args.search_depth)
        modes["hybrid"] = {
            "present": True,
            "backend": hybrid_backend.backend_id,
            **extract_competitors_above_gold(
                mode_results=hybrid_results,
                gold_chunk_ids=gold_chunk_ids,
                corpus_by_id=corpus_by_id,
                max_competitors=args.max_competitors,
            ),
        }

    diagnosis_label = assign_diagnosis_label(
        per_mode=modes,
        split_gold_same_file=split_info["is_split_gold_same_file"],
    )
    if diagnosis_label not in DIAGNOSIS_LABELS:
        raise RuntimeError(f"unexpected diagnosis label: {diagnosis_label}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query_id": args.query_id,
        "query_text": query_entry["query"],
        "query_class": query_entry["query_class"],
        "query_mode": args.query_mode,
        "search_depth": args.search_depth,
        "gold_chunk_ids": gold_chunk_ids,
        "split_gold_same_file": split_info,
        "modes": modes,
        "diagnosis_label": diagnosis_label,
        "paths": {
            "corpus": str(args.corpus),
            "queries": str(args.queries),
            "qrels": str(args.qrels),
            "semantic_manifest": str(args.semantic_manifest) if args.semantic_manifest.exists() else None,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"[diagnostic] query_id={args.query_id}")
    print(f"[diagnostic] diagnosis_label={diagnosis_label}")
    print(f"[diagnostic] output_json={args.output_json}")
    print(f"[diagnostic] output_md={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
