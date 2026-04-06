#!/usr/bin/env python3
"""Run official benchmark/audit tracks and generate comparison artifacts.

Tracks:
- B0: backend=lexical_bm25_memory, query_mode=baseline, top_k=10
- B1: backend=lexical_bm25_memory, query_mode=lexnorm, top_k=10
- S0: backend=semantic_hash_memory, query_mode=baseline, top_k=10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

from root_rag.evaluation.metrics import (
    TopKMetrics,
    aggregate_topk_metrics,
    classify_effect,
    compute_topk_metrics,
)
from root_rag.retrieval.backends import build_retrieval_backend
from root_rag.retrieval.pipeline import RetrievalPipeline
from root_rag.retrieval.transformers import build_query_transformer

TOP_K = 10
TRACKS: Dict[str, Dict[str, str]] = {
    "B0": {"backend": "lexical_bm25_memory", "query_mode": "baseline"},
    "B1": {"backend": "lexical_bm25_memory", "query_mode": "lexnorm"},
    "S0": {"backend": "semantic_hash_memory", "query_mode": "baseline"},
}
METRIC_KEYS = ("mrr_at_k", "recall_at_k", "ndcg_at_k")
EPS = 1e-12
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _run_cmd(cmd: List[str], *, py_path: str) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = py_path if not existing else f"{py_path}{os.pathsep}{existing}"
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def _metric_triplet(query_row: dict) -> dict:
    return {k: query_row[k] for k in METRIC_KEYS}


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _load_queries(path: Path) -> List[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return sorted(rows, key=lambda row: row["id"])


def _load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qrels[row["query_id"]][row["chunk_id"]] = int(row["relevance"])
    return dict(qrels)


def _load_corpus(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _compute_latency_summary_ms(samples_s: List[float]) -> dict:
    if not samples_s:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}

    sorted_samples = sorted(samples_s)
    count = len(sorted_samples)

    def percentile(p: float) -> float:
        if count == 1:
            return sorted_samples[0]
        rank = (count - 1) * p
        low = int(rank)
        high = min(low + 1, count - 1)
        frac = rank - low
        return sorted_samples[low] * (1.0 - frac) + sorted_samples[high] * frac

    to_ms = lambda value: value * 1000.0
    mean_ms = to_ms(sum(sorted_samples) / count)
    return {
        "count": count,
        "mean": mean_ms,
        "p50": to_ms(percentile(0.50)),
        "p95": to_ms(percentile(0.95)),
        "min": to_ms(sorted_samples[0]),
        "max": to_ms(sorted_samples[-1]),
    }


def _run_eval_track(
    *,
    backend_name: str,
    query_mode: str,
    corpus_rows: List[dict],
    corpus_path: Path,
    queries: List[dict],
    qrels_map: Dict[str, Dict[str, int]],
    semantic_manifest_path: Optional[Path] = None,
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    backend = build_retrieval_backend(
        backend_name,
        corpus_rows=corpus_rows,
        corpus_artifact_path=corpus_path,
        semantic_manifest_path=semantic_manifest_path,
        semantic_model_name=semantic_model_name,
        k1=1.5,
        b=0.75,
        dense_dim=512,
    )
    transformer = build_query_transformer(query_mode)
    pipeline = RetrievalPipeline(
        backend=backend,
        query_transformer=transformer,
    )

    per_query_rows = []
    metric_rows: List[TopKMetrics] = []
    latency_samples_s: List[float] = []

    for query_entry in queries:
        query_id = query_entry["id"]
        query_text = query_entry["query"]
        query_class = query_entry["query_class"]
        transformed_query = transformer.transform(query_text)

        t0 = perf_counter()
        scored = pipeline.search(query_text, top_k=TOP_K)
        latency_samples_s.append(perf_counter() - t0)

        ranked_chunk_ids = [row.chunk_id for row in scored]
        ranked_scores = [row.score for row in scored]
        relevance_map = qrels_map.get(query_id, {})
        metrics = compute_topk_metrics(
            ranked_chunk_ids,
            relevance_map,
            top_k=TOP_K,
            qrels_positive_count=len(relevance_map),
        )
        metric_rows.append(metrics)
        per_query_rows.append(
            {
                "id": query_id,
                "query": query_text,
                "query_class": query_class,
                "query_tokens": _tokenize(transformed_query),
                "mrr_at_k": metrics.mrr_at_k,
                "recall_at_k": metrics.recall_at_k,
                "ndcg_at_k": metrics.ndcg_at_k,
                "qrels_positive_count": metrics.qrels_positive_count,
                "retrieved_positive_count": metrics.retrieved_positive_count,
                "top10": ranked_chunk_ids[:TOP_K],
                "top10_scores": ranked_scores[:TOP_K],
            }
        )

    summary = aggregate_topk_metrics(metric_rows)

    by_class: Dict[str, List[dict]] = defaultdict(list)
    for row in per_query_rows:
        by_class[row["query_class"]].append(row)

    per_class = {}
    for query_class, rows in sorted(by_class.items()):
        class_metric_rows = [
            TopKMetrics(
                mrr_at_k=row["mrr_at_k"],
                recall_at_k=row["recall_at_k"],
                ndcg_at_k=row["ndcg_at_k"],
                retrieved_positive_count=row["retrieved_positive_count"],
                qrels_positive_count=row["qrels_positive_count"],
            )
            for row in rows
        ]
        per_class[query_class] = {
            "query_count": len(rows),
            "metrics": aggregate_topk_metrics(class_metric_rows),
        }

    operational = {
        "backend_id": backend.backend_id,
        "backend_metrics": backend.operational_metrics(),
        "query_latency_ms": _compute_latency_summary_ms(latency_samples_s),
    }
    backend_metrics = operational["backend_metrics"]
    bm25_meta = None
    if operational["backend_id"] == "lexical_bm25_memory":
        bm25_meta = {
            "k1": backend_metrics.get("k1", 1.5),
            "b": backend_metrics.get("b", 0.75),
            "avgdl": backend_metrics.get("avgdl"),
            "docs": int(backend_metrics["docs"]) if backend_metrics.get("docs") is not None else None,
        }

    output = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "top_k": TOP_K,
            "query_mode": query_mode,
            "backend": operational["backend_id"],
            "inputs": {
                "queries": "configs/benchmark_queries.json",
                "qrels": "configs/benchmark_qrels.jsonl",
                "corpus": str(corpus_path),
            },
        },
        "operational": operational,
        "summary": summary,
        "per_class": per_class,
        "per_query": sorted(per_query_rows, key=lambda row: row["id"]),
    }
    if bm25_meta is not None:
        output["metadata"]["bm25"] = bm25_meta
    if operational["backend_id"] in {"dense_hash_memory", "semantic_hash_memory", "semantic_faiss", "hybrid_s1"}:
        output["metadata"]["dense"] = {
            "vector_dim": backend_metrics.get("vector_dim"),
            "similarity": backend_metrics.get("similarity"),
            "avg_nonzero_dims": backend_metrics.get("avg_nonzero_dims"),
        }
    return output


def _recommendation(
    *,
    before_summary: dict,
    after_summary: dict,
    helped: List[str],
    hurt: List[str],
) -> str:
    after_not_worse = all(after_summary[m] >= before_summary[m] - EPS for m in METRIC_KEYS)
    before_not_worse = all(before_summary[m] >= after_summary[m] - EPS for m in METRIC_KEYS)
    after_strictly_better = any(after_summary[m] > before_summary[m] + EPS for m in METRIC_KEYS)
    before_strictly_better = any(before_summary[m] > after_summary[m] + EPS for m in METRIC_KEYS)

    if after_not_worse and after_strictly_better and not hurt:
        return "keep candidate"
    if before_not_worse and before_strictly_better and not helped:
        return "keep baseline"
    return "support both"


def _build_comparison_markdown(
    *,
    before_eval: dict,
    after_eval: dict,
    before_label: str,
    after_label: str,
    title: str,
    output_path: Path,
) -> None:
    before_by_id = {row["id"]: row for row in before_eval["per_query"]}
    after_by_id = {row["id"]: row for row in after_eval["per_query"]}
    query_ids = sorted(before_by_id.keys())

    helped: List[dict] = []
    hurt: List[dict] = []
    unchanged: List[str] = []

    interpretation_changed: List[str] = []
    ranking_changed: List[str] = []
    metric_changed: List[str] = []
    metric_changed_without_interpretation: List[str] = []

    for query_id in query_ids:
        before_row = before_by_id[query_id]
        after_row = after_by_id[query_id]
        before = _metric_triplet(before_row)
        after = _metric_triplet(after_row)
        effect = classify_effect(before, after)
        delta = {k: after[k] - before[k] for k in METRIC_KEYS}

        changed_interpretation = before_row["query_tokens"] != after_row["query_tokens"]
        changed_ranking = before_row["top10"] != after_row["top10"]
        changed_metrics = any(abs(delta[m]) > EPS for m in METRIC_KEYS)

        if changed_interpretation:
            interpretation_changed.append(query_id)
        if changed_ranking:
            ranking_changed.append(query_id)
        if changed_metrics:
            metric_changed.append(query_id)
            if not changed_interpretation:
                metric_changed_without_interpretation.append(query_id)

        row = {
            "id": query_id,
            "query_class": before_row["query_class"],
            "delta": delta,
        }
        if effect == "helped":
            helped.append(row)
        elif effect == "hurt":
            hurt.append(row)
        else:
            unchanged.append(query_id)

    helped.sort(key=lambda row: row["id"])
    hurt.sort(key=lambda row: row["id"])
    unchanged.sort()

    before_zero = sorted(row["id"] for row in before_eval["per_query"] if row["recall_at_k"] == 0.0)
    after_zero = sorted(row["id"] for row in after_eval["per_query"] if row["recall_at_k"] == 0.0)
    resolved_zero = sorted(set(before_zero) - set(after_zero))
    new_zero = sorted(set(after_zero) - set(before_zero))

    recommendation = _recommendation(
        before_summary=before_eval["summary"],
        after_summary=after_eval["summary"],
        helped=[row["id"] for row in helped],
        hurt=[row["id"] for row in hurt],
    )

    if metric_changed and not metric_changed_without_interpretation:
        interpretation_statement = (
            "Observed metric changes are explained by query interpretation changes "
            "(lexnorm altered transformed query tokens for the changed queries)."
        )
    elif not metric_changed:
        interpretation_statement = "No ranking-quality changes observed between B0 and B1."
    else:
        interpretation_statement = (
            "Some metric changes occurred without transformed-token changes, indicating "
            "ranking differences beyond query interpretation."
        )

    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Official tracks:",
        f"- {before_label}: backend=`{before_eval['metadata']['backend']}`, query_mode=`{before_eval['metadata']['query_mode']}`, top_k={TOP_K}",
        f"- {after_label}: backend=`{after_eval['metadata']['backend']}`, query_mode=`{after_eval['metadata']['query_mode']}`, top_k={TOP_K}",
        "",
        f"## Global Metric Deltas ({after_label} - {before_label})",
        "",
        f"| Metric | {before_label} | {after_label} | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]

    for metric in METRIC_KEYS:
        before_v = before_eval["summary"][metric]
        after_v = after_eval["summary"][metric]
        lines.append(f"| {metric} | {before_v:.4f} | {after_v:.4f} | {_fmt_delta(after_v - before_v)} |")

    lines.extend(
        [
            "",
            f"## Per-Class Deltas ({after_label} - {before_label})",
            "",
            f"| Query Class | Metric | {before_label} | {after_label} | Delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for query_class in sorted(before_eval["per_class"].keys()):
        before_metrics = before_eval["per_class"][query_class]["metrics"]
        after_metrics = after_eval["per_class"][query_class]["metrics"]
        for metric in METRIC_KEYS:
            before_v = before_metrics[metric]
            after_v = after_metrics[metric]
            lines.append(f"| {query_class} | {metric} | {before_v:.4f} | {after_v:.4f} | {_fmt_delta(after_v - before_v)} |")

    lines.extend(
        [
            "",
            "## Zero-Recall Queries",
            "",
            f"- {before_label} ({len(before_zero)}): {', '.join(before_zero) if before_zero else '(none)'}",
            f"- {after_label} ({len(after_zero)}): {', '.join(after_zero) if after_zero else '(none)'}",
            f"- Resolved by {after_label} ({len(resolved_zero)}): {', '.join(resolved_zero) if resolved_zero else '(none)'}",
            f"- New zero-recall under {after_label} ({len(new_zero)}): {', '.join(new_zero) if new_zero else '(none)'}",
            "",
            f"## Queries Helped by {after_label}",
            "",
            f"Count: {len(helped)}",
        ]
    )
    if helped:
        lines.extend(
            [
                "",
                "| Query ID | Class | dMRR | dRecall | dNDCG |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in helped:
            lines.append(
                f"| {row['id']} | {row['query_class']} | {_fmt_delta(row['delta']['mrr_at_k'])} | "
                f"{_fmt_delta(row['delta']['recall_at_k'])} | {_fmt_delta(row['delta']['ndcg_at_k'])} |"
            )
    else:
        lines.append("- (none)")

    lines.extend(
        [
            "",
            f"## Queries Hurt by {after_label}",
            "",
            f"Count: {len(hurt)}",
        ]
    )
    if hurt:
        lines.extend(
            [
                "",
                "| Query ID | Class | dMRR | dRecall | dNDCG |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in hurt:
            lines.append(
                f"| {row['id']} | {row['query_class']} | {_fmt_delta(row['delta']['mrr_at_k'])} | "
                f"{_fmt_delta(row['delta']['recall_at_k'])} | {_fmt_delta(row['delta']['ndcg_at_k'])} |"
            )
    else:
        lines.append("- (none)")

    lines.extend(
        [
            "",
            "## Interpretation vs Ranking",
            "",
            f"- Queries with transformed-query token changes: {len(interpretation_changed)} / {len(query_ids)}",
            f"- Queries with top-10 ranking changes: {len(ranking_changed)} / {len(query_ids)}",
            f"- Queries with metric changes: {len(metric_changed)} / {len(query_ids)}",
            (
                "- Metric-changed queries without transformed-token changes: "
                f"{len(metric_changed_without_interpretation)}"
            ),
            f"- Assessment: {interpretation_statement}",
            "",
            "## Recommendation",
            "",
            f"Recommendation: **{recommendation}**",
            "",
            "Rationale:",
            (
                f"- Helped={len(helped)}, Hurt={len(hurt)}, Unchanged={len(unchanged)} "
                "across fixed backend/top_k."
            ),
            (
                f"- Global deltas: dMRR={_fmt_delta(after_eval['summary']['mrr_at_k'] - before_eval['summary']['mrr_at_k'])}, "
                f"dRecall={_fmt_delta(after_eval['summary']['recall_at_k'] - before_eval['summary']['recall_at_k'])}, "
                f"dNDCG={_fmt_delta(after_eval['summary']['ndcg_at_k'] - before_eval['summary']['ndcg_at_k'])}."
            ),
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_manual_zero_recall_template(*, b0_eval: dict, output_path: Path) -> None:
    zero_recall_ids = sorted(row["id"] for row in b0_eval["per_query"] if row["recall_at_k"] == 0.0)
    lines = [
        "# Manual Zero-Recall Review Template",
        "",
        "Reference set: current B0 track zero-recall queries.",
        "",
        "Allowed values:",
        "- `qrel_correctness`: yes / maybe / no",
        "- `top1_relevance`: no / partial / yes",
        "- `expected_file_type`: header / source / either",
        "- `chunk_vs_file_issue`: yes / no",
        "",
        "| query_id | qrel_correctness | top1_relevance | expected_file_type | chunk_vs_file_issue | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if zero_recall_ids:
        for query_id in zero_recall_ids:
            lines.append(f"| {query_id} |  |  |  |  |  |")
    else:
        lines.append("| (none) |  |  |  |  |  |")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_queries_snapshot(path: Path, queries: List[dict]) -> None:
    path.write_text(json.dumps(queries, indent=2), encoding="utf-8")


def _write_qrels_snapshot(path: Path, qrels_map: Dict[str, Dict[str, int]]) -> None:
    lines = []
    for query_id in sorted(qrels_map.keys()):
        for chunk_id, relevance in sorted(qrels_map[query_id].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(json.dumps({"query_id": query_id, "chunk_id": chunk_id, "relevance": relevance}))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/corpus.jsonl"),
        help="Corpus JSONL path.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("configs/benchmark_queries.json"),
        help="Benchmark queries JSON path.",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("configs/benchmark_qrels.jsonl"),
        help="Benchmark qrels JSONL path.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Artifacts output directory.",
    )
    parser.add_argument(
        "--s1-semantic-manifest",
        type=Path,
        default=None,
        help="Optional semantic manifest to enable S1 benchmark track.",
    )
    parser.add_argument(
        "--s1-semantic-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Local embedding model name used for S1 semantic/hybrid benchmark queries.",
    )
    args = parser.parse_args()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    py_path = str((Path(__file__).resolve().parents[1] / "src"))

    corpus_rows = _load_corpus(args.corpus)
    queries = _load_queries(args.queries)
    qrels_map = _load_qrels(args.qrels)

    query_snapshot_path = args.artifacts_dir / "benchmark_queries_reconstructed.json"
    qrels_snapshot_path = args.artifacts_dir / "benchmark_qrels_reconstructed.jsonl"
    _write_queries_snapshot(query_snapshot_path, queries)
    _write_qrels_snapshot(qrels_snapshot_path, qrels_map)

    tracks = dict(TRACKS)
    if args.s1_semantic_manifest is not None:
        tracks["S1"] = {"backend": "hybrid_s1", "query_mode": "baseline"}

    for track, config in tracks.items():
        eval_path = args.artifacts_dir / f"benchmark_eval_results_{track}.json"
        eval_output = _run_eval_track(
            backend_name=config["backend"],
            query_mode=config["query_mode"],
            corpus_rows=corpus_rows,
            corpus_path=args.corpus,
            queries=queries,
            qrels_map=qrels_map,
            semantic_manifest_path=args.s1_semantic_manifest if track == "S1" else None,
            semantic_model_name=args.s1_semantic_model,
        )
        eval_path.write_text(json.dumps(eval_output, indent=2), encoding="utf-8")
        print(f"Wrote evaluation: {eval_path}")

    for track, config in tracks.items():
        audit_json = args.artifacts_dir / f"benchmark_failure_audit_{track}.json"
        audit_md = args.artifacts_dir / f"benchmark_failure_audit_{track}.md"
        cmd = [
            sys.executable,
            "scripts/audit_benchmark_failures.py",
            "--corpus",
            str(args.corpus),
            "--queries",
            str(query_snapshot_path),
            "--qrels",
            str(qrels_snapshot_path),
            "--output-json",
            str(audit_json),
            "--output-md",
            str(audit_md),
            "--top-k",
            str(TOP_K),
            "--backend",
            config["backend"],
            "--query-mode",
            config["query_mode"],
        ]
        if track == "S1" and args.s1_semantic_manifest is not None:
            cmd.extend(
                [
                    "--semantic-manifest",
                    str(args.s1_semantic_manifest),
                    "--semantic-model",
                    args.s1_semantic_model,
                ]
            )
        _run_cmd(cmd, py_path=py_path)

    b0_eval = json.loads((args.artifacts_dir / "benchmark_eval_results_B0.json").read_text(encoding="utf-8"))
    b1_eval = json.loads((args.artifacts_dir / "benchmark_eval_results_B1.json").read_text(encoding="utf-8"))
    s0_eval = json.loads((args.artifacts_dir / "benchmark_eval_results_S0.json").read_text(encoding="utf-8"))

    comparison_path = args.artifacts_dir / "benchmark_mode_comparison.md"
    _build_comparison_markdown(
        before_eval=b0_eval,
        after_eval=b1_eval,
        before_label="B0",
        after_label="B1",
        title="Benchmark Mode Comparison (B0 vs B1)",
        output_path=comparison_path,
    )
    semantic_comparison_path = args.artifacts_dir / "benchmark_semantic_comparison.md"
    _build_comparison_markdown(
        before_eval=b0_eval,
        after_eval=s0_eval,
        before_label="B0",
        after_label="S0",
        title="Benchmark Semantic Comparison (B0 vs S0)",
        output_path=semantic_comparison_path,
    )
    if args.s1_semantic_manifest is not None:
        s1_eval = json.loads((args.artifacts_dir / "benchmark_eval_results_S1.json").read_text(encoding="utf-8"))
        semantic_comparison_s1_path = args.artifacts_dir / "benchmark_semantic_comparison_S1.md"
        _build_comparison_markdown(
            before_eval=b0_eval,
            after_eval=s1_eval,
            before_label="B0",
            after_label="S1",
            title="Benchmark Semantic Comparison (B0 vs S1)",
            output_path=semantic_comparison_s1_path,
        )

    manual_template_path = args.artifacts_dir / "manual_zero_recall_review_template.md"
    _write_manual_zero_recall_template(
        b0_eval=b0_eval,
        output_path=manual_template_path,
    )

    print(f"Wrote comparison: {comparison_path}")
    print(f"Wrote semantic comparison: {semantic_comparison_path}")
    if args.s1_semantic_manifest is not None:
        print(f"Wrote semantic comparison: {args.artifacts_dir / 'benchmark_semantic_comparison_S1.md'}")
    print(f"Wrote manual review template: {manual_template_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
