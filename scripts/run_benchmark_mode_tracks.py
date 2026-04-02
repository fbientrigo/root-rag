#!/usr/bin/env python3
"""Run official benchmark/audit tracks and generate B0 vs B1 comparison artifacts.

Tracks:
- B0: backend=lexical_bm25_memory, query_mode=baseline, top_k=10
- B1: backend=lexical_bm25_memory, query_mode=lexnorm, top_k=10
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
from typing import Dict, List

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
BACKEND = "lexical_bm25_memory"
TRACKS: Dict[str, str] = {
    "B0": "baseline",
    "B1": "lexnorm",
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
    query_mode: str,
    corpus_rows: List[dict],
    corpus_path: Path,
    queries: List[dict],
    qrels_map: Dict[str, Dict[str, int]],
) -> dict:
    backend = build_retrieval_backend(
        BACKEND,
        corpus_rows=corpus_rows,
        corpus_artifact_path=corpus_path,
        k1=1.5,
        b=0.75,
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
    bm25_meta = {
        "k1": backend_metrics.get("k1", 1.5),
        "b": backend_metrics.get("b", 0.75),
        "avgdl": backend_metrics.get("avgdl"),
        "docs": int(backend_metrics["docs"]) if backend_metrics.get("docs") is not None else None,
    }

    return {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "top_k": TOP_K,
            "query_mode": query_mode,
            "backend": operational["backend_id"],
            "bm25": bm25_meta,
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


def _recommendation(
    *,
    b0_summary: dict,
    b1_summary: dict,
    helped: List[str],
    hurt: List[str],
) -> str:
    b1_not_worse = all(b1_summary[m] >= b0_summary[m] - EPS for m in METRIC_KEYS)
    b0_not_worse = all(b0_summary[m] >= b1_summary[m] - EPS for m in METRIC_KEYS)
    b1_strictly_better = any(b1_summary[m] > b0_summary[m] + EPS for m in METRIC_KEYS)
    b0_strictly_better = any(b0_summary[m] > b1_summary[m] + EPS for m in METRIC_KEYS)

    if b1_not_worse and b1_strictly_better and not hurt:
        return "keep lexnorm"
    if b0_not_worse and b0_strictly_better and not helped:
        return "keep baseline"
    return "support both"


def _build_comparison_markdown(
    *,
    b0_eval: dict,
    b1_eval: dict,
    output_path: Path,
) -> None:
    b0_by_id = {row["id"]: row for row in b0_eval["per_query"]}
    b1_by_id = {row["id"]: row for row in b1_eval["per_query"]}
    query_ids = sorted(b0_by_id.keys())

    helped: List[dict] = []
    hurt: List[dict] = []
    unchanged: List[str] = []

    interpretation_changed: List[str] = []
    ranking_changed: List[str] = []
    metric_changed: List[str] = []
    metric_changed_without_interpretation: List[str] = []

    for query_id in query_ids:
        b0_row = b0_by_id[query_id]
        b1_row = b1_by_id[query_id]
        before = _metric_triplet(b0_row)
        after = _metric_triplet(b1_row)
        effect = classify_effect(before, after)
        delta = {k: after[k] - before[k] for k in METRIC_KEYS}

        changed_interpretation = b0_row["query_tokens"] != b1_row["query_tokens"]
        changed_ranking = b0_row["top10"] != b1_row["top10"]
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
            "query_class": b0_row["query_class"],
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

    b0_zero = sorted(row["id"] for row in b0_eval["per_query"] if row["recall_at_k"] == 0.0)
    b1_zero = sorted(row["id"] for row in b1_eval["per_query"] if row["recall_at_k"] == 0.0)
    resolved_zero = sorted(set(b0_zero) - set(b1_zero))
    new_zero = sorted(set(b1_zero) - set(b0_zero))

    recommendation = _recommendation(
        b0_summary=b0_eval["summary"],
        b1_summary=b1_eval["summary"],
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
        "# Benchmark Mode Comparison (B0 vs B1)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Official tracks:",
        f"- B0: backend=`{BACKEND}`, query_mode=`baseline`, top_k={TOP_K}",
        f"- B1: backend=`{BACKEND}`, query_mode=`lexnorm`, top_k={TOP_K}",
        "",
        "## Global Metric Deltas (B1 - B0)",
        "",
        "| Metric | B0 | B1 | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]

    for metric in METRIC_KEYS:
        b0_v = b0_eval["summary"][metric]
        b1_v = b1_eval["summary"][metric]
        lines.append(f"| {metric} | {b0_v:.4f} | {b1_v:.4f} | {_fmt_delta(b1_v - b0_v)} |")

    lines.extend(
        [
            "",
            "## Per-Class Deltas (B1 - B0)",
            "",
            "| Query Class | Metric | B0 | B1 | Delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for query_class in sorted(b0_eval["per_class"].keys()):
        b0_metrics = b0_eval["per_class"][query_class]["metrics"]
        b1_metrics = b1_eval["per_class"][query_class]["metrics"]
        for metric in METRIC_KEYS:
            b0_v = b0_metrics[metric]
            b1_v = b1_metrics[metric]
            lines.append(f"| {query_class} | {metric} | {b0_v:.4f} | {b1_v:.4f} | {_fmt_delta(b1_v - b0_v)} |")

    lines.extend(
        [
            "",
            "## Zero-Recall Queries",
            "",
            f"- B0 ({len(b0_zero)}): {', '.join(b0_zero) if b0_zero else '(none)'}",
            f"- B1 ({len(b1_zero)}): {', '.join(b1_zero) if b1_zero else '(none)'}",
            f"- Resolved by lexnorm ({len(resolved_zero)}): {', '.join(resolved_zero) if resolved_zero else '(none)'}",
            f"- New zero-recall under lexnorm ({len(new_zero)}): {', '.join(new_zero) if new_zero else '(none)'}",
            "",
            "## Queries Helped by Lexnorm",
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
            "## Queries Hurt by Lexnorm",
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
                f"- Global deltas: dMRR={_fmt_delta(b1_eval['summary']['mrr_at_k'] - b0_eval['summary']['mrr_at_k'])}, "
                f"dRecall={_fmt_delta(b1_eval['summary']['recall_at_k'] - b0_eval['summary']['recall_at_k'])}, "
                f"dNDCG={_fmt_delta(b1_eval['summary']['ndcg_at_k'] - b0_eval['summary']['ndcg_at_k'])}."
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

    for track, query_mode in TRACKS.items():
        eval_path = args.artifacts_dir / f"benchmark_eval_results_{track}.json"
        eval_output = _run_eval_track(
            query_mode=query_mode,
            corpus_rows=corpus_rows,
            corpus_path=args.corpus,
            queries=queries,
            qrels_map=qrels_map,
        )
        eval_path.write_text(json.dumps(eval_output, indent=2), encoding="utf-8")
        print(f"Wrote evaluation: {eval_path}")

    for track, query_mode in TRACKS.items():
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
            BACKEND,
            "--query-mode",
            query_mode,
        ]
        _run_cmd(cmd, py_path=py_path)

    b0_eval = json.loads((args.artifacts_dir / "benchmark_eval_results_B0.json").read_text(encoding="utf-8"))
    b1_eval = json.loads((args.artifacts_dir / "benchmark_eval_results_B1.json").read_text(encoding="utf-8"))

    comparison_path = args.artifacts_dir / "benchmark_mode_comparison.md"
    _build_comparison_markdown(
        b0_eval=b0_eval,
        b1_eval=b1_eval,
        output_path=comparison_path,
    )

    manual_template_path = args.artifacts_dir / "manual_zero_recall_review_template.md"
    _write_manual_zero_recall_template(
        b0_eval=b0_eval,
        output_path=manual_template_path,
    )

    print(f"Wrote comparison: {comparison_path}")
    print(f"Wrote manual review template: {manual_template_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
