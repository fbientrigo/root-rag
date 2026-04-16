"""Semantic retrieval V1 benchmark helpers for frozen canonical chunks."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence

from root_rag.evaluation.metrics import TopKMetrics, aggregate_topk_metrics, classify_effect, compute_topk_metrics
from root_rag.retrieval.backends import build_retrieval_backend
from root_rag.retrieval.pipeline import RetrievalPipeline
from root_rag.retrieval.s1_semantic import LocalEmbedder, SemanticIndexManifest
from root_rag.retrieval.transformers import build_query_transformer

TOP_K_DEFAULT = 10
MODE_TO_BACKEND = {
    "bm25_only": "bm25_only",
    "semantic_only": "semantic_only",
    "hybrid": "hybrid",
}
TARGET_CATEGORIES = ("semantic", "bridge-light")


@dataclass(frozen=True)
class QueryEntry:
    query_id: str
    query: str
    query_class: str


def load_queries(path: Path) -> List[QueryEntry]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        QueryEntry(
            query_id=row["id"],
            query=row["query"],
            query_class=row["query_class"],
        )
        for row in rows
    ]


def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qrels[row["query_id"]][row["chunk_id"]] = int(row["relevance"])
    return dict(qrels)


def load_corpus(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _compute_latency_summary_ms(samples_s: Sequence[float]) -> dict:
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

    def to_ms(value: float) -> float:
        return value * 1000.0

    return {
        "count": count,
        "mean": to_ms(sum(sorted_samples) / count),
        "p50": to_ms(percentile(0.50)),
        "p95": to_ms(percentile(0.95)),
        "min": to_ms(sorted_samples[0]),
        "max": to_ms(sorted_samples[-1]),
    }


def evaluate_mode(
    *,
    mode: str,
    corpus_rows: List[dict],
    corpus_path: Path,
    queries: List[QueryEntry],
    qrels_map: Dict[str, Dict[str, int]],
    top_k: int,
    semantic_manifest_path: Path,
    semantic_model_name: str,
    semantic_embedder: Optional[LocalEmbedder] = None,
) -> dict:
    backend = build_retrieval_backend(
        MODE_TO_BACKEND[mode],
        corpus_rows=corpus_rows,
        corpus_artifact_path=corpus_path,
        semantic_manifest_path=semantic_manifest_path if mode != "bm25_only" else None,
        semantic_model_name=semantic_model_name,
        semantic_embedder=semantic_embedder,
        k1=1.5,
        b=0.75,
        dense_dim=512,
    )
    pipeline = RetrievalPipeline(
        backend=backend,
        query_transformer=build_query_transformer("baseline"),
    )

    per_query: List[dict] = []
    metric_rows: List[TopKMetrics] = []
    latency_samples: List[float] = []

    for query_entry in queries:
        t0 = perf_counter()
        scored = pipeline.search(query_entry.query, top_k=top_k)
        latency_samples.append(perf_counter() - t0)

        ranked_chunk_ids = [row.chunk_id for row in scored]
        ranked_scores = [row.score for row in scored]
        relevance_map = qrels_map.get(query_entry.query_id, {})
        metrics = compute_topk_metrics(
            ranked_chunk_ids,
            relevance_map,
            top_k=top_k,
            qrels_positive_count=len(relevance_map),
        )
        metric_rows.append(metrics)
        per_query.append(
            {
                "id": query_entry.query_id,
                "query": query_entry.query,
                "query_class": query_entry.query_class,
                "mrr_at_k": metrics.mrr_at_k,
                "recall_at_k": metrics.recall_at_k,
                "ndcg_at_k": metrics.ndcg_at_k,
                "retrieved_positive_count": metrics.retrieved_positive_count,
                "qrels_positive_count": metrics.qrels_positive_count,
                "top10": ranked_chunk_ids[:top_k],
                "top10_scores": ranked_scores[:top_k],
            }
        )

    per_category: Dict[str, dict] = {}
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in per_query:
        grouped[row["query_class"]].append(row)

    for query_class, rows in sorted(grouped.items()):
        class_metrics = [
            TopKMetrics(
                mrr_at_k=row["mrr_at_k"],
                recall_at_k=row["recall_at_k"],
                ndcg_at_k=row["ndcg_at_k"],
                retrieved_positive_count=row["retrieved_positive_count"],
                qrels_positive_count=row["qrels_positive_count"],
            )
            for row in rows
        ]
        per_category[query_class] = {
            "query_count": len(rows),
            "metrics": aggregate_topk_metrics(class_metrics),
        }

    return {
        "mode": mode,
        "backend": backend.backend_id,
        "summary": aggregate_topk_metrics(metric_rows),
        "per_category": per_category,
        "per_query": sorted(per_query, key=lambda row: row["id"]),
        "operational": {
            "backend_id": backend.backend_id,
            "backend_metrics": backend.operational_metrics(),
            "query_latency_ms": _compute_latency_summary_ms(latency_samples),
        },
    }


def compare_mode_runs(before: dict, after: dict) -> dict:
    before_by_id = {row["id"]: row for row in before["per_query"]}
    after_by_id = {row["id"]: row for row in after["per_query"]}
    per_query: List[dict] = []
    helped: List[str] = []
    hurt: List[str] = []
    unchanged: List[str] = []

    for query_id in sorted(after_by_id.keys()):
        before_row = before_by_id[query_id]
        after_row = after_by_id[query_id]
        before_metrics = {
            "mrr_at_k": before_row["mrr_at_k"],
            "recall_at_k": before_row["recall_at_k"],
            "ndcg_at_k": before_row["ndcg_at_k"],
        }
        after_metrics = {
            "mrr_at_k": after_row["mrr_at_k"],
            "recall_at_k": after_row["recall_at_k"],
            "ndcg_at_k": after_row["ndcg_at_k"],
        }
        effect = classify_effect(before_metrics, after_metrics)
        if effect == "helped":
            helped.append(query_id)
        elif effect == "hurt":
            hurt.append(query_id)
        else:
            unchanged.append(query_id)
        per_query.append(
            {
                "id": query_id,
                "query": after_row["query"],
                "query_class": after_row["query_class"],
                "effect": effect,
                "before": before_metrics,
                "after": after_metrics,
                "delta": {
                    "mrr_at_k": after_metrics["mrr_at_k"] - before_metrics["mrr_at_k"],
                    "recall_at_k": after_metrics["recall_at_k"] - before_metrics["recall_at_k"],
                    "ndcg_at_k": after_metrics["ndcg_at_k"] - before_metrics["ndcg_at_k"],
                },
                "before_top10": before_row["top10"],
                "after_top10": after_row["top10"],
            }
        )

    per_category = {}
    for category, after_category in after["per_category"].items():
        before_category = before["per_category"][category]
        per_category[category] = {
            "query_count": after_category["query_count"],
            "before": before_category["metrics"],
            "after": after_category["metrics"],
            "delta": {
                "mrr_at_k": after_category["metrics"]["mrr_at_k"] - before_category["metrics"]["mrr_at_k"],
                "recall_at_k": after_category["metrics"]["recall_at_k"] - before_category["metrics"]["recall_at_k"],
                "ndcg_at_k": after_category["metrics"]["ndcg_at_k"] - before_category["metrics"]["ndcg_at_k"],
            },
        }

    return {
        "before_mode": before["mode"],
        "after_mode": after["mode"],
        "summary": {
            "before": before["summary"],
            "after": after["summary"],
            "delta": {
                "mrr_at_k": after["summary"]["mrr_at_k"] - before["summary"]["mrr_at_k"],
                "recall_at_k": after["summary"]["recall_at_k"] - before["summary"]["recall_at_k"],
                "ndcg_at_k": after["summary"]["ndcg_at_k"] - before["summary"]["ndcg_at_k"],
            },
        },
        "per_category": per_category,
        "per_query": per_query,
        "effects": {
            "helped": helped,
            "hurt": hurt,
            "unchanged": unchanged,
        },
    }


def _pick_recommendation(results: dict) -> str:
    bm25 = results["modes"]["bm25_only"]
    hybrid = results["modes"]["hybrid"]
    hybrid_comparison = results["comparisons"]["hybrid_vs_bm25_only"]

    lexical_delta = hybrid_comparison["per_category"].get("lexical-control", {}).get("delta", {})
    lexical_recall_drop = lexical_delta.get("recall_at_k", 0.0)
    lexical_mrr_drop = lexical_delta.get("mrr_at_k", 0.0)
    bridge_delta = hybrid_comparison["per_category"].get("bridge-light", {}).get("delta", {})
    semantic_delta = hybrid_comparison["per_category"].get("semantic", {}).get("delta", {})
    target_help = any(
        hybrid_comparison["per_category"].get(category, {}).get("delta", {}).get("ndcg_at_k", 0.0) > 0.0
        or hybrid_comparison["per_category"].get(category, {}).get("delta", {}).get("recall_at_k", 0.0) > 0.0
        for category in TARGET_CATEGORIES
    )

    if not bm25["per_query"] or not hybrid["per_query"]:
        return "BLOCKED"
    if not target_help and hybrid["summary"]["ndcg_at_k"] <= bm25["summary"]["ndcg_at_k"]:
        return "BLOCKED"
    if (
        target_help
        and lexical_recall_drop >= -0.05
        and lexical_mrr_drop >= -0.05
        and bridge_delta.get("ndcg_at_k", 0.0) >= -0.02
        and bridge_delta.get("recall_at_k", 0.0) >= -0.05
        and semantic_delta.get("ndcg_at_k", 0.0) >= 0.0
        and hybrid["summary"]["ndcg_at_k"] >= bm25["summary"]["ndcg_at_k"]
    ):
        return "ACCEPT"
    return "ACCEPT WITH NOTES"


def run_semantic_v1_benchmark(
    *,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    semantic_manifest_path: Path,
    semantic_model_name: str,
    top_k: int = TOP_K_DEFAULT,
    commands_run: Optional[List[str]] = None,
    semantic_embedder: Optional[LocalEmbedder] = None,
) -> dict:
    corpus_rows = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrels_map = load_qrels(qrels_path)

    modes = {
        mode: evaluate_mode(
            mode=mode,
            corpus_rows=corpus_rows,
            corpus_path=corpus_path,
            queries=queries,
            qrels_map=qrels_map,
            top_k=top_k,
            semantic_manifest_path=semantic_manifest_path,
            semantic_model_name=semantic_model_name,
            semantic_embedder=semantic_embedder,
        )
        for mode in ("bm25_only", "semantic_only", "hybrid")
    }

    semantic_manifest = SemanticIndexManifest.load(semantic_manifest_path)
    results = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "top_k": top_k,
            "corpus_path": str(corpus_path),
            "queries_path": str(queries_path),
            "qrels_path": str(qrels_path),
            "semantic_manifest_path": str(semantic_manifest_path),
            "semantic_manifest_jsonl_path": semantic_manifest.records_path,
            "semantic_index_path": semantic_manifest.index_path,
            "semantic_vectors_path": semantic_manifest.vectors_path,
            "semantic_model_name": semantic_model_name,
            "commands_run": commands_run or [],
        },
        "modes": modes,
        "comparisons": {
            "semantic_only_vs_bm25_only": compare_mode_runs(modes["bm25_only"], modes["semantic_only"]),
            "hybrid_vs_bm25_only": compare_mode_runs(modes["bm25_only"], modes["hybrid"]),
        },
    }
    results["recommendation"] = _pick_recommendation(results)
    return results


def _fmt_metric(value: float) -> str:
    return f"{value:.4f}"


def _fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def _compare_results_against_baseline(current: dict, baseline: dict) -> dict:
    deltas = {"overall": {}, "per_category": {}}
    for mode in ("bm25_only", "semantic_only", "hybrid"):
        current_summary = current["modes"][mode]["summary"]
        baseline_summary = baseline["modes"][mode]["summary"]
        deltas["overall"][mode] = {
            metric: current_summary[metric] - baseline_summary[metric]
            for metric in ("mrr_at_k", "recall_at_k", "ndcg_at_k")
        }

    categories = sorted(current["modes"]["bm25_only"]["per_category"].keys())
    for category in categories:
        deltas["per_category"][category] = {}
        for mode in ("bm25_only", "semantic_only", "hybrid"):
            current_metrics = current["modes"][mode]["per_category"][category]["metrics"]
            baseline_metrics = baseline["modes"][mode]["per_category"][category]["metrics"]
            deltas["per_category"][category][mode] = {
                metric: current_metrics[metric] - baseline_metrics[metric]
                for metric in ("mrr_at_k", "recall_at_k", "ndcg_at_k")
            }
    return deltas


def _representative_wins(comparison: dict, *, limit: int = 5) -> List[dict]:
    helped = [row for row in comparison["per_query"] if row["effect"] == "helped"]
    helped.sort(key=lambda row: (-row["delta"]["ndcg_at_k"], -row["delta"]["recall_at_k"], row["id"]))
    return helped[:limit]


def _representative_failures(hybrid_run: dict, comparison: dict, *, limit: int = 5) -> List[dict]:
    hurt_ids = set(comparison["effects"]["hurt"])
    failures = []
    for row in hybrid_run["per_query"]:
        if row["recall_at_k"] == 0.0 or row["id"] in hurt_ids:
            failures.append(row)
    failures.sort(key=lambda row: (row["recall_at_k"], row["ndcg_at_k"], row["id"]))
    return failures[:limit]


def render_semantic_v1_markdown(results: dict) -> str:
    hybrid_vs_bm25 = results["comparisons"]["hybrid_vs_bm25_only"]
    wins = _representative_wins(hybrid_vs_bm25)
    failures = _representative_failures(results["modes"]["hybrid"], hybrid_vs_bm25)
    baseline = results.get("baseline_comparison")
    baseline_deltas = _compare_results_against_baseline(results, baseline) if baseline else None
    report_title = results["metadata"].get("report_title", "Semantic Retrieval V1 Results")

    lines = [
        f"# {report_title}",
        "",
        f"Generated: {results['metadata']['date']}",
        "",
        "## Commands Run",
        "",
    ]
    commands = results["metadata"]["commands_run"]
    if commands:
        lines.extend(f"- `{command}`" for command in commands)
    else:
        lines.append("- Command capture unavailable.")

    lines.extend(
        [
            "",
            "## Artifacts Used",
            "",
            f"- Corpus: `{results['metadata']['corpus_path']}`",
            f"- Queries: `{results['metadata']['queries_path']}`",
            f"- Qrels: `{results['metadata']['qrels_path']}`",
            f"- Semantic manifest JSON: `{results['metadata']['semantic_manifest_path']}`",
            f"- Semantic manifest JSONL: `{results['metadata']['semantic_manifest_jsonl_path']}`",
            f"- Semantic index: `{results['metadata']['semantic_index_path']}`",
            f"- Semantic vectors: `{results['metadata']['semantic_vectors_path']}`",
            "",
            "## Model Used",
            "",
            f"- `{results['metadata']['semantic_model_name']}`",
            "",
            "## Exact Changes Made",
            "",
        ]
    )
    exact_changes = results["metadata"].get("exact_changes", [])
    if exact_changes:
        lines.extend(f"- {change}" for change in exact_changes)
    else:
        lines.append("- Change log unavailable.")

    lines.extend(
        [
            "",
            "## Overall Metrics",
            "",
            "| Mode | MRR@10 | Recall@10 | nDCG@10 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for mode in ("bm25_only", "semantic_only", "hybrid"):
        summary = results["modes"][mode]["summary"]
        lines.append(
            f"| {mode} | {_fmt_metric(summary['mrr_at_k'])} | {_fmt_metric(summary['recall_at_k'])} | "
            f"{_fmt_metric(summary['ndcg_at_k'])} |"
        )

    lines.extend(
        [
            "",
            "## Metrics By Category",
            "",
            "| Category | Mode | MRR@10 | Recall@10 | nDCG@10 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    categories = sorted(results["modes"]["bm25_only"]["per_category"].keys())
    for category in categories:
        for mode in ("bm25_only", "semantic_only", "hybrid"):
            metrics = results["modes"][mode]["per_category"][category]["metrics"]
            lines.append(
                f"| {category} | {mode} | {_fmt_metric(metrics['mrr_at_k'])} | "
                f"{_fmt_metric(metrics['recall_at_k'])} | {_fmt_metric(metrics['ndcg_at_k'])} |"
            )

    lines.extend(
        [
            "",
            "## Hybrid Vs BM25",
            "",
            "| Category | dMRR | dRecall | dNDCG |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for category in categories:
        delta = hybrid_vs_bm25["per_category"][category]["delta"]
        lines.append(
            f"| {category} | {_fmt_delta(delta['mrr_at_k'])} | {_fmt_delta(delta['recall_at_k'])} | "
            f"{_fmt_delta(delta['ndcg_at_k'])} |"
        )

    if baseline_deltas is not None:
        lines.extend(
            [
                "",
                "## Delta Vs Previous V1",
                "",
                "| Mode | dMRR | dRecall | dNDCG |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for mode in ("bm25_only", "semantic_only", "hybrid"):
            delta = baseline_deltas["overall"][mode]
            lines.append(
                f"| {mode} | {_fmt_delta(delta['mrr_at_k'])} | {_fmt_delta(delta['recall_at_k'])} | "
                f"{_fmt_delta(delta['ndcg_at_k'])} |"
            )

        lines.extend(
            [
                "",
                "## Delta Vs Previous V1 By Category",
                "",
                "| Category | Mode | dMRR | dRecall | dNDCG |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for category in categories:
            for mode in ("bm25_only", "semantic_only", "hybrid"):
                delta = baseline_deltas["per_category"][category][mode]
                lines.append(
                    f"| {category} | {mode} | {_fmt_delta(delta['mrr_at_k'])} | {_fmt_delta(delta['recall_at_k'])} | "
                    f"{_fmt_delta(delta['ndcg_at_k'])} |"
                )

    lines.extend(
        [
            "",
            "## Representative Hybrid Wins Over BM25",
            "",
        ]
    )
    if wins:
        lines.extend(
            [
                "| Query ID | Category | dMRR | dRecall | dNDCG | Query |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in wins:
            lines.append(
                f"| {row['id']} | {row['query_class']} | {_fmt_delta(row['delta']['mrr_at_k'])} | "
                f"{_fmt_delta(row['delta']['recall_at_k'])} | {_fmt_delta(row['delta']['ndcg_at_k'])} | "
                f"{row['query']} |"
            )
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Representative Failures That Remain",
            "",
        ]
    )
    if failures:
        lines.extend(
            [
                "| Query ID | Category | Recall@10 | nDCG@10 | Top Hit | Query |",
                "| --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        for row in failures:
            top_hit = row["top10"][0] if row["top10"] else "(none)"
            lines.append(
                f"| {row['id']} | {row['query_class']} | {_fmt_metric(row['recall_at_k'])} | "
                f"{_fmt_metric(row['ndcg_at_k'])} | {top_hit} | {row['query']} |"
            )
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Final Recommendation",
            "",
            f"- `{results['recommendation']}`",
        ]
    )
    return "\n".join(lines) + "\n"
