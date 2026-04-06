#!/usr/bin/env python3
"""Run the official frozen BM25 baseline and emit stable benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List

from root_rag.evaluation.metrics import TopKMetrics, aggregate_topk_metrics, compute_topk_metrics
from root_rag.retrieval.backends import build_retrieval_backend
from root_rag.retrieval.pipeline import RetrievalPipeline
from root_rag.retrieval.transformers import build_query_transformer

REQUIRED_SUBSETS = (
    "root_basic",
    "sofie_absence_control",
    "root_sofie_integration",
    "repo_specific",
    "critical_queries",
    "fairship_only_valid",
    "extended_corpus_valid",
)
LEGACY_SUBSETS = ("sofie",)


@dataclass(frozen=True)
class QueryEntry:
    query_id: str
    query: str
    query_class: str
    category: str
    expected_behavior: str
    answer_granularity: str
    criticality: str


def _load_corpus(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_queries(path: Path) -> List[QueryEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: List[QueryEntry] = []
    for row in raw:
        category = row.get("category")
        if category is None:
            query_class = row.get("query_class", "")
            if query_class == "common_api":
                category = "root_basic"
            elif query_class in {"structural_usage", "rare_api"}:
                category = "repo_specific"
            else:
                category = "repo_specific"

        entries.append(
            QueryEntry(
                query_id=row["id"],
                query=row["query"],
                query_class=row["query_class"],
                category=category,
                expected_behavior=row.get("expected_behavior", "retrieve_present"),
                answer_granularity=row.get("answer_granularity", "file"),
                criticality=row.get("criticality", "medium"),
            )
        )
    return entries


def _load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["query_id"]][row["chunk_id"]] = int(row["relevance"])
    return dict(out)


def _load_subsets(path: Path) -> Dict[str, List[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    normalized: Dict[str, List[str]] = {}
    for key, value in raw.items():
        if not isinstance(value, list):
            raise ValueError(f"Subset '{key}' must be a list")
        normalized[key] = list(value)

    for key in REQUIRED_SUBSETS:
        if key not in normalized:
            raise ValueError(f"Missing required subset '{key}' in {path}")

    for key in LEGACY_SUBSETS:
        if key not in normalized:
            normalized[key] = []

    return normalized


def _load_profiles(path: Path) -> Dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Corpus profiles config must be a JSON object")
    return raw


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
    return {
        "count": count,
        "mean": to_ms(sum(sorted_samples) / count),
        "p50": to_ms(percentile(0.50)),
        "p95": to_ms(percentile(0.95)),
        "min": to_ms(sorted_samples[0]),
        "max": to_ms(sorted_samples[-1]),
    }


def _metric_at_10(metrics: dict) -> dict:
    return {
        "mrr_at_10": metrics["mrr_at_k"],
        "recall_at_10": metrics["recall_at_k"],
        "ndcg_at_10": metrics["ndcg_at_k"],
    }


def _resolve_commit_hash() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return output or None
    except Exception:
        return None


def _run_audit(
    *,
    script_path: Path,
    py_path: str,
    corpus: Path,
    queries: Path,
    qrels: Path,
    output_json: Path,
    output_md: Path,
    top_k: int,
) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = py_path if not existing else f"{py_path}{os.pathsep}{existing}"
    cmd = [
        sys.executable,
        str(script_path),
        "--corpus",
        str(corpus),
        "--queries",
        str(queries),
        "--qrels",
        str(qrels),
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
        "--top-k",
        str(top_k),
        "--backend",
        "lexical_bm25_memory",
        "--query-mode",
        "baseline",
    ]
    subprocess.run(cmd, check=True, env=env)


def _resolve_path(value: Path | None, profile: dict, key: str) -> Path:
    if value is not None:
        return value
    return Path(profile[key])


def _build_coverage_report(
    *,
    queries: List[QueryEntry],
    subsets: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    corpus_chunk_ids: set[str],
    scenario_name: str,
) -> dict:
    query_map = {q.query_id: q for q in queries}
    query_ids = set(query_map.keys())
    with_qrels = {qid for qid in query_ids if qrels.get(qid)}

    fairship_valid = [qid for qid in subsets.get("fairship_only_valid", []) if qid in query_ids]
    extended_valid = [qid for qid in subsets.get("extended_corpus_valid", []) if qid in query_ids]
    requires_extended = sorted([qid for qid in extended_valid if qid not in set(fairship_valid)])

    absence_controls = sorted(
        qid
        for qid, q in query_map.items()
        if q.expected_behavior == "confirm_absence"
    )

    no_qrels_by_design: List[dict] = []
    for qid, q in query_map.items():
        if qid in with_qrels:
            continue
        if qid in absence_controls:
            reason = "absence_control"
        elif qid in requires_extended:
            reason = "requires_extended_corpus"
        else:
            reason = "no_verified_evidence_in_fairship_only"
        no_qrels_by_design.append({"query_id": qid, "reason": reason})

    missing_qrel_chunks = []
    for qid, relevance in qrels.items():
        for chunk_id in relevance:
            if chunk_id not in corpus_chunk_ids:
                missing_qrel_chunks.append({"query_id": qid, "chunk_id": chunk_id})

    subset_summary = {}
    for subset_name, ids in subsets.items():
        valid_ids = [qid for qid in ids if qid in query_ids]
        qrels_count = sum(1 for qid in valid_ids if qid in with_qrels)
        subset_summary[subset_name] = {
            "query_count": len(valid_ids),
            "queries_with_qrels": qrels_count,
            "queries_without_qrels": len(valid_ids) - qrels_count,
        }

    category_summary = {}
    for category in sorted({q.category for q in queries}):
        ids = [q.query_id for q in queries if q.category == category]
        qrels_count = sum(1 for qid in ids if qid in with_qrels)
        category_summary[category] = {
            "query_count": len(ids),
            "queries_with_qrels": qrels_count,
            "queries_without_qrels": len(ids) - qrels_count,
        }

    return {
        "scenario_name": scenario_name,
        "queries_valid_in_fairship_only": fairship_valid,
        "absence_control_queries": absence_controls,
        "queries_requiring_extended_corpus": requires_extended,
        "queries_without_qrels_by_design": sorted(no_qrels_by_design, key=lambda row: row["query_id"]),
        "subset_summary": subset_summary,
        "category_summary": category_summary,
        "missing_qrel_chunks_in_corpus": sorted(
            missing_qrel_chunks,
            key=lambda row: (row["query_id"], row["chunk_id"]),
        ),
        "summary": {
            "total_queries": len(queries),
            "queries_with_qrels": len(with_qrels),
            "queries_without_qrels": len(queries) - len(with_qrels),
            "absence_controls": len(absence_controls),
            "requires_extended_corpus": len(requires_extended),
            "missing_qrel_chunks": len(missing_qrel_chunks),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-name",
        default="fairship_only_valid",
        help="Benchmark scenario/corpus profile name.",
    )
    parser.add_argument(
        "--corpus-profiles",
        type=Path,
        default=Path("configs/benchmark_corpus_profiles.json"),
        help="Scenario to corpus/qrels/output profile mapping.",
    )
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--queries", type=Path, default=Path("configs/benchmark_queries.json"))
    parser.add_argument("--qrels", type=Path, default=None)
    parser.add_argument("--subsets", type=Path, default=Path("configs/benchmark_query_subsets.json"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--experiment-name",
        default="bm25_official_baseline",
        help="Stable experiment name for the official baseline artifact.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--per-query-output", type=Path, default=None)
    parser.add_argument("--coverage-output", type=Path, default=None)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--audit-md", type=Path, default=None)
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip audit_benchmark_failures.py execution.",
    )
    args = parser.parse_args()
    if args.top_k != 10:
        parser.error("Official baseline is frozen at --top-k 10")

    profiles = _load_profiles(args.corpus_profiles)
    if args.scenario_name not in profiles:
        known = ", ".join(sorted(profiles.keys()))
        parser.error(f"Unknown --scenario-name '{args.scenario_name}'. Known: {known}")
    profile = profiles[args.scenario_name]

    corpus_path = _resolve_path(args.corpus, profile, "corpus")
    qrels_path = _resolve_path(args.qrels, profile, "qrels")
    output_path = _resolve_path(args.output, profile, "output")
    per_query_output_path = _resolve_path(args.per_query_output, profile, "per_query_output")
    coverage_output_path = _resolve_path(args.coverage_output, profile, "coverage_output")
    audit_json_path = _resolve_path(args.audit_json, profile, "audit_json")
    audit_md_path = _resolve_path(args.audit_md, profile, "audit_md")

    external_manifest = profile.get("external_manifest")
    external_manifest_exists = None
    if external_manifest:
        external_manifest_exists = Path(external_manifest).exists()

    corpus_rows = _load_corpus(corpus_path)
    queries = _load_queries(args.queries)
    qrels = _load_qrels(qrels_path)
    subsets = _load_subsets(args.subsets)

    backend = build_retrieval_backend(
        "lexical_bm25_memory",
        corpus_rows=corpus_rows,
        corpus_artifact_path=corpus_path,
        k1=1.5,
        b=0.75,
    )
    transformer = build_query_transformer("baseline")
    pipeline = RetrievalPipeline(backend=backend, query_transformer=transformer)

    per_query_rows: List[dict] = []
    metric_rows: List[TopKMetrics] = []
    latency_samples_s: List[float] = []

    for entry in queries:
        t0 = perf_counter()
        scored = pipeline.search(entry.query, top_k=args.top_k)
        latency_samples_s.append(perf_counter() - t0)

        ranked_chunk_ids = [row.chunk_id for row in scored]
        ranked_scores = [row.score for row in scored]
        relevance = qrels.get(entry.query_id, {})
        metrics = compute_topk_metrics(
            ranked_chunk_ids,
            relevance,
            top_k=args.top_k,
            qrels_positive_count=len(relevance),
        )
        metric_rows.append(metrics)

        per_query_rows.append(
            {
                "id": entry.query_id,
                "query": entry.query,
                "query_class": entry.query_class,
                "category": entry.category,
                "expected_behavior": entry.expected_behavior,
                "answer_granularity": entry.answer_granularity,
                "criticality": entry.criticality,
                "mrr_at_10": metrics.mrr_at_k,
                "recall_at_10": metrics.recall_at_k,
                "ndcg_at_10": metrics.ndcg_at_k,
                "qrels_positive_count": metrics.qrels_positive_count,
                "retrieved_positive_count": metrics.retrieved_positive_count,
                "top_k_results": ranked_chunk_ids[: args.top_k],
                "top_k_scores": ranked_scores[: args.top_k],
            }
        )

    per_query_rows.sort(key=lambda row: row["id"])
    summary = aggregate_topk_metrics(metric_rows)

    metrics_by_category: Dict[str, dict] = {}
    per_query_by_id = {row["id"]: row for row in per_query_rows}

    category_subset_map = {
        "root_basic": subsets.get("root_basic", []),
        "sofie": subsets.get("sofie", []),
        "sofie_absence_control": subsets.get("sofie_absence_control", []),
        "root_sofie_integration": subsets.get("root_sofie_integration", []),
        "repo_specific": subsets.get("repo_specific", []),
    }

    for category, ids in category_subset_map.items():
        category_rows = [per_query_by_id[qid] for qid in ids if qid in per_query_by_id]
        if category_rows:
            category_metrics = aggregate_topk_metrics(
                [
                    TopKMetrics(
                        mrr_at_k=row["mrr_at_10"],
                        recall_at_k=row["recall_at_10"],
                        ndcg_at_k=row["ndcg_at_10"],
                        retrieved_positive_count=row["retrieved_positive_count"],
                        qrels_positive_count=row["qrels_positive_count"],
                    )
                    for row in category_rows
                ]
            )
            metrics_by_category[category] = {
                "query_count": len(category_rows),
                **_metric_at_10(category_metrics),
            }
        else:
            metrics_by_category[category] = {
                "query_count": 0,
                "mrr_at_10": 0.0,
                "recall_at_10": 0.0,
                "ndcg_at_10": 0.0,
            }

    metrics_by_subset: Dict[str, dict] = {}
    for subset_name, ids in subsets.items():
        subset_rows = [per_query_by_id[qid] for qid in ids if qid in per_query_by_id]
        if subset_rows:
            subset_metrics = aggregate_topk_metrics(
                [
                    TopKMetrics(
                        mrr_at_k=row["mrr_at_10"],
                        recall_at_k=row["recall_at_10"],
                        ndcg_at_k=row["ndcg_at_10"],
                        retrieved_positive_count=row["retrieved_positive_count"],
                        qrels_positive_count=row["qrels_positive_count"],
                    )
                    for row in subset_rows
                ]
            )
            metrics_by_subset[subset_name] = {
                "query_count": len(subset_rows),
                **_metric_at_10(subset_metrics),
            }
        else:
            metrics_by_subset[subset_name] = {
                "query_count": 0,
                "mrr_at_10": 0.0,
                "recall_at_10": 0.0,
                "ndcg_at_10": 0.0,
            }

    operational = {
        "backend_id": backend.backend_id,
        "backend_metrics": backend.operational_metrics(),
        "query_latency_ms": _compute_latency_summary_ms(latency_samples_s),
    }
    backend_metrics = operational["backend_metrics"]

    chunk_ids = {row["chunk_id"] for row in corpus_rows if "chunk_id" in row}
    file_paths = {row.get("file_path", "") for row in corpus_rows if row.get("file_path")}

    coverage_report = _build_coverage_report(
        queries=queries,
        subsets=subsets,
        qrels=qrels,
        corpus_chunk_ids=chunk_ids,
        scenario_name=args.scenario_name,
    )

    report = {
        "experiment_name": args.experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit_hash": _resolve_commit_hash(),
        "backend": "lexical_bm25_memory",
        "query_mode": "baseline",
        "scenario_name": args.scenario_name,
        "corpus_profile": args.scenario_name,
        "top_k": args.top_k,
        "corpus_size_bytes": corpus_path.stat().st_size if corpus_path.exists() else None,
        "chunk_count": len(chunk_ids),
        "corpus_size": int(backend_metrics.get("docs", float(len(chunk_ids)))),
        "file_count": len(file_paths),
        "metrics_global": _metric_at_10(summary),
        "metrics_by_category": metrics_by_category,
        "metrics_by_subset": metrics_by_subset,
        "metrics_by_query": per_query_rows,
        "operational": operational,
        "coverage_report": {
            "path": str(coverage_output_path),
            "summary": coverage_report["summary"],
        },
        "inputs": {
            "corpus": str(corpus_path),
            "queries": str(args.queries),
            "qrels": str(qrels_path),
            "subsets": str(args.subsets),
            "corpus_profiles": str(args.corpus_profiles),
            "external_manifest": external_manifest,
            "external_manifest_exists": external_manifest_exists,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    per_query_output_path.parent.mkdir(parents=True, exist_ok=True)
    per_query_output_path.write_text(json.dumps(per_query_rows, indent=2), encoding="utf-8")

    coverage_output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_output_path.write_text(json.dumps(coverage_report, indent=2), encoding="utf-8")

    if not args.skip_audit:
        audit_json_path.parent.mkdir(parents=True, exist_ok=True)
        audit_md_path.parent.mkdir(parents=True, exist_ok=True)
        py_path = str((Path(__file__).resolve().parents[1] / "src"))
        _run_audit(
            script_path=Path("scripts/audit_benchmark_failures.py"),
            py_path=py_path,
            corpus=corpus_path,
            queries=args.queries,
            qrels=qrels_path,
            output_json=audit_json_path,
            output_md=audit_md_path,
            top_k=args.top_k,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
