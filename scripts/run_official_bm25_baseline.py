#!/usr/bin/env python3
"""Run the frozen official BM25 baseline and emit stable baseline artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from root_rag.evaluation.metrics import TopKMetrics, aggregate_topk_metrics, compute_topk_metrics
from root_rag.retrieval.backends import build_retrieval_backend
from root_rag.retrieval.pipeline import RetrievalPipeline
from root_rag.retrieval.transformers import build_query_transformer

MANIFEST_PATH = REPO_ROOT / "configs" / "baseline_manifest.json"
QUERIES_PATH = REPO_ROOT / "configs" / "benchmark_queries.json"
SUBSETS_PATH = REPO_ROOT / "configs" / "benchmark_query_subsets.json"
CORPUS_PROFILES_PATH = REPO_ROOT / "configs" / "benchmark_corpus_profiles.json"
OFFICIAL_COMMAND = "python scripts/run_official_bm25_baseline.py"
EXPECTED_BACKEND = "lexical_bm25_memory"
EXPECTED_QUERY_MODE = "baseline"
EXPECTED_CORPUS_PROFILE = "fairship_only_valid"
EXPECTED_TOP_K = 10
EXPECTED_ARTIFACT_FILENAMES = {
    "evaluation_report": "benchmark_eval_results_baseline.json",
    "failure_audit_json": "benchmark_failure_audit_baseline.json",
    "failure_audit_markdown": "benchmark_failure_audit_baseline.md",
    "run_manifest": "baseline_run_manifest.json",
    "summary_markdown": "baseline_summary.md",
}
EXPECTED_SUBSETS = [
    "root_basic",
    "sofie_absence_control",
    "root_sofie_integration",
    "repo_specific",
    "critical_queries",
    "fairship_only_valid",
    "extended_corpus_valid",
]


@dataclass(frozen=True)
class QueryEntry:
    query_id: str
    query: str
    query_class: str
    category: str
    expected_behavior: str
    answer_granularity: str
    criticality: str


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _resolve_commit_hash() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return output or None
    except Exception:
        return None


def _require_file(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is not a file: {path}")


def _load_queries(path: Path) -> List[QueryEntry]:
    raw = _load_json(path)
    entries: List[QueryEntry] = []
    for row in raw:
        entries.append(
            QueryEntry(
                query_id=row["id"],
                query=row["query"],
                query_class=row["query_class"],
                category=row["category"],
                expected_behavior=row["expected_behavior"],
                answer_granularity=row["answer_granularity"],
                criticality=row["criticality"],
            )
        )
    return entries


def _load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in _load_jsonl(path):
        out[row["query_id"]][row["chunk_id"]] = int(row["relevance"])
    return dict(out)


def _load_subsets(path: Path) -> Dict[str, List[str]]:
    raw = _load_json(path)
    normalized: Dict[str, List[str]] = {}
    for key, value in raw.items():
        if not isinstance(value, list):
            raise ValueError(f"Subset '{key}' must be a list")
        normalized[key] = list(value)

    for key in EXPECTED_SUBSETS:
        if key not in normalized:
            raise ValueError(f"Missing required subset '{key}' in {path}")
    return normalized


def _load_manifest(path: Path) -> dict:
    manifest = _load_json(path)
    required_keys = {
        "branch_role",
        "baseline_definition_version",
        "official_backend",
        "official_query_mode",
        "official_corpus_profile",
        "official_query_subsets",
        "official_qrels",
        "official_top_k",
        "output_artifact_directory",
        "official_artifacts",
        "deterministic_controls",
        "semantic_retrieval",
    }
    missing = sorted(required_keys - set(manifest.keys()))
    if missing:
        raise ValueError(f"baseline manifest missing keys: {missing}")

    if manifest["official_backend"] != EXPECTED_BACKEND:
        raise ValueError("official_backend drifted from frozen lexical BM25 baseline")
    if manifest["official_query_mode"] != EXPECTED_QUERY_MODE:
        raise ValueError("official_query_mode drifted from frozen baseline mode")
    if manifest["official_corpus_profile"] != EXPECTED_CORPUS_PROFILE:
        raise ValueError("official_corpus_profile drifted from frozen fairship_only_valid profile")
    if int(manifest["official_top_k"]) != EXPECTED_TOP_K:
        raise ValueError("official_top_k drifted from frozen top_k=10")
    if manifest["official_query_subsets"] != EXPECTED_SUBSETS:
        raise ValueError("official_query_subsets drifted from frozen subset ordering")
    if manifest["official_artifacts"] != EXPECTED_ARTIFACT_FILENAMES:
        raise ValueError("official_artifact filenames drifted from frozen contract")
    if EXPECTED_CORPUS_PROFILE not in manifest["official_query_subsets"]:
        raise ValueError("official_corpus_profile must also exist as the canonical official query subset")

    semantic = manifest["semantic_retrieval"]
    if semantic.get("enabled_by_default") is not False:
        raise ValueError("semantic retrieval must remain disabled by default in baseline")
    if semantic.get("allowed_in_official_baseline") is not False:
        raise ValueError("semantic retrieval must remain disallowed in official baseline")

    output_dir = Path(manifest["output_artifact_directory"])
    if output_dir.as_posix() != "artifacts/baseline_official":
        raise ValueError("output_artifact_directory drifted from artifacts/baseline_official")

    return manifest


def _validate_profile_contract(*, manifest: dict, profiles: dict) -> dict:
    if EXPECTED_CORPUS_PROFILE not in profiles:
        raise ValueError("benchmark_corpus_profiles.json is missing fairship_only_valid")
    profile = profiles[EXPECTED_CORPUS_PROFILE]
    profile_qrels = profile.get("qrels")
    if profile_qrels != manifest["official_qrels"]:
        raise ValueError(
            "Official corpus profile qrels do not match manifest official_qrels: "
            f"{profile_qrels!r} != {manifest['official_qrels']!r}"
        )
    if not profile.get("corpus"):
        raise ValueError("Official corpus profile must declare a corpus artifact path")
    return profile


def _build_query_inventory(*, queries: List[QueryEntry], qrels: Dict[str, Dict[str, int]]) -> dict:
    query_ids = {query.query_id for query in queries}
    with_qrels = {query_id for query_id in query_ids if qrels.get(query_id)}
    without_qrels = sorted(query_ids - with_qrels)
    return {
        "total_queries_defined": len(query_ids),
        "total_queries_scored": len(with_qrels),
        "total_queries_without_qrels_by_design": len(without_qrels),
        "queries_without_qrels_by_design": without_qrels,
    }


def _metric_at_10(metrics: dict) -> dict:
    return {
        "mrr_at_10": metrics["mrr_at_k"],
        "recall_at_10": metrics["recall_at_k"],
        "ndcg_at_10": metrics["ndcg_at_k"],
    }


def _aggregate_rows(rows: List[dict]) -> dict:
    metrics = aggregate_topk_metrics(
        [
            TopKMetrics(
                mrr_at_k=row["mrr_at_10"],
                recall_at_k=row["recall_at_10"],
                ndcg_at_k=row["ndcg_at_10"],
                retrieved_positive_count=row["retrieved_positive_count"],
                qrels_positive_count=row["qrels_positive_count"],
            )
            for row in rows
        ]
    )
    return _metric_at_10(metrics)


def _build_subset_summary(
    *,
    subset_name: str,
    subset_ids: List[str],
    per_query_by_id: Dict[str, dict],
) -> dict:
    defined_rows = [per_query_by_id[query_id] for query_id in subset_ids if query_id in per_query_by_id]
    scored_rows = [row for row in defined_rows if row["qrels_positive_count"] > 0]
    non_qrel_rows = [row for row in defined_rows if row["qrels_positive_count"] == 0]
    zero_recall_rows = [row for row in scored_rows if row["recall_at_10"] == 0.0]

    metrics_by_category: Dict[str, dict] = {}
    for category in sorted({row["category"] for row in scored_rows}):
        category_rows = [row for row in scored_rows if row["category"] == category]
        metrics_by_category[category] = {
            "total_queries_scored": len(category_rows),
            "zero_recall_scored_queries": sum(1 for row in category_rows if row["recall_at_10"] == 0.0),
            **_aggregate_rows(category_rows),
        }

    return {
        "subset_name": subset_name,
        "counts": {
            "total_queries_defined": len(defined_rows),
            "total_queries_scored": len(scored_rows),
            "total_queries_without_qrels_by_design": len(non_qrel_rows),
            "zero_recall_scored_queries": len(zero_recall_rows),
        },
        "defined_query_ids": [row["id"] for row in defined_rows],
        "scored_query_ids": [row["id"] for row in scored_rows],
        "queries_without_qrels_by_design": [row["id"] for row in non_qrel_rows],
        "zero_recall_scored_query_ids": [row["id"] for row in zero_recall_rows],
        "metrics_global": _aggregate_rows(scored_rows),
        "metrics_by_category": metrics_by_category,
        "metrics_by_query": scored_rows,
    }


def _build_diagnostic_subsets(
    *,
    subsets: Dict[str, List[str]],
    per_query_by_id: Dict[str, dict],
) -> dict:
    diagnostic = {}
    for subset_name in EXPECTED_SUBSETS:
        if subset_name == EXPECTED_CORPUS_PROFILE:
            continue
        summary = _build_subset_summary(
            subset_name=subset_name,
            subset_ids=subsets[subset_name],
            per_query_by_id=per_query_by_id,
        )
        summary["canonical"] = False
        summary["diagnostic_only"] = True
        if subset_name == "extended_corpus_valid":
            summary["non_canonical_reason"] = (
                "This subset is diagnostic-only in the official baseline run and must not be used "
                "as the frozen comparison anchor."
            )
        else:
            summary["non_canonical_reason"] = (
                "This subset is a diagnostic breakdown only and must not replace the official baseline."
            )
        diagnostic[subset_name] = summary
    return diagnostic


def _run_audit(*, corpus: Path, queries: Path, qrels: Path, output_json: Path, output_md: Path) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing else f"{SRC_ROOT}{os.pathsep}{existing}"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "audit_benchmark_failures.py"),
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
        str(EXPECTED_TOP_K),
        "--backend",
        EXPECTED_BACKEND,
        "--query-mode",
        EXPECTED_QUERY_MODE,
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def _write_summary(*, path: Path, report: dict, run_manifest: dict) -> None:
    official = report["official_baseline"]
    official_counts = official["counts"]
    official_metrics = official["metrics_global"]
    inventory = report["benchmark_query_inventory"]
    lines = [
        "# Official Baseline Summary",
        "",
        "## Official Baseline",
        "",
        f"Baseline definition version: `{run_manifest['baseline_definition_version']}`",
        f"Branch role: `{run_manifest['branch_role']}`",
        f"Official command: `{OFFICIAL_COMMAND}`",
        f"Backend: `{official['backend']}`",
        f"Query mode: `{official['query_mode']}`",
        f"Corpus profile: `{official['corpus_profile']}`",
        f"Canonical official subset: `{official['query_subset']}`",
        f"Official metrics source: scored queries in `{official['query_subset']}` using `{official['qrels_path']}`",
        f"Top-k: `{official['top_k']}`",
        f"Total benchmark queries defined: `{inventory['total_queries_defined']}`",
        f"Official baseline queries defined: `{official_counts['total_queries_defined']}`",
        f"Official baseline queries scored: `{official_counts['total_queries_scored']}`",
        (
            "Official baseline queries without qrels by design: "
            f"`{official_counts['total_queries_without_qrels_by_design']}`"
        ),
        f"Zero-recall scored queries: `{official_counts['zero_recall_scored_queries']}`",
        "",
        "## Official Metrics",
        "",
        f"- `MRR@10`: {official_metrics['mrr_at_10']:.6f}",
        f"- `Recall@10`: {official_metrics['recall_at_10']:.6f}",
        f"- `nDCG@10`: {official_metrics['ndcg_at_10']:.6f}",
        "",
        "## Diagnostic Only",
        "",
        "- Non-canonical subsets are emitted only as diagnostics and must not be used as the frozen comparison anchor.",
        "- `extended_corpus_valid` remains diagnostic-only in this official baseline run.",
        "",
        "## Official Artifacts",
        "",
    ]
    for _, relative_path in run_manifest["artifacts"].items():
        lines.append(f"- `{relative_path}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    manifest = _load_manifest(MANIFEST_PATH)
    profiles = _load_json(CORPUS_PROFILES_PATH)
    profile = _validate_profile_contract(manifest=manifest, profiles=profiles)

    corpus_path = REPO_ROOT / profile["corpus"]
    qrels_path = REPO_ROOT / manifest["official_qrels"]
    output_dir = REPO_ROOT / manifest["output_artifact_directory"]
    artifact_paths = {
        key: output_dir / filename
        for key, filename in manifest["official_artifacts"].items()
    }

    for path, label in [
        (MANIFEST_PATH, "baseline manifest"),
        (QUERIES_PATH, "benchmark queries"),
        (SUBSETS_PATH, "benchmark query subsets"),
        (CORPUS_PROFILES_PATH, "benchmark corpus profiles"),
        (corpus_path, "official baseline corpus"),
        (qrels_path, "official qrels"),
    ]:
        _require_file(path, label=label)

    queries = _load_queries(QUERIES_PATH)
    qrels = _load_qrels(qrels_path)
    subsets = _load_subsets(SUBSETS_PATH)
    if EXPECTED_CORPUS_PROFILE not in subsets:
        raise ValueError("Official canonical subset is missing from benchmark_query_subsets.json")

    corpus_rows = _load_jsonl(corpus_path)
    if not corpus_rows:
        raise ValueError(f"Official baseline corpus is empty: {corpus_path}")

    query_ids = {query.query_id for query in queries}
    for subset_name in EXPECTED_SUBSETS:
        unknown = [query_id for query_id in subsets[subset_name] if query_id not in query_ids]
        if unknown:
            raise ValueError(f"Subset '{subset_name}' contains unknown query ids: {unknown}")

    official_subset_ids = subsets[EXPECTED_CORPUS_PROFILE]
    if not official_subset_ids:
        raise ValueError("Official canonical subset must not be empty")

    backend = build_retrieval_backend(
        EXPECTED_BACKEND,
        corpus_rows=corpus_rows,
        corpus_artifact_path=corpus_path,
        k1=1.5,
        b=0.75,
    )
    transformer = build_query_transformer(EXPECTED_QUERY_MODE)
    pipeline = RetrievalPipeline(backend=backend, query_transformer=transformer)

    per_query_rows: List[dict] = []
    for entry in queries:
        scored = pipeline.search(entry.query, top_k=EXPECTED_TOP_K)
        ranked_chunk_ids = [row.chunk_id for row in scored]
        ranked_scores = [row.score for row in scored]
        relevance = qrels.get(entry.query_id, {})
        metrics = compute_topk_metrics(
            ranked_chunk_ids,
            relevance,
            top_k=EXPECTED_TOP_K,
            qrels_positive_count=len(relevance),
        )
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
                "top_k_results": ranked_chunk_ids[:EXPECTED_TOP_K],
                "top_k_scores": ranked_scores[:EXPECTED_TOP_K],
            }
        )

    per_query_rows.sort(key=lambda row: row["id"])
    per_query_by_id = {row["id"]: row for row in per_query_rows}
    benchmark_inventory = _build_query_inventory(queries=queries, qrels=qrels)
    official_baseline = _build_subset_summary(
        subset_name=EXPECTED_CORPUS_PROFILE,
        subset_ids=official_subset_ids,
        per_query_by_id=per_query_by_id,
    )
    if official_baseline["counts"]["total_queries_scored"] == 0:
        raise ValueError("Official baseline subset resolved to zero scored queries")

    official_baseline.update(
        {
            "canonical": True,
            "backend": EXPECTED_BACKEND,
            "query_mode": EXPECTED_QUERY_MODE,
            "corpus_profile": EXPECTED_CORPUS_PROFILE,
            "query_subset": EXPECTED_CORPUS_PROFILE,
            "qrels_path": _relative(qrels_path),
            "top_k": EXPECTED_TOP_K,
        }
    )

    report = {
        "baseline_definition_version": manifest["baseline_definition_version"],
        "branch_role": manifest["branch_role"],
        "official_command": OFFICIAL_COMMAND,
        "commit_hash": _resolve_commit_hash(),
        "inputs": {
            "baseline_manifest": _relative(MANIFEST_PATH),
            "queries": _relative(QUERIES_PATH),
            "subsets": _relative(SUBSETS_PATH),
            "qrels": _relative(qrels_path),
            "corpus": _relative(corpus_path),
            "corpus_profiles": _relative(CORPUS_PROFILES_PATH),
            "input_sha256": {
                "baseline_manifest": _sha256_file(MANIFEST_PATH),
                "queries": _sha256_file(QUERIES_PATH),
                "subsets": _sha256_file(SUBSETS_PATH),
                "qrels": _sha256_file(qrels_path),
                "corpus": _sha256_file(corpus_path),
                "corpus_profiles": _sha256_file(CORPUS_PROFILES_PATH),
            },
        },
        "benchmark_query_inventory": benchmark_inventory,
        "official_baseline": official_baseline,
        "diagnostic_subsets": _build_diagnostic_subsets(
            subsets=subsets,
            per_query_by_id=per_query_by_id,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths["evaluation_report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    _run_audit(
        corpus=corpus_path,
        queries=QUERIES_PATH,
        qrels=qrels_path,
        output_json=artifact_paths["failure_audit_json"],
        output_md=artifact_paths["failure_audit_markdown"],
    )

    run_manifest = {
        "baseline_definition_version": manifest["baseline_definition_version"],
        "branch_role": manifest["branch_role"],
        "official_command": OFFICIAL_COMMAND,
        "official_backend": EXPECTED_BACKEND,
        "official_query_mode": EXPECTED_QUERY_MODE,
        "official_corpus_profile": EXPECTED_CORPUS_PROFILE,
        "official_query_subset": EXPECTED_CORPUS_PROFILE,
        "official_qrels": _relative(qrels_path),
        "official_top_k": EXPECTED_TOP_K,
        "semantic_retrieval_disabled_by_default": True,
        "official_baseline_summary": {
            "metrics_source": f"scored queries in {EXPECTED_CORPUS_PROFILE}",
            **official_baseline["counts"],
            **official_baseline["metrics_global"],
        },
        "artifacts": {key: _relative(path) for key, path in artifact_paths.items()},
        "inputs": report["inputs"],
    }
    artifact_paths["run_manifest"].write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    _write_summary(path=artifact_paths["summary_markdown"], report=report, run_manifest=run_manifest)

    global_metrics = official_baseline["metrics_global"]
    print("Official baseline completed.")
    print(f"Artifacts directory: {_relative(output_dir)}")
    print(f"Canonical subset: {EXPECTED_CORPUS_PROFILE}")
    print(
        "Defined/scored/no-qrel-by-design="
        f"{official_baseline['counts']['total_queries_defined']}/"
        f"{official_baseline['counts']['total_queries_scored']}/"
        f"{official_baseline['counts']['total_queries_without_qrels_by_design']}"
    )
    print(f"MRR@10={global_metrics['mrr_at_10']:.6f}")
    print(f"Recall@10={global_metrics['recall_at_10']:.6f}")
    print(f"nDCG@10={global_metrics['ndcg_at_10']:.6f}")
    print(
        "Zero-recall scored queries="
        f"{official_baseline['counts']['zero_recall_scored_queries']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
