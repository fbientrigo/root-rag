"""Frozen contract checks for the official baseline formalization."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "baseline_manifest.json"
QUERIES_PATH = REPO_ROOT / "configs" / "benchmark_queries.json"
SUBSETS_PATH = REPO_ROOT / "configs" / "benchmark_query_subsets.json"
QRELS_PATH = REPO_ROOT / "configs" / "benchmark_qrels.jsonl"
CORPUS_PROFILES_PATH = REPO_ROOT / "configs" / "benchmark_corpus_profiles.json"
CORPUS_PATH = REPO_ROOT / "artifacts" / "corpus.jsonl"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_official_bm25_baseline.py"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "baseline_official"
EVALUATION_REPORT_PATH = OUTPUT_DIR / "benchmark_eval_results_baseline.json"
RUN_MANIFEST_PATH = OUTPUT_DIR / "baseline_run_manifest.json"
SUMMARY_PATH = OUTPUT_DIR / "baseline_summary.md"
EXPECTED_ARTIFACTS = {
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


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("official_baseline_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        rows.setdefault(entry["query_id"], {})[entry["chunk_id"]] = int(entry["relevance"])
    return rows


@lru_cache(maxsize=1)
def _run_official_runner_once() -> tuple[dict, dict, str]:
    subprocess.run([sys.executable, str(RUNNER_PATH)], check=True, cwd=REPO_ROOT)
    report = _load_json(EVALUATION_REPORT_PATH)
    run_manifest = _load_json(RUN_MANIFEST_PATH)
    summary_text = SUMMARY_PATH.read_text(encoding="utf-8")
    return report, run_manifest, summary_text


def test_baseline_manifest_loads_and_matches_frozen_values():
    manifest = _load_json(MANIFEST_PATH)

    assert manifest["branch_role"] == "frozen_evaluation_floor"
    assert manifest["baseline_definition_version"] == "1.0.0"
    assert manifest["official_backend"] == "lexical_bm25_memory"
    assert manifest["official_query_mode"] == "baseline"
    assert manifest["official_corpus_profile"] == "fairship_only_valid"
    assert manifest["official_qrels"] == "configs/benchmark_qrels.jsonl"
    assert manifest["official_top_k"] == 10
    assert manifest["output_artifact_directory"] == "artifacts/baseline_official"
    assert manifest["official_query_subsets"] == EXPECTED_SUBSETS
    assert manifest["official_artifacts"] == EXPECTED_ARTIFACTS


def test_official_benchmark_files_exist():
    for path in [
        MANIFEST_PATH,
        QUERIES_PATH,
        SUBSETS_PATH,
        QRELS_PATH,
        CORPUS_PROFILES_PATH,
        CORPUS_PATH,
        RUNNER_PATH,
    ]:
        assert path.exists(), f"Missing official baseline file: {path}"
        assert path.is_file(), f"Official baseline path is not a file: {path}"


def test_official_subsets_resolve_against_queries_without_unknown_ids():
    queries = _load_json(QUERIES_PATH)
    subsets = _load_json(SUBSETS_PATH)
    query_ids = {row["id"] for row in queries}

    for subset_name in EXPECTED_SUBSETS:
        assert subset_name in subsets
        ids = subsets[subset_name]
        assert ids == list(ids)
        assert len(ids) == len(set(ids)), f"Duplicate query ids in subset {subset_name}"
        unknown = [query_id for query_id in ids if query_id not in query_ids]
        assert not unknown, f"Unknown query ids in subset {subset_name}: {unknown}"


def test_official_artifact_names_are_stable():
    manifest = _load_json(MANIFEST_PATH)
    output_dir = manifest["output_artifact_directory"]
    artifact_paths = {
        key: f"{output_dir}/{filename}" for key, filename in manifest["official_artifacts"].items()
    }

    assert artifact_paths == {
        "evaluation_report": "artifacts/baseline_official/benchmark_eval_results_baseline.json",
        "failure_audit_json": "artifacts/baseline_official/benchmark_failure_audit_baseline.json",
        "failure_audit_markdown": "artifacts/baseline_official/benchmark_failure_audit_baseline.md",
        "run_manifest": "artifacts/baseline_official/baseline_run_manifest.json",
        "summary_markdown": "artifacts/baseline_official/baseline_summary.md",
    }


def test_runner_uses_manifest_driven_official_constants_without_experimental_defaults():
    runner = _load_runner_module()

    assert runner.MANIFEST_PATH == MANIFEST_PATH
    assert runner.EXPECTED_BACKEND == "lexical_bm25_memory"
    assert runner.EXPECTED_QUERY_MODE == "baseline"
    assert runner.EXPECTED_CORPUS_PROFILE == "fairship_only_valid"
    assert runner.EXPECTED_TOP_K == 10
    assert runner.EXPECTED_SUBSETS == EXPECTED_SUBSETS
    assert runner.EXPECTED_ARTIFACT_FILENAMES == EXPECTED_ARTIFACTS


def test_semantic_retrieval_cannot_become_default_in_baseline_by_accident():
    manifest = _load_json(MANIFEST_PATH)
    semantic = manifest["semantic_retrieval"]

    assert semantic["enabled_by_default"] is False
    assert semantic["allowed_in_official_baseline"] is False

    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "semantic_hash_memory" not in runner_source
    assert "semantic_faiss" not in runner_source
    assert "hybrid_s1" not in runner_source
    assert 'build_query_transformer(EXPECTED_QUERY_MODE)' in runner_source


def test_generated_official_baseline_matches_manifest_canonical_subset():
    manifest = _load_json(MANIFEST_PATH)
    report, run_manifest, _ = _run_official_runner_once()

    official = report["official_baseline"]
    assert "metrics_global" not in report
    assert official["canonical"] is True
    assert official["corpus_profile"] == manifest["official_corpus_profile"]
    assert official["query_subset"] == manifest["official_corpus_profile"]
    assert official["qrels_path"] == manifest["official_qrels"]
    assert run_manifest["official_corpus_profile"] == manifest["official_corpus_profile"]
    assert run_manifest["official_query_subset"] == manifest["official_corpus_profile"]


def test_generated_official_counts_split_defined_scored_and_non_qrel_queries():
    subsets = _load_json(SUBSETS_PATH)
    qrels = _load_qrels(QRELS_PATH)
    report, run_manifest, _ = _run_official_runner_once()

    official = report["official_baseline"]
    official_ids = subsets["fairship_only_valid"]
    expected_scored = [query_id for query_id in official_ids if qrels.get(query_id)]
    expected_without_qrels = [query_id for query_id in official_ids if not qrels.get(query_id)]

    assert report["benchmark_query_inventory"]["total_queries_defined"] == 34
    assert official["counts"]["total_queries_defined"] == len(official_ids) == 32
    assert official["counts"]["total_queries_scored"] == len(expected_scored) == 28
    assert (
        official["counts"]["total_queries_without_qrels_by_design"]
        == len(expected_without_qrels)
        == 4
    )
    assert official["queries_without_qrels_by_design"] == expected_without_qrels
    assert run_manifest["official_baseline_summary"]["total_queries_defined"] == 32
    assert run_manifest["official_baseline_summary"]["total_queries_scored"] == 28
    assert run_manifest["official_baseline_summary"]["total_queries_without_qrels_by_design"] == 4


def test_zero_recall_accounting_only_refers_to_scored_queries():
    report, run_manifest, _ = _run_official_runner_once()
    official = report["official_baseline"]
    scored_query_ids = set(official["scored_query_ids"])
    non_qrel_ids = set(official["queries_without_qrels_by_design"])
    zero_recall_ids = official["zero_recall_scored_query_ids"]
    per_query = {row["id"]: row for row in official["metrics_by_query"]}

    assert len(zero_recall_ids) == official["counts"]["zero_recall_scored_queries"] == 11
    assert len(zero_recall_ids) == len(set(zero_recall_ids))
    assert not (set(zero_recall_ids) & non_qrel_ids)
    assert set(zero_recall_ids).issubset(scored_query_ids)
    assert all(per_query[query_id]["qrels_positive_count"] > 0 for query_id in zero_recall_ids)
    assert all(per_query[query_id]["recall_at_10"] == 0.0 for query_id in zero_recall_ids)
    assert run_manifest["official_baseline_summary"]["zero_recall_scored_queries"] == 11


def test_summary_markdown_matches_canonical_subset_metrics():
    report, run_manifest, summary_text = _run_official_runner_once()
    official = report["official_baseline"]

    assert "Canonical official subset: `fairship_only_valid`" in summary_text
    assert "Official metrics source: scored queries in `fairship_only_valid`" in summary_text
    assert "Official baseline queries defined: `32`" in summary_text
    assert "Official baseline queries scored: `28`" in summary_text
    assert "Official baseline queries without qrels by design: `4`" in summary_text
    assert "Zero-recall scored queries: `11`" in summary_text
    assert f"- `MRR@10`: {official['metrics_global']['mrr_at_10']:.6f}" in summary_text
    assert f"- `Recall@10`: {official['metrics_global']['recall_at_10']:.6f}" in summary_text
    assert f"- `nDCG@10`: {official['metrics_global']['ndcg_at_10']:.6f}" in summary_text
    assert run_manifest["official_baseline_summary"]["mrr_at_10"] == official["metrics_global"]["mrr_at_10"]


def test_non_canonical_subsets_cannot_become_top_level_baseline_result():
    report, _, _ = _run_official_runner_once()
    diagnostic = report["diagnostic_subsets"]

    assert "extended_corpus_valid" in diagnostic
    assert diagnostic["extended_corpus_valid"]["canonical"] is False
    assert diagnostic["extended_corpus_valid"]["diagnostic_only"] is True
    assert "frozen comparison anchor" in diagnostic["extended_corpus_valid"]["non_canonical_reason"]
    assert report["official_baseline"]["query_subset"] == "fairship_only_valid"
