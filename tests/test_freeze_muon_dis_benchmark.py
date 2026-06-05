"""Tests for scripts/freeze_muon_dis_benchmark.py."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import yaml


MANDATORY = [
    "q02_make_muon_dis",
    "q03_run_simscript",
    "q04_shipreco",
    "q05_doca",
    "q06_sbt",
    "q07_ubt",
    "q08_muioni",
]


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "freeze_muon_dis_benchmark.py"
    spec = importlib.util.spec_from_file_location("freeze_muon_dis_benchmark", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_summary(path: Path, run_id: str = "muon_dis_emv_reconciled_01") -> None:
    payload = {
        "run_id": run_id,
        "acceptance_gate_status": "PASS",
        "index_id": "fairship_index",
        "index_dir": "data/indexes_fairship",
        "query_count": 9,
        "hit_count": 8,
        "zero_hit_count": 1,
        "error_count": 0,
        "qrels_state": "NO_CONFIRMED_QRELS",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_qrels(path: Path, confirmed_queries: list[str]) -> None:
    confirmed_rows = []
    for query_id in confirmed_queries:
        confirmed_rows.append(
            {
                "query_id": query_id,
                "qrels": [{"file_path": f"{query_id}.py", "start_line": 10, "end_line": 20, "relevance": 1}],
            }
        )
    payload = {"pack_id": "muon_dis_workflow_v1", "confirmed_qrels": confirmed_rows, "pending_qrels": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_candidates(path: Path, *, include_not_found: bool) -> None:
    rows = []
    for query_id in MANDATORY:
        rows.append(
            {
                "query_id": query_id,
                "query_text": query_id,
                "manifest_status": "HIT_OR_TEXT_EVIDENCE",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [{"file_path": f"{query_id}.py", "start_line": 10, "end_line": 20}],
            }
        )
    rows.append(
        {
            "query_id": "q09_inactivate_muon_processes",
            "query_text": "InactivateMuonProcesses",
            "manifest_status": "ZERO_HIT",
            "review_status": "NOT_FOUND_IN_INDEX" if include_not_found else "REVIEW_REQUIRED",
            "qrels": [],
        }
    )
    payload = {"pack_id": "muon_dis_workflow_v1", "candidates": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_decisions(path: Path, approved_queries: list[str]) -> None:
    rows = []
    for query_id in MANDATORY:
        rows.append(
            {
                "query_id": query_id,
                "file_path": f"{query_id}.py",
                "start_line": 10,
                "end_line": 20,
                "decision": "APPROVED" if query_id in approved_queries else "NEEDS_CONTEXT",
                "relevance": 1,
                "reason": "validated" if query_id in approved_queries else "pending",
                "reviewer_notes": "ok" if query_id in approved_queries else "pending",
            }
        )
    payload = {"review_id": "review-1", "decisions": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_plain_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _prepare(
    root: Path,
    *,
    confirmed_queries: list[str],
    approved_queries: list[str],
    include_not_found: bool,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    summary = root / "reports" / "run_vertical_slice_summary.json"
    qrels = root / "benchmarks" / "muon_dis" / "qrels.yaml"
    candidates = root / "benchmarks" / "muon_dis" / "qrels_candidates.yaml"
    decisions = root / "benchmarks" / "muon_dis" / "qrels_review_decisions.yaml"
    golden = root / "benchmarks" / "muon_dis" / "golden_queries.yaml"
    query_pack = root / "query_packs" / "muon_dis_workflow.yaml"
    output = root / "benchmarks" / "muon_dis" / "V0_FREEZE.md"
    json_output = root / "artifacts" / "benchmarks" / "muon_dis_v0_freeze.json"

    _write_summary(summary)
    _write_qrels(qrels, confirmed_queries=confirmed_queries)
    _write_candidates(candidates, include_not_found=include_not_found)
    _write_decisions(decisions, approved_queries=approved_queries)
    _write_plain_file(golden, "golden_queries: []\n")
    _write_plain_file(query_pack, "queries: []\n")
    return summary, qrels, candidates, decisions, golden, query_pack, output, json_output


def test_freeze_refuses_final_v0_when_coverage_incomplete_even_if_qrel_threshold_met(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        summary, qrels, candidates, decisions, golden, query_pack, output, json_output = _prepare(
            tmp_path,
            confirmed_queries=MANDATORY,
            approved_queries=[],
            include_not_found=False,
        )
        result = module.freeze_benchmark(
            summary_path=summary,
            qrels_path=qrels,
            candidates_path=candidates,
            decisions_path=decisions,
            golden_path=golden,
            query_pack_path=query_pack,
            output_path=output,
            json_output_path=json_output,
            min_confirmed_qrels=5,
            draft=False,
        )
        assert result["freeze_status"] == "BLOCKED_COVERAGE_INCOMPLETE"
        assert "InactivateMuonProcesses" in result["missing_v0_coverage_areas"]
        assert output.exists() is False
    finally:
        os.chdir(cwd_before)


def test_draft_freeze_still_works_and_is_marked(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        summary, qrels, candidates, decisions, golden, query_pack, output, json_output = _prepare(
            tmp_path,
            confirmed_queries=[],
            approved_queries=[],
            include_not_found=False,
        )
        result = module.freeze_benchmark(
            summary_path=summary,
            qrels_path=qrels,
            candidates_path=candidates,
            decisions_path=decisions,
            golden_path=golden,
            query_pack_path=query_pack,
            output_path=output,
            json_output_path=json_output,
            min_confirmed_qrels=5,
            draft=True,
        )
        draft_path = output.with_name("V0_FREEZE_DRAFT.md")
        text = draft_path.read_text(encoding="utf-8")
        assert result["freeze_status"] == "DRAFT_NOT_FINAL"
        assert "DRAFT / NOT A BENCHMARK FREEZE" in text
        assert output.exists() is False
    finally:
        os.chdir(cwd_before)


def test_reviewed_not_found_inactivate_satisfies_area_without_confirmed_qrel(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        summary, qrels, candidates, decisions, golden, query_pack, output, json_output = _prepare(
            tmp_path,
            confirmed_queries=[],
            approved_queries=MANDATORY,
            include_not_found=True,
        )
        result = module.freeze_benchmark(
            summary_path=summary,
            qrels_path=qrels,
            candidates_path=candidates,
            decisions_path=decisions,
            golden_path=golden,
            query_pack_path=query_pack,
            output_path=output,
            json_output_path=json_output,
            min_confirmed_qrels=5,
            draft=False,
        )
        assert result["freeze_status"] == "BLOCKED_INSUFFICIENT_QRELS"
        assert "InactivateMuonProcesses" in result["reviewed_not_found_areas"]
        assert result["confirmed_qrel_count"] == 0
    finally:
        os.chdir(cwd_before)


def test_main_draft_mode_uses_draft_json_default_path(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        summary, qrels, candidates, decisions, golden, query_pack, output, _ = _prepare(
            tmp_path,
            confirmed_queries=[],
            approved_queries=[],
            include_not_found=False,
        )
        assert (
            module.main(
                [
                    "--summary",
                    str(summary),
                    "--qrels",
                    str(qrels),
                    "--candidates",
                    str(candidates),
                    "--decisions",
                    str(decisions),
                    "--golden",
                    str(golden),
                    "--query-pack",
                    str(query_pack),
                    "--output",
                    str(output),
                    "--draft",
                ]
            )
            == 0
        )
        assert (tmp_path / "artifacts" / "benchmarks" / "muon_dis_v0_freeze_draft.json").exists()
        assert (tmp_path / "artifacts" / "benchmarks" / "muon_dis_v0_freeze.json").exists() is False
    finally:
        os.chdir(cwd_before)
