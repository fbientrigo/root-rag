"""Tests for scripts/emv_status.py."""
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
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "emv_status.py"
    spec = importlib.util.spec_from_file_location("emv_status", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_state(root: Path) -> None:
    state_path = root / "agents" / "codex_emv" / "heartbeat" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "last_verdict": "ACCEPT WITH NOTES",
                "last_run_summary": "summary",
                "next_prompt_path": "agents/codex_emv/heartbeat/next_prompt.md",
                "open_items": [],
                "last_updated": "2026-05-03",
            }
        ),
        encoding="utf-8",
    )


def _write_summary(root: Path, run_id: str = "run_pass") -> None:
    summary_path = root / "reports" / f"{run_id}_vertical_slice_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"run_id": run_id, "acceptance_gate_status": "PASS", "qrels_state": "NO_CONFIRMED_QRELS"}),
        encoding="utf-8",
    )


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
    inactivate = {
        "query_id": "q09_inactivate_muon_processes",
        "query_text": "InactivateMuonProcesses",
        "manifest_status": "ZERO_HIT",
        "review_status": "NOT_FOUND_IN_INDEX" if include_not_found else "REVIEW_REQUIRED",
        "qrels": [],
    }
    rows.append(inactivate)

    payload = {"pack_id": "muon_dis_workflow_v1", "candidates": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_decisions(path: Path, approved_queries: list[str]) -> None:
    rows = []
    for query_id in MANDATORY:
        decision = "APPROVED" if query_id in approved_queries else "NEEDS_CONTEXT"
        reason = "validated anchor" if decision == "APPROVED" else "pending review"
        notes = "manual verified" if decision == "APPROVED" else "pending"
        rows.append(
            {
                "query_id": query_id,
                "file_path": f"{query_id}.py",
                "start_line": 10,
                "end_line": 20,
                "decision": decision,
                "relevance": 1,
                "reason": reason,
                "reviewer_notes": notes,
            }
        )
    payload = {"review_id": "review-1", "decisions": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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


def _write_agent_adjudicated(path: Path, *, approved_count: int) -> None:
    rows = []
    for idx in range(approved_count):
        rows.append(
            {
                "query_id": f"q{idx + 1:02d}",
                "query_text": "x",
                "file_path": f"f{idx}.py",
                "start_line": 1,
                "end_line": 2,
                "rank": 1,
                "agent_decision": "AGENT_APPROVED",
                "proposed_relevance": 3,
                "confidence": "HIGH",
                "reasoning": "ok",
                "residual_risk": "risk",
                "label_source": "codex_agent",
                "human_review_required": True,
            }
        )
    payload = {
        "adjudicated": rows,
        "summary": {
            "agent_adjudicated_count": approved_count,
            "agent_approved_count": approved_count,
            "agent_rejected_count": 0,
            "agent_needs_context_count": 0,
            "human_approved_count": 0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _prepare(root: Path, *, approved_queries: list[str], confirmed_queries: list[str], include_not_found: bool) -> None:
    _write_state(root)
    _write_summary(root)
    _write_candidates(root / "benchmarks" / "muon_dis" / "qrels_candidates.yaml", include_not_found=include_not_found)
    _write_decisions(root / "benchmarks" / "muon_dis" / "qrels_review_decisions.yaml", approved_queries=approved_queries)
    _write_qrels(root / "benchmarks" / "muon_dis" / "qrels.yaml", confirmed_queries=confirmed_queries)


def test_no_approved_qrels_gives_no_qrels_confirmed_state(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare(tmp_path, approved_queries=[], confirmed_queries=[], include_not_found=True)
        summary = module.collect_status()
        assert summary["v0_readiness_state"] == "NO_QRELS_CONFIRMED"
        assert summary["scientific_state"] == "NO_QRELS_CONFIRMED"
        assert summary["v0_coverage_ready"] is False
    finally:
        os.chdir(cwd_before)


def test_partial_approvals_give_partial_coverage(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare(
            tmp_path,
            approved_queries=["q02_make_muon_dis", "q03_run_simscript"],
            confirmed_queries=[],
            include_not_found=True,
        )
        summary = module.collect_status()
        assert summary["v0_readiness_state"] == "PARTIAL_COVERAGE"
        assert summary["v0_coverage_ready"] is False
        assert "ShipReco" in summary["missing_v0_coverage_areas"]
    finally:
        os.chdir(cwd_before)


def test_missing_inactivate_review_blocks_coverage(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare(tmp_path, approved_queries=MANDATORY, confirmed_queries=[], include_not_found=False)
        summary = module.collect_status()
        assert summary["v0_coverage_ready"] is False
        assert "InactivateMuonProcesses" in summary["missing_v0_coverage_areas"]
        assert summary["v0_readiness_state"] == "PARTIAL_COVERAGE"
    finally:
        os.chdir(cwd_before)


def test_reviewed_not_found_inactivate_satisfies_area_without_qrel(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare(tmp_path, approved_queries=MANDATORY, confirmed_queries=[], include_not_found=True)
        summary = module.collect_status()
        assert summary["confirmed_qrel_count"] == 0
        assert "InactivateMuonProcesses" in summary["reviewed_not_found_areas"]
        assert summary["v0_coverage_ready"] is True
        assert summary["v0_readiness_state"] == "COVERAGE_READY_FOR_FREEZE"
    finally:
        os.chdir(cwd_before)


def test_markdown_mode_prints_v0_fields(tmp_path: Path, capsys) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare(tmp_path, approved_queries=[], confirmed_queries=[], include_not_found=True)
        assert module.main(["--markdown"]) == 0
        out = capsys.readouterr().out
        assert "v0_coverage_ready" in out
        assert "missing_v0_coverage_areas" in out
        assert "v0_readiness_state" in out
    finally:
        os.chdir(cwd_before)


def test_emv_status_reports_agent_adjudication_counts_and_scientific_state_gate(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare(tmp_path, approved_queries=MANDATORY, confirmed_queries=[], include_not_found=True)
        _write_agent_adjudicated(
            tmp_path / "benchmarks" / "muon_dis" / "qrels_agent_adjudicated.yaml",
            approved_count=2,
        )
        summary = module.collect_status()
        assert summary["agent_adjudicated_count"] == 2
        assert summary["agent_approved_count"] == 2
        assert summary["human_approved_count"] == 0
        assert summary["confirmed_qrel_count"] == 0
        assert summary["scientific_state"] == "NO_QRELS_CONFIRMED"
    finally:
        os.chdir(cwd_before)
