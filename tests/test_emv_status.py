"""Tests for scripts/emv_status.py."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import yaml


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "emv_status.py"
    spec = importlib.util.spec_from_file_location("emv_status", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_state(root: Path, verdict: str = "ACCEPT WITH NOTES") -> None:
    state_path = root / "agents" / "codex_emv" / "heartbeat" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "last_verdict": verdict,
                "last_run_summary": "summary",
                "next_prompt_path": "agents/codex_emv/heartbeat/next_prompt.md",
                "open_items": [],
                "last_updated": "2026-04-30",
            }
        ),
        encoding="utf-8",
    )


def _write_summary(root: Path, run_id: str, gate_status: str, stamp: int, qrels_state: str = "NO_CONFIRMED_QRELS") -> Path:
    summary_path = root / "reports" / f"{run_id}_vertical_slice_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"run_id": run_id, "acceptance_gate_status": gate_status, "qrels_state": qrels_state}),
        encoding="utf-8",
    )
    os.utime(summary_path, (stamp, stamp))
    return summary_path


def _write_candidates(path: Path) -> None:
    payload = {
        "pack_id": "muon_dis_workflow_v1",
        "candidates": [
            {
                "query_id": "q01",
                "query_text": "muonDIS",
                "manifest_status": "HIT_OR_TEXT_EVIDENCE",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [{"file_path": "muonDIS/makeMuonDIS.py", "start_line": 10, "end_line": 20}],
            },
            {
                "query_id": "q09",
                "query_text": "InactivateMuonProcesses",
                "manifest_status": "ZERO_HIT",
                "review_status": "NOT_FOUND_IN_INDEX",
                "qrels": [],
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_decisions(path: Path, decision: str = "NEEDS_CONTEXT") -> None:
    payload = {
        "review_id": "review-1",
        "decisions": [
            {
                "query_id": "q01",
                "file_path": "muonDIS/makeMuonDIS.py",
                "start_line": 10,
                "end_line": 20,
                "decision": decision,
                "relevance": 1,
                "reason": "reason",
                "reviewer_notes": "notes",
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_qrels(path: Path, confirmed_count: int = 0) -> None:
    confirmed_rows = []
    if confirmed_count > 0:
        confirmed_rows = [
            {
                "query_id": "q01",
                "qrels": [
                    {
                        "file_path": "muonDIS/makeMuonDIS.py",
                        "start_line": 10 + idx,
                        "end_line": 20 + idx,
                        "relevance": 1,
                    }
                    for idx in range(confirmed_count)
                ],
            }
        ]
    payload = {"pack_id": "muon_dis_workflow_v1", "confirmed_qrels": confirmed_rows, "pending_qrels": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _prepare_required_files(root: Path, decision: str = "NEEDS_CONTEXT", confirmed_count: int = 0) -> None:
    _write_state(root)
    _write_candidates(root / "benchmarks" / "muon_dis" / "qrels_candidates.yaml")
    _write_decisions(root / "benchmarks" / "muon_dis" / "qrels_review_decisions.yaml", decision=decision)
    _write_qrels(root / "benchmarks" / "muon_dis" / "qrels.yaml", confirmed_count=confirmed_count)


def test_status_includes_qrel_decision_counts(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare_required_files(tmp_path, decision="NEEDS_CONTEXT", confirmed_count=0)
        _write_summary(tmp_path, "run_pass", "PASS", 2000)
        summary = module.collect_status()
        assert summary["qrel_review_decision_count"] == 1
        assert summary["approved_decision_count"] == 0
        assert summary["rejected_decision_count"] == 0
        assert summary["needs_context_decision_count"] == 1
        assert summary["qrel_candidates_count"] == 1
        assert summary["not_found_in_index_count"] == 1
    finally:
        os.chdir(cwd_before)


def test_scientific_state_no_qrels_confirmed_when_no_approved(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare_required_files(tmp_path, decision="NEEDS_CONTEXT", confirmed_count=0)
        _write_summary(tmp_path, "run_pass", "PASS", 2000)
        summary = module.collect_status()
        assert summary["approved_decision_count"] == 0
        assert summary["scientific_state"] == "NO_QRELS_CONFIRMED"
    finally:
        os.chdir(cwd_before)


def test_scientific_state_ready_for_eval_when_confirmed_qrels_exist(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare_required_files(tmp_path, decision="APPROVED", confirmed_count=2)
        _write_summary(tmp_path, "run_pass", "PASS", 2000, qrels_state="HAS_CONFIRMED_QRELS")
        summary = module.collect_status()
        assert summary["confirmed_qrel_count"] == 2
        assert summary["scientific_state"] == "QRELS_CONFIRMED_READY_FOR_EVAL"
        assert summary["v0_freeze_allowed"] is True
    finally:
        os.chdir(cwd_before)


def test_markdown_mode_prints_readable_summary(tmp_path: Path, capsys) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare_required_files(tmp_path, decision="NEEDS_CONTEXT", confirmed_count=0)
        _write_summary(tmp_path, "run_pass", "PASS", 2000)
        assert module.main(["--markdown"]) == 0
        out = capsys.readouterr().out
        assert "# EMV Status" in out
        assert "scientific_state" in out
        assert "qrel_review_decision_count" in out
    finally:
        os.chdir(cwd_before)


def test_malformed_yaml_or_json_handled_clearly(tmp_path: Path, capsys) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        state_path = tmp_path / "agents" / "codex_emv" / "heartbeat" / "state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{not-json", encoding="utf-8")
        code = module.main([])
        out = capsys.readouterr().out
        assert code == 2
        assert "MALFORMED_REQUIRED_FILE" in out
    finally:
        os.chdir(cwd_before)
