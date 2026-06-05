"""Tests for scripts/review_muon_dis_qrels.py."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "review_muon_dis_qrels.py"
    spec = importlib.util.spec_from_file_location("review_muon_dis_qrels", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_candidates(path: Path) -> None:
    payload = {
        "pack_id": "muon_dis_workflow_v1",
        "candidates": [
            {
                "query_id": "q03_run_simscript",
                "query_text": "run_simScript",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [{"file_path": "macro/run_simScript.py", "start_line": 10, "end_line": 20}],
            },
            {
                "query_id": "q02_make_muon_dis",
                "query_text": "makeMuonDIS",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [{"file_path": "muonDIS/makeMuonDIS.py", "start_line": 30, "end_line": 40}],
            },
            {
                "query_id": "q09_inactivate_muon_processes",
                "query_text": "InactivateMuonProcesses",
                "review_status": "NOT_FOUND_IN_INDEX",
                "qrels": [],
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_unbalanced_candidates(path: Path) -> None:
    payload = {
        "pack_id": "muon_dis_workflow_v1",
        "candidates": [
            {
                "query_id": "q02_make_muon_dis",
                "query_text": "makeMuonDIS",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [
                    {"file_path": "muonDIS/makeMuonDIS.py", "start_line": 30, "end_line": 40},
                    {"file_path": "muonDIS/makeMuonDIS.py", "start_line": 41, "end_line": 50},
                    {"file_path": "muonDIS/makeMuonDIS.py", "start_line": 51, "end_line": 60},
                    {"file_path": "muonDIS/makeMuonDIS.py", "start_line": 61, "end_line": 70},
                ],
            },
            {
                "query_id": "q03_run_simscript",
                "query_text": "run_simScript",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [{"file_path": "macro/run_simScript.py", "start_line": 10, "end_line": 20}],
            },
            {
                "query_id": "q04_shipreco",
                "query_text": "ShipReco",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [{"file_path": "python/shipReco.py", "start_line": 10, "end_line": 20}],
            },
            {
                "query_id": "q05_doca",
                "query_text": "DOCA",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [{"file_path": "doca/module.py", "start_line": 10, "end_line": 20}],
            },
            {
                "query_id": "q06_sbt",
                "query_text": "SBT",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [{"file_path": "sbt/module.py", "start_line": 10, "end_line": 20}],
            },
            {
                "query_id": "q07_ubt",
                "query_text": "UBT",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [{"file_path": "ubt/module.py", "start_line": 10, "end_line": 20}],
            },
            {
                "query_id": "q08_muioni",
                "query_text": "muIoni",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [{"file_path": "muioni/module.py", "start_line": 10, "end_line": 20}],
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_decisions(path: Path) -> None:
    payload = {
        "review_id": "review-1",
        "decisions": [
            {
                "query_id": "q03_run_simscript",
                "file_path": "macro/run_simScript.py",
                "start_line": 10,
                "end_line": 20,
                "decision": "APPROVED",
                "relevance": 2,
                "reason": "validated",
                "reviewer_notes": "ok",
            },
            {
                "query_id": "q02_make_muon_dis",
                "file_path": "muonDIS/makeMuonDIS.py",
                "start_line": 30,
                "end_line": 40,
                "decision": "NEEDS_CONTEXT",
                "relevance": 1,
                "reason": "needs follow-up",
                "reviewer_notes": "pending",
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_list_mode_prints_rows(tmp_path: Path, capsys) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        _write_candidates(candidates)
        _write_decisions(decisions)
        assert module.main(["--list", "--candidates", str(candidates), "--decisions", str(decisions)]) == 0
        out = capsys.readouterr().out
        assert "q02_make_muon_dis" in out
        assert "q03_run_simscript" in out
    finally:
        os.chdir(cwd_before)


def test_dry_run_does_not_modify_decisions(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        _write_candidates(candidates)
        _write_decisions(decisions)

        before = decisions.read_text(encoding="utf-8")
        assert (
            module.main(
                [
                    "--candidates",
                    str(candidates),
                    "--decisions",
                    str(decisions),
                    "--set-decision",
                    "REJECTED",
                    "--reason",
                    "not relevant",
                    "--reviewer-notes",
                    "skip",
                    "--dry-run",
                ]
            )
            == 0
        )
        after = decisions.read_text(encoding="utf-8")
        assert before == after
    finally:
        os.chdir(cwd_before)


def test_approved_requires_reason_and_reviewer_notes(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        _write_candidates(candidates)
        _write_decisions(decisions)
        with pytest.raises(ValueError):
            module.main(
                [
                    "--candidates",
                    str(candidates),
                    "--decisions",
                    str(decisions),
                    "--set-decision",
                    "APPROVED",
                    "--reason",
                    "",
                    "--reviewer-notes",
                    "",
                ]
            )
    finally:
        os.chdir(cwd_before)


def test_inactivate_not_found_cannot_be_approved(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        _write_candidates(candidates)
        _write_decisions(decisions)
        with pytest.raises(ValueError):
            module.main(
                [
                    "--candidates",
                    str(candidates),
                    "--decisions",
                    str(decisions),
                    "--query-id",
                    "q09_inactivate_muon_processes",
                    "--set-decision",
                    "APPROVED",
                    "--reason",
                    "forced",
                    "--reviewer-notes",
                    "forced",
                ]
            )
    finally:
        os.chdir(cwd_before)


def test_critical_path_preset_orders_query_ids(tmp_path: Path, capsys) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        _write_candidates(candidates)
        _write_decisions(decisions)
        assert (
            module.main(
                [
                    "--list",
                    "--candidates",
                    str(candidates),
                    "--decisions",
                    str(decisions),
                    "--preset",
                    "critical-path",
                    "--limit",
                    "2",
                ]
            )
            == 0
        )
        lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("q0")]
        assert lines[0].startswith("q02_make_muon_dis")
        assert lines[1].startswith("q03_run_simscript")
    finally:
        os.chdir(cwd_before)


def test_filter_by_query_id(tmp_path: Path, capsys) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        _write_candidates(candidates)
        _write_decisions(decisions)
        assert (
            module.main(
                [
                    "--list",
                    "--candidates",
                    str(candidates),
                    "--decisions",
                    str(decisions),
                    "--query-id",
                    "q03_run_simscript",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "q03_run_simscript" in out
        assert "q02_make_muon_dis" not in out
    finally:
        os.chdir(cwd_before)


def test_filter_by_decision_status(tmp_path: Path, capsys) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        _write_candidates(candidates)
        _write_decisions(decisions)
        assert (
            module.main(
                [
                    "--list",
                    "--candidates",
                    str(candidates),
                    "--decisions",
                    str(decisions),
                    "--only",
                    "APPROVED",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "q03_run_simscript" in out
        assert "q02_make_muon_dis" not in out
    finally:
        os.chdir(cwd_before)


def test_critical_path_limit_10_is_area_balanced_and_includes_not_found_row(tmp_path: Path, capsys) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        _write_unbalanced_candidates(candidates)
        _write_decisions(decisions)
        assert (
            module.main(
                [
                    "--list",
                    "--candidates",
                    str(candidates),
                    "--decisions",
                    str(decisions),
                    "--preset",
                    "critical-path",
                    "--limit",
                    "10",
                ]
            )
            == 0
        )
        lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("q0")]
        assert len(lines) == 10
        first_batch = lines[:8]
        covered = {line.split(" | ")[0] for line in first_batch}
        assert "q02_make_muon_dis" in covered
        assert "q03_run_simscript" in covered
        assert "q04_shipreco" in covered
        assert "q05_doca" in covered
        assert "q06_sbt" in covered
        assert "q07_ubt" in covered
        assert "q08_muioni" in covered
        assert "q09_inactivate_muon_processes" in covered
        inactivate_line = next(line for line in first_batch if line.startswith("q09_inactivate_muon_processes"))
        assert "NOT_FOUND_IN_INDEX" in inactivate_line
    finally:
        os.chdir(cwd_before)
