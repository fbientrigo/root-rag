"""Tests for scripts/print_muon_dis_review_sheet.py."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import yaml


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "print_muon_dis_review_sheet.py"
    spec = importlib.util.spec_from_file_location("print_muon_dis_review_sheet", script_path)
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
                "query_id": "q01",
                "query_text": "muonDIS",
                "manifest_status": "HIT_OR_TEXT_EVIDENCE",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [
                    {
                        "file_path": "muonDIS/makeMuonDIS.py",
                        "start_line": 10,
                        "end_line": 20,
                        "relevance_candidate": 1,
                    },
                    {
                        "file_path": "macro/run_simScript.py",
                        "start_line": 30,
                        "end_line": 40,
                        "relevance_candidate": 1,
                    },
                ],
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


def _write_decisions(path: Path) -> None:
    payload = {
        "review_id": "review-1",
        "decisions": [
            {
                "query_id": "q01",
                "file_path": "muonDIS/makeMuonDIS.py",
                "start_line": 10,
                "end_line": 20,
                "decision": "NEEDS_CONTEXT",
                "relevance": 1,
                "reason": "manual review pending",
                "reviewer_notes": "check context",
            },
            {
                "query_id": "q01",
                "file_path": "macro/run_simScript.py",
                "start_line": 30,
                "end_line": 40,
                "decision": "APPROVED",
                "relevance": 2,
                "reason": "strong anchor",
                "reviewer_notes": "ok",
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_generates_markdown_sheet_from_candidates_and_decisions(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        output = Path("reports/muon_dis_qrel_review_sheet.md")
        _write_candidates(candidates)
        _write_decisions(decisions)

        assert module.main(["--candidates", str(candidates), "--decisions", str(decisions), "--output", str(output)]) == 0
        text = output.read_text(encoding="utf-8")
        assert "## Summary" in text
        assert "## Query Review Table" in text
        assert "muonDIS/makeMuonDIS.py" in text
        assert "## NOT_FOUND_IN_INDEX" in text
        assert "InactivateMuonProcesses" in text
    finally:
        os.chdir(cwd_before)


def test_filters_main_table_by_needs_context(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        output = Path("reports/muon_dis_qrel_review_sheet.md")
        _write_candidates(candidates)
        _write_decisions(decisions)

        assert (
            module.main(
                [
                    "--candidates",
                    str(candidates),
                    "--decisions",
                    str(decisions),
                    "--output",
                    str(output),
                    "--only",
                    "NEEDS_CONTEXT",
                ]
            )
            == 0
        )
        text = output.read_text(encoding="utf-8")
        assert "NEEDS_CONTEXT" in text
        assert "| q01 | muonDIS | 2 | macro/run_simScript.py | 30-40 | APPROVED |" not in text
    finally:
        os.chdir(cwd_before)


def test_detects_missing_decisions(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        output = Path("reports/muon_dis_qrel_review_sheet.md")
        _write_candidates(candidates)
        _write_decisions(decisions)
        # Drop one decision to force missing entry.
        payload = yaml.safe_load(decisions.read_text(encoding="utf-8"))
        payload["decisions"] = payload["decisions"][:1]
        decisions.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

        assert module.main(["--candidates", str(candidates), "--decisions", str(decisions), "--output", str(output)]) == 0
        text = output.read_text(encoding="utf-8")
        assert "MISSING_DECISION" in text
        assert "## Missing Decisions" in text
        assert "macro/run_simScript.py:30-40" in text
    finally:
        os.chdir(cwd_before)


def test_detects_orphan_decisions(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        output = Path("reports/muon_dis_qrel_review_sheet.md")
        _write_candidates(candidates)
        _write_decisions(decisions)

        payload = yaml.safe_load(decisions.read_text(encoding="utf-8"))
        payload["decisions"].append(
            {
                "query_id": "q_orphan",
                "file_path": "orphan/path.py",
                "start_line": 1,
                "end_line": 2,
                "decision": "REJECTED",
                "relevance": 0,
                "reason": "orphan",
                "reviewer_notes": "fix",
            }
        )
        decisions.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

        assert module.main(["--candidates", str(candidates), "--decisions", str(decisions), "--output", str(output)]) == 0
        text = output.read_text(encoding="utf-8")
        assert "## Orphan Decisions" in text
        assert "orphan/path.py:1-2" in text
    finally:
        os.chdir(cwd_before)


def test_does_not_modify_input_files(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        output = Path("reports/muon_dis_qrel_review_sheet.md")
        _write_candidates(candidates)
        _write_decisions(decisions)

        before_candidates = candidates.read_text(encoding="utf-8")
        before_decisions = decisions.read_text(encoding="utf-8")
        assert module.main(["--candidates", str(candidates), "--decisions", str(decisions), "--output", str(output)]) == 0
        after_candidates = candidates.read_text(encoding="utf-8")
        after_decisions = decisions.read_text(encoding="utf-8")

        assert before_candidates == after_candidates
        assert before_decisions == after_decisions
    finally:
        os.chdir(cwd_before)

