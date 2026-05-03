"""Tests for scripts/promote_muon_dis_qrels.py."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "promote_muon_dis_qrels.py"
    spec = importlib.util.spec_from_file_location("promote_muon_dis_qrels", script_path)
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
                "query_id": "q01_muondis_anchor",
                "qrels": [
                    {
                        "file_path": "muonDIS/makeMuonDIS.py",
                        "start_line": 351,
                        "end_line": 355,
                        "relevance_candidate": 1,
                    }
                ],
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_qrels(path: Path) -> None:
    payload = {
        "pack_id": "muon_dis_workflow_v1",
        "updated": "2026-04-30",
        "confirmed_qrels": [],
        "pending_qrels": [
            {
                "query_id": "q01_muondis_anchor",
                "pending_label": True,
                "qrels": [],
                "review_status": "NOT FOUND IN INDEX",
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_decisions(path: Path, *, decision: str, reason: str, reviewer_notes: str, file_path: str = "muonDIS/makeMuonDIS.py") -> None:
    payload = {
        "review_id": "review-1",
        "source_candidates_file": "benchmarks/muon_dis/qrels_candidates.yaml",
        "reviewed_by": "reviewer",
        "reviewed_at": "2026-04-30T00:00:00Z",
        "decisions": [
            {
                "query_id": "q01_muondis_anchor",
                "file_path": file_path,
                "start_line": 351,
                "end_line": 355,
                "decision": decision,
                "relevance": 2,
                "reason": reason,
                "reviewer_notes": reviewer_notes,
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_scaffold_defaults_to_needs_context() -> None:
    payload = yaml.safe_load(Path("benchmarks/muon_dis/qrels_review_decisions.yaml").read_text(encoding="utf-8"))
    decisions = payload["decisions"]
    assert decisions
    assert all(row["decision"] == "NEEDS_CONTEXT" for row in decisions)
    assert all(row["decision"] != "APPROVED" for row in decisions)


def test_dry_run_does_not_modify_qrels(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        qrels = Path("benchmarks/muon_dis/qrels.yaml")
        _write_candidates(candidates)
        _write_decisions(decisions, decision="APPROVED", reason="valid anchor", reviewer_notes="reviewed")
        _write_qrels(qrels)

        before = qrels.read_text(encoding="utf-8")
        summary = module.promote_qrels(
            decisions_path=decisions,
            qrels_path=qrels,
            dry_run=True,
            candidates_path=candidates,
        )
        after = qrels.read_text(encoding="utf-8")
        assert before == after
        assert summary["promoted_count"] == 1
    finally:
        os.chdir(cwd_before)


def test_approved_decision_is_promoted(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        qrels = Path("benchmarks/muon_dis/qrels.yaml")
        _write_candidates(candidates)
        _write_decisions(decisions, decision="APPROVED", reason="valid anchor", reviewer_notes="reviewed")
        _write_qrels(qrels)

        summary = module.promote_qrels(
            decisions_path=decisions,
            qrels_path=qrels,
            dry_run=False,
            candidates_path=candidates,
        )
        payload = yaml.safe_load(qrels.read_text(encoding="utf-8"))
        assert summary["promoted_count"] == 1
        assert payload["confirmed_qrels"][0]["query_id"] == "q01_muondis_anchor"
        assert payload["confirmed_qrels"][0]["qrels"][0]["relevance"] == 2
    finally:
        os.chdir(cwd_before)


def test_needs_context_is_not_promoted(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        qrels = Path("benchmarks/muon_dis/qrels.yaml")
        _write_candidates(candidates)
        _write_decisions(
            decisions,
            decision="NEEDS_CONTEXT",
            reason="needs more evidence",
            reviewer_notes="manual follow-up",
        )
        _write_qrels(qrels)

        summary = module.promote_qrels(
            decisions_path=decisions,
            qrels_path=qrels,
            dry_run=False,
            candidates_path=candidates,
        )
        payload = yaml.safe_load(qrels.read_text(encoding="utf-8"))
        assert summary["promoted_count"] == 0
        assert payload["confirmed_qrels"] == []
    finally:
        os.chdir(cwd_before)


def test_rejected_is_not_promoted(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        qrels = Path("benchmarks/muon_dis/qrels.yaml")
        _write_candidates(candidates)
        _write_decisions(decisions, decision="REJECTED", reason="not relevant", reviewer_notes="skip")
        _write_qrels(qrels)

        summary = module.promote_qrels(
            decisions_path=decisions,
            qrels_path=qrels,
            dry_run=False,
            candidates_path=candidates,
        )
        payload = yaml.safe_load(qrels.read_text(encoding="utf-8"))
        assert summary["promoted_count"] == 0
        assert payload["confirmed_qrels"] == []
    finally:
        os.chdir(cwd_before)


def test_invalid_candidate_range_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        qrels = Path("benchmarks/muon_dis/qrels.yaml")
        _write_candidates(candidates)
        _write_decisions(
            decisions,
            decision="APPROVED",
            reason="valid",
            reviewer_notes="reviewed",
            file_path="invalid/path.py",
        )
        _write_qrels(qrels)

        with pytest.raises(ValueError):
            module.promote_qrels(
                decisions_path=decisions,
                qrels_path=qrels,
                dry_run=False,
                candidates_path=candidates,
            )
    finally:
        os.chdir(cwd_before)


def test_approved_without_reason_or_notes_fails(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        qrels = Path("benchmarks/muon_dis/qrels.yaml")
        _write_candidates(candidates)
        _write_decisions(decisions, decision="APPROVED", reason="", reviewer_notes="")
        _write_qrels(qrels)

        with pytest.raises(ValueError):
            module.promote_qrels(
                decisions_path=decisions,
                qrels_path=qrels,
                dry_run=False,
                candidates_path=candidates,
            )
    finally:
        os.chdir(cwd_before)
