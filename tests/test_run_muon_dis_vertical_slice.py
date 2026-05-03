"""Tests for scripts/run_muon_dis_vertical_slice.py."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_muon_dis_vertical_slice.py"
    spec = importlib.util.spec_from_file_location("run_muon_dis_vertical_slice", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vertical_slice_passes_acceptance_when_artifacts_exist(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        idx_path = tmp_path / "data" / "indexes_fairship" / "idx_a"
        idx_path.mkdir(parents=True)
        (idx_path / "fts.sqlite").write_text("", encoding="utf-8")

        def fake_run(command, capture_output, text, check):
            cmd_text = " ".join(command)
            if "run_query_pack.py" in cmd_text:
                evidence_dir = tmp_path / "evidence" / "run_pass"
                evidence_dir.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "queries": [
                        {"id": "q1", "status": "HIT_OR_TEXT_EVIDENCE"},
                        {"id": "q2", "status": "ZERO_HIT"},
                    ]
                }
                (evidence_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                return module.subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            if "generate_weekly_report.py" in cmd_text:
                report_path = tmp_path / "reports" / "run_pass.md"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text("# report\n", encoding="utf-8")
                return module.subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            if "evaluate_muon_dis_retrieval.py" in cmd_text:
                eval_path = tmp_path / "reports" / "run_pass_eval.json"
                eval_path.parent.mkdir(parents=True, exist_ok=True)
                eval_path.write_text(
                    json.dumps({"summary": {"qrels_state": "NO_CONFIRMED_QRELS", "text_evidence_unsupported_count": 0}}),
                    encoding="utf-8",
                )
                return module.subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            return module.subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(module.subprocess, "run", fake_run)

        config = module.VerticalSliceConfig(
            index_id=None,
            index_dir=Path("data/indexes_fairship"),
            run_id="run_pass",
            top_k=10,
            skip_tests=True,
        )
        summary = module.run_vertical_slice(config)

        assert summary["acceptance_gate_status"] == "PASS"
        assert summary["query_count"] == 2
        assert summary["error_count"] == 0
        assert (tmp_path / "reports" / "run_pass_vertical_slice_summary.json").exists()
    finally:
        os.chdir(cwd_before)


def test_vertical_slice_fails_when_all_queries_are_error(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        idx_path = tmp_path / "data" / "indexes_fairship" / "idx_a"
        idx_path.mkdir(parents=True)
        (idx_path / "index_manifest.json").write_text("{}", encoding="utf-8")

        def fake_run(command, capture_output, text, check):
            cmd_text = " ".join(command)
            if "run_query_pack.py" in cmd_text:
                evidence_dir = tmp_path / "evidence" / "run_fail"
                evidence_dir.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "queries": [
                        {"id": "q1", "status": "ERROR"},
                        {"id": "q2", "status": "ERROR"},
                    ]
                }
                (evidence_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                return module.subprocess.CompletedProcess(command, 1, stdout="", stderr="")
            if "generate_weekly_report.py" in cmd_text:
                report_path = tmp_path / "reports" / "run_fail.md"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text("# report\n", encoding="utf-8")
                return module.subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            if "evaluate_muon_dis_retrieval.py" in cmd_text:
                eval_path = tmp_path / "reports" / "run_fail_eval.json"
                eval_path.parent.mkdir(parents=True, exist_ok=True)
                eval_path.write_text(
                    json.dumps({"summary": {"qrels_state": "NO_CONFIRMED_QRELS", "text_evidence_unsupported_count": 0}}),
                    encoding="utf-8",
                )
                return module.subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            return module.subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(module.subprocess, "run", fake_run)

        config = module.VerticalSliceConfig(
            index_id=None,
            index_dir=Path("data/indexes_fairship"),
            run_id="run_fail",
            top_k=10,
            skip_tests=True,
        )
        summary = module.run_vertical_slice(config)

        assert summary["acceptance_gate_status"] == "FAIL"
        assert summary["query_count"] == 2
        assert summary["error_count"] == 2
    finally:
        os.chdir(cwd_before)


def test_valid_index_child_detected(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "data" / "indexes_fairship"
    valid = root / "valid_idx"
    valid.mkdir(parents=True)
    (valid / "fts.sqlite").write_text("", encoding="utf-8")

    config = module.VerticalSliceConfig(index_id=None, index_dir=root, run_id="x", top_k=10, skip_tests=True)
    resolved_dir, index_id, notes = module._resolve_index_target(config)
    assert resolved_dir == root
    assert index_id == "valid_idx"
    assert any("auto-detected" in note for note in notes)


def test_invalid_parent_child_without_markers_ignored(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "data" / "indexes_fairship"
    invalid = root / "invalid_idx"
    invalid.mkdir(parents=True)

    config = module.VerticalSliceConfig(index_id=None, index_dir=root, run_id="x", top_k=10, skip_tests=True)
    resolved_dir, index_id, _ = module._resolve_index_target(config)
    assert resolved_dir is None
    assert index_id is None


def test_data_alone_does_not_pick_indexes_fairship_as_index_id(tmp_path: Path) -> None:
    module = _load_module()
    # Simulate data/indexes_fairship existing as parent folder only (no marker files in child).
    (tmp_path / "data" / "indexes_fairship").mkdir(parents=True)

    config = module.VerticalSliceConfig(index_id=None, index_dir=tmp_path / "data", run_id="x", top_k=10, skip_tests=True)
    resolved_dir, index_id, _ = module._resolve_index_target(config)
    assert resolved_dir is None
    assert index_id is None


def test_missing_valid_index_fails_clearly(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        (tmp_path / "data").mkdir(parents=True)
        config = module.VerticalSliceConfig(index_id=None, index_dir=Path("data"), run_id="missing_idx", top_k=10, skip_tests=True)
        summary = module.run_vertical_slice(config)
        assert summary["acceptance_gate_status"] == "FAIL_MISSING_VALID_INDEX"
    finally:
        os.chdir(cwd_before)
