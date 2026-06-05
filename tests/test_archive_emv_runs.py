"""Tests for scripts/archive_emv_runs.py."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "archive_emv_runs.py"
    spec = importlib.util.spec_from_file_location("archive_emv_runs", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_state(root: Path) -> None:
    path = root / "agents" / "codex_emv" / "heartbeat" / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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


def _write_qrel_files(root: Path) -> None:
    qrel_dir = root / "benchmarks" / "muon_dis"
    qrel_dir.mkdir(parents=True, exist_ok=True)
    for name in ("qrels.yaml", "qrels_candidates.yaml", "qrels_review_decisions.yaml"):
        (qrel_dir / name).write_text("{}", encoding="utf-8")


def _write_summary(root: Path, run_id: str, status: str, stamp: int) -> None:
    path = root / "reports" / f"{run_id}_vertical_slice_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": run_id, "acceptance_gate_status": status}), encoding="utf-8")
    os.utime(path, (stamp, stamp))


def _write_run_files(root: Path, run_id: str) -> None:
    (root / "reports" / f"{run_id}.md").write_text("report", encoding="utf-8")
    (root / "reports" / f"{run_id}_eval.json").write_text("{}", encoding="utf-8")
    evidence_dir = root / "evidence" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "manifest.json").write_text("{}", encoding="utf-8")
    artifact = root / "artifacts" / f"{run_id}_artifact.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}", encoding="utf-8")


def _prepare_fixture(root: Path) -> None:
    _write_state(root)
    _write_qrel_files(root)
    _write_summary(root, "run_pass_latest", "PASS", 2000)
    _write_summary(root, "run_fail_old", "FAIL", 1000)
    _write_run_files(root, "run_pass_latest")
    _write_run_files(root, "run_fail_old")


def test_dry_run_moves_nothing(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare_fixture(tmp_path)
        before = (tmp_path / "reports" / "run_fail_old.md").read_text(encoding="utf-8")
        manifest = module.archive_runs(execute=False, keep_latest_pass=True, include_failed=True, older_than_days=None)
        after = (tmp_path / "reports" / "run_fail_old.md").read_text(encoding="utf-8")
        assert before == after
        assert manifest["execute"] is False
        assert manifest["moved_items"] == []
    finally:
        os.chdir(cwd_before)


def test_latest_pass_is_preserved(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare_fixture(tmp_path)
        manifest = module.archive_runs(execute=False, keep_latest_pass=True, include_failed=True, older_than_days=None)
        assert manifest["latest_pass_run_id"] == "run_pass_latest"
        planned_paths = [row["path"] for row in manifest["planned_items"]]
        assert not any("run_pass_latest" in path for path in planned_paths)
    finally:
        os.chdir(cwd_before)


def test_failed_stale_run_is_listed(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare_fixture(tmp_path)
        manifest = module.archive_runs(execute=False, keep_latest_pass=True, include_failed=True, older_than_days=None)
        planned_paths = [row["path"] for row in manifest["planned_items"]]
        assert any("run_fail_old" in path for path in planned_paths)
    finally:
        os.chdir(cwd_before)


def test_execute_moves_only_allowed_files(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare_fixture(tmp_path)
        manifest = module.archive_runs(execute=True, keep_latest_pass=True, include_failed=True, older_than_days=None)
        assert (tmp_path / "reports" / "run_fail_old.md").exists() is False
        assert (tmp_path / "reports" / "archive" / "run_fail_old.md").exists() is True
        assert (tmp_path / "evidence" / "run_fail_old").exists() is False
        assert (tmp_path / "evidence" / "archive" / "run_fail_old").exists() is True
        assert (tmp_path / "reports" / "run_pass_latest.md").exists() is True
        assert (tmp_path / "benchmarks" / "muon_dis" / "qrels.yaml").exists() is True
        assert manifest["moved_items"]
    finally:
        os.chdir(cwd_before)


def test_archive_manifest_is_written(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _prepare_fixture(tmp_path)
        module.archive_runs(execute=False, keep_latest_pass=True, include_failed=True, older_than_days=None)
        manifest_path = tmp_path / "reports" / "emv_archive_manifest.json"
        assert manifest_path.exists()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "planned_items" in payload
        assert "latest_pass_run_id" in payload
    finally:
        os.chdir(cwd_before)


def test_overlapping_run_ids_do_not_cross_claim_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        _write_state(tmp_path)
        _write_qrel_files(tmp_path)
        _write_summary(tmp_path, "run", "FAIL", 1000)
        _write_summary(tmp_path, "run_2", "FAIL", 900)
        _write_run_files(tmp_path, "run")
        _write_run_files(tmp_path, "run_2")

        manifest = module.archive_runs(execute=False, keep_latest_pass=False, include_failed=True, older_than_days=None)

        planned_by_run: dict[str, list[str]] = {}
        for row in manifest["planned_items"]:
            planned_by_run.setdefault(str(row["run_id"]), []).append(str(row["path"]))

        run_paths = planned_by_run.get("run", [])
        run2_paths = planned_by_run.get("run_2", [])
        assert run_paths
        assert run2_paths
        assert not any("run_2" in path for path in run_paths)
        assert all("run_2" in path for path in run2_paths)
    finally:
        os.chdir(cwd_before)
