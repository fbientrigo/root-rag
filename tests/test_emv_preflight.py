"""Tests for scripts/emv_preflight.py."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "emv_preflight.py"
    spec = importlib.util.spec_from_file_location("emv_preflight", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_preflight_detects_latest_index_and_help(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    index_root = tmp_path / "indexes_fairship"
    old_dir = index_root / "old_idx"
    new_dir = index_root / "new_idx"
    old_dir.mkdir(parents=True)
    new_dir.mkdir()
    os.utime(old_dir, (1000, 1000))
    os.utime(new_dir, (2000, 2000))

    def fake_run(command, capture_output, text, check):
        return module.subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    summary = module.collect_preflight(index_root)

    assert summary["checks"]["pyyaml"]["ok"] is True
    assert summary["checks"]["root_rag_ask_help"]["ok"] is True
    assert summary["checks"]["root_rag_search_help"]["ok"] is True
    assert summary["checks"]["fairship_indexes"]["latest_index_id"] == "new_idx"
    assert summary["detected_latest_index_id"] == "new_idx"


def test_should_fail_when_root_rag_missing() -> None:
    module = _load_module()
    summary = {
        "checks": {
            "pyyaml": {"ok": True},
            "root_rag_ask_help": {"return_code": None},
            "root_rag_search_help": {"return_code": 0},
        }
    }
    assert module._should_fail(summary) is True
