"""Tests for scripts/agent_preflight.py."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "agent_preflight.py"
    spec = importlib.util.spec_from_file_location("agent_preflight", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_preflight_lists_latest_index_dir(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    index_root = tmp_path / "indexes_fairship"
    older = index_root / "idx_old"
    newer = index_root / "idx_new"
    older.mkdir(parents=True)
    newer.mkdir()
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    def fake_run(command, capture_output, text, check):
        return module.subprocess.CompletedProcess(command, 0, stdout="help", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.collect_preflight(index_root)

    assert result["python_executable"]
    assert result["pyyaml"]["ok"] is True
    assert result["root_rag_ask_help"]["ok"] is True
    assert result["fairship_indexes"]["exists"] is True
    assert result["fairship_indexes"]["latest_index_id"] == "idx_new"
    assert len(result["fairship_indexes"]["dirs"]) == 2
