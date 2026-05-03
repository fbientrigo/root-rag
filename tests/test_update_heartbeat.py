"""Tests for scripts/update_heartbeat.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "update_heartbeat.py"
    spec = importlib.util.spec_from_file_location("update_heartbeat", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_update_state_requires_next_prompt_for_accept_with_notes(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(ValueError):
        module.update_state(
            path=tmp_path / "state.json",
            verdict="ACCEPT WITH NOTES",
            run_summary="summary",
            open_items=["x"],
            next_prompt_file=None,
        )


def test_update_state_writes_json(tmp_path: Path) -> None:
    module = _load_module()
    state_path = tmp_path / "state.json"
    module.update_state(
        path=state_path,
        verdict="ACCEPT",
        run_summary="done",
        open_items=["follow up"],
        next_prompt_file=None,
    )
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["last_verdict"] == "ACCEPT"
    assert payload["last_run_summary"] == "done"
    assert payload["open_items"] == ["follow up"]


def test_from_json_mode_updates_state(tmp_path: Path) -> None:
    module = _load_module()
    state_path = tmp_path / "state.json"
    payload_path = tmp_path / "heartbeat_payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "verdict": "ACCEPT WITH NOTES",
                "run_summary": "from-json-summary",
                "open_items": ["item1", "item2"],
                "next_prompt_file": "agents/codex_emv/heartbeat/next_prompt.md",
            }
        ),
        encoding="utf-8",
    )

    assert module.main(["--state-file", str(state_path), "--from-json", str(payload_path)]) == 0
    written = json.loads(state_path.read_text(encoding="utf-8"))
    assert written["last_verdict"] == "ACCEPT WITH NOTES"
    assert written["last_run_summary"] == "from-json-summary"
    assert written["open_items"] == ["item1", "item2"]
    assert written["next_prompt_path"] == "agents/codex_emv/heartbeat/next_prompt.md"


def test_preset_mode_updates_state(tmp_path: Path) -> None:
    module = _load_module()
    state_path = tmp_path / "state.json"
    assert module.main(["--state-file", str(state_path), "--preset", "qrel_review_pending"]) == 0
    written = json.loads(state_path.read_text(encoding="utf-8"))
    assert written["last_verdict"] == "ACCEPT WITH NOTES"
    assert written["last_run_summary"] == "EMV harness operational; manual qrel review remains pending."
    assert written["next_prompt_path"] == "agents/codex_emv/heartbeat/next_prompt.md"
    assert "Manual qrel review pending" in written["open_items"]


def test_accept_with_notes_from_json_requires_next_prompt(tmp_path: Path) -> None:
    module = _load_module()
    state_path = tmp_path / "state.json"
    payload_path = tmp_path / "heartbeat_payload_invalid.json"
    payload_path.write_text(
        json.dumps(
            {
                "verdict": "ACCEPT WITH NOTES",
                "run_summary": "missing-next-prompt",
                "open_items": ["item1"],
                "next_prompt_file": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        module.main(["--state-file", str(state_path), "--from-json", str(payload_path)])


def test_written_state_is_valid_json_with_cli_mode(tmp_path: Path) -> None:
    module = _load_module()
    state_path = tmp_path / "state.json"
    assert (
        module.main(
            [
                "--state-file",
                str(state_path),
                "--verdict",
                "ACCEPT",
                "--run-summary",
                "legacy-mode",
                "--open-item",
                "pending-1",
            ]
        )
        == 0
    )
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data["last_verdict"] == "ACCEPT"
