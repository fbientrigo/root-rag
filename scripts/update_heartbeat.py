"""Small helper to update EMV heartbeat state.json deterministically."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Sequence


ALLOWED_VERDICTS = {"ACCEPT", "ACCEPT WITH NOTES", "BLOCKED"}
PRESETS: Dict[str, Dict[str, Any]] = {
    "qrel_review_pending": {
        "verdict": "ACCEPT WITH NOTES",
        "run_summary": "EMV harness operational; manual qrel review remains pending.",
        "open_items": [
            "Manual qrel review pending",
            "Guarded qrel promotion workflow pending",
            "Wiki claim promotion blocked until reviewed evidence exists",
        ],
        "next_prompt_file": "agents/codex_emv/heartbeat/next_prompt.md",
    }
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update agents/codex_emv/heartbeat/state.json")
    parser.add_argument(
        "--state-file",
        default=Path("agents/codex_emv/heartbeat/state.json"),
        type=Path,
        help="Heartbeat state JSON file path.",
    )
    parser.add_argument("--from-json", type=Path, default=None, help="Read update payload from JSON file.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None, help="Apply named heartbeat preset.")
    parser.add_argument("--verdict", choices=sorted(ALLOWED_VERDICTS))
    parser.add_argument("--run-summary", default=None, help="One-line summary for last_run_summary.")
    parser.add_argument("--open-item", action="append", default=[], help="Open item entry (repeatable).")
    parser.add_argument("--next-prompt-file", default=None, help="Required for ACCEPT WITH NOTES/BLOCKED.")
    return parser.parse_args(argv)


def _load_state(path: Path) -> Dict[str, Any]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return {}


def _load_update_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")

    required = {"verdict", "run_summary", "open_items", "next_prompt_file"}
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing keys in {path}: {', '.join(missing)}")

    verdict = payload["verdict"]
    if verdict not in ALLOWED_VERDICTS:
        raise ValueError(f"Invalid verdict in {path}: {verdict}")

    if not isinstance(payload["run_summary"], str):
        raise ValueError(f"Field run_summary must be string: {path}")

    open_items = payload["open_items"]
    if not isinstance(open_items, list) or not all(isinstance(item, str) for item in open_items):
        raise ValueError(f"Field open_items must be list[str]: {path}")

    next_prompt_file = payload["next_prompt_file"]
    if next_prompt_file is not None and not isinstance(next_prompt_file, str):
        raise ValueError(f"Field next_prompt_file must be string or null: {path}")

    return payload


def _resolve_update_fields(args: argparse.Namespace) -> Dict[str, Any]:
    if args.from_json is not None:
        return _load_update_payload(args.from_json)

    if args.preset is not None:
        return dict(PRESETS[args.preset])

    if not args.verdict:
        raise ValueError("--verdict is required unless --from-json or --preset is used")
    if not args.run_summary:
        raise ValueError("--run-summary is required unless --from-json or --preset is used")

    return {
        "verdict": args.verdict,
        "run_summary": args.run_summary,
        "open_items": list(args.open_item),
        "next_prompt_file": args.next_prompt_file,
    }


def update_state(
    *,
    path: Path,
    verdict: str,
    run_summary: str,
    open_items: list[str],
    next_prompt_file: str | None,
) -> Dict[str, Any]:
    if verdict in {"ACCEPT WITH NOTES", "BLOCKED"} and not next_prompt_file:
        raise ValueError("--next-prompt-file is required for ACCEPT WITH NOTES or BLOCKED")

    state = _load_state(path)
    state["last_verdict"] = verdict
    state["last_run_summary"] = run_summary
    state["open_items"] = open_items
    if next_prompt_file is not None:
        state["next_prompt_path"] = next_prompt_file
    state["last_updated"] = date.today().isoformat()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fields = _resolve_update_fields(args)
    update_state(
        path=args.state_file,
        verdict=str(fields["verdict"]),
        run_summary=str(fields["run_summary"]),
        open_items=list(fields["open_items"]),
        next_prompt_file=str(fields["next_prompt_file"]) if fields["next_prompt_file"] is not None else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
