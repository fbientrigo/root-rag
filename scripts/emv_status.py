"""Authoritative EMV status command for Muon DIS harness state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


HEARTBEAT_PATH = Path("agents/codex_emv/heartbeat/state.json")
NEXT_PROMPT_DEFAULT = Path("agents/codex_emv/heartbeat/next_prompt.md")
REPORTS_DIR = Path("reports")
QREL_DIR = Path("benchmarks/muon_dis")
QREL_GLOB = "qrels_candidates*.yaml"
QREL_CANDIDATES_PATH = QREL_DIR / "qrels_candidates.yaml"
QREL_DECISIONS_PATH = QREL_DIR / "qrels_review_decisions.yaml"
QRELS_PATH = QREL_DIR / "qrels.yaml"
V0_FREEZE_PATH = QREL_DIR / "V0_FREEZE.md"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report Codex EMV harness status.")
    parser.add_argument("--markdown", action="store_true", help="Print markdown report instead of JSON.")
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return payload


def _iter_vertical_summaries() -> List[Path]:
    if not REPORTS_DIR.exists():
        return []
    return sorted(
        REPORTS_DIR.glob("*_vertical_slice_summary.json"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )


def _classify_summary(summary: Mapping[str, Any]) -> str:
    status = str(summary.get("acceptance_gate_status", "")).strip().upper()
    return "PASS" if status == "PASS" else "FAIL"


def _pick_latest(
    summaries: Sequence[Tuple[Path, Dict[str, Any], str]],
    wanted: str,
) -> Optional[Tuple[Path, Dict[str, Any], str]]:
    for row in summaries:
        if row[2] == wanted:
            return row
    return None


def _resolve_candidate_file() -> Optional[Path]:
    if QREL_CANDIDATES_PATH.exists():
        return QREL_CANDIDATES_PATH
    candidates = sorted(QREL_DIR.glob(QREL_GLOB), key=lambda row: row.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _count_candidates(payload: Mapping[str, Any]) -> Tuple[int, int]:
    rows = payload.get("candidates")
    if rows is None:
        return 0, 0
    if not isinstance(rows, list):
        raise ValueError("Candidates payload must contain list field: candidates")

    candidate_count = 0
    not_found = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        qrels = row.get("qrels")
        review_status = row.get("review_status")
        manifest_status = row.get("manifest_status")
        if not isinstance(qrels, list):
            qrels = []
        candidate_count += len(qrels)
        if review_status == "NOT_FOUND_IN_INDEX" or manifest_status == "ZERO_HIT" or len(qrels) == 0:
            not_found += 1
    return candidate_count, not_found


def _count_decisions(payload: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    rows = payload.get("decisions")
    if rows is None:
        return 0, 0, 0, 0
    if not isinstance(rows, list):
        raise ValueError("Decision payload must contain list field: decisions")

    approved = 0
    rejected = 0
    needs_context = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        decision = str(row.get("decision", "")).strip().upper()
        if decision == "APPROVED":
            approved += 1
        elif decision == "REJECTED":
            rejected += 1
        elif decision == "NEEDS_CONTEXT":
            needs_context += 1
    return len(rows), approved, rejected, needs_context


def _count_confirmed_qrels(payload: Mapping[str, Any]) -> int:
    rows = payload.get("confirmed_qrels")
    if rows is None:
        return 0
    if not isinstance(rows, list):
        raise ValueError("Qrels payload must contain list field: confirmed_qrels")
    count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        qrels = row.get("qrels")
        if isinstance(qrels, list):
            count += len(qrels)
    return count


def _scientific_state(
    *,
    confirmed_qrel_count: int,
    approved_decision_count: int,
    v0_freeze_exists: bool,
) -> str:
    if v0_freeze_exists:
        return "V0_FROZEN"
    if confirmed_qrel_count > 0:
        return "QRELS_CONFIRMED_READY_FOR_EVAL"
    if approved_decision_count > 0:
        return "QRELS_REVIEW_IN_PROGRESS"
    return "NO_QRELS_CONFIRMED"


def _build_notes(*, latest_pass_exists: bool, latest_fail_exists: bool, malformed_paths: Iterable[str]) -> List[str]:
    notes: List[str] = []
    if not latest_pass_exists:
        notes.append("No PASS vertical slice summary found.")
    if latest_fail_exists:
        notes.append("Latest FAIL summary present; inspect failure artifact.")
    malformed = list(malformed_paths)
    if malformed:
        notes.append("Malformed state file(s) detected.")
    return notes


def collect_status() -> Dict[str, Any]:
    malformed_paths: List[str] = []
    if not HEARTBEAT_PATH.exists():
        raise FileNotFoundError(f"Missing heartbeat state file: {HEARTBEAT_PATH}")
    heartbeat = _load_json_object(HEARTBEAT_PATH)

    summary_rows: List[Tuple[Path, Dict[str, Any], str]] = []
    for summary_path in _iter_vertical_summaries():
        try:
            payload = _load_json_object(summary_path)
        except (json.JSONDecodeError, ValueError):
            malformed_paths.append(str(summary_path))
            continue
        summary_rows.append((summary_path, payload, _classify_summary(payload)))

    latest_pass = _pick_latest(summary_rows, "PASS")
    latest_fail = _pick_latest(summary_rows, "FAIL")

    decision_payload = _load_yaml_mapping(QREL_DECISIONS_PATH)
    qrels_payload = _load_yaml_mapping(QRELS_PATH)

    candidate_files = sorted([str(path) for path in QREL_DIR.glob(QREL_GLOB)], reverse=True)
    candidate_path = _resolve_candidate_file()
    candidate_payload: Dict[str, Any] = {}
    if candidate_path is not None:
        candidate_payload = _load_yaml_mapping(candidate_path)
    candidate_count, not_found_count = _count_candidates(candidate_payload)
    review_count, approved_count, rejected_count, needs_context_count = _count_decisions(decision_payload)
    confirmed_qrel_count = _count_confirmed_qrels(qrels_payload)

    heartbeat_verdict = str(heartbeat.get("last_verdict", "UNKNOWN"))
    raw_next_prompt = heartbeat.get("next_prompt_path")
    next_prompt_path = str(raw_next_prompt) if isinstance(raw_next_prompt, str) else str(NEXT_PROMPT_DEFAULT)

    latest_pass_run_id = str(latest_pass[1].get("run_id")) if latest_pass else None
    latest_fail_run_id = str(latest_fail[1].get("run_id")) if latest_fail else None
    latest_pass_summary_path = str(latest_pass[0]) if latest_pass else None
    latest_fail_summary_path = str(latest_fail[0]) if latest_fail else None
    acceptance_state = "PASS" if latest_pass else "FAIL"
    qrels_state = str((latest_pass[1] if latest_pass else {}).get("qrels_state", "NO_CONFIRMED_QRELS"))

    v0_freeze_exists = V0_FREEZE_PATH.exists()
    v0_freeze_allowed = bool(latest_pass and confirmed_qrel_count > 0)
    scientific_state = _scientific_state(
        confirmed_qrel_count=confirmed_qrel_count,
        approved_decision_count=approved_count,
        v0_freeze_exists=v0_freeze_exists,
    )

    notes = _build_notes(
        latest_pass_exists=latest_pass is not None,
        latest_fail_exists=latest_fail is not None,
        malformed_paths=malformed_paths,
    )

    return {
        "heartbeat_verdict": heartbeat_verdict,
        "next_prompt_path": next_prompt_path,
        "latest_pass_run_id": latest_pass_run_id,
        "latest_fail_run_id": latest_fail_run_id,
        "acceptance_state": acceptance_state,
        "qrel_candidates_count": candidate_count,
        "qrel_review_decision_count": review_count,
        "approved_decision_count": approved_count,
        "rejected_decision_count": rejected_count,
        "needs_context_decision_count": needs_context_count,
        "confirmed_qrel_count": confirmed_qrel_count,
        "not_found_in_index_count": not_found_count,
        "v0_freeze_exists": v0_freeze_exists,
        "v0_freeze_allowed": v0_freeze_allowed,
        "scientific_state": scientific_state,
        "qrels_state": qrels_state,
        "latest_pass_summary_path": latest_pass_summary_path,
        "latest_fail_summary_path": latest_fail_summary_path,
        "qrel_candidate_files": candidate_files,
        "notes": notes,
        "_malformed_state_paths": malformed_paths,
    }


def _render_markdown(status: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# EMV Status")
    lines.append("")
    lines.append("## Harness")
    lines.append("")
    lines.append(f"- heartbeat_verdict: `{status['heartbeat_verdict']}`")
    lines.append(f"- next_prompt_path: `{status['next_prompt_path']}`")
    lines.append(f"- latest_pass_run_id: `{status['latest_pass_run_id']}`")
    lines.append(f"- latest_fail_run_id: `{status['latest_fail_run_id']}`")
    lines.append(f"- acceptance_state: `{status['acceptance_state']}`")
    lines.append("")
    lines.append("## Qrels")
    lines.append("")
    lines.append(f"- qrel_candidates_count: `{status['qrel_candidates_count']}`")
    lines.append(f"- qrel_review_decision_count: `{status['qrel_review_decision_count']}`")
    lines.append(f"- approved_decision_count: `{status['approved_decision_count']}`")
    lines.append(f"- rejected_decision_count: `{status['rejected_decision_count']}`")
    lines.append(f"- needs_context_decision_count: `{status['needs_context_decision_count']}`")
    lines.append(f"- confirmed_qrel_count: `{status['confirmed_qrel_count']}`")
    lines.append(f"- not_found_in_index_count: `{status['not_found_in_index_count']}`")
    lines.append(f"- qrels_state: `{status['qrels_state']}`")
    lines.append("")
    lines.append("## Scientific State")
    lines.append("")
    lines.append(f"- scientific_state: `{status['scientific_state']}`")
    lines.append(f"- v0_freeze_exists: `{status['v0_freeze_exists']}`")
    lines.append(f"- v0_freeze_allowed: `{status['v0_freeze_allowed']}`")
    if status.get("notes"):
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        for note in status["notes"]:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _error_payload(message: str) -> Dict[str, Any]:
    return {
        "heartbeat_verdict": "UNKNOWN",
        "next_prompt_path": str(NEXT_PROMPT_DEFAULT),
        "latest_pass_run_id": None,
        "latest_fail_run_id": None,
        "acceptance_state": "FAIL",
        "qrel_candidates_count": 0,
        "qrel_review_decision_count": 0,
        "approved_decision_count": 0,
        "rejected_decision_count": 0,
        "needs_context_decision_count": 0,
        "confirmed_qrel_count": 0,
        "not_found_in_index_count": 0,
        "v0_freeze_exists": False,
        "v0_freeze_allowed": False,
        "scientific_state": "NO_QRELS_CONFIRMED",
        "notes": [f"MALFORMED_REQUIRED_FILE: {message}"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        status = collect_status()
    except (FileNotFoundError, json.JSONDecodeError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps(_error_payload(str(exc)), indent=2))
        return 2

    public_status = {key: value for key, value in status.items() if not key.startswith("_")}
    if args.markdown:
        print(_render_markdown(public_status))
    else:
        print(json.dumps(public_status, indent=2))

    if status["_malformed_state_paths"]:
        return 2
    if status["latest_pass_run_id"] is None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
