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
QREL_AGENT_ADJUDICATED_PATH = QREL_DIR / "qrels_agent_adjudicated.yaml"
V0_FREEZE_PATH = QREL_DIR / "V0_FREEZE.md"

MANDATORY_V0_AREAS: Mapping[str, str] = {
    "q02_make_muon_dis": "makeMuonDIS",
    "q03_run_simscript": "run_simScript",
    "q04_shipreco": "ShipReco",
    "q05_doca": "DOCA",
    "q06_sbt": "SBT",
    "q07_ubt": "UBT",
    "q08_muioni": "muIoni",
    "q09_inactivate_muon_processes": "InactivateMuonProcesses",
}


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


def _count_confirmed_qrels(payload: Mapping[str, Any]) -> Tuple[int, Dict[str, int]]:
    rows = payload.get("confirmed_qrels")
    if rows is None:
        return 0, {}
    if not isinstance(rows, list):
        raise ValueError("Qrels payload must contain list field: confirmed_qrels")
    count = 0
    by_query: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        query_id = row.get("query_id")
        qrels = row.get("qrels")
        if isinstance(query_id, str) and isinstance(qrels, list):
            by_query[query_id] = by_query.get(query_id, 0) + len(qrels)
            count += len(qrels)
    return count, by_query


def _count_agent_adjudicated(payload: Mapping[str, Any]) -> Tuple[int, int, int, int, int]:
    rows = payload.get("adjudicated")
    if not isinstance(rows, list):
        return 0, 0, 0, 0, 0
    approved = 0
    rejected = 0
    needs_context = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        decision = str(row.get("agent_decision", "")).strip().upper()
        if decision == "AGENT_APPROVED":
            approved += 1
        elif decision == "AGENT_REJECTED":
            rejected += 1
        elif decision == "AGENT_NEEDS_CONTEXT":
            needs_context += 1
    summary = payload.get("summary")
    human_approved_count = 0
    if isinstance(summary, dict):
        raw = summary.get("human_approved_count")
        if isinstance(raw, int):
            human_approved_count = raw
    return len(rows), approved, rejected, needs_context, human_approved_count


def _approved_by_query(decisions_payload: Mapping[str, Any]) -> Dict[str, int]:
    rows = decisions_payload.get("decisions")
    if not isinstance(rows, list):
        return {}
    output: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        query_id = row.get("query_id")
        decision = str(row.get("decision", "")).strip().upper()
        if isinstance(query_id, str) and decision == "APPROVED":
            output[query_id] = output.get(query_id, 0) + 1
    return output


def _not_found_reviewed_queries(candidates_payload: Mapping[str, Any]) -> set[str]:
    rows = candidates_payload.get("candidates")
    if not isinstance(rows, list):
        return set()
    output: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        query_id = row.get("query_id")
        review_status = str(row.get("review_status", "")).strip().upper()
        if isinstance(query_id, str) and review_status == "NOT_FOUND_IN_INDEX":
            output.add(query_id)
    return output


def _compute_v0_coverage(
    *,
    confirmed_by_query: Mapping[str, int],
    approved_by_query: Mapping[str, int],
    not_found_reviewed_queries: set[str],
) -> Tuple[bool, List[str], List[str]]:
    missing_areas: List[str] = []
    reviewed_not_found_areas: List[str] = []
    for query_id, area_name in MANDATORY_V0_AREAS.items():
        if query_id == "q09_inactivate_muon_processes":
            if query_id in not_found_reviewed_queries:
                reviewed_not_found_areas.append(area_name)
            elif confirmed_by_query.get(query_id, 0) > 0 or approved_by_query.get(query_id, 0) > 0:
                pass
            else:
                missing_areas.append(area_name)
            continue

        has_confirmed = confirmed_by_query.get(query_id, 0) > 0
        has_approved = approved_by_query.get(query_id, 0) > 0
        if not (has_confirmed or has_approved):
            missing_areas.append(area_name)

    return (len(missing_areas) == 0), missing_areas, reviewed_not_found_areas


def _derive_v0_readiness_state(
    *,
    v0_freeze_exists: bool,
    confirmed_qrel_count: int,
    approved_decision_count: int,
    v0_coverage_ready: bool,
) -> str:
    if v0_freeze_exists:
        return "V0_FROZEN"
    if confirmed_qrel_count == 0 and approved_decision_count == 0:
        return "NO_QRELS_CONFIRMED"
    if v0_coverage_ready:
        return "COVERAGE_READY_FOR_FREEZE"
    return "PARTIAL_COVERAGE"


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


def _eval_artifact_status_from_summary(summary_payload: Mapping[str, Any]) -> str:
    eval_path_raw = summary_payload.get("eval_path")
    if not isinstance(eval_path_raw, str) or not eval_path_raw.strip():
        return "NOT_AVAILABLE"
    eval_path = Path(eval_path_raw)
    if not eval_path.exists():
        return "MISSING_EVAL_FILE"
    try:
        payload = _load_json_object(eval_path)
    except (json.JSONDecodeError, ValueError):
        return "MALFORMED_EVAL_FILE"
    return "AVAILABLE" if isinstance(payload, dict) else "MALFORMED_EVAL_FILE"


def _derive_metrics_status(*, confirmed_qrel_count: int, eval_artifact_status: str) -> str:
    if confirmed_qrel_count <= 0:
        return "NO_CONFIRMED_QRELS"
    if eval_artifact_status != "AVAILABLE":
        return "NOT_AVAILABLE"
    return "AVAILABLE"


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
    adjudicated_payload: Dict[str, Any] = {}
    if QREL_AGENT_ADJUDICATED_PATH.exists():
        adjudicated_payload = _load_yaml_mapping(QREL_AGENT_ADJUDICATED_PATH)

    candidate_files = sorted([str(path) for path in QREL_DIR.glob(QREL_GLOB)], reverse=True)
    candidate_path = _resolve_candidate_file()
    candidate_payload: Dict[str, Any] = {}
    if candidate_path is not None:
        candidate_payload = _load_yaml_mapping(candidate_path)

    candidate_count, not_found_count = _count_candidates(candidate_payload)
    review_count, approved_count, rejected_count, needs_context_count = _count_decisions(decision_payload)
    confirmed_qrel_count, confirmed_by_query = _count_confirmed_qrels(qrels_payload)
    (
        agent_adjudicated_count,
        agent_approved_count,
        agent_rejected_count,
        agent_needs_context_count,
        human_approved_count,
    ) = _count_agent_adjudicated(adjudicated_payload)
    approved_by_query = _approved_by_query(decision_payload)
    reviewed_not_found_queries = _not_found_reviewed_queries(candidate_payload)
    v0_coverage_ready, missing_areas, reviewed_not_found_areas = _compute_v0_coverage(
        confirmed_by_query=confirmed_by_query,
        approved_by_query=approved_by_query,
        not_found_reviewed_queries=reviewed_not_found_queries,
    )

    heartbeat_verdict = str(heartbeat.get("last_verdict", "UNKNOWN"))
    raw_next_prompt = heartbeat.get("next_prompt_path")
    next_prompt_path = str(raw_next_prompt) if isinstance(raw_next_prompt, str) else str(NEXT_PROMPT_DEFAULT)

    latest_pass_run_id = str(latest_pass[1].get("run_id")) if latest_pass else None
    latest_fail_run_id = str(latest_fail[1].get("run_id")) if latest_fail else None
    latest_pass_summary_path = str(latest_pass[0]) if latest_pass else None
    latest_fail_summary_path = str(latest_fail[0]) if latest_fail else None
    acceptance_state = "PASS" if latest_pass else "FAIL"
    qrels_state = str((latest_pass[1] if latest_pass else {}).get("qrels_state", "NO_CONFIRMED_QRELS"))
    eval_artifact_status = _eval_artifact_status_from_summary(latest_pass[1]) if latest_pass else "NOT_AVAILABLE"
    metrics_status = _derive_metrics_status(
        confirmed_qrel_count=confirmed_qrel_count,
        eval_artifact_status=eval_artifact_status,
    )

    v0_freeze_exists = V0_FREEZE_PATH.exists()
    v0_freeze_allowed = bool(latest_pass and v0_coverage_ready and confirmed_qrel_count > 0)
    v0_readiness_state = _derive_v0_readiness_state(
        v0_freeze_exists=v0_freeze_exists,
        confirmed_qrel_count=confirmed_qrel_count,
        approved_decision_count=approved_count,
        v0_coverage_ready=v0_coverage_ready,
    )
    scientific_state = v0_readiness_state if confirmed_qrel_count > 0 else "NO_QRELS_CONFIRMED"

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
        "agent_adjudicated_count": agent_adjudicated_count,
        "agent_approved_count": agent_approved_count,
        "agent_rejected_count": agent_rejected_count,
        "agent_needs_context_count": agent_needs_context_count,
        "human_approved_count": human_approved_count,
        "not_found_in_index_count": not_found_count,
        "v0_coverage_ready": v0_coverage_ready,
        "missing_v0_coverage_areas": missing_areas,
        "reviewed_not_found_areas": reviewed_not_found_areas,
        "v0_readiness_state": v0_readiness_state,
        "v0_freeze_exists": v0_freeze_exists,
        "v0_freeze_allowed": v0_freeze_allowed,
        "scientific_state": scientific_state,
        "qrels_state": qrels_state,
        "eval_artifact_status": eval_artifact_status,
        "metrics_status": metrics_status,
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
    lines.append(f"- agent_adjudicated_count: `{status['agent_adjudicated_count']}`")
    lines.append(f"- agent_approved_count: `{status['agent_approved_count']}`")
    lines.append(f"- agent_rejected_count: `{status['agent_rejected_count']}`")
    lines.append(f"- agent_needs_context_count: `{status['agent_needs_context_count']}`")
    lines.append(f"- human_approved_count: `{status['human_approved_count']}`")
    lines.append(f"- not_found_in_index_count: `{status['not_found_in_index_count']}`")
    lines.append(f"- qrels_state: `{status['qrels_state']}`")
    lines.append(f"- eval_artifact_status: `{status['eval_artifact_status']}`")
    lines.append(f"- metrics_status: `{status['metrics_status']}`")
    lines.append("")
    lines.append("## V0 Coverage")
    lines.append("")
    lines.append(f"- v0_coverage_ready: `{status['v0_coverage_ready']}`")
    lines.append(f"- missing_v0_coverage_areas: `{status['missing_v0_coverage_areas']}`")
    lines.append(f"- reviewed_not_found_areas: `{status['reviewed_not_found_areas']}`")
    lines.append(f"- v0_readiness_state: `{status['v0_readiness_state']}`")
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
        "agent_adjudicated_count": 0,
        "agent_approved_count": 0,
        "agent_rejected_count": 0,
        "agent_needs_context_count": 0,
        "human_approved_count": 0,
        "not_found_in_index_count": 0,
        "v0_coverage_ready": False,
        "missing_v0_coverage_areas": list(MANDATORY_V0_AREAS.values()),
        "reviewed_not_found_areas": [],
        "v0_readiness_state": "NO_QRELS_CONFIRMED",
        "v0_freeze_exists": False,
        "v0_freeze_allowed": False,
        "scientific_state": "NO_QRELS_CONFIRMED",
        "eval_artifact_status": "NOT_AVAILABLE",
        "metrics_status": "NO_CONFIRMED_QRELS",
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
