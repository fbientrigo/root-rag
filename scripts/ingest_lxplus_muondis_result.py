"""Ingest saved LXPLUS MuonDIS preflight/probe artifacts into a reproducible Markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


RUNTIME_STATES = {
    "PREFLIGHT_FAILED",
    "PREFLIGHT_PASS",
    "ROOT_PROBE_PASS",
    "ROOT_PROBE_FAILED",
    "RUNTIME_OUTPUT_INSPECTED",
    "RUNTIME_UNVALIDATED",
}

ORACLE_FAILURE_STATUSES = {
    "NO_FILE",
    "ROOT_IMPORT_FAILED",
    "ROOT_FILE_OPEN_FAILED",
}

ORACLE_FIELDS = [
    "transport_status",
    "dis_tree_exists",
    "n_DISParticles",
    "n_SoftParticles",
    "n_SBT_hits",
    "n_UBT_hits",
    "n_veto_hits",
    "fiducial_fail",
    "wall_like_fail",
    "output_valid",
    "error_reason",
]


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def summarize_tag_lines(lines: list[str], token: str) -> list[str]:
    return [line for line in lines if token in line]


def summarize_preflight_log(path: Path) -> dict[str, Any]:
    lines = _read_lines(path)
    pass_lines = summarize_tag_lines(lines, "[OK]")
    warn_lines = summarize_tag_lines(lines, "[WARN]")
    fail_lines = summarize_tag_lines(lines, "[FAIL]")
    root_fail_lines = [line for line in lines if "[FAIL] import ROOT" in line]
    fairship_missing = any("FAIRSHIP is missing or not a directory" in line for line in lines)
    shipsoft_missing = any("SHIPSOFT is missing or not a directory" in line for line in lines)

    if fail_lines:
        outcome = "FAIL"
    elif warn_lines:
        outcome = "WARN"
    else:
        outcome = "PASS"

    return {
        "path": str(path),
        "line_count": len(lines),
        "pass_lines": pass_lines,
        "warn_lines": warn_lines,
        "fail_lines": fail_lines,
        "root_fail_lines": root_fail_lines,
        "fairship_missing": fairship_missing,
        "shipsoft_missing": shipsoft_missing,
        "root_import_failed": bool(root_fail_lines),
        "outcome": outcome,
    }


def load_oracle_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"oracle json not found: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"oracle json decode failure: {exc}"
    if not isinstance(payload, dict):
        return None, "oracle json root must be an object"
    return payload, None


def summarize_command_log(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "line_count": 0, "fail_lines": [], "warn_lines": [], "pass_lines": []}
    if not path.exists():
        return {"path": str(path), "exists": False, "line_count": 0, "fail_lines": [], "warn_lines": [], "pass_lines": []}
    lines = _read_lines(path)
    return {
        "path": str(path),
        "exists": True,
        "line_count": len(lines),
        "fail_lines": summarize_tag_lines(lines, "[FAIL]"),
        "warn_lines": summarize_tag_lines(lines, "[WARN]"),
        "pass_lines": summarize_tag_lines(lines, "[OK]"),
    }


def classify_runtime_status(preflight: dict[str, Any], oracle: dict[str, Any] | None, oracle_error: str | None) -> str:
    if preflight["fail_lines"]:
        return "PREFLIGHT_FAILED"

    if oracle is None:
        if oracle_error:
            return "RUNTIME_UNVALIDATED"
        return "PREFLIGHT_PASS"

    transport_status = str(oracle.get("transport_status", "UNSET"))
    output_valid = oracle.get("output_valid")

    if transport_status in ORACLE_FAILURE_STATUSES:
        return "ROOT_PROBE_FAILED"

    if output_valid is False:
        return "ROOT_PROBE_FAILED"

    if output_valid is True and transport_status == "DIS_AND_CBMSIM_FOUND":
        return "ROOT_PROBE_PASS"

    if output_valid is True:
        return "RUNTIME_OUTPUT_INSPECTED"

    return "PREFLIGHT_PASS"


def _format_oracle_summary(oracle: dict[str, Any] | None, oracle_error: str | None) -> list[str]:
    rows: list[str] = []
    if oracle is None:
        rows.append(f"- oracle_status: `MISSING_OR_INVALID`")
        rows.append(f"- oracle_error: `{oracle_error}`")
        return rows
    for key in ORACLE_FIELDS:
        rows.append(f"- {key}: `{oracle.get(key)}`")
    return rows


def _suggested_wiki_updates(runtime_status: str) -> list[str]:
    if runtime_status == "PREFLIGHT_FAILED":
        return [
            "- Update `docs/wiki/fairship/runtime/LXPLUS_preflight.md` with preflight fail evidence link.",
            "- Keep `docs/wiki/WIKI_STATE.md` LXPLUS status as `UNRESOLVED`.",
        ]
    if runtime_status == "ROOT_PROBE_PASS":
        return [
            "- Propose update to `docs/wiki/fairship/runtime/oracle_probe_runtime.md` with probe artifact path.",
            "- Propose map note update in `docs/wiki/maps/Oracle_observables_map.md` for runtime-inspected fields.",
        ]
    if runtime_status == "RUNTIME_OUTPUT_INSPECTED":
        return [
            "- Propose partial update to `docs/wiki/fairship/runtime/oracle_probe_runtime.md` with constrained status.",
            "- Keep canonical chain and truth-label claims unresolved.",
        ]
    if runtime_status == "ROOT_PROBE_FAILED":
        return [
            "- Keep runtime nodes unresolved and attach failure evidence in runtime notes.",
            "- Add failure-specific remediation note under preflight/probe runtime notes.",
        ]
    return [
        "- Keep runtime claims unchanged; add artifact links only.",
        "- Preserve non-promotion policy in `docs/wiki/WIKI_STATE.md`.",
    ]


def _suggested_open_question_updates(runtime_status: str) -> list[str]:
    if runtime_status in {"PREFLIGHT_FAILED", "ROOT_PROBE_FAILED"}:
        return [
            "- Append failure mode and error snippet to `docs/wiki/open_questions.md` under LXPLUS execution.",
            "- Add any DY gating ambiguity observed under DY/Yheight question.",
        ]
    if runtime_status in {"ROOT_PROBE_PASS", "RUNTIME_OUTPUT_INSPECTED"}:
        return [
            "- Link oracle artifact under ROOT schema validation and oracle truth labels questions.",
            "- Keep wall/fiducial/Geant4 ambiguity open unless explicit boundary evidence appears.",
        ]
    return [
        "- Keep existing open-question statuses; add artifact path references only.",
    ]


def build_report(
    *,
    preflight: dict[str, Any],
    oracle: dict[str, Any] | None,
    oracle_error: str | None,
    command_log: dict[str, Any],
    runtime_status: str,
    preflight_path: Path,
    oracle_path: Path,
    command_path: Path | None,
) -> str:
    if runtime_status not in RUNTIME_STATES:
        raise ValueError(f"invalid runtime status: {runtime_status}")

    lines: list[str] = []
    lines.append("# FairShip LXPLUS Runtime Ingest Report")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- preflight_log: `{preflight_path}`")
    lines.append(f"- oracle_json: `{oracle_path}`")
    lines.append(f"- command_log: `{command_path}`")
    lines.append("")
    lines.append("## Runtime Status")
    lines.append(f"- runtime_status: `{runtime_status}`")
    lines.append("")
    lines.append("## Preflight Summary")
    lines.append(f"- preflight_outcome: `{preflight['outcome']}`")
    lines.append(f"- pass_line_count: `{len(preflight['pass_lines'])}`")
    lines.append(f"- warn_line_count: `{len(preflight['warn_lines'])}`")
    lines.append(f"- fail_line_count: `{len(preflight['fail_lines'])}`")
    lines.append(f"- fairship_missing: `{preflight['fairship_missing']}`")
    lines.append(f"- shipsoft_missing: `{preflight['shipsoft_missing']}`")
    lines.append(f"- root_import_failed: `{preflight['root_import_failed']}`")
    lines.append("")
    lines.append("### PASS lines")
    for row in preflight["pass_lines"] or ["(none)"]:
        lines.append(f"- {row}")
    lines.append("")
    lines.append("### WARN lines")
    for row in preflight["warn_lines"] or ["(none)"]:
        lines.append(f"- {row}")
    lines.append("")
    lines.append("### FAIL lines")
    for row in preflight["fail_lines"] or ["(none)"]:
        lines.append(f"- {row}")
    lines.append("")
    lines.append("## Oracle Summary")
    lines.extend(_format_oracle_summary(oracle, oracle_error))
    lines.append("")
    lines.append("## Command Log Summary")
    lines.append(f"- command_log_exists: `{command_log['exists']}`")
    lines.append(f"- command_log_path: `{command_log['path']}`")
    lines.append(f"- command_log_line_count: `{command_log['line_count']}`")
    lines.append(f"- command_log_warn_count: `{len(command_log['warn_lines'])}`")
    lines.append(f"- command_log_fail_count: `{len(command_log['fail_lines'])}`")
    lines.append("")
    lines.append("## Policy Guardrails")
    lines.append("- Wiki is not updated automatically by this ingest script.")
    lines.append("- Suggested wiki updates are listed separately.")
    lines.append("- Suggested open question updates are listed separately.")
    lines.append("")
    lines.append("## Suggested Wiki Updates (Manual)")
    lines.extend(_suggested_wiki_updates(runtime_status))
    lines.append("")
    lines.append("## Suggested Open Question Updates (Manual)")
    lines.extend(_suggested_open_question_updates(runtime_status))
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest LXPLUS MuonDIS preflight/probe artifacts into Markdown.")
    parser.add_argument("--preflight-log", required=True, type=Path, help="Path to preflight log file.")
    parser.add_argument("--oracle-json", required=True, type=Path, help="Path to oracle probe JSON file.")
    parser.add_argument("--command-log", default=None, type=Path, help="Optional path to command log file.")
    parser.add_argument("--output", required=True, type=Path, help="Output Markdown report path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preflight = summarize_preflight_log(args.preflight_log)
    oracle, oracle_error = load_oracle_json(args.oracle_json)
    command_log = summarize_command_log(args.command_log)
    runtime_status = classify_runtime_status(preflight, oracle, oracle_error)

    report = build_report(
        preflight=preflight,
        oracle=oracle,
        oracle_error=oracle_error,
        command_log=command_log,
        runtime_status=runtime_status,
        preflight_path=args.preflight_log,
        oracle_path=args.oracle_json,
        command_path=args.command_log,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"runtime_status={runtime_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
