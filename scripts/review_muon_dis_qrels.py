"""Interactive/manual review CLI for Muon DIS qrel candidates."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml


DEFAULT_CANDIDATES = Path("benchmarks/muon_dis/qrels_candidates.yaml")
DEFAULT_DECISIONS = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
DEFAULT_OUTPUT = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")

ALLOWED_DECISIONS = ("APPROVED", "REJECTED", "NEEDS_CONTEXT")
ALLOWED_ONLY = ("ALL",) + ALLOWED_DECISIONS
PRESET_CRITICAL_PATH = [
    "q02_make_muon_dis",
    "q03_run_simscript",
    "q04_shipreco",
    "q05_doca",
    "q06_sbt",
    "q07_ubt",
    "q08_muioni",
    "q09_inactivate_muon_processes",
    "q01_muondis_anchor",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review Muon DIS qrel candidates.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES, help="Qrel candidates YAML path.")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS, help="Qrel decisions YAML path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output decisions YAML path.")
    parser.add_argument("--interactive", action="store_true", help="Run prompt-driven review mode.")
    parser.add_argument("--dry-run", action="store_true", help="Print proposed changes without writing files.")
    parser.add_argument("--list", action="store_true", help="List filtered review rows without editing.")
    parser.add_argument("--query-id", default=None, help="Filter rows by one query id.")
    parser.add_argument("--only", default="ALL", choices=ALLOWED_ONLY, help="Filter by current decision status.")
    parser.add_argument("--preset", choices=("critical-path",), default=None, help="Apply built-in review ordering.")
    parser.add_argument(
        "--top-per-area",
        type=int,
        default=None,
        help="When using --preset critical-path, take up to N rows per area before repeats.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit listed/processed rows.")
    parser.add_argument(
        "--set-decision",
        choices=ALLOWED_DECISIONS,
        default=None,
        help="Apply one decision to all selected rows (non-interactive mode).",
    )
    parser.add_argument("--reason", default="", help="Reason text used with --set-decision.")
    parser.add_argument("--reviewer-notes", default="", help="Reviewer notes used with --set-decision.")
    parser.add_argument("--relevance", type=int, default=1, help="Relevance value used with --set-decision.")
    return parser.parse_args(argv)


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in: {path}")
    return payload


def _candidate_key(query_id: str, file_path: str, start_line: int, end_line: int) -> Tuple[str, str, int, int]:
    return (query_id, file_path, start_line, end_line)


def _build_decisions_lookup(decisions_payload: Mapping[str, Any]) -> Dict[Tuple[str, str, int, int], Dict[str, Any]]:
    rows = decisions_payload.get("decisions")
    if rows is None:
        return {}
    if not isinstance(rows, list):
        raise ValueError("Decision payload must contain list field: decisions")
    lookup: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        query_id = row.get("query_id")
        file_path = row.get("file_path")
        start_line = row.get("start_line")
        end_line = row.get("end_line")
        if not isinstance(query_id, str) or not isinstance(file_path, str):
            continue
        if not isinstance(start_line, int) or not isinstance(end_line, int):
            continue
        lookup[_candidate_key(query_id, file_path, start_line, end_line)] = row
    return lookup


def _flatten_rows(
    candidates_payload: Mapping[str, Any],
    decisions_lookup: Mapping[Tuple[str, str, int, int], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates = candidates_payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Candidates payload must contain list field: candidates")

    rows: List[Dict[str, Any]] = []
    for query in candidates:
        if not isinstance(query, dict):
            continue
        query_id = query.get("query_id")
        query_text = query.get("query_text", "")
        review_status = query.get("review_status", "")
        qrels = query.get("qrels")
        if not isinstance(query_id, str):
            continue
        if not isinstance(qrels, list):
            qrels = []

        if len(qrels) == 0:
            if query_id == "q09_inactivate_muon_processes" or review_status == "NOT_FOUND_IN_INDEX":
                rows.append(
                    {
                        "query_id": query_id,
                        "query_text": str(query_text),
                        "rank": 0,
                        "file_path": "NOT_FOUND_IN_INDEX",
                        "start_line": None,
                        "end_line": None,
                        "current_decision": "NOT_FOUND_IN_INDEX",
                        "current_relevance": "",
                        "reason": "NOT FOUND IN INDEX",
                        "reviewer_notes": "",
                        "key": None,
                        "is_not_found": True,
                    }
                )
            continue

        for idx, entry in enumerate(qrels, start=1):
            if not isinstance(entry, dict):
                continue
            file_path = entry.get("file_path")
            start_line = entry.get("start_line")
            end_line = entry.get("end_line")
            if not isinstance(file_path, str) or not isinstance(start_line, int) or not isinstance(end_line, int):
                continue

            key = _candidate_key(query_id, file_path, start_line, end_line)
            decision = decisions_lookup.get(key, {})
            rows.append(
                {
                    "query_id": query_id,
                    "query_text": str(query_text),
                    "rank": int(entry.get("rank", idx)),
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "current_decision": str(decision.get("decision", "MISSING_DECISION")),
                    "current_relevance": decision.get("relevance", entry.get("relevance_candidate", "")),
                    "reason": str(decision.get("reason", "")),
                    "reviewer_notes": str(decision.get("reviewer_notes", "")),
                    "key": key,
                    "is_not_found": False,
                }
            )

    if not any(row["query_id"] == "q09_inactivate_muon_processes" for row in rows):
        rows.append(
            {
                "query_id": "q09_inactivate_muon_processes",
                "query_text": "InactivateMuonProcesses",
                "rank": 0,
                "file_path": "NOT_FOUND_IN_INDEX",
                "start_line": None,
                "end_line": None,
                "current_decision": "NOT_FOUND_IN_INDEX",
                "current_relevance": "",
                "reason": "NOT FOUND IN INDEX",
                "reviewer_notes": "",
                "key": None,
                "is_not_found": True,
            }
        )
    return rows


def _apply_ordering(
    rows: List[Dict[str, Any]],
    preset: Optional[str],
    *,
    top_per_area: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if preset != "critical-path":
        return sorted(rows, key=lambda row: (row["query_id"], row["rank"], row["file_path"]))
    index = {query_id: idx for idx, query_id in enumerate(PRESET_CRITICAL_PATH)}
    ordered = sorted(
        rows,
        key=lambda row: (
            index.get(row["query_id"], len(PRESET_CRITICAL_PATH)),
            row["rank"],
            row["file_path"],
        ),
    )
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in ordered:
        grouped.setdefault(row["query_id"], []).append(row)

    cap = 1
    if isinstance(top_per_area, int) and top_per_area > 0:
        cap = top_per_area

    output: List[Dict[str, Any]] = []
    for query_id in PRESET_CRITICAL_PATH:
        queue = grouped.get(query_id, [])
        if not queue:
            continue
        take = min(cap, len(queue))
        output.extend(queue[:take])
        del queue[:take]

    # Round-robin leftovers to keep later slices balanced as limits grow.
    while True:
        progressed = False
        for query_id in PRESET_CRITICAL_PATH:
            queue = grouped.get(query_id, [])
            if not queue:
                continue
            output.append(queue.pop(0))
            progressed = True
        if not progressed:
            break

    tail = [row for row in ordered if row["query_id"] not in PRESET_CRITICAL_PATH]
    output.extend(tail)
    return output


def _filter_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    query_id: Optional[str],
    only: str,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        if query_id and row["query_id"] != query_id:
            continue
        if only != "ALL" and row["current_decision"] != only:
            continue
        filtered.append(row)
    if isinstance(limit, int) and limit >= 0:
        return filtered[:limit]
    return filtered


def _line_range(row: Mapping[str, Any]) -> str:
    start_line = row.get("start_line")
    end_line = row.get("end_line")
    if isinstance(start_line, int) and isinstance(end_line, int):
        return f"{start_line}-{end_line}"
    return "NOT_FOUND_IN_INDEX"


def _print_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        print("No rows matched filters.")
        return
    print(
        "query_id | query_text | rank | file_path | line_range | current_decision | "
        "current_relevance | reason | reviewer_notes"
    )
    print("-" * 160)
    for row in rows:
        print(
            f"{row['query_id']} | {row['query_text']} | {row['rank']} | {row['file_path']} | "
            f"{_line_range(row)} | {row['current_decision']} | {row['current_relevance']} | "
            f"{str(row['reason']).strip()} | {str(row['reviewer_notes']).strip()}"
        )


def _validate_update_target(row: Mapping[str, Any], decision: str) -> None:
    if decision == "APPROVED" and (row.get("is_not_found") or row.get("key") is None):
        raise ValueError(
            "APPROVED is not allowed for NOT_FOUND_IN_INDEX or missing file/range rows: "
            f"{row.get('query_id')}"
        )


def _decision_entry(
    row: Mapping[str, Any],
    *,
    decision: str,
    relevance: int,
    reason: str,
    reviewer_notes: str,
) -> Dict[str, Any]:
    key = row.get("key")
    if not isinstance(key, tuple) or len(key) != 4:
        raise ValueError(f"Cannot write decision without file/range anchor: {row.get('query_id')}")
    query_id, file_path, start_line, end_line = key
    return {
        "query_id": query_id,
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "decision": decision,
        "relevance": relevance,
        "reason": reason.strip(),
        "reviewer_notes": reviewer_notes.strip(),
    }


def _validate_reason_notes(decision: str, reason: str, reviewer_notes: str) -> None:
    if decision != "APPROVED":
        return
    if not reason.strip():
        raise ValueError("APPROVED decisions require non-empty reason.")
    if not reviewer_notes.strip():
        raise ValueError("APPROVED decisions require non-empty reviewer_notes.")


def _merge_decisions(
    decisions_payload: MutableMapping[str, Any],
    updates: Sequence[Mapping[str, Any]],
) -> MutableMapping[str, Any]:
    existing = decisions_payload.get("decisions")
    if existing is None:
        existing = []
        decisions_payload["decisions"] = existing
    if not isinstance(existing, list):
        raise ValueError("Decision payload must contain list field: decisions")

    lookup: Dict[Tuple[str, str, int, int], int] = {}
    for idx, row in enumerate(existing):
        if not isinstance(row, dict):
            continue
        query_id = row.get("query_id")
        file_path = row.get("file_path")
        start_line = row.get("start_line")
        end_line = row.get("end_line")
        if isinstance(query_id, str) and isinstance(file_path, str) and isinstance(start_line, int) and isinstance(end_line, int):
            lookup[_candidate_key(query_id, file_path, start_line, end_line)] = idx

    for update in updates:
        key = _candidate_key(
            str(update["query_id"]),
            str(update["file_path"]),
            int(update["start_line"]),
            int(update["end_line"]),
        )
        if key in lookup:
            existing[lookup[key]] = dict(update)
        else:
            existing.append(dict(update))

    decisions_payload["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    return decisions_payload


def _interactive_updates(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    updates: List[Dict[str, Any]] = []
    print("Interactive mode: enter to skip, a=APPROVED, r=REJECTED, n=NEEDS_CONTEXT, q=quit.")
    for row in rows:
        print("")
        print(f"query_id: {row['query_id']}")
        print(f"query_text: {row['query_text']}")
        print(f"rank: {row['rank']}")
        print(f"file_path: {row['file_path']}")
        print(f"start_line-end_line: {_line_range(row)}")
        print(f"current_decision: {row['current_decision']}")
        print(f"current_relevance: {row['current_relevance']}")
        print(f"reason: {row['reason']}")
        print(f"reviewer_notes: {row['reviewer_notes']}")

        if row.get("is_not_found"):
            print("NOT_FOUND_IN_INDEX row detected; APPROVED is not allowed.")

        action = input("action [enter/a/r/n/q]: ").strip().lower()
        if action == "":
            continue
        if action == "q":
            break
        mapping = {"a": "APPROVED", "r": "REJECTED", "n": "NEEDS_CONTEXT"}
        if action not in mapping:
            print("Skipped: invalid action.")
            continue
        decision = mapping[action]
        try:
            _validate_update_target(row, decision)
        except ValueError as exc:
            print(str(exc))
            continue
        if row.get("key") is None:
            print("Skipped: row has no file/range anchor; decision not writable.")
            continue

        raw_relevance = input(f"relevance [0-3, default {row['current_relevance'] or 1}]: ").strip()
        if raw_relevance:
            relevance = int(raw_relevance)
        else:
            relevance = int(row["current_relevance"] or 1)

        reason_default = str(row.get("reason", ""))
        notes_default = str(row.get("reviewer_notes", ""))
        reason = input(f"reason [{reason_default}]: ").strip() or reason_default
        reviewer_notes = input(f"reviewer_notes [{notes_default}]: ").strip() or notes_default
        try:
            _validate_reason_notes(decision, reason, reviewer_notes)
        except ValueError as exc:
            print(str(exc))
            continue
        updates.append(
            _decision_entry(
                row,
                decision=decision,
                relevance=relevance,
                reason=reason,
                reviewer_notes=reviewer_notes,
            )
        )
    return updates


def _bulk_updates(
    rows: Sequence[Mapping[str, Any]],
    *,
    decision: str,
    reason: str,
    reviewer_notes: str,
    relevance: int,
) -> List[Dict[str, Any]]:
    _validate_reason_notes(decision, reason, reviewer_notes)
    updates: List[Dict[str, Any]] = []
    for row in rows:
        _validate_update_target(row, decision)
        if row.get("key") is None:
            continue
        updates.append(
            _decision_entry(
                row,
                decision=decision,
                relevance=relevance,
                reason=reason,
                reviewer_notes=reviewer_notes,
            )
        )
    return updates


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list and args.interactive:
        raise ValueError("--list and --interactive cannot be used together.")
    if args.set_decision and args.interactive:
        raise ValueError("--set-decision is non-interactive; do not combine with --interactive.")

    candidates_payload = _load_yaml_mapping(args.candidates)
    decisions_payload = _load_yaml_mapping(args.decisions)
    decisions_lookup = _build_decisions_lookup(decisions_payload)

    rows = _flatten_rows(candidates_payload, decisions_lookup)
    rows = _apply_ordering(rows, args.preset, top_per_area=args.top_per_area)
    selected = _filter_rows(rows, query_id=args.query_id, only=args.only, limit=args.limit)

    if args.list or (not args.interactive and not args.set_decision):
        _print_rows(selected)
        print(
            json.dumps(
                {
                    "mode": "list",
                    "selected_count": len(selected),
                    "output": str(args.output),
                    "dry_run": bool(args.dry_run),
                },
                indent=2,
            )
        )
        return 0

    updates: List[Dict[str, Any]] = []
    if args.interactive:
        updates = _interactive_updates(selected)
    elif args.set_decision:
        updates = _bulk_updates(
            selected,
            decision=args.set_decision,
            reason=args.reason,
            reviewer_notes=args.reviewer_notes,
            relevance=args.relevance,
        )

    if not updates:
        print(json.dumps({"updated_count": 0, "write_performed": False, "output": str(args.output)}, indent=2))
        return 0

    merged = _merge_decisions(deepcopy(decisions_payload), updates)
    write_performed = False
    if not args.dry_run:
        _write_yaml(args.output, merged)
        write_performed = True

    print(
        json.dumps(
            {
                "updated_count": len(updates),
                "write_performed": write_performed,
                "output": str(args.output),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
