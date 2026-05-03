"""Render human-friendly Muon DIS qrel review sheet from candidate/decision YAML."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml


DECISION_VALUES = {"NEEDS_CONTEXT", "APPROVED", "REJECTED"}
ONLY_VALUES = {"ALL", *DECISION_VALUES}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print Muon DIS qrel review sheet.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("benchmarks/muon_dis/qrels_candidates.yaml"),
        help="Qrel candidates YAML path.",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("benchmarks/muon_dis/qrels_review_decisions.yaml"),
        help="Qrel review decisions YAML path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/muon_dis_qrel_review_sheet.md"),
        help="Markdown review sheet output path.",
    )
    parser.add_argument(
        "--top-per-query",
        type=int,
        default=5,
        help="Maximum rows per query in main review table.",
    )
    parser.add_argument(
        "--only",
        default="ALL",
        choices=sorted(ONLY_VALUES),
        help="Filter by decision label in main review table.",
    )
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


def _escape_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _suggested_action(decision: str) -> str:
    if decision == "APPROVED":
        return "Ready for guarded promotion."
    if decision == "REJECTED":
        return "Keep excluded from qrels promotion."
    if decision == "NEEDS_CONTEXT":
        return "Manual evidence inspection required."
    if decision == "MISSING_DECISION":
        return "Add decision row before promotion."
    if decision == "ORPHAN_DECISION":
        return "Fix or remove unmatched decision row."
    return "Manual triage required."


def _flatten_candidates(payload: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = payload.get("candidates")
    if rows is None:
        return [], []
    if not isinstance(rows, list):
        raise ValueError("Candidates payload must contain list field: candidates")

    candidates: List[Dict[str, Any]] = []
    not_found: List[Dict[str, Any]] = []
    for query_row in rows:
        if not isinstance(query_row, dict):
            continue
        query_id = query_row.get("query_id")
        query_text = query_row.get("query_text")
        manifest_status = query_row.get("manifest_status")
        review_status = query_row.get("review_status")
        qrels = query_row.get("qrels")

        if not isinstance(query_id, str):
            continue
        if not isinstance(qrels, list):
            qrels = []

        if review_status == "NOT_FOUND_IN_INDEX" or manifest_status == "ZERO_HIT" or len(qrels) == 0:
            not_found.append(
                {
                    "query_id": query_id,
                    "query_text": query_text or "",
                    "manifest_status": manifest_status or "",
                    "review_status": review_status or "",
                }
            )

        for idx, qrel in enumerate(qrels, start=1):
            if not isinstance(qrel, dict):
                continue
            file_path = qrel.get("file_path")
            start_line = qrel.get("start_line")
            end_line = qrel.get("end_line")
            if not isinstance(file_path, str) or not isinstance(start_line, int) or not isinstance(end_line, int):
                continue
            candidates.append(
                {
                    "query_id": query_id,
                    "query_text": query_text or "",
                    "manifest_status": manifest_status or "",
                    "review_status": review_status or "",
                    "rank": qrel.get("rank", idx),
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "relevance_candidate": qrel.get("relevance_candidate"),
                    "key": _candidate_key(query_id, file_path, start_line, end_line),
                }
            )
    return candidates, not_found


def _build_decision_lookup(payload: Mapping[str, Any]) -> Dict[Tuple[str, str, int, int], Dict[str, Any]]:
    rows = payload.get("decisions")
    if rows is None:
        return {}
    if not isinstance(rows, list):
        raise ValueError("Decisions payload must contain list field: decisions")

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


def _filter_rows(rows: Iterable[Dict[str, Any]], only: str, top_per_query: int) -> List[Dict[str, Any]]:
    filtered = [row for row in rows if only == "ALL" or row["current_decision"] == only]
    if top_per_query <= 0:
        return filtered
    output: List[Dict[str, Any]] = []
    per_query: Dict[str, int] = defaultdict(int)
    for row in filtered:
        query_id = str(row["query_id"])
        if per_query[query_id] >= top_per_query:
            continue
        per_query[query_id] += 1
        output.append(row)
    return output


def build_review_data(
    *,
    candidates_payload: Mapping[str, Any],
    decisions_payload: Mapping[str, Any],
    only: str = "ALL",
    top_per_query: int = 5,
) -> Dict[str, Any]:
    candidate_rows, not_found_rows = _flatten_candidates(candidates_payload)
    decisions_lookup = _build_decision_lookup(decisions_payload)
    remaining = dict(decisions_lookup)

    review_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    for candidate in candidate_rows:
        key = candidate["key"]
        decision_row = remaining.pop(key, None)
        if decision_row is None:
            current_decision = "MISSING_DECISION"
            relevance = candidate.get("relevance_candidate", "")
            reason = ""
            reviewer_notes = ""
            missing_rows.append(candidate)
        else:
            current_decision = str(decision_row.get("decision", "MISSING_DECISION"))
            relevance = decision_row.get("relevance", candidate.get("relevance_candidate", ""))
            reason = decision_row.get("reason", "")
            reviewer_notes = decision_row.get("reviewer_notes", "")

        review_rows.append(
            {
                "query_id": candidate["query_id"],
                "query_text": candidate["query_text"],
                "rank": candidate["rank"],
                "file_path": candidate["file_path"],
                "line_range": f"{candidate['start_line']}-{candidate['end_line']}",
                "current_decision": current_decision,
                "relevance": relevance,
                "reason": reason,
                "reviewer_notes": reviewer_notes,
                "suggested_action": _suggested_action(current_decision),
            }
        )

    orphan_rows: List[Dict[str, Any]] = []
    for row in remaining.values():
        orphan_rows.append(
            {
                "query_id": row.get("query_id", ""),
                "file_path": row.get("file_path", ""),
                "line_range": f"{row.get('start_line', '')}-{row.get('end_line', '')}",
                "decision": row.get("decision", ""),
                "relevance": row.get("relevance", ""),
                "reason": row.get("reason", ""),
                "reviewer_notes": row.get("reviewer_notes", ""),
                "suggested_action": _suggested_action("ORPHAN_DECISION"),
            }
        )

    filtered_rows = _filter_rows(review_rows, only=only, top_per_query=top_per_query)
    decision_counts: Dict[str, int] = defaultdict(int)
    for row in review_rows:
        decision_counts[str(row["current_decision"])] += 1

    return {
        "total_candidates": len(candidate_rows),
        "filtered_count": len(filtered_rows),
        "review_rows": filtered_rows,
        "not_found_rows": not_found_rows,
        "missing_rows": missing_rows,
        "orphan_rows": orphan_rows,
        "decision_counts": dict(decision_counts),
    }


def render_markdown(review_data: Mapping[str, Any], *, only: str, top_per_query: int) -> str:
    lines: List[str] = []
    lines.append("# Muon DIS Qrel Review Sheet")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total candidate ranges: {review_data['total_candidates']}")
    lines.append(f"- Main table rows shown: {review_data['filtered_count']}")
    lines.append(f"- Filter (`--only`): {only}")
    lines.append(f"- Top rows per query (`--top-per-query`): {top_per_query}")
    lines.append(f"- NOT_FOUND_IN_INDEX queries: {len(review_data['not_found_rows'])}")
    lines.append(f"- Missing decisions: {len(review_data['missing_rows'])}")
    lines.append(f"- Orphan decisions: {len(review_data['orphan_rows'])}")
    lines.append("- Decision counts:")
    for key in sorted(review_data["decision_counts"].keys()):
        lines.append(f"  - {key}: {review_data['decision_counts'][key]}")
    lines.append("")
    lines.append("## Query Review Table")
    lines.append("")
    lines.append(
        "| query_id | query_text | rank | file_path | line range | current decision | relevance | reason | reviewer_notes | suggested action |"
    )
    lines.append("|---|---|---:|---|---|---|---:|---|---|---|")
    for row in review_data["review_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_cell(row["query_id"]),
                    _escape_cell(row["query_text"]),
                    _escape_cell(row["rank"]),
                    _escape_cell(row["file_path"]),
                    _escape_cell(row["line_range"]),
                    _escape_cell(row["current_decision"]),
                    _escape_cell(row["relevance"]),
                    _escape_cell(row["reason"]),
                    _escape_cell(row["reviewer_notes"]),
                    _escape_cell(row["suggested_action"]),
                ]
            )
            + " |"
        )
    if not review_data["review_rows"]:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")

    lines.append("")
    lines.append("## NOT_FOUND_IN_INDEX")
    lines.append("")
    for row in review_data["not_found_rows"]:
        lines.append(
            f"- `{row['query_id']}` `{row['query_text']}` "
            f"(manifest_status=`{row['manifest_status']}`, review_status=`{row['review_status']}`)."
        )
    if not review_data["not_found_rows"]:
        lines.append("- None.")

    lines.append("")
    lines.append("## Missing Decisions")
    lines.append("")
    for row in review_data["missing_rows"]:
        lines.append(
            f"- `{row['query_id']}` `{row['file_path']}:{row['start_line']}-{row['end_line']}` "
            f"-> `MISSING_DECISION`."
        )
    if not review_data["missing_rows"]:
        lines.append("- None.")

    lines.append("")
    lines.append("## Orphan Decisions")
    lines.append("")
    for row in review_data["orphan_rows"]:
        lines.append(
            f"- `{row['query_id']}` `{row['file_path']}:{row['line_range']}` "
            f"decision=`{row['decision']}`."
        )
    if not review_data["orphan_rows"]:
        lines.append("- None.")

    lines.append("")
    lines.append("## Suggested Manual Review Order")
    lines.append("")
    lines.append(f"1. Resolve `MISSING_DECISION` rows first ({len(review_data['missing_rows'])}).")
    lines.append(
        "2. Review `NEEDS_CONTEXT` rows next "
        f"({review_data['decision_counts'].get('NEEDS_CONTEXT', 0)})."
    )
    lines.append(
        "3. Recheck `APPROVED` rows before guarded promotion "
        f"({review_data['decision_counts'].get('APPROVED', 0)})."
    )
    lines.append(
        "4. Audit `ORPHAN_DECISION` rows and fix stale anchors "
        f"({len(review_data['orphan_rows'])})."
    )
    lines.append("5. Keep wiki/workflow graph claim promotion blocked until manual review complete.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        candidates_payload = _load_yaml_mapping(args.candidates)
        decisions_payload = _load_yaml_mapping(args.decisions)
        review_data = build_review_data(
            candidates_payload=candidates_payload,
            decisions_payload=decisions_payload,
            only=args.only,
            top_per_query=args.top_per_query,
        )
        markdown = render_markdown(review_data, only=args.only, top_per_query=args.top_per_query)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output": str(args.output),
                "total_candidates": review_data["total_candidates"],
                "main_rows": review_data["filtered_count"],
                "missing_decisions": len(review_data["missing_rows"]),
                "orphan_decisions": len(review_data["orphan_rows"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

