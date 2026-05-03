"""Promote manually approved Muon DIS qrel decisions into confirmed qrels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml


DEFAULT_DECISIONS_PATH = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
DEFAULT_QRELS_PATH = Path("benchmarks/muon_dis/qrels.yaml")
DEFAULT_CANDIDATES_PATH = Path("benchmarks/muon_dis/qrels_candidates.yaml")
ALLOWED_DECISIONS = {"APPROVED", "REJECTED", "NEEDS_CONTEXT"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote reviewed Muon DIS qrels with manual approval gates.")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS_PATH, help="Review decisions YAML file.")
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS_PATH, help="Qrels YAML file.")
    parser.add_argument("--dry-run", action="store_true", help="Show promotions without modifying qrels.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for promoted qrels; defaults to --qrels when not set.",
    )
    return parser.parse_args(argv)


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in: {path}")
    return payload


def _candidate_key(query_id: str, file_path: str, start_line: int, end_line: int) -> Tuple[str, str, int, int]:
    return (query_id, file_path, start_line, end_line)


def _build_candidate_lookup(candidates_payload: Mapping[str, Any]) -> set[Tuple[str, str, int, int]]:
    rows = candidates_payload.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("qrels_candidates.yaml must contain candidates list.")
    allowed: set[Tuple[str, str, int, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        query_id = row.get("query_id")
        qrels = row.get("qrels")
        if not isinstance(query_id, str) or not isinstance(qrels, list):
            continue
        for qrel in qrels:
            if not isinstance(qrel, dict):
                continue
            file_path = qrel.get("file_path")
            start_line = qrel.get("start_line")
            end_line = qrel.get("end_line")
            if isinstance(file_path, str) and isinstance(start_line, int) and isinstance(end_line, int):
                allowed.add(_candidate_key(query_id, file_path, start_line, end_line))
    return allowed


def _normalize_decisions(decisions_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = decisions_payload.get("decisions")
    if not isinstance(rows, list):
        raise ValueError("qrels_review_decisions.yaml must contain decisions list.")
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each decision row must be a mapping.")
        normalized.append(row)
    return normalized


def _canonical_anchor(entry: Mapping[str, Any]) -> Tuple[str, int, int] | None:
    file_path = entry.get("file_path")
    start_line = entry.get("start_line")
    end_line = entry.get("end_line")
    if isinstance(file_path, str) and isinstance(start_line, int) and isinstance(end_line, int):
        return (file_path, start_line, end_line)
    return None


def _validate_and_collect_promotions(
    *,
    decision_rows: Sequence[Mapping[str, Any]],
    allowed_candidates: set[Tuple[str, str, int, int]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    approved_count = 0
    rejected_count = 0
    needs_context_count = 0
    promotions: List[Dict[str, Any]] = []

    for row in decision_rows:
        query_id = row.get("query_id")
        file_path = row.get("file_path")
        start_line = row.get("start_line")
        end_line = row.get("end_line")
        decision = row.get("decision")
        relevance = row.get("relevance")

        if not isinstance(query_id, str):
            raise ValueError("Decision row missing string query_id.")
        if not isinstance(file_path, str):
            raise ValueError(f"Decision row for {query_id} missing string file_path.")
        if not isinstance(start_line, int) or not isinstance(end_line, int):
            raise ValueError(f"Decision row for {query_id} must include integer start_line/end_line.")
        if not isinstance(decision, str) or decision not in ALLOWED_DECISIONS:
            raise ValueError(f"Decision row for {query_id} has invalid decision: {decision}")
        if not isinstance(relevance, int) or relevance < 0 or relevance > 3:
            raise ValueError(f"Decision row for {query_id} has invalid relevance: {relevance}")

        key = _candidate_key(query_id, file_path, start_line, end_line)
        if key not in allowed_candidates:
            raise ValueError(
                "Decision row references file/range not present in qrels_candidates.yaml: "
                f"{query_id} {file_path}:{start_line}-{end_line}"
            )

        if decision == "APPROVED":
            approved_count += 1
            reason = row.get("reason")
            reviewer_notes = row.get("reviewer_notes")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"APPROVED decision missing non-empty reason: {query_id} {file_path}:{start_line}-{end_line}"
                )
            if not isinstance(reviewer_notes, str) or not reviewer_notes.strip():
                raise ValueError(
                    "APPROVED decision missing non-empty reviewer_notes: "
                    f"{query_id} {file_path}:{start_line}-{end_line}"
                )
            promotions.append(
                {
                    "query_id": query_id,
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "relevance": relevance,
                    "reason": reason.strip(),
                    "reviewer_notes": reviewer_notes.strip(),
                }
            )
        elif decision == "REJECTED":
            rejected_count += 1
        else:
            needs_context_count += 1

    counts = {
        "approved_count": approved_count,
        "rejected_count": rejected_count,
        "needs_context_count": needs_context_count,
    }
    return promotions, counts


def _merge_promotions_into_qrels(
    qrels_payload: Mapping[str, Any],
    promotions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(qrels_payload)
    confirmed_rows = qrels_payload.get("confirmed_qrels")
    pending_rows = qrels_payload.get("pending_qrels")
    if not isinstance(confirmed_rows, list):
        confirmed_rows = []
    if not isinstance(pending_rows, list):
        pending_rows = []

    normalized_confirmed: List[Dict[str, Any]] = []
    query_to_index: Dict[str, int] = {}
    for row in confirmed_rows:
        if not isinstance(row, dict):
            continue
        query_id = row.get("query_id")
        qrels = row.get("qrels")
        if not isinstance(query_id, str):
            continue
        if not isinstance(qrels, list):
            qrels = []
        clean_row = {"query_id": query_id, "qrels": qrels}
        query_to_index[query_id] = len(normalized_confirmed)
        normalized_confirmed.append(clean_row)

    for promotion in promotions:
        query_id = str(promotion["query_id"])
        entry = {
            "file_path": str(promotion["file_path"]),
            "start_line": int(promotion["start_line"]),
            "end_line": int(promotion["end_line"]),
            "relevance": int(promotion["relevance"]),
            "reason": str(promotion["reason"]),
            "reviewer_notes": str(promotion["reviewer_notes"]),
        }
        if query_id not in query_to_index:
            query_to_index[query_id] = len(normalized_confirmed)
            normalized_confirmed.append({"query_id": query_id, "qrels": []})
        target = normalized_confirmed[query_to_index[query_id]]
        existing = target["qrels"]

        updated = False
        for existing_row in existing:
            if not isinstance(existing_row, dict):
                continue
            anchor = _canonical_anchor(existing_row)
            if anchor == _canonical_anchor(entry):
                existing_row["relevance"] = entry["relevance"]
                existing_row["reason"] = entry["reason"]
                existing_row["reviewer_notes"] = entry["reviewer_notes"]
                updated = True
                break
        if not updated:
            existing.append(entry)

    for row in normalized_confirmed:
        row["qrels"] = sorted(
            [qrel for qrel in row["qrels"] if isinstance(qrel, dict)],
            key=lambda qrel: (
                str(qrel.get("file_path", "")),
                int(qrel.get("start_line", 0)),
                int(qrel.get("end_line", 0)),
            ),
        )
    normalized_confirmed.sort(key=lambda row: row["query_id"])

    result["confirmed_qrels"] = normalized_confirmed
    result["pending_qrels"] = pending_rows
    return result


def promote_qrels(
    *,
    decisions_path: Path,
    qrels_path: Path,
    dry_run: bool = False,
    output_path: Path | None = None,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
) -> Dict[str, Any]:
    decisions_payload = _load_yaml_mapping(decisions_path)
    qrels_payload = _load_yaml_mapping(qrels_path)
    candidates_payload = _load_yaml_mapping(candidates_path)

    decision_rows = _normalize_decisions(decisions_payload)
    allowed_candidates = _build_candidate_lookup(candidates_payload)
    promotions, counts = _validate_and_collect_promotions(
        decision_rows=decision_rows,
        allowed_candidates=allowed_candidates,
    )

    final_output = output_path if output_path is not None else qrels_path
    summary: Dict[str, Any] = {
        "approved_count": counts["approved_count"],
        "rejected_count": counts["rejected_count"],
        "needs_context_count": counts["needs_context_count"],
        "promoted_count": len(promotions),
        "qrels_output_path": str(final_output),
    }

    if dry_run:
        summary["would_promote"] = promotions
        return summary

    promoted_payload = _merge_promotions_into_qrels(qrels_payload, promotions)
    should_write = True
    if len(promotions) == 0:
        try:
            should_write = final_output.resolve() != qrels_path.resolve()
        except FileNotFoundError:
            should_write = str(final_output) != str(qrels_path)
    if should_write:
        final_output.parent.mkdir(parents=True, exist_ok=True)
        final_output.write_text(yaml.safe_dump(promoted_payload, sort_keys=False), encoding="utf-8")
    summary["write_performed"] = should_write
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = promote_qrels(
            decisions_path=args.decisions,
            qrels_path=args.qrels,
            dry_run=args.dry_run,
            output_path=args.output,
        )
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
