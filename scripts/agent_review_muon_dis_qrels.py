"""Agentic review assistant for Muon DIS qrel candidates."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

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
PROPOSED_DECISIONS = {
    "PROPOSE_APPROVED",
    "PROPOSE_REJECTED",
    "PROPOSE_NEEDS_CONTEXT",
    "PROPOSE_NOT_FOUND_IN_INDEX",
}


@dataclass(frozen=True)
class CandidateRow:
    query_id: str
    query_text: str
    rank: int
    file_path: str
    start_line: int | None
    end_line: int | None
    is_not_found: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review Muon DIS qrel candidates with source snippets.")
    parser.add_argument("--candidates", type=Path, default=Path("benchmarks/muon_dis/qrels_candidates.yaml"))
    parser.add_argument("--decisions", type=Path, default=Path("benchmarks/muon_dis/qrels_review_decisions.yaml"))
    parser.add_argument("--fairship-path", type=Path, default=Path("..\\FairShip"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/muon_dis/qrels_agent_review_proposals.yaml"))
    parser.add_argument("--report", type=Path, default=Path("reports/muon_dis_agent_qrel_review.md"))
    parser.add_argument("--preset", choices=("critical-path",), default=None)
    parser.add_argument("--top-per-area", type=int, default=1)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in: {path}")
    return payload


def _flatten_candidates(payload: Mapping[str, Any]) -> List[CandidateRow]:
    rows: List[CandidateRow] = []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Candidates payload must contain list field: candidates")
    for query in candidates:
        if not isinstance(query, dict):
            continue
        query_id = str(query.get("query_id", ""))
        query_text = str(query.get("query_text", ""))
        qrels = query.get("qrels")
        review_status = str(query.get("review_status", ""))
        if not isinstance(qrels, list):
            qrels = []
        if len(qrels) == 0 and (review_status == "NOT_FOUND_IN_INDEX" or query_id == "q09_inactivate_muon_processes"):
            rows.append(
                CandidateRow(
                    query_id=query_id,
                    query_text=query_text,
                    rank=0,
                    file_path="NOT_FOUND_IN_INDEX",
                    start_line=None,
                    end_line=None,
                    is_not_found=True,
                )
            )
            continue
        for i, qrel in enumerate(qrels, start=1):
            if not isinstance(qrel, dict):
                continue
            file_path = qrel.get("file_path")
            start_line = qrel.get("start_line")
            end_line = qrel.get("end_line")
            if not isinstance(file_path, str) or not isinstance(start_line, int) or not isinstance(end_line, int):
                continue
            rows.append(
                CandidateRow(
                    query_id=query_id,
                    query_text=query_text,
                    rank=int(qrel.get("rank", i)),
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    is_not_found=False,
                )
            )
    return rows


def _apply_ordering(rows: List[CandidateRow], preset: Optional[str], top_per_area: int, limit: int) -> List[CandidateRow]:
    if preset != "critical-path":
        ordered = sorted(rows, key=lambda r: (r.query_id, r.rank, r.file_path))
        return ordered[:limit] if limit > 0 else ordered
    index = {qid: i for i, qid in enumerate(PRESET_CRITICAL_PATH)}
    grouped: Dict[str, List[CandidateRow]] = {}
    ordered = sorted(rows, key=lambda r: (index.get(r.query_id, 999), r.rank, r.file_path))
    for row in ordered:
        grouped.setdefault(row.query_id, []).append(row)
    cap = top_per_area if top_per_area > 0 else 1
    out: List[CandidateRow] = []
    for qid in PRESET_CRITICAL_PATH:
        queue = grouped.get(qid, [])
        out.extend(queue[:cap])
        del queue[:cap]
    while True:
        progressed = False
        for qid in PRESET_CRITICAL_PATH:
            queue = grouped.get(qid, [])
            if queue:
                out.append(queue.pop(0))
                progressed = True
        if not progressed:
            break
    tail = [r for r in ordered if r.query_id not in PRESET_CRITICAL_PATH]
    out.extend(tail)
    return out[:limit] if limit and limit > 0 else out


def _read_snippet(file_path: Path, start_line: int, end_line: int, window: int = 20) -> Tuple[int, int, str, str]:
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    start = max(1, start_line - window)
    end = min(total, end_line + window)

    context = "\n".join(f"{ln:>6}: {lines[ln - 1]}" for ln in range(start, end + 1))
    exact = "\n".join(f"{ln:>6}: {lines[ln - 1]}" for ln in range(start_line, min(end_line, total) + 1))
    return start, end, context, exact


def _propose_from_snippet(row: CandidateRow, context: str, exact: str) -> Tuple[str, int, str, str, str]:
    exact_lower = exact.lower()
    context_lower = context.lower()
    file_lower = row.file_path.lower()
    key = row.query_text.lower()
    exact_lines = [line for line in exact.splitlines() if line.strip()]
    non_comment_lines = [line for line in exact_lines if "#" not in line]
    looks_like_doc = file_lower.endswith("readme.md") or file_lower.endswith("changelog.md")
    explicit_symbol_anchor = any(token in exact_lower for token in ("def ", "class ", "(", "import "))
    # Stricter approval: in-snippet match, enough context, non-doc path, and explicit executable anchor.
    if (
        key
        and key in exact_lower
        and len(exact_lines) >= 8
        and len(non_comment_lines) >= 4
        and not looks_like_doc
        and explicit_symbol_anchor
    ):
        return (
            "PROPOSE_APPROVED",
            3,
            "HIGH",
            "Exact snippet contains the query symbol with sufficient executable context for qrel relevance.",
            "Human should still confirm this claim is qrel relevance, not a workflow-order/data-flow assertion.",
        )
    if key and key in exact_lower and looks_like_doc:
        return (
            "PROPOSE_REJECTED",
            0,
            "HIGH",
            "Exact range match appears in documentation/changelog context rather than executable workflow code.",
            "If documentation evidence is intentionally allowed for this query, human must override this rejection.",
        )
    if key and key in exact_lower and len(exact_lines) < 8:
        return (
            "PROPOSE_NEEDS_CONTEXT",
            1,
            "MEDIUM",
            "Query symbol is present but snippet is too short for safe qrel approval.",
            "Expand to a wider function/method block before any approval.",
        )
    if key and key in context_lower:
        return (
            "PROPOSE_NEEDS_CONTEXT",
            1,
            "MEDIUM",
            "Query target appears in nearby context but not clearly in the exact candidate range anchor.",
            "Human should inspect broader function boundaries before approval.",
        )
    if row.query_id == "q09_inactivate_muon_processes":
        return (
            "PROPOSE_NOT_FOUND_IN_INDEX",
            0,
            "HIGH",
            "NOT FOUND IN INDEX; this row is an explicit not-found coverage candidate, not a qrel.",
            "Require explicit human confirmation if this satisfies V0 not-found coverage.",
        )
    return (
        "PROPOSE_REJECTED",
        0,
        "MEDIUM",
        "Snippet does not show explicit query target in exact range or surrounding context.",
        "Human should re-check retrieval candidate ranking if this was expected to be relevant.",
    )


def _review_rows(rows: Iterable[CandidateRow], fairship_path: Path) -> List[Dict[str, Any]]:
    proposals: List[Dict[str, Any]] = []
    for row in rows:
        if row.is_not_found:
            proposals.append(
                {
                    "query_id": row.query_id,
                    "query_text": row.query_text,
                    "file_path": row.file_path,
                    "start_line": row.start_line,
                    "end_line": row.end_line,
                    "rank": row.rank,
                    "proposed_decision": "PROPOSE_NOT_FOUND_IN_INDEX",
                    "proposed_relevance": 0,
                    "confidence": "HIGH",
                    "reasoning": "NOT FOUND IN INDEX; this is not a retrievable qrel snippet.",
                    "risk_notes": "Require explicit human confirmation for V0 not-found coverage semantics.",
                    "required_human_check": "Confirm whether NOT_FOUND_IN_INDEX is sufficient for q09 coverage.",
                    "snippet_excerpt": "NOT FOUND IN INDEX",
                    "evidence_file": "NOT_FOUND_IN_INDEX",
                    "evidence_start_line": None,
                    "evidence_end_line": None,
                }
            )
            continue

        evidence_file = fairship_path / row.file_path
        if not evidence_file.exists():
            proposals.append(
                {
                    "query_id": row.query_id,
                    "query_text": row.query_text,
                    "file_path": row.file_path,
                    "start_line": row.start_line,
                    "end_line": row.end_line,
                    "rank": row.rank,
                    "proposed_decision": "PROPOSE_NEEDS_CONTEXT",
                    "proposed_relevance": 0,
                    "confidence": "LOW",
                    "reasoning": "Evidence file missing under --fairship-path; cannot validate snippet.",
                    "risk_notes": "Path mismatch between candidate index and local FairShip checkout.",
                    "required_human_check": "Verify FairShip path and retry snippet extraction.",
                    "snippet_excerpt": "FILE_NOT_FOUND",
                    "evidence_file": str(evidence_file),
                    "evidence_start_line": row.start_line,
                    "evidence_end_line": row.end_line,
                }
            )
            continue

        assert row.start_line is not None and row.end_line is not None
        ev_start, ev_end, context, exact = _read_snippet(evidence_file, row.start_line, row.end_line)
        decision, relevance, confidence, reasoning, risk = _propose_from_snippet(row, context, exact)
        proposals.append(
            {
                "query_id": row.query_id,
                "query_text": row.query_text,
                "file_path": row.file_path,
                "start_line": row.start_line,
                "end_line": row.end_line,
                "rank": row.rank,
                "proposed_decision": decision,
                "proposed_relevance": relevance,
                "confidence": confidence,
                "reasoning": reasoning,
                "risk_notes": risk,
                "required_human_check": "Inspect snippet and surrounding function to confirm workflow relevance.",
                "snippet_excerpt": exact if exact.strip() else context,
                "evidence_file": str(evidence_file),
                "evidence_start_line": ev_start,
                "evidence_end_line": ev_end,
            }
        )
    return proposals


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _write_report(path: Path, proposals: Sequence[Mapping[str, Any]], next_command: str) -> None:
    lines: List[str] = []
    lines.append("# Muon DIS Agentic Qrel Review")
    lines.append("")
    lines.append("| query_id | rank | proposed_decision | relevance | confidence | file |")
    lines.append("|---|---:|---|---:|---|---|")
    for p in proposals:
        lines.append(
            f"| {p['query_id']} | {p['rank']} | {p['proposed_decision']} | {p['proposed_relevance']} | "
            f"{p['confidence']} | {p['file_path']}:{p['start_line']}-{p['end_line']} |"
        )
    lines.append("")
    for p in proposals:
        lines.append(f"## {p['query_id']} rank {p['rank']}")
        lines.append("")
        lines.append(f"- proposed_decision: `{p['proposed_decision']}`")
        lines.append(f"- confidence: `{p['confidence']}`")
        lines.append(f"- reasoning: {p['reasoning']}")
        lines.append(f"- risk_notes: {p['risk_notes']}")
        lines.append("- snippet_excerpt:")
        lines.append("```text")
        lines.append(str(p["snippet_excerpt"]))
        lines.append("```")
        lines.append("")
    lines.append("## Next Command")
    lines.append("")
    lines.append(f"`{next_command}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidates_payload = _load_yaml_mapping(args.candidates)
    _ = _load_yaml_mapping(args.decisions)
    rows = _flatten_candidates(candidates_payload)
    selected = _apply_ordering(rows, args.preset, args.top_per_area, args.limit)
    proposals = _review_rows(selected, args.fairship_path)

    for proposal in proposals:
        if proposal["proposed_decision"] not in PROPOSED_DECISIONS:
            raise ValueError(f"Invalid proposal decision: {proposal['proposed_decision']}")

    out_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_candidates": str(args.candidates),
        "fairship_path": str(args.fairship_path),
        "dry_run": bool(args.dry_run),
        "proposals": [
            {
                "query_id": p["query_id"],
                "query_text": p["query_text"],
                "file_path": p["file_path"],
                "start_line": p["start_line"],
                "end_line": p["end_line"],
                "rank": p["rank"],
                "proposed_decision": p["proposed_decision"],
                "proposed_relevance": p["proposed_relevance"],
                "confidence": p["confidence"],
                "reasoning": p["reasoning"],
                "risk_notes": p["risk_notes"],
                "required_human_check": p["required_human_check"],
                "snippet_excerpt": p["snippet_excerpt"],
            }
            for p in proposals
        ],
    }

    _write_yaml(args.output, out_payload)
    next_cmd = ".venv\\Scripts\\python.exe scripts/review_muon_dis_qrels.py --list --preset critical-path --limit 10"
    _write_report(args.report, proposals, next_cmd)
    print(
        yaml.safe_dump(
            {
                "proposals_count": len(proposals),
                "output": str(args.output),
                "report": str(args.report),
                "dry_run": bool(args.dry_run),
            },
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
