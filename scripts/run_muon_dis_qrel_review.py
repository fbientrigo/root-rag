"""Build Muon DIS qrel candidates from parsed evidence artifacts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qrel review candidates from evidence artifacts.")
    parser.add_argument("--evidence-dir", required=True, type=Path, help="Evidence directory with manifest.json.")
    parser.add_argument("--run-id", default=None, help="Run id label for output naming.")
    parser.add_argument(
        "--candidates-path",
        default=Path("benchmarks/muon_dis/qrels_candidates.yaml"),
        type=Path,
        help="Primary candidate qrels output file.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting candidates-path.")
    return parser.parse_args(argv)


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _resolve_output_file(output_ref: str | None, evidence_dir: Path, query_id: str) -> Path:
    if output_ref:
        output_path = Path(str(output_ref))
    else:
        output_path = evidence_dir / f"{query_id}.json"
    if output_path.is_absolute():
        return output_path

    candidates = [
        output_path,
        evidence_dir / output_path,
        evidence_dir / output_path.name,
        Path.cwd() / output_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return output_path


def _extract_hits(query_payload: Mapping[str, Any]) -> List[Dict[str, int | str]]:
    hits = query_payload.get("hits")
    if not isinstance(hits, list):
        return []
    normalized: List[Dict[str, int | str]] = []
    seen: set[Tuple[str, int, int]] = set()
    for row in hits:
        if not isinstance(row, dict):
            continue
        file_path = row.get("file_path", row.get("file"))
        start_line = row.get("start_line", row.get("line_start"))
        end_line = row.get("end_line", row.get("line_end"))
        if not isinstance(file_path, str) or not isinstance(start_line, int) or not isinstance(end_line, int):
            continue
        key = (file_path, start_line, end_line)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
            }
        )
    return normalized


def _choose_candidates_path(path: Path, overwrite: bool) -> Path:
    if overwrite or (not path.exists()):
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def generate_qrel_review(
    *,
    evidence_dir: Path,
    run_id: str,
    candidates_path: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    manifest_path = evidence_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    queries = manifest.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Manifest field queries must be list.")

    rows: List[Dict[str, Any]] = []
    total_candidates = 0
    not_found_count = 0
    for query in queries:
        if not isinstance(query, dict):
            continue
        query_id = str(query.get("id", "UNKNOWN"))
        query_text = str(query.get("query", ""))
        status = str(query.get("status", "ERROR"))
        output_file = _resolve_output_file(query.get("output_file"), evidence_dir, query_id)

        hits: List[Dict[str, int | str]] = []
        if output_file.exists():
            try:
                payload = _load_json(output_file)
                hits = _extract_hits(payload)
            except (json.JSONDecodeError, ValueError):
                hits = []

        qrel_entries = [
            {
                "file_path": hit["file_path"],
                "start_line": hit["start_line"],
                "end_line": hit["end_line"],
                "review_required": True,
                "relevance_candidate": 1,
            }
            for hit in hits
        ]
        total_candidates += len(qrel_entries)

        review_status = "REVIEW_REQUIRED" if qrel_entries else "NOT_FOUND_IN_INDEX"
        if query_id == "q09_inactivate_muon_processes" and len(qrel_entries) == 0:
            review_status = "NOT_FOUND_IN_INDEX"
        if review_status == "NOT_FOUND_IN_INDEX":
            not_found_count += 1

        rows.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "manifest_status": status,
                "review_required": True,
                "review_status": review_status,
                "qrels": qrel_entries,
            }
        )

    output_path = _choose_candidates_path(candidates_path, overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pack_id": manifest.get("pack_id"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_evidence_dir": str(evidence_dir),
        "review_required": True,
        "candidates": rows,
    }
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report_path = Path("reports") / f"{run_id}_qrel_review.md"
    report_lines: List[str] = []
    report_lines.append(f"# Muon DIS Qrel Candidate Review - {run_id}")
    report_lines.append("")
    report_lines.append(f"- Evidence directory: `{evidence_dir}`")
    report_lines.append(f"- Candidate file: `{output_path}`")
    report_lines.append(f"- Candidate qrel entries: {total_candidates}")
    report_lines.append(f"- NOT_FOUND_IN_INDEX queries: {not_found_count}")
    report_lines.append("- Confirmed qrels modified: no")
    report_lines.append("")
    report_lines.append("## Query Review Rows")
    for row in rows:
        report_lines.append(
            f"- `{row['query_id']}` status `{row['manifest_status']}` -> `{row['review_status']}` "
            f"({len(row['qrels'])} candidate ranges)."
        )
    report_lines.append("")
    report_lines.append("## Policy")
    report_lines.append("- All candidates remain unconfirmed and require manual review.")
    report_lines.append("- This script does not modify `benchmarks/muon_dis/qrels.yaml`.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "evidence_dir": str(evidence_dir),
        "candidate_file": str(output_path),
        "report_file": str(report_path),
        "candidate_count": total_candidates,
        "not_found_in_index_count": not_found_count,
        "confirmed_qrels_modified": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or args.evidence_dir.name
    summary = generate_qrel_review(
        evidence_dir=args.evidence_dir,
        run_id=run_id,
        candidates_path=args.candidates_path,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
