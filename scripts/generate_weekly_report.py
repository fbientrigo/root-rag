"""Generate deterministic weekly Markdown report from query-pack evidence outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for report generation."""
    parser = argparse.ArgumentParser(description="Generate weekly Markdown report from evidence artifacts.")
    parser.add_argument("--evidence-dir", required=True, type=Path, help="Evidence run directory containing manifest.json.")
    parser.add_argument("--output", required=True, type=Path, help="Output Markdown report path.")
    parser.add_argument("--title", default=None, help="Optional report title override.")
    return parser.parse_args(argv)


def load_manifest(evidence_dir: Path) -> Dict[str, Any]:
    """Load and validate evidence manifest.json."""
    manifest_path = evidence_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object.")

    required_fields = ["pack_path", "pack_id", "timestamp", "queries"]
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Manifest missing required field: {field}")

    if not isinstance(manifest["queries"], list):
        raise ValueError("Manifest field queries must be a list.")

    return manifest


def _load_pack_rq(pack_path: Path) -> str:
    """Read research question from query pack when available."""
    if not pack_path.exists():
        return "TODO: Query pack file not available in this environment."

    with pack_path.open("r", encoding="utf-8") as handle:
        pack_payload = yaml.safe_load(handle)

    if not isinstance(pack_payload, dict):
        return "TODO: Query pack payload is not a mapping."

    rq = pack_payload.get("rq")
    if isinstance(rq, str) and rq.strip():
        return rq.strip()

    return "TODO: Add research question (`rq`) to query pack."


def _resolve_pack_path(pack_ref: str, evidence_dir: Path) -> Path:
    """Resolve query-pack path from manifest value across common relative forms."""
    pack_path = Path(pack_ref)
    if pack_path.is_absolute():
        return pack_path

    candidates = [
        pack_path,
        evidence_dir / pack_path,
        Path.cwd() / pack_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return pack_path


def _format_line_range(hit: Mapping[str, Any]) -> str:
    """Format line range from hit payload when fields exist."""
    start = hit.get("start_line", hit.get("line_start"))
    end = hit.get("end_line", hit.get("line_end"))
    if isinstance(start, int) and isinstance(end, int):
        return f"{start}-{end}"
    return "N/A"


def _extract_score(hit: Mapping[str, Any]) -> str:
    """Format score with stable precision when available."""
    score = hit.get("score", hit.get("bm25_score"))
    if isinstance(score, (int, float)):
        return f"{float(score):.6f}"
    return "N/A"


def _safe_cell(value: str) -> str:
    """Escape markdown table pipes for deterministic rendering."""
    return value.replace("|", "\\|")


def _resolve_query_output_file(output_ref: str | None, evidence_dir: Path, query_id: str) -> Path:
    """Resolve per-query output path across absolute, repo-relative, and evidence-relative forms."""
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


def _normalize_hits_from_payload(payload: Any) -> Tuple[str, List[Mapping[str, Any]], str]:
    """Normalize supported evidence payload shapes into a list of hit mappings."""
    if isinstance(payload, list):
        hits = [hit for hit in payload if isinstance(hit, dict)]
        if not hits:
            return "ZERO_HIT", [], ""
        return "HIT", hits, ""

    if isinstance(payload, dict):
        if payload.get("evidence_format") == "text-wrapper":
            return_code = payload.get("return_code")
            if return_code == 0:
                parsed_hits = payload.get("hits")
                if isinstance(parsed_hits, list):
                    hits = [hit for hit in parsed_hits if isinstance(hit, dict)]
                    if hits:
                        return "HIT", hits, ""
                return "HIT_OR_TEXT_EVIDENCE", [], ""
            if return_code == 5:
                return "ZERO_HIT", [], ""
            if isinstance(return_code, int):
                return "ERROR", [], f"return-code:{return_code}"
            return "ERROR", [], "missing-return-code"

        if payload.get("dry_run") is True:
            return "ERROR", [], "dry-run-artifact"

        for key in ("evidence", "hits", "results", "items"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                hits = [hit for hit in candidate if isinstance(hit, dict)]
                if not hits:
                    return "ZERO_HIT", [], ""
                return "HIT", hits, ""

        return "ERROR", [], "unsupported-json-shape"

    return "ERROR", [], "unsupported-json-type"


def _read_query_hits(query_output_file: Path) -> Tuple[str, List[Mapping[str, Any]], str]:
    """Read one per-query JSON output and return status, hits, and error info."""
    if not query_output_file.exists():
        return "ERROR", [], f"missing-file:{query_output_file.name}"
    if query_output_file.stat().st_size == 0:
        return "ERROR", [], f"empty-file:{query_output_file.name}"

    try:
        payload = json.loads(query_output_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "ERROR", [], f"invalid-json:{query_output_file.name}"

    status, hits, detail = _normalize_hits_from_payload(payload)
    if status == "ERROR" and detail:
        return "ERROR", [], f"{detail}:{query_output_file.name}"
    return status, hits, detail


def build_report(markdown_title: str, manifest: Mapping[str, Any], evidence_dir: Path) -> str:
    """Build deterministic markdown report text from manifest and query outputs."""
    pack_path = _resolve_pack_path(str(manifest["pack_path"]), evidence_dir)
    rq_text = _load_pack_rq(pack_path)

    rows: List[Dict[str, str]] = []
    candidate_nodes: Dict[str, Dict[str, str]] = {}
    zero_hit_queries: List[str] = []
    error_queries: List[str] = []
    unresolved_lines: List[str] = []
    hit_query_count = 0

    for query_entry in manifest["queries"]:
        query_id = str(query_entry.get("id", "UNKNOWN"))
        query_str = str(query_entry.get("query", ""))
        return_code = int(query_entry.get("return_code", 1))
        output_file = _resolve_query_output_file(query_entry.get("output_file"), evidence_dir, query_id)

        status: str
        hits: List[Mapping[str, Any]]
        error_info: str

        entry_format = query_entry.get("evidence_format", manifest.get("evidence_format"))
        if entry_format == "text-wrapper":
            status, hits, error_info = _read_query_hits(output_file)
        elif return_code != 0:
            if output_file.exists() and output_file.stat().st_size == 0:
                status, hits, error_info = "ERROR", [], f"empty-file:{output_file.name}"
            else:
                status, hits, error_info = "ERROR", [], f"return-code:{return_code}"
        else:
            status, hits, error_info = _read_query_hits(output_file)

        hit_count = str(len(hits))
        top_file = "N/A"
        top_range = "N/A"
        top_score = "N/A"

        if status == "HIT" and hits:
            hit_query_count += 1
            top_hit = hits[0]
            top_file = str(top_hit.get("file_path", top_hit.get("file", "N/A")))
            top_range = _format_line_range(top_hit)
            top_score = _extract_score(top_hit)

            node_key = f"{top_file}:{top_range}"
            if node_key not in candidate_nodes:
                candidate_nodes[node_key] = {
                    "file": top_file,
                    "line_range": top_range,
                    "query_ids": query_id,
                }
            else:
                candidate_nodes[node_key]["query_ids"] = f"{candidate_nodes[node_key]['query_ids']}, {query_id}"

        if status == "HIT_OR_TEXT_EVIDENCE":
            hit_query_count += 1
            hit_count = "N/A"
            top_file = "TEXT_OUTPUT"

        if status == "ZERO_HIT":
            hit_count = "0"
            zero_hit_queries.append(f"- `{query_id}`: `{query_str}`")
            unresolved_lines.append(f"- TODO `{query_id}`: no hits for query `{query_str}`.")
        elif status == "ERROR":
            detail = error_info or str(query_entry.get("stderr", "")).strip() or "unknown-error"
            error_queries.append(f"- `{query_id}`: `{detail}`")
            unresolved_lines.append(f"- TODO `{query_id}`: execution/evidence error (`{detail}`).")

        rows.append(
            {
                "query_id": query_id,
                "query": query_str,
                "hits": hit_count,
                "top_file": top_file,
                "top_range": top_range,
                "score": top_score,
                "status": status,
            }
        )

    lines: List[str] = []
    lines.append(f"# {markdown_title}")
    lines.append("")

    lines.append("## Research Question")
    lines.append(rq_text)
    lines.append("")

    lines.append("## Query Pack Summary")
    lines.append(f"- Pack ID: `{manifest.get('pack_id', 'N/A')}`")
    lines.append(f"- Pack Path: `{manifest.get('pack_path', 'N/A')}`")
    lines.append(f"- Run Timestamp: `{manifest.get('timestamp', 'N/A')}`")
    lines.append(f"- Query Count: {len(rows)}")
    lines.append("")

    lines.append("## Evidence Summary")
    lines.append(f"- Queries with hits: {hit_query_count}")
    lines.append(f"- Zero-hit queries: {len(zero_hit_queries)}")
    lines.append(f"- Error queries: {len(error_queries)}")
    lines.append("")

    lines.append("## Retrieval Table")
    lines.append("| query_id | query string | number of hits | top file | top line range | score | status |")
    lines.append("|---|---|---:|---|---|---:|---|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _safe_cell(row["query_id"]),
                    _safe_cell(row["query"]),
                    row["hits"],
                    _safe_cell(row["top_file"]),
                    _safe_cell(row["top_range"]),
                    row["score"],
                    row["status"],
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Candidate Workflow Nodes")
    if candidate_nodes:
        for node in sorted(candidate_nodes.keys()):
            node_item = candidate_nodes[node]
            lines.append(
                f"- `{node_item['file']}:{node_item['line_range']}` from query_ids: {node_item['query_ids']}."
            )
    else:
        lines.append("- TODO: no candidate nodes from current retrieval metadata.")
    lines.append("")

    lines.append("## Missing or Zero-Hit Queries")
    if zero_hit_queries:
        lines.extend(zero_hit_queries)
    else:
        lines.append("- None.")
    lines.append("")

    lines.append("## Errors")
    if error_queries:
        lines.extend(error_queries)
    else:
        lines.append("- None.")
    lines.append("")

    lines.append("## Unresolved Questions")
    if unresolved_lines:
        lines.extend(unresolved_lines)
    else:
        lines.append("- TODO: add unresolved questions after manual evidence review.")
    lines.append("")

    lines.append("## Next Query Seeds")
    if zero_hit_queries:
        for query_line in zero_hit_queries:
            query_id = query_line.split("`")[1]
            query_text = query_line.split("`")[-2]
            lines.append(f"- `{query_id}` seed: `{query_text}` + synonym/abbrev variants (TODO manual curation).")
    else:
        lines.append("- TODO: derive new seeds from unresolved findings after analyst review.")

    return "\n".join(lines).rstrip() + "\n"


def generate_report(evidence_dir: Path, output_path: Path, title: str | None = None) -> Path:
    """Generate and write report from one evidence directory."""
    manifest = load_manifest(evidence_dir)
    default_title = "Weekly Research Report"
    report_text = build_report(title or default_title, manifest, evidence_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    try:
        generate_report(args.evidence_dir, args.output, args.title)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
