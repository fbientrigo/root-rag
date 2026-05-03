"""Evaluate Muon DIS retrieval runs against scaffolded golden queries and qrels."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Muon DIS retrieval outputs against qrels.")
    parser.add_argument("--evidence-dir", required=True, type=Path, help="Evidence run directory with manifest.json.")
    parser.add_argument(
        "--golden",
        default=Path("benchmarks/muon_dis/golden_queries.yaml"),
        type=Path,
        help="Golden queries YAML path.",
    )
    parser.add_argument(
        "--qrels",
        default=Path("benchmarks/muon_dis/qrels.yaml"),
        type=Path,
        help="Muon DIS qrels YAML path.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional explicit top-k override.")
    return parser.parse_args(argv)


def _load_manifest(evidence_dir: Path) -> Dict[str, Any]:
    manifest_path = evidence_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a JSON object.")
    if "queries" not in payload or not isinstance(payload["queries"], list):
        raise ValueError("Manifest field queries must be a list.")
    return payload


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in: {path}")
    return payload


def _resolve_output_file(output_ref: str | None, evidence_dir: Path, query_id: str) -> Path:
    if output_ref:
        output_path = Path(str(output_ref))
    else:
        output_path = evidence_dir / f"{query_id}.json"

    if output_path.is_absolute():
        return output_path

    candidates = [output_path, evidence_dir / output_path, evidence_dir / output_path.name, Path.cwd() / output_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return output_path


def _normalize_hits(payload: Any) -> Tuple[str, List[Mapping[str, Any]], str]:
    if isinstance(payload, list):
        hits = [row for row in payload if isinstance(row, dict)]
        if not hits:
            return "ZERO_HIT", [], ""
        return "HIT", hits, ""
    if isinstance(payload, dict):
        if payload.get("evidence_format") == "text-wrapper":
            return_code = payload.get("return_code")
            if return_code == 0:
                parsed_hits = payload.get("hits")
                if isinstance(parsed_hits, list):
                    hits = [row for row in parsed_hits if isinstance(row, dict)]
                    if hits:
                        return "HIT", hits, ""
                return "TEXT_EVIDENCE_UNSUPPORTED_FOR_METRICS", [], "text-wrapper"
            if return_code == 5:
                return "ZERO_HIT", [], ""
            if isinstance(return_code, int):
                return "ERROR", [], f"return-code:{return_code}"
            return "ERROR", [], "missing-return-code"

        if payload.get("dry_run") is True:
            return "ERROR", [], "dry-run-artifact"
        for key in ("evidence", "hits", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                hits = [row for row in value if isinstance(row, dict)]
                if not hits:
                    return "ZERO_HIT", [], ""
                return "HIT", hits, ""
        return "ERROR", [], "unsupported-json-shape"
    return "ERROR", [], "unsupported-json-type"


def _read_hits(path: Path) -> Tuple[str, List[Mapping[str, Any]], str]:
    if not path.exists():
        return "ERROR", [], f"missing-file:{path.name}"
    if path.stat().st_size == 0:
        return "ERROR", [], f"empty-file:{path.name}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "ERROR", [], f"invalid-json:{path.name}"
    status, hits, detail = _normalize_hits(payload)
    if status == "ERROR":
        return status, hits, f"{detail}:{path.name}"
    return status, hits, detail


def _canonical_hit_anchor(hit: Mapping[str, Any]) -> str:
    file_path = hit.get("file_path", hit.get("file"))
    start = hit.get("start_line", hit.get("line_start"))
    end = hit.get("end_line", hit.get("line_end"))
    if isinstance(file_path, str) and isinstance(start, int) and isinstance(end, int):
        return f"{file_path}:{start}-{end}"
    if isinstance(file_path, str):
        return file_path
    return ""


def _canonical_qrel_anchor(qrel: Mapping[str, Any]) -> str:
    chunk_id = qrel.get("chunk_id")
    if isinstance(chunk_id, str) and chunk_id:
        return chunk_id

    file_path = qrel.get("file_path", qrel.get("file"))
    start = qrel.get("start_line")
    end = qrel.get("end_line")
    if isinstance(file_path, str) and isinstance(start, int) and isinstance(end, int):
        return f"{file_path}:{start}-{end}"
    if isinstance(file_path, str):
        return file_path
    return ""


def _load_golden_queries(golden_payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = golden_payload.get("golden_queries")
    if not isinstance(rows, list):
        raise ValueError("golden_queries.yaml must contain golden_queries list.")
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("query_id"), str):
            by_id[row["query_id"]] = row
    return by_id


def _load_qrels_payload(qrels_payload: Mapping[str, Any]) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    confirmed_map: Dict[str, Dict[str, int]] = {}
    pending_ids: List[str] = []

    for row in qrels_payload.get("confirmed_qrels", []):
        if not isinstance(row, dict):
            continue
        query_id = row.get("query_id")
        entries = row.get("qrels")
        if not isinstance(query_id, str) or not isinstance(entries, list):
            continue
        rel_map: Dict[str, int] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            anchor = _canonical_qrel_anchor(entry)
            if not anchor:
                continue
            relevance = entry.get("relevance", 1)
            if isinstance(relevance, int):
                rel_map[anchor] = relevance
        confirmed_map[query_id] = rel_map

    for row in qrels_payload.get("pending_qrels", []):
        if not isinstance(row, dict):
            continue
        query_id = row.get("query_id")
        if isinstance(query_id, str) and row.get("pending_label") is True:
            pending_ids.append(query_id)

    return confirmed_map, sorted(set(pending_ids))


def _compute_query_metrics(hits: List[Mapping[str, Any]], relevance_map: Mapping[str, int], top_k: int) -> Dict[str, Any]:
    top_hits = hits[:top_k]
    retrieved = [_canonical_hit_anchor(hit) for hit in top_hits]
    retrieved = [row for row in retrieved if row]
    positives = [anchor for anchor, gain in relevance_map.items() if gain > 0]
    positive_set = set(positives)
    retrieved_positive = [anchor for anchor in retrieved if anchor in positive_set]

    precision_at_k = (len(retrieved_positive) / top_k) if top_k > 0 else 0.0
    recall_at_k = (len(retrieved_positive) / len(positive_set)) if positive_set else 0.0
    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "retrieved_positive_count": len(retrieved_positive),
        "qrels_positive_count": len(positive_set),
        "retrieved_anchors": retrieved,
    }


def evaluate_run(
    *,
    evidence_dir: Path,
    golden_path: Path,
    qrels_path: Path,
    top_k_override: int | None = None,
) -> Dict[str, Any]:
    """Evaluate one manifest run and return deterministic JSON-serializable summary."""
    manifest = _load_manifest(evidence_dir)
    golden_queries = _load_golden_queries(_load_yaml_mapping(golden_path))
    confirmed_qrels, pending_ids = _load_qrels_payload(_load_yaml_mapping(qrels_path))

    top_k = int(top_k_override if top_k_override is not None else manifest.get("top_k", 10))

    per_query: List[Dict[str, Any]] = []
    unresolved: List[str] = []
    pending: List[str] = []
    text_unsupported: List[str] = []
    scored_rows: List[Dict[str, float]] = []

    for query_row in manifest["queries"]:
        query_id = str(query_row.get("id", "UNKNOWN"))
        query_text = str(query_row.get("query", ""))
        return_code = int(query_row.get("return_code", 1))
        output_file = _resolve_output_file(query_row.get("output_file"), evidence_dir, query_id)
        entry_format = query_row.get("evidence_format", manifest.get("evidence_format"))

        if entry_format == "text-wrapper":
            status, hits, error_detail = _read_hits(output_file)
        elif return_code != 0:
            if output_file.exists() and output_file.stat().st_size == 0:
                status, hits, error_detail = "ERROR", [], f"empty-file:{output_file.name}"
            else:
                status, hits, error_detail = "ERROR", [], f"return-code:{return_code}"
        else:
            status, hits, error_detail = _read_hits(output_file)

        has_golden = query_id in golden_queries
        is_pending = query_id in pending_ids or bool(golden_queries.get(query_id, {}).get("pending_label"))
        relevance_map = confirmed_qrels.get(query_id, {})

        row: Dict[str, Any] = {
            "query_id": query_id,
            "query": query_text,
            "status": status,
            "hit_count": len(hits),
            "golden_defined": has_golden,
            "pending_label": is_pending,
            "scored": False,
        }

        if status == "ERROR":
            row["error"] = error_detail
            unresolved.append(f"{query_id}: {error_detail}")
        elif status == "TEXT_EVIDENCE_UNSUPPORTED_FOR_METRICS":
            text_unsupported.append(query_id)
            row["metrics_state"] = "TEXT_EVIDENCE_UNSUPPORTED_FOR_METRICS"
            if is_pending and not relevance_map:
                pending.append(query_id)
                row["evaluation_state"] = "pending_qrels"
            else:
                unresolved.append(f"{query_id}: TEXT_EVIDENCE_UNSUPPORTED_FOR_METRICS")
                row["evaluation_state"] = "TEXT_EVIDENCE_UNSUPPORTED_FOR_METRICS"
        elif is_pending and not relevance_map:
            pending.append(query_id)
            row["evaluation_state"] = "pending_qrels"
        elif not has_golden:
            unresolved.append(f"{query_id}: missing-golden-query")
            row["evaluation_state"] = "missing_golden_query"
        elif not relevance_map:
            unresolved.append(f"{query_id}: missing-confirmed-qrels")
            row["evaluation_state"] = "missing_confirmed_qrels"
        else:
            metrics = _compute_query_metrics(hits, relevance_map, top_k)
            row.update(metrics)
            row["scored"] = True
            row["evaluation_state"] = "scored"
            scored_rows.append(metrics)

        per_query.append(row)

    per_query.sort(key=lambda row: row["query_id"])
    unresolved = sorted(set(unresolved))
    pending = sorted(set(pending))
    text_unsupported = sorted(set(text_unsupported))

    if scored_rows:
        macro_precision = sum(row["precision_at_k"] for row in scored_rows) / len(scored_rows)
        macro_recall = sum(row["recall_at_k"] for row in scored_rows) / len(scored_rows)
    else:
        macro_precision = 0.0
        macro_recall = 0.0

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "evidence_dir": str(evidence_dir),
            "manifest_pack_id": manifest.get("pack_id"),
            "manifest_timestamp": manifest.get("timestamp"),
            "golden_path": str(golden_path),
            "qrels_path": str(qrels_path),
            "top_k": top_k,
        },
        "summary": {
            "manifest_query_count": len(manifest["queries"]),
            "golden_query_count": len(golden_queries),
            "scored_query_count": len(scored_rows),
            "pending_query_count": len(pending),
            "text_evidence_unsupported_count": len(text_unsupported),
            "unresolved_count": len(unresolved),
            "macro_precision_at_k": macro_precision,
            "macro_recall_at_k": macro_recall,
            "qrels_state": "NO_CONFIRMED_QRELS" if not confirmed_qrels else "CONFIRMED_QRELS_PRESENT",
        },
        "per_query": per_query,
        "pending_queries": pending,
        "text_evidence_unsupported_queries": text_unsupported,
        "unresolved": unresolved,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = evaluate_run(
            evidence_dir=args.evidence_dir,
            golden_path=args.golden,
            qrels_path=args.qrels,
            top_k_override=args.top_k,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
