"""Guarded Muon DIS V0 benchmark freeze generator."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml


DEFAULT_SUMMARY = Path("reports/muon_dis_emv_reconciled_01_vertical_slice_summary.json")
DEFAULT_QRELS = Path("benchmarks/muon_dis/qrels.yaml")
DEFAULT_GOLDEN = Path("benchmarks/muon_dis/golden_queries.yaml")
DEFAULT_QUERY_PACK = Path("query_packs/muon_dis_workflow.yaml")
DEFAULT_OUTPUT = Path("benchmarks/muon_dis/V0_FREEZE.md")
DEFAULT_JSON_OUTPUT = Path("artifacts/benchmarks/muon_dis_v0_freeze.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create guarded Muon DIS V0 benchmark freeze.")
    parser.add_argument("--summary", type=Path, default=None, help="PASS vertical slice summary JSON.")
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS, help="Qrels YAML path.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="Golden queries YAML path.")
    parser.add_argument("--query-pack", type=Path, default=DEFAULT_QUERY_PACK, help="Query pack path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Final freeze markdown output path.")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT, help="Freeze metadata JSON path.")
    parser.add_argument("--min-confirmed-qrels", type=int, default=5, help="Minimum confirmed qrels required.")
    parser.add_argument("--draft", action="store_true", help="Write DRAFT freeze document and do not claim final freeze.")
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


def _count_confirmed_qrels(qrels_payload: Mapping[str, Any]) -> int:
    rows = qrels_payload.get("confirmed_qrels")
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


def _load_emv_status_module():
    module_path = Path(__file__).with_name("emv_status.py")
    spec = importlib.util.spec_from_file_location("emv_status_for_freeze", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load emv_status module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_summary_path(summary_path: Optional[Path]) -> Path:
    if summary_path is not None:
        return summary_path
    module = _load_emv_status_module()
    status = module.collect_status()
    latest = status.get("latest_pass_summary_path")
    if not isinstance(latest, str) or not latest:
        raise ValueError("No valid PASS vertical slice summary found; freeze is blocked.")
    return Path(latest)


def _read_eval_status(summary_payload: Mapping[str, Any]) -> Dict[str, Any]:
    eval_path_raw = summary_payload.get("eval_path")
    if not isinstance(eval_path_raw, str) or not eval_path_raw.strip():
        return {"eval_path": None, "metrics_status": "NOT_AVAILABLE"}
    eval_path = Path(eval_path_raw)
    if not eval_path.exists():
        return {"eval_path": str(eval_path), "metrics_status": "MISSING_EVAL_FILE"}
    try:
        payload = _load_json_object(eval_path)
    except (json.JSONDecodeError, ValueError):
        return {"eval_path": str(eval_path), "metrics_status": "MALFORMED_EVAL_FILE"}

    return {
        "eval_path": str(eval_path),
        "metrics_status": "AVAILABLE",
        "pending_query_count": payload.get("pending_query_count"),
        "scored_query_count": payload.get("scored_query_count"),
        "qrels_state": payload.get("qrels_state"),
    }


def _build_markdown(metadata: Mapping[str, Any], *, draft_mode: bool) -> str:
    header = "# Muon DIS V0 Benchmark Freeze"
    if draft_mode:
        header = "# Muon DIS V0 Benchmark Freeze DRAFT"
    lines = [header, ""]
    if draft_mode:
        lines.append("> DRAFT / NOT A BENCHMARK FREEZE")
        lines.append("")
    lines.extend(
        [
            f"- Generated at: `{metadata['generated_at']}`",
            f"- Run id: `{metadata['run_id']}`",
            f"- FairShip index id: `{metadata['index_id']}`",
            f"- Index dir: `{metadata['index_dir']}`",
            f"- Query pack path: `{metadata['query_pack_path']}`",
            f"- Golden queries path: `{metadata['golden_queries_path']}`",
            f"- Qrels path: `{metadata['qrels_path']}`",
            f"- Confirmed qrel count: `{metadata['confirmed_qrel_count']}`",
            f"- Min confirmed qrels threshold: `{metadata['min_confirmed_qrels']}`",
            f"- Query count: `{metadata['query_count']}`",
            f"- Hit count: `{metadata['hit_count']}`",
            f"- Zero-hit count: `{metadata['zero_hit_count']}`",
            f"- Error count: `{metadata['error_count']}`",
            f"- qrels_state: `{metadata['qrels_state']}`",
            f"- Evaluator/metrics status: `{metadata['metrics_status']}`",
            f"- Freeze status: `{metadata['freeze_status']}`",
            "",
            "## Limitations",
            "",
            "- Qrel confirmation is manual and reviewer-dependent.",
            "- Candidate qrels and decisions may change with future evidence runs.",
            "- Benchmark freeze does not certify semantic completeness of workflow claims.",
            "- Wiki claims are not automatically confirmed by this freeze.",
            "- Workflow graph claims are not automatically confirmed by this freeze.",
            "",
        ]
    )
    if metadata.get("block_reason"):
        lines.append("## Block Reason")
        lines.append("")
        lines.append(f"- {metadata['block_reason']}")
        lines.append("")
    return "\n".join(lines)


def freeze_benchmark(
    *,
    summary_path: Optional[Path],
    qrels_path: Path,
    golden_path: Path,
    query_pack_path: Path,
    output_path: Path,
    json_output_path: Path,
    min_confirmed_qrels: int,
    draft: bool,
) -> Dict[str, Any]:
    resolved_summary_path = _resolve_summary_path(summary_path)
    if not resolved_summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {resolved_summary_path}")

    summary_payload = _load_json_object(resolved_summary_path)
    gate_status = str(summary_payload.get("acceptance_gate_status", "")).strip().upper()
    if gate_status != "PASS":
        raise ValueError(f"Freeze requires PASS summary; found {gate_status or 'UNKNOWN'} in {resolved_summary_path}")

    qrels_payload = _load_yaml_mapping(qrels_path)
    confirmed_qrel_count = _count_confirmed_qrels(qrels_payload)
    eval_status = _read_eval_status(summary_payload)
    qrels_state = str(summary_payload.get("qrels_state") or eval_status.get("qrels_state") or "NO_CONFIRMED_QRELS")

    enough_qrels = confirmed_qrel_count >= min_confirmed_qrels
    freeze_status = "READY_FOR_FREEZE"
    block_reason = ""
    if not enough_qrels:
        freeze_status = "BLOCKED_INSUFFICIENT_QRELS"
        block_reason = (
            f"confirmed_qrel_count {confirmed_qrel_count} is below min_confirmed_qrels {min_confirmed_qrels}."
        )
    if draft:
        freeze_status = "DRAFT_NOT_FINAL"

    freeze_doc_path = output_path
    if draft:
        freeze_doc_path = output_path.with_name("V0_FREEZE_DRAFT.md")

    metadata: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": summary_payload.get("run_id"),
        "summary_path": str(resolved_summary_path),
        "index_id": summary_payload.get("index_id"),
        "index_dir": summary_payload.get("index_dir"),
        "query_pack_path": str(query_pack_path),
        "golden_queries_path": str(golden_path),
        "qrels_path": str(qrels_path),
        "confirmed_qrel_count": confirmed_qrel_count,
        "min_confirmed_qrels": min_confirmed_qrels,
        "query_count": summary_payload.get("query_count"),
        "hit_count": summary_payload.get("hit_count"),
        "zero_hit_count": summary_payload.get("zero_hit_count"),
        "error_count": summary_payload.get("error_count"),
        "qrels_state": qrels_state,
        "metrics_status": eval_status.get("metrics_status"),
        "eval_path": eval_status.get("eval_path"),
        "pending_query_count": eval_status.get("pending_query_count"),
        "scored_query_count": eval_status.get("scored_query_count"),
        "freeze_status": freeze_status,
        "draft": draft,
        "freeze_output_path": str(freeze_doc_path),
        "json_output_path": str(json_output_path),
        "block_reason": block_reason,
    }

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if draft:
        freeze_doc_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_doc_path.write_text(_build_markdown(metadata, draft_mode=True), encoding="utf-8")
        return metadata

    if not enough_qrels:
        return metadata

    freeze_doc_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_doc_path.write_text(_build_markdown(metadata, draft_mode=False), encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = freeze_benchmark(
            summary_path=args.summary,
            qrels_path=args.qrels,
            golden_path=args.golden,
            query_pack_path=args.query_pack,
            output_path=args.output,
            json_output_path=args.json_output,
            min_confirmed_qrels=args.min_confirmed_qrels,
            draft=args.draft,
        )
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"freeze_status": "ERROR", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2))
    if summary["freeze_status"] == "BLOCKED_INSUFFICIENT_QRELS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

