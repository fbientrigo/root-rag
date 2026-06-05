"""Guarded Muon DIS V0 benchmark freeze generator."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml


DEFAULT_SUMMARY = Path("reports/muon_dis_emv_reconciled_01_vertical_slice_summary.json")
DEFAULT_QRELS = Path("benchmarks/muon_dis/qrels.yaml")
DEFAULT_CANDIDATES = Path("benchmarks/muon_dis/qrels_candidates.yaml")
DEFAULT_DECISIONS = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
DEFAULT_AGENT_ADJUDICATED = Path("benchmarks/muon_dis/qrels_agent_adjudicated.yaml")
DEFAULT_GOLDEN = Path("benchmarks/muon_dis/golden_queries.yaml")
DEFAULT_QUERY_PACK = Path("query_packs/muon_dis_workflow.yaml")
DEFAULT_OUTPUT = Path("benchmarks/muon_dis/V0_FREEZE.md")
DEFAULT_JSON_OUTPUT = Path("artifacts/benchmarks/muon_dis_v0_freeze.json")
DEFAULT_DRAFT_JSON_OUTPUT = Path("artifacts/benchmarks/muon_dis_v0_freeze_draft.json")

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
    parser = argparse.ArgumentParser(description="Create guarded Muon DIS V0 benchmark freeze.")
    parser.add_argument("--summary", type=Path, default=None, help="PASS vertical slice summary JSON.")
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS, help="Qrels YAML path.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES, help="Qrel candidates YAML path.")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS, help="Qrel decisions YAML path.")
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


def _count_confirmed_qrels(qrels_payload: Mapping[str, Any]) -> Tuple[int, Dict[str, int]]:
    rows = qrels_payload.get("confirmed_qrels")
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


def _count_agent_adjudicated(payload: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    rows = payload.get("adjudicated")
    if not isinstance(rows, list):
        return 0, 0, 0, 0
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
    return len(rows), approved, rejected, needs_context


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

        if confirmed_by_query.get(query_id, 0) > 0 or approved_by_query.get(query_id, 0) > 0:
            continue
        missing_areas.append(area_name)
    return (len(missing_areas) == 0), missing_areas, reviewed_not_found_areas


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
        return {"eval_path": None, "eval_artifact_status": "NOT_AVAILABLE"}
    eval_path = Path(eval_path_raw)
    if not eval_path.exists():
        return {"eval_path": str(eval_path), "eval_artifact_status": "MISSING_EVAL_FILE"}
    try:
        payload = _load_json_object(eval_path)
    except (json.JSONDecodeError, ValueError):
        return {"eval_path": str(eval_path), "eval_artifact_status": "MALFORMED_EVAL_FILE"}

    return {
        "eval_path": str(eval_path),
        "eval_artifact_status": "AVAILABLE",
        "pending_query_count": payload.get("pending_query_count"),
        "scored_query_count": payload.get("scored_query_count"),
        "qrels_state": payload.get("qrels_state"),
    }


def _derive_metrics_status(*, confirmed_qrel_count: int, eval_artifact_status: str) -> str:
    if confirmed_qrel_count <= 0:
        return "NO_CONFIRMED_QRELS"
    if eval_artifact_status != "AVAILABLE":
        return "NOT_AVAILABLE"
    return "AVAILABLE"


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
            f"- Candidates path: `{metadata['candidates_path']}`",
            f"- Decisions path: `{metadata['decisions_path']}`",
            f"- Agent adjudicated path: `{metadata['agent_adjudicated_path']}`",
            f"- Confirmed qrel count: `{metadata['confirmed_qrel_count']}`",
            f"- agent_adjudicated_count: `{metadata['agent_adjudicated_count']}`",
            f"- agent_approved_count: `{metadata['agent_approved_count']}`",
            f"- agentic_bootstrap_available: `{metadata['agentic_bootstrap_available']}`",
            f"- Min confirmed qrels threshold: `{metadata['min_confirmed_qrels']}`",
            f"- v0_coverage_ready: `{metadata['v0_coverage_ready']}`",
            f"- missing_v0_coverage_areas: `{metadata['missing_v0_coverage_areas']}`",
            f"- reviewed_not_found_areas: `{metadata['reviewed_not_found_areas']}`",
            f"- Query count: `{metadata['query_count']}`",
            f"- Hit count: `{metadata['hit_count']}`",
            f"- Zero-hit count: `{metadata['zero_hit_count']}`",
            f"- Error count: `{metadata['error_count']}`",
            f"- qrels_state: `{metadata['qrels_state']}`",
            f"- Eval artifact status: `{metadata['eval_artifact_status']}`",
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
    candidates_path: Path = DEFAULT_CANDIDATES,
    decisions_path: Path = DEFAULT_DECISIONS,
    agent_adjudicated_path: Path = DEFAULT_AGENT_ADJUDICATED,
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
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")
    if not decisions_path.exists():
        raise FileNotFoundError(f"Decisions file not found: {decisions_path}")

    summary_payload = _load_json_object(resolved_summary_path)
    gate_status = str(summary_payload.get("acceptance_gate_status", "")).strip().upper()
    if gate_status != "PASS":
        raise ValueError(f"Freeze requires PASS summary; found {gate_status or 'UNKNOWN'} in {resolved_summary_path}")

    qrels_payload = _load_yaml_mapping(qrels_path)
    candidates_payload = _load_yaml_mapping(candidates_path)
    decisions_payload = _load_yaml_mapping(decisions_path)
    agent_adjudicated_payload: Dict[str, Any] = {}
    if agent_adjudicated_path.exists():
        agent_adjudicated_payload = _load_yaml_mapping(agent_adjudicated_path)
    (
        agent_adjudicated_count,
        agent_approved_count,
        agent_rejected_count,
        agent_needs_context_count,
    ) = _count_agent_adjudicated(agent_adjudicated_payload)
    confirmed_qrel_count, confirmed_by_query = _count_confirmed_qrels(qrels_payload)
    approved_by_query = _approved_by_query(decisions_payload)
    reviewed_not_found_queries = _not_found_reviewed_queries(candidates_payload)
    v0_coverage_ready, missing_areas, reviewed_not_found_areas = _compute_v0_coverage(
        confirmed_by_query=confirmed_by_query,
        approved_by_query=approved_by_query,
        not_found_reviewed_queries=reviewed_not_found_queries,
    )

    eval_status = _read_eval_status(summary_payload)
    eval_artifact_status = str(eval_status.get("eval_artifact_status", "NOT_AVAILABLE"))
    metrics_status = _derive_metrics_status(
        confirmed_qrel_count=confirmed_qrel_count,
        eval_artifact_status=eval_artifact_status,
    )
    qrels_state = str(summary_payload.get("qrels_state") or eval_status.get("qrels_state") or "NO_CONFIRMED_QRELS")

    enough_qrels = confirmed_qrel_count >= min_confirmed_qrels
    freeze_status = "READY_FOR_FREEZE"
    block_reason = ""
    if draft:
        freeze_status = "DRAFT_NOT_FINAL"
    elif not v0_coverage_ready:
        freeze_status = "BLOCKED_COVERAGE_INCOMPLETE"
        block_reason = f"Missing mandatory V0 coverage areas: {', '.join(missing_areas)}."
    elif not enough_qrels:
        freeze_status = "BLOCKED_INSUFFICIENT_QRELS"
        block_reason = (
            f"confirmed_qrel_count {confirmed_qrel_count} is below min_confirmed_qrels {min_confirmed_qrels}."
        )

    freeze_doc_path = output_path.with_name("V0_FREEZE_DRAFT.md") if draft else output_path

    metadata: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": summary_payload.get("run_id"),
        "summary_path": str(resolved_summary_path),
        "index_id": summary_payload.get("index_id"),
        "index_dir": summary_payload.get("index_dir"),
        "query_pack_path": str(query_pack_path),
        "golden_queries_path": str(golden_path),
        "qrels_path": str(qrels_path),
        "candidates_path": str(candidates_path),
        "decisions_path": str(decisions_path),
        "agent_adjudicated_path": str(agent_adjudicated_path),
        "confirmed_qrel_count": confirmed_qrel_count,
        "agent_adjudicated_count": agent_adjudicated_count,
        "agent_approved_count": agent_approved_count,
        "agent_rejected_count": agent_rejected_count,
        "agent_needs_context_count": agent_needs_context_count,
        "agentic_bootstrap_available": agent_adjudicated_count > 0,
        "min_confirmed_qrels": min_confirmed_qrels,
        "v0_coverage_ready": v0_coverage_ready,
        "missing_v0_coverage_areas": missing_areas,
        "reviewed_not_found_areas": reviewed_not_found_areas,
        "query_count": summary_payload.get("query_count"),
        "hit_count": summary_payload.get("hit_count"),
        "zero_hit_count": summary_payload.get("zero_hit_count"),
        "error_count": summary_payload.get("error_count"),
        "qrels_state": qrels_state,
        "eval_artifact_status": eval_artifact_status,
        "metrics_status": metrics_status,
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

    if freeze_status.startswith("BLOCKED_"):
        return metadata

    freeze_doc_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_doc_path.write_text(_build_markdown(metadata, draft_mode=False), encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    json_output_path = args.json_output
    if args.draft and json_output_path == DEFAULT_JSON_OUTPUT:
        json_output_path = DEFAULT_DRAFT_JSON_OUTPUT
    try:
        summary = freeze_benchmark(
            summary_path=args.summary,
            qrels_path=args.qrels,
            candidates_path=args.candidates,
            decisions_path=args.decisions,
            agent_adjudicated_path=DEFAULT_AGENT_ADJUDICATED,
            golden_path=args.golden,
            query_pack_path=args.query_pack,
            output_path=args.output,
            json_output_path=json_output_path,
            min_confirmed_qrels=args.min_confirmed_qrels,
            draft=args.draft,
        )
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"freeze_status": "ERROR", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2))
    if summary["freeze_status"].startswith("BLOCKED_"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
