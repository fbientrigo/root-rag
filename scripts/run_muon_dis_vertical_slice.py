"""Run the Muon DIS EMV vertical slice in one deterministic command."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


DEFAULT_TEST_ARGS = [
    "-m",
    "pytest",
    "tests/test_run_query_pack.py",
    "tests/test_generate_weekly_report.py",
    "tests/test_muon_dis_phase2.py",
    "tests/test_lint_wiki_claims.py",
    "tests/test_validate_workflow_graph.py",
    "-q",
]

INDEX_MARKER_FILES = ("fts.sqlite", "index_manifest.json")


@dataclass(frozen=True)
class VerticalSliceConfig:
    index_id: str | None
    index_dir: Path
    run_id: str
    top_k: int
    skip_tests: bool


def parse_args(argv: Sequence[str] | None = None) -> VerticalSliceConfig:
    parser = argparse.ArgumentParser(description="Run Muon DIS vertical slice checks.")
    parser.add_argument("--index-id", default=None, help="Explicit FairShip index id to use.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/indexes_fairship"),
        help="FairShip index root directory.",
    )
    parser.add_argument(
        "--run-id",
        default=f"muon_dis_vertical_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Run id used for evidence/reports output naming.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k value for query-pack run.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip targeted pytest bundle.")
    args = parser.parse_args(argv)
    return VerticalSliceConfig(
        index_id=args.index_id,
        index_dir=args.index_dir,
        run_id=args.run_id,
        top_k=args.top_k,
        skip_tests=args.skip_tests,
    )


def _is_valid_index_dir(path: Path) -> bool:
    return path.is_dir() and any((path / marker).exists() for marker in INDEX_MARKER_FILES)


def _find_valid_child_indexes(index_dir: Path) -> List[Path]:
    if not index_dir.exists() or not index_dir.is_dir():
        return []
    children = [candidate for candidate in index_dir.iterdir() if candidate.is_dir()]
    valid = [candidate for candidate in children if _is_valid_index_dir(candidate)]
    return sorted(valid, key=lambda candidate: candidate.stat().st_mtime, reverse=True)


def _resolve_index_target(config: VerticalSliceConfig) -> Tuple[Path | None, str | None, List[str]]:
    notes: List[str] = []

    if config.index_id:
        explicit_target = config.index_dir / config.index_id
        if _is_valid_index_dir(explicit_target):
            return config.index_dir, config.index_id, notes
        notes.append(
            f"Explicit index target is invalid or missing markers: {explicit_target} (required one of {INDEX_MARKER_FILES})."
        )
        return None, None, notes

    if _is_valid_index_dir(config.index_dir):
        notes.append(f"Index auto-detected from explicit index directory: {config.index_dir.name}.")
        return config.index_dir.parent, config.index_dir.name, notes

    valid_children = _find_valid_child_indexes(config.index_dir)
    if valid_children:
        detected = valid_children[0]
        notes.append(
            f"Index auto-detected from {config.index_dir}: {detected.name}."
        )
        return config.index_dir, detected.name, notes

    notes.append(
        f"No valid index directories found under {config.index_dir}. Required marker files: {INDEX_MARKER_FILES}."
    )
    return None, None, notes


def _run(command: List[str]) -> Dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _collect_manifest_counts(manifest: Mapping[str, Any] | None) -> Dict[str, int]:
    if not manifest:
        return {"query_count": 0, "hit_count": 0, "zero_hit_count": 0, "error_count": 0}
    rows = manifest.get("queries", [])
    if not isinstance(rows, list):
        return {"query_count": 0, "hit_count": 0, "zero_hit_count": 0, "error_count": 0}

    query_count = len(rows)
    zero_hit_count = 0
    error_count = 0
    hit_count = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", ""))
        if status == "ZERO_HIT":
            zero_hit_count += 1
        elif status == "ERROR":
            error_count += 1
        else:
            hit_count += 1

    return {
        "query_count": query_count,
        "hit_count": hit_count,
        "zero_hit_count": zero_hit_count,
        "error_count": error_count,
    }


def _build_summary(
    *,
    run_id: str,
    index_id: str | None,
    index_dir: Path,
    test_status: str,
    counts: Mapping[str, int],
    qrels_state: str,
    text_evidence_unsupported_count: int,
    report_path: Path,
    eval_path: Path,
    evidence_dir: Path,
    acceptance_gate_status: str,
    notes: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "index_id": index_id,
        "index_dir": str(index_dir),
        "test_status": test_status,
        "query_count": counts["query_count"],
        "hit_count": counts["hit_count"],
        "zero_hit_count": counts["zero_hit_count"],
        "error_count": counts["error_count"],
        "qrels_state": qrels_state,
        "text_evidence_unsupported_count": text_evidence_unsupported_count,
        "report_path": str(report_path),
        "eval_path": str(eval_path),
        "evidence_dir": str(evidence_dir),
        "acceptance_gate_status": acceptance_gate_status,
        "notes": notes or [],
    }


def run_vertical_slice(config: VerticalSliceConfig) -> Dict[str, Any]:
    resolved_index_dir, index_id, detection_notes = _resolve_index_target(config)
    evidence_dir = Path("evidence") / config.run_id
    report_path = Path("reports") / f"{config.run_id}.md"
    eval_path = Path("reports") / f"{config.run_id}_eval.json"
    summary_path = Path("reports") / f"{config.run_id}_vertical_slice_summary.json"

    if evidence_dir.exists() and any(evidence_dir.iterdir()):
        summary = _build_summary(
            run_id=config.run_id,
            index_id=index_id,
            index_dir=resolved_index_dir or config.index_dir,
            test_status="not_run",
            counts={"query_count": 0, "hit_count": 0, "zero_hit_count": 0, "error_count": 0},
            qrels_state="UNKNOWN",
            text_evidence_unsupported_count=0,
            report_path=report_path,
            eval_path=eval_path,
            evidence_dir=evidence_dir,
            acceptance_gate_status="FAIL_REUSED_EVIDENCE_DIR",
            notes=detection_notes,
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    if resolved_index_dir is None or index_id is None:
        summary = _build_summary(
            run_id=config.run_id,
            index_id=None,
            index_dir=config.index_dir,
            test_status="not_run",
            counts={"query_count": 0, "hit_count": 0, "zero_hit_count": 0, "error_count": 0},
            qrels_state="UNKNOWN",
            text_evidence_unsupported_count=0,
            report_path=report_path,
            eval_path=eval_path,
            evidence_dir=evidence_dir,
            acceptance_gate_status="FAIL_MISSING_VALID_INDEX",
            notes=detection_notes,
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    test_status = "skipped"
    if not config.skip_tests:
        test_result = _run([sys.executable, *DEFAULT_TEST_ARGS])
        test_status = "passed" if test_result["return_code"] == 0 else "failed"

    query_pack_result = _run(
        [
            sys.executable,
            "scripts/run_query_pack.py",
            "--pack",
            "query_packs/muon_dis_workflow.yaml",
            "--output-dir",
            str(evidence_dir),
            "--top-k",
            str(config.top_k),
            "--index-dir",
            str(resolved_index_dir),
            "--index-id",
            index_id,
        ]
    )
    smoke_ran = True

    report_result = _run(
        [
            sys.executable,
            "scripts/generate_weekly_report.py",
            "--evidence-dir",
            str(evidence_dir),
            "--output",
            str(report_path),
            "--title",
            "Muon DIS EMV Vertical Slice",
        ]
    )

    eval_result = _run(
        [
            sys.executable,
            "scripts/evaluate_muon_dis_retrieval.py",
            "--evidence-dir",
            str(evidence_dir),
            "--output",
            str(eval_path),
        ]
    )

    wiki_lint_result = _run([sys.executable, "scripts/lint_wiki_claims.py", "docs/wiki"])
    graph_validate_result = _run(
        [sys.executable, "scripts/validate_workflow_graph.py", "workflow_graphs/muon_dis_workflow.json"]
    )

    manifest = _load_json(evidence_dir / "manifest.json")
    counts = _collect_manifest_counts(manifest)
    eval_payload = _load_json(eval_path) or {}
    eval_summary = eval_payload.get("summary", {})
    qrels_state = "UNKNOWN"
    text_unsupported = 0
    if isinstance(eval_summary, dict):
        qrels_state = str(eval_summary.get("qrels_state", "UNKNOWN"))
        raw_count = eval_summary.get("text_evidence_unsupported_count", 0)
        text_unsupported = int(raw_count) if isinstance(raw_count, int) else 0

    all_queries_error = counts["query_count"] > 0 and counts["error_count"] == counts["query_count"]
    acceptance_pass = (
        (test_status in {"passed", "skipped"})
        and smoke_ran
        and report_result["return_code"] == 0
        and report_path.exists()
        and eval_result["return_code"] == 0
        and eval_path.exists()
        and counts["query_count"] > 0
        and counts["error_count"] < counts["query_count"]
        and (not all_queries_error)
        and wiki_lint_result["return_code"] == 0
        and graph_validate_result["return_code"] == 0
    )

    acceptance_gate_status = "PASS" if acceptance_pass else "FAIL"
    summary = _build_summary(
        run_id=config.run_id,
        index_id=index_id,
        index_dir=resolved_index_dir,
        test_status=test_status,
        counts=counts,
        qrels_state=qrels_state,
        text_evidence_unsupported_count=text_unsupported,
        report_path=report_path,
        eval_path=eval_path,
        evidence_dir=evidence_dir,
        acceptance_gate_status=acceptance_gate_status,
        notes=detection_notes,
    )
    summary["command_status"] = {
        "tests": test_status,
        "run_query_pack": query_pack_result["return_code"],
        "generate_report": report_result["return_code"],
        "evaluate": eval_result["return_code"],
        "wiki_lint": wiki_lint_result["return_code"],
        "workflow_graph_validate": graph_validate_result["return_code"],
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    summary = run_vertical_slice(config)
    print(json.dumps(summary, indent=2))
    if summary["acceptance_gate_status"] == "PASS":
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
