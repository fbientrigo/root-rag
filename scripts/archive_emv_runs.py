"""Conservative archiver for stale EMV run artifacts."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REPORTS_DIR = Path("reports")
EVIDENCE_DIR = Path("evidence")
ARTIFACTS_DIR = Path("artifacts")

ARCHIVE_REPORTS_DIR = REPORTS_DIR / "archive"
ARCHIVE_EVIDENCE_DIR = EVIDENCE_DIR / "archive"
ARCHIVE_ARTIFACTS_DIR = ARTIFACTS_DIR / "archive"

MANIFEST_PATH = REPORTS_DIR / "emv_archive_manifest.json"

PROTECTED_REPORTS = {
    "emv_archive_manifest.json",
    "emv_foundation_preimplementation_audit.md",
}
PROTECTED_EVIDENCE_DIRS = {"archive"}
PROTECTED_ARTIFACT_PATH_PARTS = {"benchmarks", "archive"}


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    status: str
    summary_path: Path
    modified_at: datetime


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive stale EMV run artifacts.")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True, help="List archival plan without moving files.")
    parser.add_argument("--execute", dest="dry_run", action="store_false", help="Move files to archive folders.")
    parser.add_argument("--keep-latest-pass", action="store_true", default=True, help="Always preserve latest PASS run.")
    parser.add_argument("--include-failed", action="store_true", help="Include FAILED runs in archive candidates.")
    parser.add_argument("--older-than-days", type=int, default=None, help="Only archive runs older than N days.")
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _iter_run_summaries() -> List[RunSummary]:
    rows: List[RunSummary] = []
    if not REPORTS_DIR.exists():
        return rows
    for summary_path in REPORTS_DIR.glob("*_vertical_slice_summary.json"):
        try:
            payload = _load_json_object(summary_path)
        except (json.JSONDecodeError, ValueError):
            continue
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        status = str(payload.get("acceptance_gate_status", "FAIL")).strip().upper()
        modified = datetime.fromtimestamp(summary_path.stat().st_mtime, tz=timezone.utc)
        rows.append(RunSummary(run_id=run_id, status=status, summary_path=summary_path, modified_at=modified))
    return sorted(rows, key=lambda row: row.modified_at, reverse=True)


def _latest_pass_from_status_module() -> Optional[str]:
    module_path = Path(__file__).with_name("emv_status.py")
    try:
        spec = importlib.util.spec_from_file_location("emv_status_for_archive", module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        status = module.collect_status()
    except Exception:
        return None
    run_id = status.get("latest_pass_run_id")
    if isinstance(run_id, str) and run_id.strip():
        return run_id
    return None


def _latest_pass_from_summaries(rows: Sequence[RunSummary]) -> Optional[str]:
    for row in rows:
        if row.status == "PASS":
            return row.run_id
    return None


def _should_skip_by_age(row: RunSummary, older_than_days: Optional[int]) -> bool:
    if older_than_days is None:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    return row.modified_at > cutoff


def _relative_to(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return Path(path.name)


def _run_id_matches_candidate(candidate: Path, run_id: str) -> bool:
    candidate_text = candidate.as_posix()
    return run_id in candidate_text or run_id in candidate.name


def _resolve_owner_run_id(candidate: Path, run_ids: Sequence[str]) -> Optional[str]:
    matches = [run_id for run_id in run_ids if _run_id_matches_candidate(candidate, run_id)]
    if not matches:
        return None
    return max(matches, key=len)


def _select_report_paths(run_id: str, run_ids: Sequence[str]) -> List[Path]:
    selected: List[Path] = []
    if not REPORTS_DIR.exists():
        return selected
    for candidate in REPORTS_DIR.iterdir():
        if not candidate.is_file():
            continue
        if candidate.name in PROTECTED_REPORTS:
            continue
        if candidate.parent == ARCHIVE_REPORTS_DIR:
            continue
        if _resolve_owner_run_id(candidate, run_ids) != run_id:
            continue
        selected.append(candidate)
    return sorted(selected)


def _select_evidence_paths(run_id: str, run_ids: Sequence[str]) -> List[Path]:
    selected: List[Path] = []
    candidate = EVIDENCE_DIR / run_id
    if _resolve_owner_run_id(candidate, run_ids) != run_id:
        return selected
    if candidate.exists() and candidate.name not in PROTECTED_EVIDENCE_DIRS:
        selected.append(candidate)
    return selected


def _is_protected_artifact(path: Path) -> bool:
    rel = _relative_to(path, ARTIFACTS_DIR)
    parts = set(rel.parts)
    return bool(parts & PROTECTED_ARTIFACT_PATH_PARTS)


def _select_artifact_paths(run_id: str, run_ids: Sequence[str]) -> List[Path]:
    selected: List[Path] = []
    if not ARTIFACTS_DIR.exists():
        return selected
    for candidate in ARTIFACTS_DIR.rglob("*"):
        if candidate.is_dir():
            continue
        if _resolve_owner_run_id(candidate, run_ids) != run_id:
            continue
        if _is_protected_artifact(candidate):
            continue
        selected.append(candidate)
    return sorted(set(selected))


def _move_to_archive(path: Path, destination_root: Path, source_root: Path) -> Path:
    relative = _relative_to(path, source_root)
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    return destination


def _build_manifest(
    *,
    execute: bool,
    latest_pass_run_id: Optional[str],
    kept_runs: Sequence[str],
    archived_runs: Sequence[str],
    plans: Sequence[Dict[str, Any]],
    moved: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execute": execute,
        "latest_pass_run_id": latest_pass_run_id,
        "kept_runs": list(kept_runs),
        "archived_runs": list(archived_runs),
        "planned_items": list(plans),
        "moved_items": list(moved),
    }


def archive_runs(
    *,
    execute: bool,
    keep_latest_pass: bool,
    include_failed: bool,
    older_than_days: Optional[int],
) -> Dict[str, Any]:
    rows = _iter_run_summaries()
    run_ids = [row.run_id for row in rows]
    latest_pass = _latest_pass_from_status_module() or _latest_pass_from_summaries(rows)

    kept_runs: List[str] = []
    archived_runs: List[str] = []
    plans: List[Dict[str, Any]] = []
    moved: List[Dict[str, Any]] = []

    for row in rows:
        if keep_latest_pass and latest_pass and row.run_id == latest_pass:
            kept_runs.append(row.run_id)
            continue
        if row.status != "PASS" and not include_failed:
            kept_runs.append(row.run_id)
            continue
        if _should_skip_by_age(row, older_than_days):
            kept_runs.append(row.run_id)
            continue

        run_report_paths = _select_report_paths(row.run_id, run_ids)
        run_evidence_paths = _select_evidence_paths(row.run_id, run_ids)
        run_artifact_paths = _select_artifact_paths(row.run_id, run_ids)
        if not (run_report_paths or run_evidence_paths or run_artifact_paths):
            kept_runs.append(row.run_id)
            continue

        archived_runs.append(row.run_id)
        for path in run_report_paths:
            plans.append({"run_id": row.run_id, "kind": "report", "path": str(path)})
        for path in run_evidence_paths:
            plans.append({"run_id": row.run_id, "kind": "evidence", "path": str(path)})
        for path in run_artifact_paths:
            plans.append({"run_id": row.run_id, "kind": "artifact", "path": str(path)})

        if execute:
            for path in run_report_paths:
                target = _move_to_archive(path, ARCHIVE_REPORTS_DIR, REPORTS_DIR)
                moved.append({"kind": "report", "source": str(path), "target": str(target)})
            for path in run_evidence_paths:
                target = _move_to_archive(path, ARCHIVE_EVIDENCE_DIR, EVIDENCE_DIR)
                moved.append({"kind": "evidence", "source": str(path), "target": str(target)})
            for path in run_artifact_paths:
                target = _move_to_archive(path, ARCHIVE_ARTIFACTS_DIR, ARTIFACTS_DIR)
                moved.append({"kind": "artifact", "source": str(path), "target": str(target)})

    manifest = _build_manifest(
        execute=execute,
        latest_pass_run_id=latest_pass,
        kept_runs=sorted(set(kept_runs)),
        archived_runs=sorted(set(archived_runs)),
        plans=plans,
        moved=moved,
    )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    execute = not bool(args.dry_run)
    manifest = archive_runs(
        execute=execute,
        keep_latest_pass=bool(args.keep_latest_pass),
        include_failed=bool(args.include_failed),
        older_than_days=args.older_than_days,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
