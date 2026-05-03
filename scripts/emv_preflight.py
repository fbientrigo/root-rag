"""Run non-mutating EMV preflight checks for local Muon DIS harness workflows."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _check_pyyaml() -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc), "version": None}
    return {"ok": True, "error": None, "version": getattr(yaml, "__version__", "unknown")}


def _run_help_command(command: List[str]) -> Dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return {
            "ok": False,
            "return_code": None,
            "stdout": "",
            "stderr": f"executable not found: {command[0]}",
        }
    return {
        "ok": completed.returncode == 0,
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _list_latest_indexes(index_dir: Path, limit: int = 5) -> Dict[str, Any]:
    if not index_dir.exists():
        return {
            "exists": False,
            "latest_index_id": None,
            "indexes": [],
        }

    dirs = sorted(
        [path for path in index_dir.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return {
        "exists": True,
        "latest_index_id": dirs[0].name if dirs else None,
        "indexes": [
            {"name": path.name, "path": str(path), "mtime": path.stat().st_mtime}
            for path in dirs[:limit]
        ],
    }


def collect_preflight(index_dir: Path) -> Dict[str, Any]:
    pyyaml = _check_pyyaml()
    ask_help = _run_help_command(["root-rag", "ask", "--help"])
    search_help = _run_help_command(["root-rag", "search", "--help"])
    index_info = _list_latest_indexes(index_dir)

    return {
        "python_executable": sys.executable,
        "checks": {
            "pyyaml": pyyaml,
            "root_rag_ask_help": ask_help,
            "root_rag_search_help": search_help,
            "fairship_indexes": index_info,
        },
        "detected_latest_index_id": index_info["latest_index_id"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EMV preflight checks without modifying files.")
    parser.add_argument(
        "--index-dir",
        default=Path("data/indexes_fairship"),
        type=Path,
        help="FairShip index root directory (default: data/indexes_fairship).",
    )
    return parser.parse_args(argv)


def _should_fail(summary: Dict[str, Any]) -> bool:
    pyyaml_ok = bool(summary["checks"]["pyyaml"]["ok"])
    ask_help = summary["checks"]["root_rag_ask_help"]
    search_help = summary["checks"]["root_rag_search_help"]
    ask_missing = ask_help["return_code"] is None
    search_missing = search_help["return_code"] is None
    return (not pyyaml_ok) or ask_missing or search_missing


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = collect_preflight(args.index_dir)
    print(json.dumps(summary, indent=2))

    if _should_fail(summary):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
