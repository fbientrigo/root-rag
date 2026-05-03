"""Non-mutating preflight checks for Codex EMV sessions."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Sequence


def _check_pyyaml() -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - exact import failure depends on environment
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "version": getattr(yaml, "__version__", "unknown")}


def _check_root_rag_help() -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            ["root-rag", "ask", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {"ok": False, "return_code": None, "error": "root-rag executable not found"}
    return {
        "ok": completed.returncode == 0,
        "return_code": completed.returncode,
        "stderr": completed.stderr.strip(),
    }


def _list_index_dirs(index_root: Path) -> Dict[str, Any]:
    if not index_root.exists():
        return {"exists": False, "latest_index_id": None, "dirs": []}

    dirs = sorted(
        [path for path in index_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return {
        "exists": True,
        "latest_index_id": dirs[0].name if dirs else None,
        "dirs": [{"name": path.name, "path": str(path)} for path in dirs[:3]],
    }


def collect_preflight(index_root: Path = Path("data/indexes_fairship")) -> Dict[str, Any]:
    """Collect preflight facts without modifying files or requiring network."""
    return {
        "python_executable": sys.executable,
        "pyyaml": _check_pyyaml(),
        "root_rag_ask_help": _check_root_rag_help(),
        "fairship_indexes": _list_index_dirs(index_root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print JSON preflight output and return success unless Python itself failed."""
    args = list(argv or [])
    index_root = Path(args[0]) if args else Path("data/indexes_fairship")
    print(json.dumps(collect_preflight(index_root), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
