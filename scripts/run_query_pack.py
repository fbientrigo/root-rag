"""Execute workflow query packs through root-rag CLI and persist evidence artifacts."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class QuerySpec:
    """Normalized query definition loaded from query pack YAML."""

    query_id: str
    bm25_tokens: List[str]


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration for query-pack execution."""

    pack_path: Path
    output_dir: Path
    top_k: int
    index_dir: Path | None
    index_id: str | None
    root_ref: str | None
    evidence_format: str
    dry_run: bool
    fail_fast: bool
    root_rag_cmd: str | None


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    """Parse CLI arguments into a strongly-typed runtime config."""
    parser = argparse.ArgumentParser(description="Run root-rag query pack and save evidence JSON outputs.")
    parser.add_argument("--pack", required=True, type=Path, help="Path to query pack YAML file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Evidence output directory for this run.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K results per query (default: 10).")
    parser.add_argument("--index-dir", type=Path, default=None, help="Directory containing indexes for root-rag ask.")
    parser.add_argument("--index-id", default=None, help="Explicit index ID for root-rag ask.")
    parser.add_argument("--root-ref", default=None, help="Root reference for root-rag ask when no index ID is provided.")
    parser.add_argument(
        "--evidence-format",
        choices=["text-wrapper"],
        default="text-wrapper",
        help="Evidence artifact format to write (default: text-wrapper).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write manifest/placeholder outputs without calling CLI.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed query execution.")
    parser.add_argument(
        "--root-rag-cmd",
        default=None,
        help="Optional executable command for root-rag CLI (default: auto-resolve).",
    )

    parsed = parser.parse_args(argv)
    return RunConfig(
        pack_path=parsed.pack,
        output_dir=parsed.output_dir,
        top_k=parsed.top_k,
        index_dir=parsed.index_dir,
        index_id=parsed.index_id,
        root_ref=parsed.root_ref,
        evidence_format=parsed.evidence_format,
        dry_run=parsed.dry_run,
        fail_fast=parsed.fail_fast,
        root_rag_cmd=parsed.root_rag_cmd,
    )


def load_query_pack(pack_path: Path) -> Dict[str, Any]:
    """Load and minimally validate a query pack YAML document."""
    with pack_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Query pack must be a mapping at top level.")

    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Query pack missing list field: queries")

    for idx, query in enumerate(queries):
        if not isinstance(query, dict):
            raise ValueError(f"Query entry {idx} must be a mapping.")
        if "id" not in query:
            raise ValueError(f"Query entry {idx} missing field: id")
        tokens = query.get("bm25_tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError(f"Query entry {idx} field bm25_tokens must be list[str].")

    return payload


def extract_queries(pack_payload: Mapping[str, Any]) -> List[QuerySpec]:
    """Extract normalized query specs from validated query-pack payload."""
    query_specs: List[QuerySpec] = []
    for query in pack_payload["queries"]:
        query_specs.append(
            QuerySpec(
                query_id=str(query["id"]),
                bm25_tokens=[str(token) for token in query["bm25_tokens"]],
            )
        )
    return query_specs


def _resolve_root_rag_command(explicit_cmd: str | None) -> Tuple[List[str], str, str | None]:
    """Resolve root-rag command with explicit > PATH > python module fallback priority."""
    if explicit_cmd:
        return [explicit_cmd], "path_executable", None
    if shutil.which("root-rag"):
        return ["root-rag"], "path_executable", None
    return [sys.executable, "-m", "root_rag.cli"], "python_module_fallback", sys.executable


def _build_command(
    *,
    root_rag_prefix: Sequence[str],
    query_text: str,
    top_k: int,
    index_dir: Path | None,
    index_id: str | None,
    root_ref: str | None,
) -> List[str]:
    """Build subprocess argument vector for root-rag query execution."""
    command = [*root_rag_prefix, "ask", query_text, "--top-k", str(top_k)]
    if index_dir is not None:
        command.extend(["--index-dir", str(index_dir)])
    if index_id:
        command.extend(["--index-id", index_id])
    if root_ref:
        command.extend(["--root-ref", root_ref])
    return command


def _iso_utc_timestamp() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _execute_command(command: Sequence[str], dry_run: bool) -> Tuple[int, str, str]:
    """Execute CLI command or synthesize dry-run result."""
    if dry_run:
        dry_payload = {
            "dry_run": True,
            "command": list(command),
        }
        return 0, json.dumps(dry_payload, indent=2), ""

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.returncode, completed.stdout, completed.stderr


def _status_from_return_code(return_code: int) -> str:
    """Classify root-rag ask return codes for manifest consumers."""
    if return_code == 0:
        return "HIT_OR_TEXT_EVIDENCE"
    if return_code == 5:
        return "ZERO_HIT"
    return "ERROR"


_TEXT_HIT_PATTERN = re.compile(r"^\[(?P<rank>\d+)\]\s+(?P<file>.+?):(?P<start>\d+)-(?P<end>\d+)\s*$")


def _parse_text_wrapper_hits(stdout: str) -> List[Dict[str, Any]]:
    """Parse root-rag human-readable stdout lines into structured hit entries."""
    hits: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        match = _TEXT_HIT_PATTERN.match(line.strip())
        if not match:
            continue
        hits.append(
            {
                "rank": int(match.group("rank")),
                "file": match.group("file"),
                "start_line": int(match.group("start")),
                "end_line": int(match.group("end")),
            }
        )
    return hits


def _build_text_wrapper(
    *,
    query_id: str,
    query_text: str,
    command: Sequence[str],
    evidence_format: str,
    return_code: int,
    stdout: str,
    stderr: str,
) -> Dict[str, Any]:
    """Wrap text-only CLI output in deterministic JSON for downstream tools."""
    return {
        "query_id": query_id,
        "query": query_text,
        "command": list(command),
        "evidence_format": evidence_format,
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "hits": _parse_text_wrapper_hits(stdout),
        "notes": "Raw CLI text output; not structured JSON.",
    }


def run_query_pack(config: RunConfig) -> Tuple[Dict[str, Any], int]:
    """Run all queries in pack, write evidence files, and return manifest + exit code."""
    pack_payload = load_query_pack(config.pack_path)
    query_specs = extract_queries(pack_payload)
    timestamp = _iso_utc_timestamp()

    existing_manifest_path = config.output_dir / "manifest.json"
    output_dir_reused = existing_manifest_path.exists()
    reuse_warning = ""
    if output_dir_reused:
        reuse_warning = (
            "WARNING: output-dir already contains manifest.json; "
            "artifacts may be overwritten. Prefer a fresh evidence directory."
        )
        print(reuse_warning)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "pack_path": str(config.pack_path),
        "pack_id": pack_payload.get("pack_id"),
        "timestamp": timestamp,
        "top_k": config.top_k,
        "index_dir": str(config.index_dir) if config.index_dir is not None else None,
        "index_id": config.index_id,
        "root_ref": config.root_ref,
        "evidence_format": config.evidence_format,
        "dry_run": config.dry_run,
        "fail_fast": config.fail_fast,
        "output_dir_reused": output_dir_reused,
        "warnings": [reuse_warning] if reuse_warning else [],
        "queries": [],
    }
    root_rag_prefix, resolution_mode, fallback_python_executable = _resolve_root_rag_command(config.root_rag_cmd)
    manifest["command_resolution_mode"] = resolution_mode
    manifest["root_rag_command_prefix"] = list(root_rag_prefix)
    manifest["root_rag_cmd"] = config.root_rag_cmd
    manifest["python_executable"] = fallback_python_executable

    had_failure = False
    for query_spec in query_specs:
        query_text = " ".join(query_spec.bm25_tokens)
        command = _build_command(
            root_rag_prefix=root_rag_prefix,
            query_text=query_text,
            top_k=config.top_k,
            index_dir=config.index_dir,
            index_id=config.index_id,
            root_ref=config.root_ref,
        )
        return_code, stdout, stderr = _execute_command(command, config.dry_run)
        status = _status_from_return_code(return_code)

        output_file = config.output_dir / f"{query_spec.query_id}.json"
        wrapper = _build_text_wrapper(
            query_id=query_spec.query_id,
            query_text=query_text,
            command=command,
            evidence_format=config.evidence_format,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
        )
        _write_text(output_file, json.dumps(wrapper, indent=2))

        query_entry: Dict[str, Any] = {
            "id": query_spec.query_id,
            "command": command,
            "query": query_text,
            "output_file": str(output_file),
            "evidence_format": config.evidence_format,
            "return_code": return_code,
            "status": status,
            "stderr": stderr,
        }
        manifest["queries"].append(query_entry)

        if status == "ERROR":
            had_failure = True
            if config.fail_fast:
                break

    manifest_path = config.output_dir / "manifest.json"
    _write_text(manifest_path, json.dumps(manifest, indent=2))

    return manifest, 1 if had_failure else 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    config = parse_args(argv)
    _, exit_code = run_query_pack(config)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
