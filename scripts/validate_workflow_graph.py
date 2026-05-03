"""Validate lightweight workflow graph JSON artifacts."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


ALLOWED_KINDS = {"script", "function", "class", "config", "data", "stage", "unknown"}
ALLOWED_STATUS = {"CONFIRMED", "PROVISIONAL", "UNRESOLVED"}
ALLOWED_RELATIONS = {"calls", "reads", "writes", "configures", "produces", "consumes", "precedes", "unknown"}
SOURCE_FORMAT_RE = re.compile(r"^[A-Za-z0-9_.\-/]+:[0-9]+-[0-9]+$")


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _validate_sources(
    *,
    container: Dict[str, Any],
    path_prefix: str,
    errors: List[str],
    require_non_empty: bool,
) -> List[str]:
    sources = container.get("sources")
    if not isinstance(sources, list):
        errors.append(f"{path_prefix}.sources must be a list")
        return []

    normalized: List[str] = []
    for idx, source in enumerate(sources):
        if not isinstance(source, str):
            errors.append(f"{path_prefix}.sources[{idx}] must be a string")
            continue
        if not SOURCE_FORMAT_RE.match(source):
            errors.append(
                f"{path_prefix}.sources[{idx}] invalid format '{source}', expected path/to/file.ext:start-end"
            )
            continue
        normalized.append(source)

    if require_non_empty and len(normalized) == 0:
        errors.append(f"{path_prefix} requires at least one source for CONFIRMED status")
    return normalized


def validate_graph_payload(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    for field in ("graph_id", "created", "scope", "nodes", "edges"):
        if field not in payload:
            errors.append(f"missing required field: {field}")

    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list):
        errors.append("nodes must be a list")
        nodes = []
    if not isinstance(edges, list):
        errors.append("edges must be a list")
        edges = []

    node_ids: set[str] = set()
    for idx, node in enumerate(nodes):
        path_prefix = f"nodes[{idx}]"
        if not isinstance(node, dict):
            errors.append(f"{path_prefix} must be an object")
            continue

        node_id = node.get("id")
        if not _is_non_empty_string(node_id):
            errors.append(f"{path_prefix}.id must be a non-empty string")
        else:
            if node_id in node_ids:
                errors.append(f"{path_prefix}.id duplicates existing node id '{node_id}'")
            node_ids.add(node_id)

        if not _is_non_empty_string(node.get("label")):
            errors.append(f"{path_prefix}.label must be a non-empty string")

        kind = node.get("kind")
        if kind not in ALLOWED_KINDS:
            errors.append(f"{path_prefix}.kind invalid enum '{kind}'")

        status = node.get("status")
        if status not in ALLOWED_STATUS:
            errors.append(f"{path_prefix}.status invalid enum '{status}'")

        _validate_sources(
            container=node,
            path_prefix=path_prefix,
            errors=errors,
            require_non_empty=(status == "CONFIRMED"),
        )

    for idx, edge in enumerate(edges):
        path_prefix = f"edges[{idx}]"
        if not isinstance(edge, dict):
            errors.append(f"{path_prefix} must be an object")
            continue

        source_id = edge.get("source")
        target_id = edge.get("target")
        if not _is_non_empty_string(source_id):
            errors.append(f"{path_prefix}.source must be a non-empty string")
        if not _is_non_empty_string(target_id):
            errors.append(f"{path_prefix}.target must be a non-empty string")

        if isinstance(source_id, str) and source_id not in node_ids:
            errors.append(f"{path_prefix}.source references missing node '{source_id}'")
        if isinstance(target_id, str) and target_id not in node_ids:
            errors.append(f"{path_prefix}.target references missing node '{target_id}'")

        relation = edge.get("relation")
        if relation not in ALLOWED_RELATIONS:
            errors.append(f"{path_prefix}.relation invalid enum '{relation}'")

        status = edge.get("status")
        if status not in ALLOWED_STATUS:
            errors.append(f"{path_prefix}.status invalid enum '{status}'")

        normalized_sources = _validate_sources(
            container=edge,
            path_prefix=path_prefix,
            errors=errors,
            require_non_empty=(status == "CONFIRMED"),
        )

        notes = edge.get("notes", "")
        if not isinstance(notes, str):
            errors.append(f"{path_prefix}.notes must be a string")
            notes = ""

        if status == "CONFIRMED" and len(normalized_sources) == 1 and notes.strip() == "":
            errors.append(f"{path_prefix} with single source requires explanatory notes")

    return errors


def validate_workflow_graph(graph_path: Path) -> List[str]:
    if not graph_path.exists():
        return [f"{graph_path}: file does not exist"]
    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{graph_path}: invalid json: {exc}"]
    if not isinstance(payload, dict):
        return [f"{graph_path}: root JSON must be an object"]
    return validate_graph_payload(payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate workflow graph JSON.")
    parser.add_argument("graph_path", type=Path, help="Path to workflow graph JSON file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    errors = validate_workflow_graph(args.graph_path)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"OK: workflow graph valid: {args.graph_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
