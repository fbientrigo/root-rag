"""Tests for scripts/validate_workflow_graph.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_workflow_graph.py"
    spec = importlib.util.spec_from_file_location("validate_workflow_graph", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _base_graph() -> dict:
    return {
        "graph_id": "g1",
        "created": "2026-04-28",
        "scope": "test",
        "nodes": [
            {"id": "a", "label": "A", "kind": "stage", "status": "PROVISIONAL", "sources": []},
            {"id": "b", "label": "B", "kind": "stage", "status": "PROVISIONAL", "sources": []},
        ],
        "edges": [
            {
                "source": "a",
                "target": "b",
                "relation": "precedes",
                "status": "PROVISIONAL",
                "sources": [],
                "notes": "",
            }
        ],
    }


def test_valid_graph(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(_base_graph(), indent=2), encoding="utf-8")
    assert module.validate_workflow_graph(path) == []


def test_missing_node_referenced_by_edge(tmp_path: Path) -> None:
    module = _load_module()
    payload = _base_graph()
    payload["edges"][0]["target"] = "missing"
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    errors = module.validate_workflow_graph(path)
    assert any("references missing node 'missing'" in error for error in errors)


def test_confirmed_node_without_source(tmp_path: Path) -> None:
    module = _load_module()
    payload = _base_graph()
    payload["nodes"][0]["status"] = "CONFIRMED"
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    errors = module.validate_workflow_graph(path)
    assert any("nodes[0] requires at least one source for CONFIRMED status" in error for error in errors)


def test_confirmed_edge_without_source(tmp_path: Path) -> None:
    module = _load_module()
    payload = _base_graph()
    payload["edges"][0]["status"] = "CONFIRMED"
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    errors = module.validate_workflow_graph(path)
    assert any("edges[0] requires at least one source for CONFIRMED status" in error for error in errors)


def test_invalid_enum(tmp_path: Path) -> None:
    module = _load_module()
    payload = _base_graph()
    payload["nodes"][0]["kind"] = "pipeline"
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    errors = module.validate_workflow_graph(path)
    assert any("nodes[0].kind invalid enum 'pipeline'" in error for error in errors)
