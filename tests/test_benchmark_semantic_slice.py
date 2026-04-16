"""Tiny validation checks for semantic benchmark slice."""

from __future__ import annotations

import json
from pathlib import Path


def _load_queries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_qrels(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_semantic_benchmark_slice_files_parse_and_balance():
    queries_path = Path("configs/benchmark_queries_semantic.json")
    qrels_path = Path("configs/benchmark_qrels_semantic.jsonl")

    assert queries_path.exists()
    assert qrels_path.exists()

    queries = _load_queries(queries_path)
    qrels = _load_qrels(qrels_path)

    assert len(queries) == 30
    assert {row["query_class"] for row in queries} == {
        "lexical-control",
        "semantic",
        "bridge-light",
    }
    assert sum(1 for row in queries if row["query_class"] == "lexical-control") == 10
    assert sum(1 for row in queries if row["query_class"] == "semantic") == 12
    assert sum(1 for row in queries if row["query_class"] == "bridge-light") == 8

    query_ids = {row["id"] for row in queries}
    assert len(query_ids) == 30

    qrels_by_query: dict[str, list[dict]] = {}
    for row in qrels:
        assert row["query_id"] in query_ids
        assert row["relevance"] in {1, 2}
        qrels_by_query.setdefault(row["query_id"], []).append(row)

    assert len(qrels) == 60
    assert all(len(rows) == 2 for rows in qrels_by_query.values())
    assert set(qrels_by_query) == query_ids
