"""Contract checks for frozen benchmark/query metadata used by BM25 baseline."""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_SUBSETS = {
    "root_basic",
    "sofie_absence_control",
    "root_sofie_integration",
    "repo_specific",
    "critical_queries",
    "fairship_only_valid",
    "extended_corpus_valid",
}

LEGACY_SUBSETS = {"sofie"}

REQUIRED_QUERY_CATEGORIES = {
    "root_basic",
    "sofie_absence_control",
    "root_sofie_integration",
    "repo_specific",
}

EXPECTED_BEHAVIORS = {
    "retrieve_present",
    "confirm_absence",
    "cross_source_integration",
}

ANSWER_GRANULARITIES = {"file", "section", "code_symbol", "workflow"}
CRITICALITIES = {"high", "medium", "low"}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_qrels(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_benchmark_queries_have_required_contract_fields():
    queries = _load_json(Path("configs/benchmark_queries.json"))
    assert queries, "benchmark_queries.json must not be empty"

    ids = []
    for row in queries:
        assert "id" in row
        assert "query" in row
        assert "query_class" in row
        assert "category" in row
        assert "expected_behavior" in row
        assert "answer_granularity" in row
        assert "criticality" in row

        assert row["category"] in REQUIRED_QUERY_CATEGORIES
        assert row["expected_behavior"] in EXPECTED_BEHAVIORS
        assert row["answer_granularity"] in ANSWER_GRANULARITIES
        assert row["criticality"] in CRITICALITIES
        ids.append(row["id"])

    assert len(ids) == len(set(ids)), "query IDs must be unique"


def test_query_subsets_file_has_required_scenarios_and_categories():
    queries = _load_json(Path("configs/benchmark_queries.json"))
    query_ids = {row["id"] for row in queries}

    subsets = _load_json(Path("configs/benchmark_query_subsets.json"))
    assert REQUIRED_SUBSETS.issubset(set(subsets.keys()))
    assert LEGACY_SUBSETS.issubset(set(subsets.keys()))

    for subset_name, ids in subsets.items():
        assert isinstance(ids, list), f"subset {subset_name} must be a list"
        assert len(ids) == len(set(ids)), f"subset {subset_name} has duplicate query IDs"
        unknown = [qid for qid in ids if qid not in query_ids]
        assert not unknown, f"subset {subset_name} contains unknown IDs: {unknown}"

    fairship = set(subsets["fairship_only_valid"])
    extended = set(subsets["extended_corpus_valid"])
    assert fairship.issubset(extended), "fairship_only_valid must be subset of extended_corpus_valid"


def test_qrels_reference_existing_queries_base_and_extended():
    queries = _load_json(Path("configs/benchmark_queries.json"))
    query_ids = {row["id"] for row in queries}

    for qrels_path in [
        Path("configs/benchmark_qrels.jsonl"),
        Path("configs/benchmark_qrels_extended.jsonl"),
    ]:
        qrels = _load_qrels(qrels_path)
        assert qrels, f"{qrels_path} must not be empty"
        for row in qrels:
            assert row["query_id"] in query_ids, (
                f"qrel references unknown query_id {row['query_id']} in {qrels_path}"
            )
            assert "chunk_id" in row and row["chunk_id"]
            assert int(row["relevance"]) > 0


def test_sofie_absence_controls_have_no_fairship_only_qrels():
    queries = _load_json(Path("configs/benchmark_queries.json"))
    absence_ids = {
        row["id"] for row in queries if row["expected_behavior"] == "confirm_absence"
    }
    qrels = _load_qrels(Path("configs/benchmark_qrels.jsonl"))
    qrel_ids = {row["query_id"] for row in qrels}

    assert absence_ids
    assert absence_ids.isdisjoint(qrel_ids), (
        "Absence-control queries must not have positive qrels in FairShip-only profile"
    )


def test_root_sofie_integration_queries_have_extended_qrels():
    queries = _load_json(Path("configs/benchmark_queries.json"))
    integration_ids = {
        row["id"]
        for row in queries
        if row["expected_behavior"] == "cross_source_integration"
    }

    base_qrels = _load_qrels(Path("configs/benchmark_qrels.jsonl"))
    extended_qrels = _load_qrels(Path("configs/benchmark_qrels_extended.jsonl"))

    base_qrel_ids = {row["query_id"] for row in base_qrels}
    extended_qrel_ids = {row["query_id"] for row in extended_qrels}

    assert integration_ids
    assert integration_ids.isdisjoint(base_qrel_ids)
    assert integration_ids.issubset(extended_qrel_ids)


def test_corpus_profiles_reference_expected_paths():
    profiles = _load_json(Path("configs/benchmark_corpus_profiles.json"))
    assert set(profiles.keys()) == {"fairship_only_valid", "extended_corpus_valid"}

    fairship = profiles["fairship_only_valid"]
    extended = profiles["extended_corpus_valid"]

    assert fairship["qrels"] == "configs/benchmark_qrels.jsonl"
    assert extended["qrels"] == "configs/benchmark_qrels_extended.jsonl"
    assert extended["external_manifest"] == "configs/root_sofie_minimal_external_manifest.json"


def test_external_manifest_and_minimal_corpus_contract():
    manifest = _load_json(Path("configs/root_sofie_minimal_external_manifest.json"))
    corpus_path = Path(manifest["local_corpus"]["path"])
    assert corpus_path.exists(), f"Missing minimal external corpus {corpus_path}"

    required_ids = set(manifest["local_corpus"]["required_chunk_ids"])
    rows = _load_qrels(Path("configs/benchmark_qrels_extended.jsonl"))
    qrel_chunk_ids = {row["chunk_id"] for row in rows if row["query_id"] in {"b009", "b010"}}

    assert required_ids.issubset(qrel_chunk_ids), (
        "Extended qrels for b009/b010 must use required external chunk IDs"
    )
