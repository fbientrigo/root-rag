from __future__ import annotations

from pathlib import Path

from root_rag.evaluation.split_geometry_audit import load_bridge_light_geometry_audit


def test_bridge_light_split_geometry_audit_labels_and_distances():
    report = load_bridge_light_geometry_audit(
        corpus_path=Path("artifacts/corpus.jsonl"),
        queries_path=Path("configs/benchmark_queries_semantic.json"),
        qrels_path=Path("configs/benchmark_qrels_semantic.jsonl"),
    )

    assert report["class_summary"]["n_queries"] == 6
    assert report["class_summary"]["dominant_label"] == "adjacent_split"
    assert report["class_summary"]["dominant_label_count"] == 3

    rows = {row["query_id"]: row for row in report["per_query"]}
    assert rows["br003"]["label"] == "one_gold_self_sufficient"
    assert rows["br003"]["self_sufficient"] is True
    assert rows["br003"]["chunk_distance"] == 17
    assert rows["br003"]["minimum_contiguous_window_size"] == 18

    assert rows["br005"]["label"] == "local_nonadjacent_split"
    assert rows["br005"]["chunk_distance"] == 4
    assert rows["br005"]["minimum_contiguous_window_size"] == 5

    for qid in ("br004", "br006", "br007"):
        assert rows[qid]["label"] == "adjacent_split"
        assert rows[qid]["self_sufficient"] is False
        assert rows[qid]["chunk_distance"] == 1

    assert rows["br008"]["label"] == "far_same_file_split"
    assert rows["br008"]["self_sufficient"] is False
    assert rows["br008"]["chunk_distance"] == 7

