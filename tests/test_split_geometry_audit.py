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
    assert report["class_summary"]["dominant_label"] == "far_same_file_split"
    assert report["class_summary"]["dominant_label_count"] == 4

    rows = {row["query_id"]: row for row in report["per_query"]}
    assert rows["br003"]["label"] == "one_gold_self_sufficient"
    assert rows["br003"]["self_sufficient"] is True
    assert rows["br003"]["chunk_distance"] == 62
    assert rows["br003"]["minimum_contiguous_window_size"] == 63

    assert rows["br007"]["label"] == "local_nonadjacent_split"
    assert rows["br007"]["chunk_distance"] == 4
    assert rows["br007"]["minimum_contiguous_window_size"] == 5

    for qid in ("br004", "br005", "br006", "br008"):
        assert rows[qid]["label"] == "far_same_file_split"
        assert rows[qid]["self_sufficient"] is False

