from __future__ import annotations

from root_rag.evaluation.focal_chunk_experiment import (
    build_shadow_corpus_rows,
    compute_gold_coverage,
    summarize_class_rates,
    summarize_query,
)


def test_build_shadow_corpus_rows_merges_adjacent_same_file_pairs():
    corpus_rows = [
        {
            "chunk_id": "file_a_001",
            "file_path": "foo/bar.cxx",
            "line_range": [1, 3],
            "text": "L1\nL2\nL3",
            "headers_used": ["A.h"],
            "usage_count": 1,
        },
        {
            "chunk_id": "file_a_002",
            "file_path": "foo/bar.cxx",
            "line_range": [3, 5],
            "text": "L3\nL4\nL5",
            "headers_used": ["B.h"],
            "usage_count": 2,
        },
        {
            "chunk_id": "file_b_001",
            "file_path": "foo/baz.cxx",
            "line_range": [1, 2],
            "text": "Z1\nZ2",
            "headers_used": ["C.h"],
            "usage_count": 3,
        },
    ]

    shadow_rows, membership = build_shadow_corpus_rows(corpus_rows)

    assert len(shadow_rows) == 2
    pair_row = next(row for row in shadow_rows if row["file_path"] == "foo/bar.cxx")
    assert pair_row["source_chunk_ids"] == ["file_a_001", "file_a_002"]
    assert pair_row["line_range"] == [1, 5]
    assert pair_row["text"] == "L1\nL2\nL3\nL4\nL5"
    assert membership["file_a_001"] == [pair_row["chunk_id"]]
    assert membership["file_a_002"] == [pair_row["chunk_id"]]


def test_summarize_query_counts_original_gold_coverage_from_shadow_results():
    corpus_by_id = {
        "gold_a": {"file_path": "foo/bar.cxx"},
        "gold_b": {"file_path": "foo/bar.cxx"},
        "gold_c": {"file_path": "foo/bar.cxx"},
    }
    shadow_membership = {
        "gold_a": ["shadow::gold_a::gold_b"],
        "gold_b": ["shadow::gold_a::gold_b"],
        "gold_c": ["shadow::gold_c"],
    }
    query_row = {"id": "br001", "query": "q", "query_class": "bridge-light"}
    runs = {
        "bm25_only": {"br001": ["shadow::gold_c", "shadow::gold_a::gold_b"]},
        "semantic_only": {"br001": ["shadow::gold_a::gold_b"]},
        "hybrid": {"br001": ["shadow::gold_c"]},
    }

    summary = summarize_query(
        query_row=query_row,
        gold_chunk_ids=["gold_a", "gold_b", "gold_c"],
        corpus_by_id=corpus_by_id,
        shadow_membership=shadow_membership,
        runs=runs,
    )

    assert summary["same_file_split"] is True
    assert summary["aggregate"]["best_mode_gold_count"] == 3
    assert summary["aggregate"]["best_mode_best_rank"] == 1
    assert summary["aggregate"]["late_rank"] is False

    class_summary = summarize_class_rates([summary])
    assert class_summary["both_golds_found_rate"] == 1.0
    assert class_summary["one_gold_found_rate"] == 0.0
    assert class_summary["zero_gold_found_rate"] == 0.0
    assert class_summary["same_file_split_rate"] == 1.0
    assert class_summary["late_rank_rate"] == 0.0


def test_compute_gold_coverage_handles_missing_gold_hits():
    membership = {
        "gold_a": ["shadow::gold_a"],
        "gold_b": ["shadow::gold_b"],
    }
    coverage, count, best_rank = compute_gold_coverage(
        ranked_shadow_chunk_ids=["shadow::gold_b", "shadow::other"],
        gold_chunk_ids=["gold_a", "gold_b"],
        shadow_membership=membership,
    )

    assert count == 1
    assert best_rank == 1
    assert coverage["gold_a"].present is False
    assert coverage["gold_b"].present is True
    assert coverage["gold_b"].best_rank == 1
