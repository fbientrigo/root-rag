from root_rag.evaluation.competition_diagnostics import (
    analyze_split_gold_same_file,
    assign_diagnosis_label,
)


def test_analyze_split_gold_same_file_detects_repeated_file():
    corpus_by_id = {
        "a": {"file_path": "ship/A.cxx"},
        "b": {"file_path": "ship/A.cxx"},
        "c": {"file_path": "ship/B.cxx"},
    }
    out = analyze_split_gold_same_file(gold_chunk_ids=["a", "b", "c"], corpus_by_id=corpus_by_id)
    assert out["is_split_gold_same_file"] is True
    assert out["repeated_gold_file_counts"] == {"ship/A.cxx": 2}


def test_assign_diagnosis_label_prefers_split_gold_same_file():
    modes = {
        "bm25": {
            "present": True,
            "all_gold_found_in_depth": False,
            "gold_rank_positions": {"a": 8, "b": None},
            "best_gold_rank": 8,
            "competitors_above_gold": [{"delta_vs_best_gold_score": 0.3, "same_file_as_any_gold": False}],
        },
        "semantic": {
            "present": False,
        },
        "hybrid": {
            "present": False,
        },
    }
    assert assign_diagnosis_label(per_mode=modes, split_gold_same_file=True) == "split_gold_same_file"


def test_assign_diagnosis_label_high_generic_competition():
    modes = {
        "bm25": {
            "present": True,
            "all_gold_found_in_depth": True,
            "gold_rank_positions": {"a": 5, "b": 8},
            "best_gold_rank": 5,
            "competitors_above_gold": [{"delta_vs_best_gold_score": 0.2, "same_file_as_any_gold": False}],
        },
        "semantic": {
            "present": True,
            "all_gold_found_in_depth": True,
            "gold_rank_positions": {"a": 6, "b": 9},
            "best_gold_rank": 6,
            "competitors_above_gold": [{"delta_vs_best_gold_score": 0.1, "same_file_as_any_gold": False}],
        },
        "hybrid": {
            "present": False,
        },
    }
    assert assign_diagnosis_label(per_mode=modes, split_gold_same_file=False) == "high_generic_competition"
