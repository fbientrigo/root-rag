"""Tests for retrieval evaluation metrics."""

from root_rag.evaluation.metrics import (
    TopKMetrics,
    aggregate_topk_metrics,
    classify_effect,
    compute_topk_metrics,
)


def test_compute_topk_metrics_returns_expected_values():
    ranked = ["a", "b", "c", "d"]
    qrels = {"b": 2, "d": 1}

    metrics = compute_topk_metrics(ranked, qrels, top_k=4, qrels_positive_count=2)

    assert metrics.mrr_at_k == 0.5
    assert metrics.recall_at_k == 1.0
    assert metrics.ndcg_at_k > 0.0
    assert metrics.retrieved_positive_count == 2
    assert metrics.qrels_positive_count == 2


def test_aggregate_topk_metrics_macro_averages_rows():
    rows = [
        TopKMetrics(1.0, 1.0, 1.0, 2, 2),
        TopKMetrics(0.5, 0.5, 0.5, 1, 2),
    ]
    summary = aggregate_topk_metrics(rows)
    assert summary["mrr_at_k"] == 0.75
    assert summary["recall_at_k"] == 0.75
    assert summary["ndcg_at_k"] == 0.75


def test_classify_effect_flags_helped_hurt_and_unchanged():
    before = {"mrr_at_k": 0.2, "recall_at_k": 0.3, "ndcg_at_k": 0.4}

    assert classify_effect(before, {"mrr_at_k": 0.3, "recall_at_k": 0.3, "ndcg_at_k": 0.4}) == "helped"
    assert classify_effect(before, {"mrr_at_k": 0.1, "recall_at_k": 0.3, "ndcg_at_k": 0.4}) == "hurt"
    assert classify_effect(before, {"mrr_at_k": 0.2, "recall_at_k": 0.3, "ndcg_at_k": 0.4}) == "unchanged"
