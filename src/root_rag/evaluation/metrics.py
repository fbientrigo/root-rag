"""Core retrieval quality metrics for benchmark evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


IDCG_TWO_DOCS_GRADED = ((2**2 - 1) / math.log2(2)) + ((2**1 - 1) / math.log2(3))


@dataclass(frozen=True)
class TopKMetrics:
    """Per-query retrieval metrics at a fixed cutoff k."""

    mrr_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    retrieved_positive_count: int
    qrels_positive_count: int


def compute_topk_metrics(
    ranked_chunk_ids: list[str],
    relevance_by_chunk: Mapping[str, int],
    *,
    top_k: int,
    qrels_positive_count: int = 2,
) -> TopKMetrics:
    """Compute MRR@k, Recall@k, and nDCG@k for one query."""
    ranked = ranked_chunk_ids[:top_k]
    gains = [relevance_by_chunk.get(chunk_id, 0) for chunk_id in ranked]
    positive_ranks = [idx for idx, gain in enumerate(gains, start=1) if gain > 0]

    mrr = 1.0 / positive_ranks[0] if positive_ranks else 0.0
    retrieved_positive_count = sum(1 for gain in gains if gain > 0)
    recall = retrieved_positive_count / float(qrels_positive_count) if qrels_positive_count else 0.0
    dcg = sum(
        ((2**gain) - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
        if gain > 0
    )
    ndcg = dcg / IDCG_TWO_DOCS_GRADED if IDCG_TWO_DOCS_GRADED else 0.0

    return TopKMetrics(
        mrr_at_k=mrr,
        recall_at_k=recall,
        ndcg_at_k=ndcg,
        retrieved_positive_count=retrieved_positive_count,
        qrels_positive_count=qrels_positive_count,
    )


def aggregate_topk_metrics(rows: Iterable[TopKMetrics]) -> dict[str, float]:
    """Compute macro-average MRR/Recall/nDCG from per-query metrics."""
    rows = list(rows)
    if not rows:
        return {"mrr_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0}
    n = len(rows)
    return {
        "mrr_at_k": sum(row.mrr_at_k for row in rows) / n,
        "recall_at_k": sum(row.recall_at_k for row in rows) / n,
        "ndcg_at_k": sum(row.ndcg_at_k for row in rows) / n,
    }


def classify_effect(before: Mapping[str, float], after: Mapping[str, float], *, eps: float = 1e-12) -> str:
    """Classify before/after change as helped, hurt, or unchanged."""
    improved = (
        after["mrr_at_k"] > before["mrr_at_k"] + eps
        or after["recall_at_k"] > before["recall_at_k"] + eps
        or after["ndcg_at_k"] > before["ndcg_at_k"] + eps
    )
    worsened = (
        after["mrr_at_k"] + eps < before["mrr_at_k"]
        or after["recall_at_k"] + eps < before["recall_at_k"]
        or after["ndcg_at_k"] + eps < before["ndcg_at_k"]
    )
    if improved and not worsened:
        return "helped"
    if worsened and not improved:
        return "hurt"
    return "unchanged"
