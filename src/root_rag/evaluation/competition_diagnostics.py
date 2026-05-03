"""Diagnostics helpers for query-level competition pressure analysis."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Mapping, Sequence

from root_rag.retrieval.models import EvidenceCandidate

DIAGNOSIS_LABELS = {
    "weak_gold_signal",
    "high_generic_competition",
    "split_gold_same_file",
    "unknown",
}


def build_rank_map(results: Sequence[EvidenceCandidate]) -> Dict[str, int]:
    """Map chunk_id -> 1-indexed rank from scored results."""
    return {row.chunk_id: idx + 1 for idx, row in enumerate(results)}


def build_score_map(results: Sequence[EvidenceCandidate]) -> Dict[str, float]:
    """Map chunk_id -> score from scored results."""
    return {row.chunk_id: float(row.score) for row in results}


def analyze_split_gold_same_file(
    *,
    gold_chunk_ids: Sequence[str],
    corpus_by_id: Mapping[str, dict],
) -> dict:
    """Return whether multiple gold chunks belong to the same file."""
    file_by_gold: Dict[str, str | None] = {}
    for chunk_id in gold_chunk_ids:
        row = corpus_by_id.get(chunk_id)
        file_by_gold[chunk_id] = row.get("file_path") if row else None

    file_counts = Counter(path for path in file_by_gold.values() if path)
    repeated = {path: count for path, count in sorted(file_counts.items()) if count > 1}
    return {
        "is_split_gold_same_file": bool(repeated),
        "gold_files": file_by_gold,
        "repeated_gold_file_counts": repeated,
    }


def extract_competitors_above_gold(
    *,
    mode_results: Sequence[EvidenceCandidate],
    gold_chunk_ids: Sequence[str],
    corpus_by_id: Mapping[str, dict],
    max_competitors: int = 5,
) -> dict:
    """Extract top competing chunks above best-ranked gold chunk."""
    gold_set = set(gold_chunk_ids)
    rank_map = build_rank_map(mode_results)
    score_map = build_score_map(mode_results)
    gold_ranks = {chunk_id: rank_map.get(chunk_id) for chunk_id in gold_chunk_ids}
    gold_scores = {chunk_id: score_map.get(chunk_id) for chunk_id in gold_chunk_ids}
    gold_presence: Dict[str, dict] = {}

    for chunk_id in gold_chunk_ids:
        rank = gold_ranks.get(chunk_id)
        score = gold_scores.get(chunk_id)
        if rank is None:
            gold_presence[chunk_id] = {
                "present": False,
                "rank": None,
                "score": None,
                "prev_competitor_chunk_id": None,
                "prev_competitor_score": None,
                "gap_to_prev_competitor": None,
            }
            continue

        prev_chunk_id = None
        prev_score = None
        gap = None
        if rank > 1 and rank - 2 < len(mode_results):
            prev = mode_results[rank - 2]
            prev_chunk_id = prev.chunk_id
            prev_score = float(prev.score)
            if score is not None:
                gap = float(score) - float(prev.score)

        gold_presence[chunk_id] = {
            "present": True,
            "rank": rank,
            "score": score,
            "prev_competitor_chunk_id": prev_chunk_id,
            "prev_competitor_score": prev_score,
            "gap_to_prev_competitor": gap,
        }

    ranked_gold = [rank for rank in gold_ranks.values() if rank is not None]
    best_gold_rank = min(ranked_gold) if ranked_gold else None
    best_gold_score = None
    if best_gold_rank is not None:
        for chunk_id, rank in gold_ranks.items():
            if rank == best_gold_rank:
                best_gold_score = gold_scores.get(chunk_id)
                break

    gold_files = {
        corpus_by_id[chunk_id]["file_path"]
        for chunk_id in gold_chunk_ids
        if chunk_id in corpus_by_id and corpus_by_id[chunk_id].get("file_path")
    }

    if best_gold_rank is None:
        pool = mode_results
    else:
        pool = mode_results[: max(0, best_gold_rank - 1)]

    competitors: List[dict] = []
    for idx, row in enumerate(pool, start=1):
        if row.chunk_id in gold_set:
            continue
        same_file = row.file_path in gold_files
        delta_vs_best = None
        if best_gold_score is not None:
            delta_vs_best = float(row.score) - float(best_gold_score)
        competitors.append(
            {
                "chunk_id": row.chunk_id,
                "rank": idx,
                "score": float(row.score),
                "file_path": row.file_path,
                "same_file_as_any_gold": same_file,
                "delta_vs_best_gold_score": delta_vs_best,
            }
        )
        if len(competitors) >= max_competitors:
            break

    top_score = float(mode_results[0].score) if mode_results else None
    return {
        "gold_rank_positions": gold_ranks,
        "gold_scores": gold_scores,
        "gold_presence": gold_presence,
        "best_gold_rank": best_gold_rank,
        "best_gold_score": best_gold_score,
        "top_score": top_score,
        "top_minus_best_gold": None if top_score is None or best_gold_score is None else top_score - best_gold_score,
        "all_gold_found_in_depth": all(rank is not None for rank in gold_ranks.values()),
        "competitors_above_gold": competitors,
    }


def assign_diagnosis_label(
    *,
    per_mode: Mapping[str, dict],
    split_gold_same_file: bool,
) -> str:
    """Assign deterministic diagnosis label from mode-level evidence."""
    modes = [row for row in per_mode.values() if row.get("present")]
    if not modes:
        return "unknown"

    if split_gold_same_file:
        split_signals = 0
        for mode in modes:
            if not mode.get("all_gold_found_in_depth", False):
                split_signals += 1
                continue
            ranks = [r for r in mode.get("gold_rank_positions", {}).values() if r is not None]
            if len(ranks) >= 2 and (max(ranks) - min(ranks) >= 3):
                split_signals += 1
        if split_signals >= 1:
            return "split_gold_same_file"

    weak_modes = 0
    high_comp_modes = 0
    for mode in modes:
        best_gold_rank = mode.get("best_gold_rank")
        competitors = mode.get("competitors_above_gold", [])
        if best_gold_rank is None:
            weak_modes += 1
            continue
        if not competitors:
            continue
        top_comp = competitors[0]
        delta = top_comp.get("delta_vs_best_gold_score")
        if delta is not None and delta >= 0 and not top_comp.get("same_file_as_any_gold", False):
            high_comp_modes += 1
        if delta is not None and delta >= 0 and top_comp.get("same_file_as_any_gold", False):
            weak_modes += 1

    if high_comp_modes >= max(1, len(modes) // 2):
        return "high_generic_competition"
    if weak_modes >= max(1, len(modes) // 2):
        return "weak_gold_signal"
    return "unknown"
