"""Shadow experiment helpers for focal chunk-granularity diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from root_rag.retrieval.backends import build_retrieval_backend
from root_rag.retrieval.pipeline import RetrievalPipeline
from root_rag.retrieval.s1_semantic import LocalEmbedder
from root_rag.retrieval.transformers import build_query_transformer


def _line_range(row: Mapping[str, object]) -> tuple[int, int]:
    line_range = row.get("line_range")
    if isinstance(line_range, Sequence) and len(line_range) >= 1:
        start = int(line_range[0])
        end = int(line_range[1]) if len(line_range) > 1 else start
        return start, end
    start = int(row.get("start_line", 1))
    end = int(row.get("end_line", start))
    return start, end


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def _merge_pair_text(left: Mapping[str, object], right: Mapping[str, object]) -> str:
    left_text = str(left.get("text") or left.get("content") or "")
    right_text = str(right.get("text") or right.get("content") or "")
    left_lines = left_text.splitlines()
    right_lines = right_text.splitlines()
    left_start, left_end = _line_range(left)
    right_start, _right_end = _line_range(right)

    overlap = max(0, left_end - right_start + 1)
    overlap = min(overlap, len(right_lines))
    merged_lines = left_lines + right_lines[overlap:]
    return "\n".join(merged_lines).strip() or left_text.strip() or right_text.strip()


def _make_shadow_row(
    *,
    left: Mapping[str, object],
    right: Mapping[str, object] | None,
    file_path: str,
) -> dict:
    left_start, left_end = _line_range(left)
    if right is None:
        chunk_id = f"shadow::{left['chunk_id']}"
        merged_text = str(left.get("text") or left.get("content") or "")
        member_chunk_ids = [str(left["chunk_id"])]
        line_range = [left_start, left_end]
        provenance = "shadow_singleton"
    else:
        right_start, right_end = _line_range(right)
        chunk_id = f"shadow::{left['chunk_id']}::{right['chunk_id']}"
        merged_text = _merge_pair_text(left, right)
        member_chunk_ids = [str(left["chunk_id"]), str(right["chunk_id"])]
        line_range = [min(left_start, right_start), max(left_end, right_end)]
        provenance = "shadow_adjacent_pair"

    headers_used = _dedupe_keep_order(
        list(left.get("headers_used", [])) + (list(right.get("headers_used", [])) if right is not None else [])
    )

    return {
        "chunk_id": chunk_id,
        "text": merged_text,
        "file_path": file_path,
        "line_range": line_range,
        "headers_used": headers_used,
        "tier": "shadow",
        "provenance": provenance,
        "usage_count": int(left.get("usage_count", 0)) + (int(right.get("usage_count", 0)) if right is not None else 0),
        "source_chunk_ids": member_chunk_ids,
        "composition_rule": "same_file_adjacent_pair",
        "root_ref": left.get("root_ref") if left.get("root_ref") is not None else (right.get("root_ref") if right is not None else None),
        "resolved_commit": left.get("resolved_commit")
        if left.get("resolved_commit") is not None
        else (right.get("resolved_commit") if right is not None else None),
    }


def build_shadow_corpus_rows(corpus_rows: Sequence[dict]) -> tuple[List[dict], Dict[str, List[str]]]:
    """Build a local shadow corpus from same-file adjacent chunk pairs.

    Each file is sorted by line range. Adjacent rows are merged into a shadow
    superchunk whose content is the union of the two slices. Single-row files
    are preserved as singletons.

    Returns:
        shadow_rows: corpus rows for retrieval
        membership: map of original chunk_id -> shadow chunk_ids containing it
    """
    by_file: Dict[str, List[dict]] = defaultdict(list)
    for row in corpus_rows:
        by_file[str(row["file_path"])].append(dict(row))

    shadow_rows: List[dict] = []
    membership: Dict[str, List[str]] = defaultdict(list)

    for file_path, rows in sorted(by_file.items()):
        ordered = sorted(rows, key=lambda row: (*_line_range(row), str(row["chunk_id"])))
        if len(ordered) == 1:
            shadow_row = _make_shadow_row(left=ordered[0], right=None, file_path=file_path)
            shadow_rows.append(shadow_row)
            membership[str(ordered[0]["chunk_id"])].append(shadow_row["chunk_id"])
            continue

        for left, right in zip(ordered, ordered[1:]):
            shadow_row = _make_shadow_row(left=left, right=right, file_path=file_path)
            shadow_rows.append(shadow_row)
            membership[str(left["chunk_id"])].append(shadow_row["chunk_id"])
            membership[str(right["chunk_id"])].append(shadow_row["chunk_id"])

    for chunk_id, shadow_ids in membership.items():
        membership[chunk_id] = _dedupe_keep_order(shadow_ids)

    return shadow_rows, dict(membership)


def build_shadow_qrels_rows(
    *,
    qrels_map: Mapping[str, Mapping[str, int]],
    shadow_membership: Mapping[str, Sequence[str]],
) -> List[dict]:
    """Derive a shadow qrels snapshot for the local experiment."""
    merged: Dict[tuple[str, str], int] = {}
    for query_id, gold_map in qrels_map.items():
        for gold_chunk_id, relevance in gold_map.items():
            shadow_ids = shadow_membership.get(gold_chunk_id, [gold_chunk_id])
            for shadow_chunk_id in shadow_ids:
                key = (str(query_id), str(shadow_chunk_id))
                merged[key] = max(merged.get(key, 0), int(relevance))

    rows = [
        {"query_id": query_id, "chunk_id": chunk_id, "relevance": relevance}
        for (query_id, chunk_id), relevance in sorted(merged.items())
    ]
    return rows


@dataclass(frozen=True)
class GoldCoverage:
    gold_chunk_id: str
    present: bool
    best_rank: int | None
    covering_shadow_chunk_id: str | None


def compute_gold_coverage(
    *,
    ranked_shadow_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
    shadow_membership: Mapping[str, Sequence[str]],
) -> tuple[Dict[str, GoldCoverage], int, int | None]:
    """Map ranked shadow results back onto original gold chunk ids."""
    rank_by_shadow_chunk = {chunk_id: idx + 1 for idx, chunk_id in enumerate(ranked_shadow_chunk_ids)}
    per_gold: Dict[str, GoldCoverage] = {}
    gold_ranks: List[int] = []

    for gold_chunk_id in gold_chunk_ids:
        shadow_ids = shadow_membership.get(gold_chunk_id, [gold_chunk_id])
        covered_rank: int | None = None
        covered_shadow_id: str | None = None
        for shadow_chunk_id in shadow_ids:
            rank = rank_by_shadow_chunk.get(shadow_chunk_id)
            if rank is None:
                continue
            if covered_rank is None or rank < covered_rank:
                covered_rank = rank
                covered_shadow_id = shadow_chunk_id
        present = covered_rank is not None
        if present:
            gold_ranks.append(covered_rank)
        per_gold[gold_chunk_id] = GoldCoverage(
            gold_chunk_id=gold_chunk_id,
            present=present,
            best_rank=covered_rank,
            covering_shadow_chunk_id=covered_shadow_id,
        )

    gold_count = sum(1 for coverage in per_gold.values() if coverage.present)
    best_rank = min(gold_ranks) if gold_ranks else None
    return per_gold, gold_count, best_rank


def build_mode_pipelines(
    *,
    corpus_rows: Sequence[dict],
    corpus_path: Path,
    semantic_manifest_path: Path,
    semantic_model_name: str,
    semantic_embedder: LocalEmbedder,
) -> Dict[str, RetrievalPipeline]:
    """Build frozen benchmark pipelines for the three retrieval modes."""
    pipelines: Dict[str, RetrievalPipeline] = {}
    for mode in ("bm25_only", "semantic_only", "hybrid"):
        backend = build_retrieval_backend(
            mode,
            corpus_rows=list(corpus_rows),
            corpus_artifact_path=corpus_path,
            semantic_manifest_path=semantic_manifest_path if mode != "bm25_only" else None,
            semantic_model_name=semantic_model_name,
            semantic_embedder=semantic_embedder,
            k1=1.5,
            b=0.75,
            dense_dim=512,
        )
        pipelines[mode] = RetrievalPipeline(
            backend=backend,
            query_transformer=build_query_transformer("baseline"),
        )
    return pipelines


def search_pipelines(
    *,
    pipelines: Mapping[str, RetrievalPipeline],
    queries: Sequence[dict],
    top_k: int,
) -> Dict[str, Dict[str, List[str]]]:
    """Return ranked chunk ids per query for each retrieval mode."""
    runs: Dict[str, Dict[str, List[str]]] = {mode: {} for mode in pipelines}
    for query_row in queries:
        query_id = str(query_row["id"])
        query_text = str(query_row["query"])
        for mode, pipeline in pipelines.items():
            ranked = pipeline.search(query_text, top_k=top_k)
            runs[mode][query_id] = [row.chunk_id for row in ranked]
    return runs


def summarize_query(
    *,
    query_row: dict,
    gold_chunk_ids: Sequence[str],
    corpus_by_id: Mapping[str, Mapping[str, object]],
    shadow_membership: Mapping[str, Sequence[str]],
    runs: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict:
    """Summarize one query across modes using original gold ids."""
    mode_rows: Dict[str, dict] = {}
    mode_gold_counts: Dict[str, int] = {}
    mode_best_ranks: Dict[str, int | None] = {}

    for mode, mode_run in runs.items():
        ranked_shadow_ids = list(mode_run.get(str(query_row["id"]), []))
        gold_presence, gold_count, best_rank = compute_gold_coverage(
            ranked_shadow_chunk_ids=ranked_shadow_ids,
            gold_chunk_ids=gold_chunk_ids,
            shadow_membership=shadow_membership,
        )
        mode_gold_counts[mode] = gold_count
        mode_best_ranks[mode] = best_rank
        mode_rows[mode] = {
            "present": True,
            "backend": {
                "bm25_only": "lexical_bm25_memory",
                "semantic_only": "semantic_faiss",
                "hybrid": "hybrid_s1",
            }[mode],
            "gold_chunk_ids": list(gold_chunk_ids),
            "gold_presence": {
                gold_chunk_id: {
                    "present": coverage.present,
                    "rank": coverage.best_rank,
                    "shadow_chunk_id": coverage.covering_shadow_chunk_id,
                }
                for gold_chunk_id, coverage in gold_presence.items()
            },
            "best_gold_count": gold_count,
            "best_gold_rank": best_rank,
            "top10": ranked_shadow_ids[:10],
        }

    best_mode_gold_count = max(mode_gold_counts.values()) if mode_gold_counts else 0
    found_ranks = [rank for rank in mode_best_ranks.values() if rank is not None]
    best_mode_best_rank = min(found_ranks) if found_ranks else None
    late_rank = bool(found_ranks) and best_mode_best_rank is not None and best_mode_best_rank > 5

    return {
        "query_id": str(query_row["id"]),
        "query_text": str(query_row["query"]),
        "query_class": str(query_row["query_class"]),
        "gold_chunk_ids": list(gold_chunk_ids),
        "same_file_split": _same_file_split(gold_chunk_ids, corpus_by_id),
        "aggregate": {
            "best_mode_gold_count": best_mode_gold_count,
            "best_mode_best_rank": best_mode_best_rank,
            "late_rank": late_rank,
        },
        "modes": mode_rows,
    }


def _same_file_split(
    gold_chunk_ids: Sequence[str],
    corpus_by_id: Mapping[str, Mapping[str, object]],
) -> bool:
    if len(gold_chunk_ids) < 2:
        return False
    gold_files = {
        str(corpus_by_id[chunk_id]["file_path"])
        for chunk_id in gold_chunk_ids
        if chunk_id in corpus_by_id and corpus_by_id[chunk_id].get("file_path")
    }
    return len(gold_files) == 1


def summarize_class_rates(per_query_rows: Sequence[dict]) -> dict:
    """Compute the focal-chunk class summary rates."""
    n_queries = len(per_query_rows)
    both = sum(1 for row in per_query_rows if row["aggregate"]["best_mode_gold_count"] >= 2)
    one = sum(1 for row in per_query_rows if row["aggregate"]["best_mode_gold_count"] == 1)
    zero = sum(1 for row in per_query_rows if row["aggregate"]["best_mode_gold_count"] == 0)
    same_file_split = sum(1 for row in per_query_rows if row["same_file_split"])
    late_rank = sum(1 for row in per_query_rows if row["aggregate"]["late_rank"])

    def rate(value: int) -> float:
        return value / n_queries if n_queries else 0.0

    return {
        "n_queries": n_queries,
        "both_golds_found_rate": rate(both),
        "one_gold_found_rate": rate(one),
        "zero_gold_found_rate": rate(zero),
        "same_file_split_rate": rate(same_file_split),
        "late_rank_rate": rate(late_rank),
    }


def compare_rate_blocks(before: Mapping[str, float], after: Mapping[str, float]) -> dict:
    """Return delta between two rate dictionaries."""
    keys = (
        "both_golds_found_rate",
        "one_gold_found_rate",
        "zero_gold_found_rate",
        "same_file_split_rate",
        "late_rank_rate",
    )
    return {key: float(after[key]) - float(before[key]) for key in keys}
