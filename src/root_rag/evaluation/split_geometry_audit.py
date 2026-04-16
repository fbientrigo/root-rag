"""Split-gold geometry audit helpers for bridge-light diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from root_rag.evaluation.semantic_v1 import load_corpus, load_qrels, load_queries

LABELS = {
    "adjacent_split",
    "local_nonadjacent_split",
    "far_same_file_split",
    "one_gold_self_sufficient",
    "query_requires_noncontiguous_relation",
}

GEOMETRY_LABELS = {
    "adjacent_split": "adjacent",
    "local_nonadjacent_split": "near-but-nonadjacent",
    "far_same_file_split": "far apart",
    "one_gold_self_sufficient": "self-sufficient",
    "query_requires_noncontiguous_relation": "noncontiguous relation",
}

SELF_SUFFICIENT_NOTE = {
    "br003": (
        "Chunk 021 already shows the parsed 'region' directive dispatching to "
        "defineRegionField(...); the later definition chunk is supporting detail."
    ),
}

QUERY_NOTES = {
    "br003": "Parse-time dispatch already exposes the answer; the definition chunk is extra context.",
    "br004": "File-open and branch-binding are split across two distant regions of the same file.",
    "br005": "The helper definition and the concrete ShipMCTrack/ShipParticle calls are separated.",
    "br006": "Top-volume lookup and AddNode attachment live in distinct parts of ShipCave.cxx.",
    "br007": "ProcessHits and AddHit are separated by intervening detector-stepping code.",
    "br008": "Setter methods and runtime decayer setup are far apart in TEvtGenDecayer.cxx.",
}


@dataclass(frozen=True)
class GeometryAuditRow:
    query_id: str
    query_text: str
    gold_chunk_ids: list[str]
    file_path: str
    gold_order_positions: dict[str, int]
    chunk_distance: int
    minimum_contiguous_window_size: int
    geometry_relation: str
    self_sufficient: bool
    label: str
    note: str


def _index_corpus(rows: Sequence[dict]) -> tuple[Dict[str, dict], Dict[str, List[dict]]]:
    by_id = {row["chunk_id"]: row for row in rows}
    by_file: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_file[str(row["file_path"])].append(row)
    for file_path, file_rows in by_file.items():
        file_rows.sort(key=lambda row: (tuple(row.get("line_range", [0, 0])), row["chunk_id"]))
    return by_id, dict(by_file)


def _file_positions(file_rows: Sequence[dict]) -> dict[str, int]:
    return {row["chunk_id"]: idx + 1 for idx, row in enumerate(file_rows)}


def _same_file_gold_ids(*, query_id: str, qrels_map: Mapping[str, Mapping[str, int]], corpus_by_id: Mapping[str, dict]) -> list[str]:
    gold_ids = sorted(qrels_map[query_id].keys())
    gold_files = {
        str(corpus_by_id[chunk_id]["file_path"])
        for chunk_id in gold_ids
        if chunk_id in corpus_by_id and corpus_by_id[chunk_id].get("file_path")
    }
    if len(gold_ids) != 2 or len(gold_files) != 1:
        return []
    return gold_ids


def _classify_geometry(*, chunk_distance: int, self_sufficient: bool) -> str:
    if self_sufficient:
        return "one_gold_self_sufficient"
    if chunk_distance == 1:
        return "adjacent_split"
    if 2 <= chunk_distance <= 4:
        return "local_nonadjacent_split"
    return "far_same_file_split"


def load_bridge_light_geometry_audit(
    *,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
) -> dict:
    """Load frozen inputs and build the bridge-light split geometry audit."""
    corpus_rows = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrels_map = load_qrels(qrels_path)
    corpus_by_id, corpus_by_file = _index_corpus(corpus_rows)

    bridge_light_queries = [row for row in queries if row.query_class == "bridge-light"]
    per_query: List[GeometryAuditRow] = []

    for query_row in bridge_light_queries:
        gold_ids = _same_file_gold_ids(
            query_id=query_row.query_id,
            qrels_map=qrels_map,
            corpus_by_id=corpus_by_id,
        )
        if not gold_ids:
            continue

        file_path = str(corpus_by_id[gold_ids[0]]["file_path"])
        file_positions = _file_positions(corpus_by_file[file_path])
        gold_positions = {chunk_id: file_positions[chunk_id] for chunk_id in gold_ids}
        chunk_distance = abs(gold_positions[gold_ids[0]] - gold_positions[gold_ids[1]])
        minimum_window = chunk_distance + 1
        self_sufficient = query_row.query_id in SELF_SUFFICIENT_NOTE
        label = _classify_geometry(chunk_distance=chunk_distance, self_sufficient=self_sufficient)
        geometry_relation = GEOMETRY_LABELS.get(label, "other")
        note = QUERY_NOTES.get(query_row.query_id, "")
        if self_sufficient:
            note = SELF_SUFFICIENT_NOTE[query_row.query_id]

        per_query.append(
            GeometryAuditRow(
                query_id=query_row.query_id,
                query_text=query_row.query,
                gold_chunk_ids=gold_ids,
                file_path=file_path,
                gold_order_positions=gold_positions,
                chunk_distance=chunk_distance,
                minimum_contiguous_window_size=minimum_window,
                geometry_relation=geometry_relation,
                self_sufficient=self_sufficient,
                label=label,
                note=note,
            )
        )

    per_query.sort(key=lambda row: row.query_id)
    label_counts = Counter(row.label for row in per_query)
    dominant_label = label_counts.most_common(1)[0][0] if label_counts else None

    return {
        "generated_at": None,
        "scope": {
            "query_class": "bridge-light",
            "same_file_split_only": True,
            "query_ids": [row.query_id for row in per_query],
        },
        "geometry_definitions": {
            "adjacent_split": "gold chunks are consecutive in file order",
            "local_nonadjacent_split": "gold chunks are separated by a small in-file gap",
            "far_same_file_split": "gold chunks are separated by a large in-file gap",
            "one_gold_self_sufficient": "one gold chunk already answers the query well enough",
            "query_requires_noncontiguous_relation": "answer needs stitching noncontiguous file regions",
        },
        "per_query": [row.__dict__ for row in per_query],
        "class_summary": {
            "n_queries": len(per_query),
            "label_counts": dict(sorted(label_counts.items())),
            "dominant_label": dominant_label,
            "dominant_label_count": label_counts[dominant_label] if dominant_label else 0,
        },
    }
