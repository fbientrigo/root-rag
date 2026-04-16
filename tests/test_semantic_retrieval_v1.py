"""Smoke tests for semantic retrieval V1 over canonical chunk ids."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from root_rag.evaluation.semantic_v1 import run_semantic_v1_benchmark
from root_rag.retrieval.s1_semantic import (
    SemanticFaissSearcher,
    SemanticIndexManifest,
    build_semantic_index_artifacts,
    normalize_vectors,
)


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class StubEmbedder:
    model_name: str = "stub-semantic-model"
    dim: int = 16

    def embedding_dimension(self) -> int:
        return self.dim

    def embed(self, texts):
        import numpy as np

        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token in TOKEN_RE.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, byteorder="big", signed=False) % self.dim
                vectors[row_idx, index] += 1.0
        return normalize_vectors(vectors)


def _corpus_rows() -> list[dict]:
    return [
        {
            "chunk_id": "field_ShipFieldMaker.cxx_007",
            "text": (
                "// build overall magnetic field from config components\n"
                "defineGlobalField(fieldMap, regionName);\n"
                "compose global field from named components;"
            ),
            "file_path": "field\\ShipFieldMaker.cxx",
            "line_range": [70, 92],
            "headers_used": ["ShipCompField.h", "FairField.h"],
            "tier": "tier_1",
            "provenance": "FAIRSHIP_LOCAL",
            "usage_count": 12,
        },
        {
            "chunk_id": "field_ShipFieldMaker.cxx_021",
            "text": (
                "// attach one local field to region volume\n"
                "defineLocalField(regionName, volumeName, localField);\n"
                "volume assignment uses parsed config line;"
            ),
            "file_path": "field\\ShipFieldMaker.cxx",
            "line_range": [210, 236],
            "headers_used": ["ShipCompField.h", "FairField.h"],
            "tier": "tier_1",
            "provenance": "FAIRSHIP_LOCAL",
            "usage_count": 12,
        },
        {
            "chunk_id": "shipdata_ShipStack.cxx_019",
            "text": (
                "// keep transport particles in stack\n"
                "PushTrack(trackId, motherId, particleId);\n"
                "track vector output stored separately;"
            ),
            "file_path": "shipdata\\ShipStack.cxx",
            "line_range": [180, 205],
            "headers_used": ["TClonesArray.h", "ShipMCTrack.h"],
            "tier": "tier_1",
            "provenance": "FAIRSHIP_LOCAL",
            "usage_count": 8,
        },
    ]


def _write_queries(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {"id": "lc001", "query": "defineGlobalField", "query_class": "lexical-control"},
                {
                    "id": "sm001",
                    "query": "where overall magnetic field is built from named components",
                    "query_class": "semantic",
                },
                {
                    "id": "br001",
                    "query": "where parsed region directive becomes local field assignment",
                    "query_class": "bridge-light",
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_qrels(path: Path) -> None:
    lines = [
        {"query_id": "lc001", "chunk_id": "field_ShipFieldMaker.cxx_007", "relevance": 2},
        {"query_id": "sm001", "chunk_id": "field_ShipFieldMaker.cxx_007", "relevance": 2},
        {"query_id": "br001", "chunk_id": "field_ShipFieldMaker.cxx_021", "relevance": 2},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")


def _write_corpus(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_semantic_manifest_preserves_canonical_chunk_ids_and_row_count(tmp_path):
    corpus_rows = _corpus_rows()
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(corpus_path, corpus_rows)

    manifest = build_semantic_index_artifacts(
        corpus_rows=corpus_rows,
        corpus_path=corpus_path,
        output_dir=tmp_path / "semantic",
        embedder=StubEmbedder(),
        corpus_source_identifier="toy-corpus",
    )

    rows = [
        json.loads(line)
        for line in Path(manifest.records_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest.row_count == len(corpus_rows)
    assert {row["chunk_id"] for row in rows} == {row["chunk_id"] for row in corpus_rows}
    assert all("semantic_text" in row for row in rows)
    assert all("path_tokens" in row for row in rows)


def test_semantic_search_returns_benchmark_compatible_chunk_ids(tmp_path):
    corpus_rows = _corpus_rows()
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(corpus_path, corpus_rows)

    manifest = build_semantic_index_artifacts(
        corpus_rows=corpus_rows,
        corpus_path=corpus_path,
        output_dir=tmp_path / "semantic",
        embedder=StubEmbedder(),
        corpus_source_identifier="toy-corpus",
    )
    searcher = SemanticFaissSearcher(
        manifest_path=tmp_path / "semantic" / "semantic_manifest.json",
        embedder=StubEmbedder(),
    )

    results = searcher.search("overall magnetic field named components", top_k=2)

    assert results
    assert {row.chunk_id for row in results}.issubset({row["chunk_id"] for row in corpus_rows})
    assert results[0].chunk_id == "field_ShipFieldMaker.cxx_007"
    assert SemanticIndexManifest.load(tmp_path / "semantic" / "semantic_manifest.json").row_count == 3


def test_semantic_v1_benchmark_runs_all_three_modes(tmp_path):
    corpus_rows = _corpus_rows()
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.json"
    qrels_path = tmp_path / "qrels.jsonl"
    _write_corpus(corpus_path, corpus_rows)
    _write_queries(queries_path)
    _write_qrels(qrels_path)

    build_semantic_index_artifacts(
        corpus_rows=corpus_rows,
        corpus_path=corpus_path,
        output_dir=tmp_path / "semantic",
        embedder=StubEmbedder(),
        corpus_source_identifier="toy-corpus",
    )

    results = run_semantic_v1_benchmark(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        semantic_manifest_path=tmp_path / "semantic" / "semantic_manifest.json",
        semantic_model_name="stub-semantic-model",
        semantic_embedder=StubEmbedder(),
        commands_run=["build", "benchmark"],
    )

    assert set(results["modes"]) == {"bm25_only", "semantic_only", "hybrid"}
    assert all(results["modes"][mode]["per_query"] for mode in results["modes"])
    assert "lexical-control" in results["modes"]["bm25_only"]["per_category"]
    assert "semantic" in results["modes"]["semantic_only"]["per_category"]
    assert "bridge-light" in results["modes"]["hybrid"]["per_category"]
    assert results["comparisons"]["hybrid_vs_bm25_only"]["summary"]["after"] == results["modes"]["hybrid"]["summary"]
