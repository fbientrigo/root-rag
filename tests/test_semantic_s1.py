"""Focused tests for S1 semantic helpers and hybrid guardrails."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from root_rag.retrieval.backends import HybridS1Backend
from root_rag.retrieval.models import EvidenceCandidate
from root_rag.retrieval.s1_semantic import (
    SemanticIndexManifest,
    build_embedding_text,
    fuse_ranked_results,
    is_symbol_like_query,
)


def _evidence(chunk_id: str, score: float, *, file_path: Optional[str] = None) -> EvidenceCandidate:
    return EvidenceCandidate(
        chunk_id=chunk_id,
        file_path=file_path or f"{chunk_id}.cpp",
        start_line=1,
        end_line=5,
        symbol_path=None,
        doc_origin="source_impl",
        language="cpp",
        root_ref="v0.1",
        resolved_commit="abc123" + "0" * 34,
        score=score,
    )


def test_build_embedding_text_is_deterministic_and_metadata_rich():
    row = {
        "chunk_id": "chunk_1",
        "file_path": "tree/src/TTree.cxx",
        "start_line": 10,
        "end_line": 22,
        "symbol_path": "TTree::Draw",
        "doc_origin": "source_impl",
        "language": "cpp",
        "headers_used": ["TObject.h", "TTree.h"],
        "content": "Long64_t TTree::Draw(const char* expr) { return 0; }",
    }

    text_a = build_embedding_text(row)
    text_b = build_embedding_text(row)

    assert text_a == text_b
    assert "path: tree/src/TTree.cxx" in text_a
    assert "symbol: TTree::Draw" in text_a
    assert "headers: TObject.h, TTree.h" in text_a
    assert "content:" in text_a


def test_build_embedding_text_adds_local_relation_hints():
    row = {
        "chunk_id": "chunk_2",
        "file_path": "passive/ShipCave.cxx",
        "start_line": 38,
        "end_line": 45,
        "symbol_path": None,
        "doc_origin": "source_impl",
        "language": "cpp",
        "headers_used": ["TGeoBBox.h", "TGeoManager.h"],
        "content": (
            "TGeoVolume* top = gGeoManager->GetTopVolume();\n"
            "ShipGeo::InitMedium(\"Concrete\");\n"
            "TGeoMedium* concrete = gGeoManager->GetMedium(\"Concrete\");"
        ),
    }

    text = build_embedding_text(row)

    assert "relations:" in text
    assert "fetches top geometry volume before attachment" in text
    assert "loads ROOT medium for geometry build" in text


def test_symbol_like_query_heuristic_prefers_exact_code_shapes():
    assert is_symbol_like_query("TTree::Draw")
    assert is_symbol_like_query("ShipStack::PushTrack()")
    assert is_symbol_like_query("field/ShipFieldMaker.h")
    assert not is_symbol_like_query("how to build detector geometry")


def test_weighted_rrf_pins_lexical_results_for_symbol_queries():
    lexical = [
        _evidence("lex_1", 10.0),
        _evidence("lex_2", 9.0),
        _evidence("lex_3", 8.0),
    ]
    semantic = [
        _evidence("sem_1", 0.95),
        _evidence("sem_2", 0.90),
        _evidence("lex_2", 0.89),
    ]

    fused = fuse_ranked_results(
        lexical_results=lexical,
        semantic_results=semantic,
        top_k=5,
        lexical_weight=0.85,
        semantic_weight=0.15,
        symbol_safe=True,
        lexical_pin_count=3,
    )

    assert [row.chunk_id for row in fused[:3]] == ["lex_1", "lex_2", "lex_3"]


@dataclass
class StubBackend:
    backend_id: str
    rows: list[EvidenceCandidate]

    def search(self, query: str, top_k: int):
        return self.rows[:top_k]

    def operational_metrics(self):
        return {}


def test_hybrid_backend_uses_symbol_safe_policy():
    backend = HybridS1Backend(
        lexical_backend=StubBackend(
            backend_id="lexical_bm25_memory",
            rows=[_evidence("lex_1", 5.0), _evidence("lex_2", 4.0), _evidence("lex_3", 3.0)],
        ),
        semantic_backend=StubBackend(
            backend_id="semantic_faiss",
            rows=[_evidence("sem_1", 0.95), _evidence("sem_2", 0.90), _evidence("sem_3", 0.85)],
        ),
    )

    results = backend.search("TTree::Draw", top_k=5)

    assert [row.chunk_id for row in results[:3]] == ["lex_1", "lex_2", "lex_3"]


def test_hybrid_backend_operational_metrics_include_search_depth():
    backend = HybridS1Backend(
        lexical_backend=StubBackend(backend_id="lexical_bm25_memory", rows=[]),
        semantic_backend=StubBackend(backend_id="semantic_faiss", rows=[]),
        search_depth=50,
    )

    metrics = backend.operational_metrics()

    assert metrics["search_depth"] == 50


def test_semantic_manifest_round_trip(tmp_path):
    manifest = SemanticIndexManifest(
        schema_version="1.0.0",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        normalization="l2",
        corpus_source_identifier="test-corpus",
        corpus_path="artifacts/corpus.jsonl",
        corpus_sha256="abc123",
        row_count=2,
        faiss_index_type="IndexFlatIP",
        index_path="semantic/index.faiss",
        records_path="semantic/records.jsonl",
        vectors_path="semantic/vectors.npy",
        created_at="2026-04-04T00:00:00+00:00",
        build_backend="sentence_transformers_local",
        python_version="3.12.0",
        platform="test",
    )

    path = tmp_path / "semantic_manifest.json"
    manifest.save(path)
    loaded = SemanticIndexManifest.load(path)

    assert loaded.model_name == manifest.model_name
    assert loaded.embedding_dimension == 384
    assert Path(loaded.index_path).name == "index.faiss"
