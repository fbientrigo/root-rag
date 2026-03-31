"""Contract tests for retrieval backend interface behavior."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from root_rag.index.fts import create_fts5_db, insert_chunks_into_fts
from root_rag.index.schemas import Chunk
from root_rag.retrieval.backends import (
    DenseHashMemoryBackend,
    FTS5LexicalBackend,
    InMemoryBM25LexicalBackend,
    build_retrieval_backend,
)
from root_rag.retrieval.interfaces import BaseRetrievalBackend
from root_rag.retrieval.models import EvidenceCandidate
from root_rag.retrieval.pipeline import RetrievalPipeline


def _evidence(chunk_id: str, score: float) -> EvidenceCandidate:
    return EvidenceCandidate(
        chunk_id=chunk_id,
        file_path=f"{chunk_id}.cpp",
        start_line=1,
        end_line=2,
        symbol_path=None,
        doc_origin="source_impl",
        language="cpp",
        root_ref="v0.1",
        resolved_commit="abc123" + "0" * 34,
        score=score,
    )


def _write_single_chunk_fts(db_path: Path) -> None:
    create_fts5_db(db_path)
    chunk = Chunk.from_file_slice(
        file_path="tree/src/TTree.cxx",
        start_line=1,
        end_line=3,
        content="Long64_t TTree::Draw(const char* expr) { return 0; }",
        root_ref="v0.1",
        resolved_commit="abc123" + "0" * 34,
        language="cpp",
        doc_origin="source_impl",
    )
    insert_chunks_into_fts(db_path, [chunk])


@dataclass
class StubTransformer:
    suffix: str = ""

    def transform(self, query: str) -> str:
        return f"{query}{self.suffix}"


@dataclass
class OverReturningBackend:
    call_count: int = 0
    last_top_k: Optional[int] = None
    last_query: Optional[str] = None

    def search(self, query: str, top_k: int) -> List[EvidenceCandidate]:
        self.call_count += 1
        self.last_top_k = top_k
        self.last_query = query
        return [
            _evidence("chunk_1", 2.0),
            _evidence("chunk_2", 1.0),
            _evidence("chunk_3", 0.5),
        ]

    def operational_metrics(self) -> Dict[str, float]:
        return {}


def test_pipeline_short_circuits_non_positive_top_k():
    backend = OverReturningBackend()
    pipeline = RetrievalPipeline(backend=backend, query_transformer=StubTransformer(" normalized"))

    assert pipeline.search("TTree", top_k=0) == []
    assert pipeline.search("TTree", top_k=-5) == []
    assert backend.call_count == 0


def test_pipeline_truncates_backend_over_return_to_top_k():
    backend = OverReturningBackend()
    pipeline = RetrievalPipeline(backend=backend, query_transformer=StubTransformer(" normalized"))

    results = pipeline.search("TTree", top_k=2)

    assert backend.last_query == "TTree normalized"
    assert backend.last_top_k == 2
    assert len(results) == 2
    assert [row.chunk_id for row in results] == ["chunk_1", "chunk_2"]


def test_normalize_operational_metrics_coerces_non_finite_floats():
    metrics = BaseRetrievalBackend.normalize_operational_metrics(
        {
            "finite": 1.0,
            "nan_value": float("nan"),
            "inf_value": float("inf"),
        }
    )

    assert metrics["finite"] == 1.0
    assert metrics["nan_value"] is None
    assert metrics["inf_value"] is None


def test_fts_backend_returns_empty_for_non_positive_top_k(tmp_path):
    db_path = tmp_path / "test.sqlite"
    _write_single_chunk_fts(db_path)
    backend = FTS5LexicalBackend(db_path=db_path)

    assert len(backend.search("TTree Draw", top_k=10)) >= 1
    assert backend.search("TTree Draw", top_k=0) == []
    assert backend.search("TTree Draw", top_k=-1) == []


def test_bm25_backend_returns_empty_for_non_positive_top_k():
    backend = InMemoryBM25LexicalBackend(
        corpus_rows=[
            {
                "chunk_id": "chunk_1",
                "text": "TTree Draw long64",
                "file_path": "tree/src/TTree.cxx",
                "line_range": [1, 3],
            }
        ]
    )

    assert len(backend.search("TTree", top_k=10)) == 1
    assert backend.search("TTree", top_k=0) == []
    assert backend.search("TTree", top_k=-1) == []


def test_factory_returns_expected_backend_ids(tmp_path):
    db_path = tmp_path / "test.sqlite"
    _write_single_chunk_fts(db_path)

    fts_backend = build_retrieval_backend("lexical_fts5", db_path=db_path)
    bm25_backend = build_retrieval_backend(
        "lexical_bm25_memory",
        corpus_rows=[
            {
                "chunk_id": "chunk_1",
                "text": "TTree Draw",
                "file_path": "tree/src/TTree.cxx",
                "line_range": [1, 3],
            }
        ],
    )
    dense_backend = build_retrieval_backend(
        "dense_hash_memory",
        corpus_rows=[
            {
                "chunk_id": "chunk_1",
                "text": "TTree Draw",
                "file_path": "tree/src/TTree.cxx",
                "line_range": [1, 3],
            }
        ],
        dense_dim=128,
    )

    assert fts_backend.backend_id == "lexical_fts5"
    assert bm25_backend.backend_id == "lexical_bm25_memory"
    assert dense_backend.backend_id == "dense_hash_memory"


def test_dense_backend_returns_evidence_candidates():
    backend = DenseHashMemoryBackend(
        corpus_rows=[
            {
                "chunk_id": "chunk_1",
                "text": "TTree Draw long64",
                "file_path": "tree/src/TTree.cxx",
                "line_range": [1, 3],
            },
            {
                "chunk_id": "chunk_2",
                "text": "TH1F histogram bins",
                "file_path": "hist/h1.cxx",
                "line_range": [5, 8],
            },
        ],
        vector_dim=64,
    )

    results = backend.search("TTree Draw", top_k=5)
    assert len(results) >= 1
    assert isinstance(results[0], EvidenceCandidate)
    assert results[0].chunk_id == "chunk_1"


def test_operational_metrics_values_are_json_scalars(tmp_path):
    db_path = tmp_path / "test.sqlite"
    _write_single_chunk_fts(db_path)

    backends = [
        FTS5LexicalBackend(db_path=db_path),
        InMemoryBM25LexicalBackend(
            corpus_rows=[
                {
                    "chunk_id": "chunk_1",
                    "text": "TTree Draw",
                    "file_path": "tree/src/TTree.cxx",
                    "line_range": [1, 3],
                }
            ]
        ),
        DenseHashMemoryBackend(
            corpus_rows=[
                {
                    "chunk_id": "chunk_1",
                    "text": "TTree Draw",
                    "file_path": "tree/src/TTree.cxx",
                    "line_range": [1, 3],
                }
            ],
            vector_dim=64,
        ),
    ]

    for backend in backends:
        metrics = backend.operational_metrics()
        assert all(isinstance(key, str) for key in metrics.keys())
        assert all(
            value is None or isinstance(value, (int, float, str))
            for value in metrics.values()
        )


def test_dense_operational_metrics_include_dense_specific_keys():
    backend = DenseHashMemoryBackend(
        corpus_rows=[
            {
                "chunk_id": "chunk_1",
                "text": "TTree Draw",
                "file_path": "tree/src/TTree.cxx",
                "line_range": [1, 3],
            }
        ],
        vector_dim=64,
    )

    metrics = backend.operational_metrics()
    assert metrics["vector_dim"] == 64
    assert metrics["similarity"] == "cosine_hash"
    assert metrics["avg_nonzero_dims"] is not None
