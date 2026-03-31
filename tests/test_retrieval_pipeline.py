"""Tests for retrieval pipeline composition."""

from dataclasses import dataclass
from typing import Optional

from root_rag.retrieval.models import EvidenceCandidate
from root_rag.retrieval.pipeline import RetrievalPipeline


@dataclass
class StubTransformer:
    suffix: str

    def transform(self, query: str) -> str:
        return f"{query} {self.suffix}".strip()


@dataclass
class StubBackend:
    last_query: Optional[str] = None

    def search(self, query: str, top_k: int):
        self.last_query = query
        return [
            EvidenceCandidate(
                chunk_id="chunk_1",
                file_path="test.cpp",
                start_line=1,
                end_line=2,
                symbol_path=None,
                doc_origin="source_impl",
                language="cpp",
                root_ref="v0.1",
                resolved_commit="abc123" + "0" * 34,
                score=1.0,
            )
        ][:top_k]


def test_retrieval_pipeline_applies_transformer_before_backend():
    backend = StubBackend()
    pipeline = RetrievalPipeline(
        backend=backend,
        query_transformer=StubTransformer("expanded"),
    )

    results = pipeline.search("TTree", top_k=5)

    assert backend.last_query == "TTree expanded"
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_1"
