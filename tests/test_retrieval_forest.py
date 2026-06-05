import pytest
from pathlib import Path
from root_rag.retrieval.forest import RetrievalForestBackend
from root_rag.retrieval.models import EvidenceCandidate

class MockBackend:
    def __init__(self, results):
        self.results = results
    def search(self, query, top_k):
        return self.results
    def operational_metrics(self):
        from root_rag.retrieval.interfaces import OperationalMetrics
        return OperationalMetrics()

def test_forest_fusion_rrf():
    c1 = EvidenceCandidate(chunk_id="a", file_path="f1", start_line=1, end_line=10, symbol_path=None, doc_origin="d", language="p", root_ref="r", resolved_commit="c", score=0.5)
    c2 = EvidenceCandidate(chunk_id="b", file_path="f2", start_line=1, end_line=10, symbol_path=None, doc_origin="d", language="p", root_ref="r", resolved_commit="c", score=0.4)
    
    b1 = MockBackend([c1, c2])
    b2 = MockBackend([c2, c1])
    
    forest = RetrievalForestBackend(profiles=[b1, b2], profile_names=["p1", "p2"])
    results = forest.search("query", top_k=2)
    
    assert len(results) == 2
    # Both c1 and c2 should have same RRF score if ranks are swapped
    assert results[0].chunk_id in ["a", "b"]

def test_forest_dedup():
    c1 = EvidenceCandidate(chunk_id="a", file_path="f1", start_line=1, end_line=10, symbol_path=None, doc_origin="d", language="p", root_ref="r", resolved_commit="c", score=0.5)
    c2 = EvidenceCandidate(chunk_id="b", file_path="f1", start_line=5, end_line=15, symbol_path=None, doc_origin="d", language="p", root_ref="r", resolved_commit="c", score=0.4)
    
    b1 = MockBackend([c1, c2])
    
    forest = RetrievalForestBackend(profiles=[b1], profile_names=["p1"], dedup_method="line_overlap")
    results = forest.search("query", top_k=2)
    
    assert len(results) == 1
    assert results[0].chunk_id == "a"
