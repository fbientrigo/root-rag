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

def test_harness_single_index_rrf():
    """Verify that RRF logic still works with a single profile (baseline + harness)."""
    c1 = EvidenceCandidate(chunk_id="a", file_path="f1", start_line=1, end_line=10, symbol_path=None, doc_origin="d", language="p", root_ref="r", resolved_commit="c", score=0.5)
    
    b1 = MockBackend([c1])
    forest = RetrievalForestBackend(profiles=[b1], profile_names=["baseline"], fusion_method="rrf")
    results = forest.search("query", top_k=1)
    
    assert len(results) == 1
    assert results[0].chunk_id == "a"
    assert results[0].source_profile == "baseline"

def test_enhanced_tie_breaker():
    """Verify the enhanced tie-breaker rules."""
    # Two candidates
    c1 = EvidenceCandidate(chunk_id="a", file_path="important_file.cxx", start_line=1, end_line=100, symbol_path=None, doc_origin="d", language="p", root_ref="r", resolved_commit="c", score=0.5)
    c2 = EvidenceCandidate(chunk_id="b", file_path="other.cxx", start_line=1, end_line=10, symbol_path=None, doc_origin="d", language="p", root_ref="r", resolved_commit="c", score=0.5)
    
    # Simulate a tie by having them swap ranks in two profiles
    # Profile 1: a=1, b=2
    # Profile 2: b=1, a=2
    b1 = MockBackend([c1, c2])
    b2 = MockBackend([c2, c1])
    
    forest = RetrievalForestBackend(profiles=[b1, b2], profile_names=["p1", "p2"], tie_breaker="enhanced")
    
    # Query contains 'important'
    results = forest.search("important", top_k=2)
    
    # c1 should be first because of query term in path
    assert results[0].chunk_id == "a"
    
    # If query doesn't match path, rule 2 (smaller span) should win
    results = forest.search("nothing", top_k=2)
    assert results[0].chunk_id == "b" # c2 has span 9, c1 has span 99
