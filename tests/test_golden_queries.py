"""Golden query tests for ROOT 6.36.08 seed corpus.

Tests that core FairShip-relevant ROOT classes can be found via retrieval.
Acceptance: 5/6 queries should return correct file in top-3 results.
"""
import pytest
from pathlib import Path

# Golden queries with expected results
GOLDEN_QUERIES = [
    {
        "id": "q001",
        "query": "TTree::Fill",
        "expected_files": [
            "tree/tree/inc/TTree.h",
            "tree/tree/src/TTree.cxx",
        ],
        "category": "symbol_lookup",
    },
    {
        "id": "q002",
        "query": "TFile",
        "expected_files": [
            "io/io/inc/TFile.h",
            "io/io/src/TFile.cxx",
        ],
        "category": "class_lookup",
    },
    {
        "id": "q003",
        "query": "TH1F",
        "expected_files": [
            "hist/hist/inc/TH1F.h",
            "hist/hist/inc/TH1.h",
        ],
        "category": "class_lookup",
    },
    {
        "id": "q004",
        "query": "TVector3 Mag",
        "expected_files": [
            "math/physics/inc/TVector3.h",
            "math/physics/src/TVector3.cxx",
        ],
        "category": "method_lookup",
    },
    {
        "id": "q005",
        "query": "TGeoManager",
        "expected_files": [
            "geom/geom/inc/TGeoManager.h",
            "geom/geom/src/TGeoManager.cxx",
        ],
        "category": "class_lookup",
    },
    {
        "id": "q006",
        "query": "TLorentzVector",
        "expected_files": [
            "math/physics/inc/TLorentzVector.h",
            "math/physics/src/TLorentzVector.cxx",
        ],
        "category": "class_lookup",
    },
]


def test_golden_queries_exist():
    """Verify golden query set is loaded."""
    assert len(GOLDEN_QUERIES) == 6
    assert all("query" in q for q in GOLDEN_QUERIES)
    assert all("expected_files" in q for q in GOLDEN_QUERIES)


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("data/indexes").exists(),
    reason="No indexes found - run 'root-rag index' first"
)
def test_golden_queries_retrieval():
    """Test retrieval quality on golden queries.
    
    Requires:
        - ROOT 6.36.08 indexed (run: root-rag index)
    
    Acceptance:
        - At least 5/6 queries return expected file in top-3
    """
    from root_rag.retrieval import lexical_search
    from root_rag.index.locator import resolve_index
    
    # Resolve index for v6-36-08
    try:
        manifest = resolve_index(
            indexes_root=Path("data/indexes"),
            root_ref="v6-36-08",
            index_id=None,
        )
    except Exception as e:
        pytest.skip(f"Index not found for v6-36-08: {e}")
    
    db_path = Path(manifest.fts_db_path)
    if not db_path.exists():
        pytest.skip(f"FTS database not found: {db_path}")
    
    # Run queries and check results
    passed = 0
    failures = []
    
    for q in GOLDEN_QUERIES:
        query_str = q["query"]
        expected_files = q["expected_files"]
        
        # Perform search
        results = lexical_search(
            db_path=str(db_path),
            query=query_str,
            top_k=3,
        )
        
        if not results:
            failures.append({
                "id": q["id"],
                "query": query_str,
                "reason": "no_results",
                "expected": expected_files,
            })
            continue
        
        # Check if any expected file is in top-3
        returned_files = [r.file_path for r in results]
        found = any(
            any(exp in ret for exp in expected_files)
            for ret in returned_files
        )
        
        if found:
            passed += 1
        else:
            failures.append({
                "id": q["id"],
                "query": query_str,
                "reason": "expected_not_in_top3",
                "expected": expected_files,
                "returned": returned_files,
            })
    
    # Report results
    pass_rate = passed / len(GOLDEN_QUERIES)
    print(f"\nGolden Query Results:")
    print(f"  Passed: {passed}/{len(GOLDEN_QUERIES)} ({pass_rate:.1%})")
    
    if failures:
        print(f"\n  Failures:")
        for f in failures:
            print(f"    [{f['id']}] {f['query']}: {f['reason']}")
            print(f"      Expected: {f['expected']}")
            if f['reason'] == 'expected_not_in_top3':
                print(f"      Returned: {f['returned']}")
    
    # Acceptance: at least 5/6 queries (83%)
    assert pass_rate >= 0.83, f"Pass rate {pass_rate:.1%} < 83%. Failures: {failures}"


@pytest.mark.integration
def test_version_tagging():
    """Verify all retrieval results are tagged with ROOT version."""
    from root_rag.retrieval import lexical_search
    from root_rag.index.locator import resolve_index
    
    try:
        manifest = resolve_index(
            indexes_root=Path("data/indexes"),
            root_ref="v6-36-08",
            index_id=None,
        )
    except Exception:
        pytest.skip("Index not found for v6-36-08")
    
    db_path = Path(manifest.fts_db_path)
    if not db_path.exists():
        pytest.skip(f"FTS database not found: {db_path}")
    
    # Test query
    results = lexical_search(
        db_path=str(db_path),
        query="TTree",
        top_k=5,
    )
    
    if not results:
        pytest.skip("No results for test query")
    
    # Check version tagging
    for r in results:
        assert r.root_ref is not None, "Missing root_ref"
        assert r.resolved_commit is not None, "Missing resolved_commit"
        assert r.file_path is not None, "Missing file_path"
        assert r.start_line > 0, "Invalid start_line"
        assert r.end_line >= r.start_line, "Invalid line range"


@pytest.mark.integration
def test_no_hallucinations():
    """Verify retrieval never returns uncited results."""
    from root_rag.retrieval import lexical_search
    from root_rag.index.locator import resolve_index
    
    try:
        manifest = resolve_index(
            indexes_root=Path("data/indexes"),
            root_ref="v6-36-08",
            index_id=None,
        )
    except Exception:
        pytest.skip("Index not found for v6-36-08")
    
    db_path = Path(manifest.fts_db_path)
    if not db_path.exists():
        pytest.skip(f"FTS database not found: {db_path}")
    
    # Test with query that should have no results
    results = lexical_search(
        db_path=str(db_path),
        query="ThisClassDefinitelyDoesNotExistInROOT12345",
        top_k=5,
    )
    
    # Should return empty list, not hallucinated results
    assert isinstance(results, list), "Results should be a list"
    # It's OK if FTS5 returns partial matches, but they must have valid citations
    for r in results:
        assert r.file_path is not None
        assert r.start_line > 0
        assert r.end_line >= r.start_line
