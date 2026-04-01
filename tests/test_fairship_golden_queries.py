"""
Tests for FairShip golden queries using cross-index search.

These tests validate that our ROOT + FairShip indices can answer
real-world questions about FairShip usage patterns.
"""

import json
from pathlib import Path

import pytest

from root_rag.retrieval.cross_index import create_standard_search


@pytest.fixture
def fairship_golden_queries():
    """Load FairShip golden queries from config."""
    config_path = Path("configs/fairship_golden_queries.json")
    with open(config_path) as f:
        data = json.load(f)
    return data["fairship_golden_queries"]


@pytest.fixture
def cross_search():
    """Create cross-index search with ROOT + FairShip."""
    return create_standard_search(
        include_tier1=True,
        include_sofie=False,
        include_fairship=True,
    )


def test_fairship_detector_hit_query(cross_search, fairship_golden_queries):
    """Test query: How to implement a detector hit class in FairShip?"""
    query = next(q for q in fairship_golden_queries if q["query_id"] == "fs_01_detector_hit")
    
    results = cross_search.search(query["query"], top_k=10)
    
    assert len(results) > 0, "Should find detector hit examples"
    
    # Should have FairShip results (likely DetectorHit classes)
    fairship_results = [r for r in results if r.root_ref == "master"]
    assert len(fairship_results) > 0, "Should find FairShip detector hit code"
    
    # Check for expected file patterns
    hit_files = [r for r in fairship_results if "Hit" in r.file_path]
    assert len(hit_files) > 0, "Should find *Hit.* files"
    
    print(f"[OK] Detector hit query:")
    print(f"  Total: {len(results)} results")
    print(f"  FairShip: {len(fairship_results)} results")
    print(f"  Hit files: {len(hit_files)} results")
    if hit_files:
        print(f"  Example: {hit_files[0].file_path}:{hit_files[0].start_line}")


def test_fairship_tgeomanager_query(cross_search, fairship_golden_queries):
    """Test query: How does FairShip use TGeoManager?"""
    query = next(q for q in fairship_golden_queries if q["query_id"] == "fs_02_tgeomanager_usage")
    
    results = cross_search.search(query["query"], top_k=15)
    
    assert len(results) > 0, "Should find TGeoManager usage"
    
    # Should have results (may be from ROOT or FairShip)
    root_results = [r for r in results if r.root_ref == "v6-36-08"]
    fairship_results = [r for r in results if r.root_ref == "master"]
    
    print(f"[OK] TGeoManager query:")
    print(f"  Total: {len(results)} results")
    print(f"  ROOT: {len(root_results)} results (API definition)")
    print(f"  FairShip: {len(fairship_results)} results (actual usage)")


def test_fairship_ttree_fill_query(cross_search, fairship_golden_queries):
    """Test query: How does FairShip save data to TTree?"""
    query = next(q for q in fairship_golden_queries if q["query_id"] == "fs_03_ttree_fill")
    
    results = cross_search.search(query["query"], top_k=15)
    
    assert len(results) > 0, "Should find TTree usage"
    
    # Should have both ROOT (TTree implementation) and FairShip (TTree usage)
    root_results = [r for r in results if r.root_ref == "v6-36-08"]
    fairship_results = [r for r in results if r.root_ref == "master"]
    
    print(f"[OK] TTree::Fill query:")
    print(f"  Total: {len(results)} results")
    print(f"  ROOT: {len(root_results)} results")
    print(f"  FairShip: {len(fairship_results)} results")


def test_fairship_tvector3_query(cross_search, fairship_golden_queries):
    """Test query: TVector3 physics calculations in FairShip"""
    query = next(q for q in fairship_golden_queries if q["query_id"] == "fs_04_tvector3_physics")
    
    results = cross_search.search(query["query"], top_k=15)
    
    assert len(results) > 0, "Should find TVector3 usage"
    
    fairship_results = [r for r in results if r.root_ref == "master"]
    
    print(f"[OK] TVector3 physics query:")
    print(f"  Total: {len(results)} results")
    print(f"  FairShip: {len(fairship_results)} results")


def test_fairship_detector_point_query(cross_search, fairship_golden_queries):
    """Test query: What is DetectorPoint in FairShip?"""
    query = next(q for q in fairship_golden_queries if q["query_id"] == "fs_05_detector_point")
    
    results = cross_search.search(query["query"], top_k=10)
    
    assert len(results) > 0, "Should find Point-related code"
    
    # FairShip likely has Point-related files
    fairship_results = [r for r in results if r.root_ref == "master"]
    
    print(f"[OK] DetectorPoint query:")
    print(f"  Total: {len(results)} results")
    print(f"  FairShip: {len(fairship_results)} results")
    if fairship_results:
        print(f"  Example: {fairship_results[0].file_path}:{fairship_results[0].start_line}")


def test_fairship_geometry_construction_query(cross_search, fairship_golden_queries):
    """Test query: FairShip detector geometry construction workflow"""
    query = next(q for q in fairship_golden_queries if q["query_id"] == "fs_06_geometry_construction")
    
    results = cross_search.search(query["query"], top_k=15)
    
    assert len(results) > 0, "Should find geometry construction code"
    
    fairship_results = [r for r in results if r.root_ref == "master"]
    assert len(fairship_results) > 0, "Should find FairShip geometry code"
    
    print(f"[OK] Geometry construction query:")
    print(f"  Total: {len(results)} results")
    print(f"  FairShip: {len(fairship_results)} results")


def test_fairship_root_io_pattern_query(cross_search, fairship_golden_queries):
    """Test query: FairShip ROOT I/O patterns for event storage"""
    query = next(q for q in fairship_golden_queries if q["query_id"] == "fs_07_root_io_pattern")
    
    results = cross_search.search(query["query"], top_k=15)
    
    assert len(results) > 0, "Should find I/O patterns"
    
    # Should have both ROOT (TFile/TTree API) and FairShip (usage)
    root_results = [r for r in results if r.root_ref == "v6-36-08"]
    fairship_results = [r for r in results if r.root_ref == "master"]
    
    print(f"[OK] ROOT I/O pattern query:")
    print(f"  Total: {len(results)} results")
    print(f"  ROOT: {len(root_results)} results")
    print(f"  FairShip: {len(fairship_results)} results")


def test_all_golden_queries_return_results(cross_search, fairship_golden_queries):
    """Test that all golden queries return at least some results."""
    failures = []
    
    for query_config in fairship_golden_queries:
        query_id = query_config["query_id"]
        query_text = query_config["query"]
        
        results = cross_search.search(query_text, top_k=10)
        
        if len(results) == 0:
            failures.append(query_id)
    
    assert len(failures) == 0, f"Queries returned no results: {failures}"
    
    print(f"[OK] All {len(fairship_golden_queries)} golden queries returned results")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
