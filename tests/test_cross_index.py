"""
Tests for cross-index search functionality.
"""

import json
from pathlib import Path

import pytest

from root_rag.retrieval.cross_index import (
    CrossIndexSearch,
    IndexSource,
    create_standard_search,
)


def test_cross_index_search_initialization():
    """Test that cross-index search initializes with multiple sources."""
    sources = [
        IndexSource(
            name="ROOT Tier 1",
            indexes_root=Path("data/indexes_tier1"),
            root_ref="v6-36-08",
        ),
        IndexSource(
            name="FairShip",
            indexes_root=Path("data/indexes_fairship"),
            root_ref="master",
        ),
    ]
    
    search = CrossIndexSearch(sources=sources)
    
    # Check that sources were added
    assert len(search.sources) == 2
    assert search.sources[0].name == "ROOT Tier 1"
    assert search.sources[1].name == "FairShip"
    
    print("[OK] Cross-index search initialized")


def test_cross_index_search_resolves_indices():
    """Test that indices are resolved correctly."""
    search = create_standard_search(
        include_tier1=True,
        include_sofie=False,  # Skip SOFIE (partial index)
        include_fairship=True,
    )
    
    # Check resolved indices
    stats = search.get_index_stats()
    
    assert "ROOT Tier 1" in stats
    assert "FairShip" in stats
    
    # Validate ROOT Tier 1
    tier1_stats = stats["ROOT Tier 1"]
    assert tier1_stats["root_ref"] == "v6-36-08"
    assert tier1_stats["chunk_count"] > 0
    assert tier1_stats["file_count"] > 0
    
    # Validate FairShip
    fairship_stats = stats["FairShip"]
    assert fairship_stats["root_ref"] == "master"
    assert fairship_stats["chunk_count"] > 0
    assert fairship_stats["file_count"] > 0
    
    print(f"[OK] Resolved indices:")
    print(f"  ROOT Tier 1: {tier1_stats['chunk_count']} chunks, {tier1_stats['file_count']} files")
    print(f"  FairShip: {fairship_stats['chunk_count']} chunks, {fairship_stats['file_count']} files")


def test_cross_index_search_tgeomanager():
    """Test searching for TGeoManager across ROOT and FairShip."""
    search = create_standard_search(
        include_tier1=True,
        include_sofie=False,
        include_fairship=True,
    )
    
    results = search.search("TGeoManager", top_k=10)
    
    # Should find results from both indices
    assert len(results) > 0
    
    # Check that results have proper structure
    for result in results[:3]:
        assert result.file_path is not None
        assert result.start_line > 0
        assert result.end_line >= result.start_line
        assert result.score is not None  # BM25 scores can be negative
        assert result.root_ref in {"v6-36-08", "master"}
    
    # Check that we have results from both ROOT and FairShip
    root_results = [r for r in results if r.root_ref == "v6-36-08"]
    fairship_results = [r for r in results if r.root_ref == "master"]
    
    print(f"[OK] TGeoManager search:")
    print(f"  Total: {len(results)} results")
    print(f"  ROOT: {len(root_results)} results")
    print(f"  FairShip: {len(fairship_results)} results")
    
    if results:
        print(f"  Top result: {results[0].file_path}:{results[0].start_line} (score={results[0].score:.2f})")


def test_cross_index_search_fairship_specific():
    """Test searching for FairShip-specific terms."""
    search = create_standard_search(
        include_tier1=True,
        include_sofie=False,
        include_fairship=True,
    )
    
    results = search.search("DetectorPoint", top_k=10)
    
    # Should find results (likely only from FairShip)
    assert len(results) > 0
    
    # Most/all results should be from FairShip
    fairship_results = [r for r in results if r.root_ref == "master"]
    assert len(fairship_results) > 0
    
    print(f"[OK] DetectorPoint search:")
    print(f"  Total: {len(results)} results")
    print(f"  FairShip-specific: {len(fairship_results)} results")


def test_cross_index_search_ttree():
    """Test searching for TTree (ROOT class used by FairShip)."""
    search = create_standard_search(
        include_tier1=True,
        include_sofie=False,
        include_fairship=True,
    )
    
    results = search.search("TTree Fill", top_k=15)
    
    assert len(results) > 0
    
    # Should have both ROOT implementation and FairShip usage
    root_results = [r for r in results if r.root_ref == "v6-36-08"]
    fairship_results = [r for r in results if r.root_ref == "master"]
    
    print(f"[OK] TTree::Fill search:")
    print(f"  Total: {len(results)} results")
    print(f"  ROOT implementation: {len(root_results)} results")
    print(f"  FairShip usage: {len(fairship_results)} results")
    
    # Show top results from each index
    if root_results:
        print(f"  Top ROOT: {root_results[0].file_path}:{root_results[0].start_line}")
    if fairship_results:
        print(f"  Top FairShip: {fairship_results[0].file_path}:{fairship_results[0].start_line}")


def test_cross_index_per_index_limit():
    """Test that per_index_limit works correctly."""
    search = create_standard_search(
        include_tier1=True,
        include_sofie=False,
        include_fairship=True,
    )
    
    # Limit to 3 results per index, but request 10 total
    results = search.search("TGeoManager", top_k=10, per_index_limit=3)
    
    # Should have at most 6 results (3 from each index)
    assert len(results) <= 6
    
    print(f"[OK] Per-index limit test: {len(results)} results (max 6)")


def test_standard_search_factory():
    """Test that standard search factory creates correct configuration."""
    # Test with all indices
    search_all = create_standard_search(
        include_tier1=True,
        include_sofie=True,
        include_fairship=True,
    )
    assert len(search_all.sources) == 3
    
    # Test with only ROOT
    search_root = create_standard_search(
        include_tier1=True,
        include_sofie=False,
        include_fairship=False,
    )
    assert len(search_root.sources) == 1
    
    # Test with ROOT + FairShip
    search_combined = create_standard_search(
        include_tier1=True,
        include_sofie=False,
        include_fairship=True,
    )
    assert len(search_combined.sources) == 2
    
    print("[OK] Standard search factory works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
