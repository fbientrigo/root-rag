"""
Tests for FairShip corpus indexing and retrieval.
"""

import json
import sqlite3
from pathlib import Path

import pytest


def test_fairship_index_exists():
    """Test that FairShip index was created."""
    index_dir = Path("data/indexes_fairship")
    
    # Find the latest index manifest
    manifests = list(index_dir.glob("*/index_manifest.json"))
    assert len(manifests) > 0, "No FairShip index manifests found"
    
    latest_manifest = max(manifests, key=lambda p: p.stat().st_mtime)
    manifest_data = json.loads(latest_manifest.read_text())
    
    # Validate manifest
    assert manifest_data["root_ref"] == "master"
    assert len(manifest_data["resolved_commit"]) >= 12
    assert manifest_data["chunk_count"] > 0
    assert manifest_data["file_count"] > 0
    
    print(f"✓ FairShip index: {manifest_data['chunk_count']} chunks, {manifest_data['file_count']} files")


def test_fairship_fts_database():
    """Test that FairShip FTS5 database is queryable."""
    index_dir = Path("data/indexes_fairship")
    
    # Find the latest FTS database
    fts_dbs = list(index_dir.glob("*/fts.sqlite"))
    assert len(fts_dbs) > 0, "No FairShip FTS databases found"
    
    latest_fts = max(fts_dbs, key=lambda p: p.stat().st_mtime)
    
    # Query the database
    conn = sqlite3.connect(latest_fts)
    cursor = conn.cursor()
    
    # Check FTS5 table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
    assert cursor.fetchone() is not None, "FTS5 table 'chunks_fts' not found"
    
    # Count chunks
    cursor.execute("SELECT COUNT(*) FROM chunks_fts")
    chunk_count = cursor.fetchone()[0]
    assert chunk_count > 0, "No chunks in FTS5 database"
    
    print(f"✓ FTS5 database: {chunk_count} chunks indexed")
    
    conn.close()


def test_fairship_search_tgeomanager():
    """Test searching for TGeoManager usage in FairShip."""
    index_dir = Path("data/indexes_fairship")
    fts_dbs = list(index_dir.glob("*/fts.sqlite"))
    assert len(fts_dbs) > 0
    
    latest_fts = max(fts_dbs, key=lambda p: p.stat().st_mtime)
    
    conn = sqlite3.connect(latest_fts)
    cursor = conn.cursor()
    
    # Search for TGeoManager
    cursor.execute("""
        SELECT file_path, start_line, end_line
        FROM chunks_fts
        WHERE chunks_fts MATCH 'TGeoManager'
        LIMIT 5
    """)
    
    results = cursor.fetchall()
    assert len(results) > 0, "No TGeoManager results found in FairShip"
    
    print(f"✓ Found {len(results)} TGeoManager matches in FairShip:")
    for file_path, start, end in results[:3]:
        print(f"  {file_path}:{start}-{end}")
    
    conn.close()


def test_fairship_search_detector():
    """Test searching for detector-related code in FairShip."""
    index_dir = Path("data/indexes_fairship")
    fts_dbs = list(index_dir.glob("*/fts.sqlite"))
    assert len(fts_dbs) > 0
    
    latest_fts = max(fts_dbs, key=lambda p: p.stat().st_mtime)
    
    conn = sqlite3.connect(latest_fts)
    cursor = conn.cursor()
    
    # Search for detector classes
    cursor.execute("""
        SELECT DISTINCT file_path
        FROM chunks_fts
        WHERE chunks_fts MATCH 'detector OR DetectorPoint OR DetectorHit'
        LIMIT 10
    """)
    
    results = cursor.fetchall()
    assert len(results) > 0, "No detector-related code found"
    
    print(f"✓ Found {len(results)} detector-related files:")
    for (file_path,) in results[:5]:
        print(f"  {file_path}")
    
    conn.close()


def test_fairship_chunks_metadata():
    """Test that FairShip chunks have proper metadata."""
    index_dir = Path("data/indexes_fairship")
    fts_dbs = list(index_dir.glob("*/fts.sqlite"))
    assert len(fts_dbs) > 0
    
    latest_fts = max(fts_dbs, key=lambda p: p.stat().st_mtime)
    
    conn = sqlite3.connect(latest_fts)
    cursor = conn.cursor()
    
    # Check metadata columns
    cursor.execute("""
        SELECT file_path, start_line, end_line, root_ref, resolved_commit, language
        FROM chunks_fts
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    assert row is not None
    
    file_path, start_line, end_line, root_ref, resolved_commit, language = row
    
    # Validate metadata
    assert file_path is not None and len(file_path) > 0
    assert start_line > 0
    assert end_line >= start_line
    assert root_ref == "master"
    assert len(resolved_commit) >= 12
    assert language in {"c", "cpp"}
    
    print(f"✓ Chunk metadata valid: {file_path}:{start_line}-{end_line} ({language})")
    
    conn.close()


def test_fairship_coverage():
    """Test that FairShip index covers expected file types."""
    index_dir = Path("data/indexes_fairship")
    fts_dbs = list(index_dir.glob("*/fts.sqlite"))
    assert len(fts_dbs) > 0
    
    latest_fts = max(fts_dbs, key=lambda p: p.stat().st_mtime)
    
    conn = sqlite3.connect(latest_fts)
    cursor = conn.cursor()
    
    # Count files by extension
    cursor.execute("""
        SELECT DISTINCT file_path
        FROM chunks_fts
    """)
    
    files = [row[0] for row in cursor.fetchall()]
    
    # Check we have both headers and implementations
    headers = [f for f in files if f.endswith(('.h', '.hpp', '.hh'))]
    impls = [f for f in files if f.endswith(('.cxx', '.cpp', '.cc', '.c'))]
    
    assert len(headers) > 0, "No header files found"
    assert len(impls) > 0, "No implementation files found"
    
    print(f"✓ Coverage: {len(headers)} headers, {len(impls)} implementations")
    
    conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
