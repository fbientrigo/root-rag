"""Tests for chunking logic (test_chunking.py)."""
from pathlib import Path

import pytest

from root_rag.index.schemas import Chunk
from root_rag.parser.chunks import chunk_corpus, chunk_file


def test_chunk_invariants_line_ranges(cpp_repo_fixture):
    """Test: Chunk line ranges follow invariants.
    
    Given: C++ repo with files
    When: File is chunked
    Then:
      - start_line >= 1
      - end_line >= start_line
      - end_line <= total lines
    """
    repo_path = cpp_repo_fixture["path"]
    resolved_commit = cpp_repo_fixture["resolved_commit"]
    root_ref = cpp_repo_fixture["root_ref"]
    
    tree_h = cpp_repo_fixture["files"]["tree.h"]
    
    chunks = chunk_file(
        file_path=tree_h,
        root_ref=root_ref,
        resolved_commit=resolved_commit,
        repo_root=repo_path,
        window_lines=10,
        overlap_lines=2,
    )
    
    assert len(chunks) > 0, "Should produce at least one chunk"
    
    # Read file to get total lines
    lines = tree_h.read_text().splitlines()
    total_lines = len(lines)
    
    for chunk in chunks:
        assert chunk.start_line >= 1, f"start_line must be >= 1, got {chunk.start_line}"
        assert chunk.end_line >= chunk.start_line, \
            f"end_line ({chunk.end_line}) must be >= start_line ({chunk.start_line})"
        assert chunk.end_line <= total_lines, \
            f"end_line ({chunk.end_line}) must be <= total_lines ({total_lines})"


def test_chunk_content_matches_exact_file_lines(cpp_repo_fixture):
    """Test: Chunk content exactly matches lines from file.
    
    Given: C++ file and computed chunks
    When: Chunk content is compared to file lines
    Then: Content exactly matches lines [start_line:end_line]
    """
    repo_path = cpp_repo_fixture["path"]
    resolved_commit = cpp_repo_fixture["resolved_commit"]
    root_ref = cpp_repo_fixture["root_ref"]
    
    tree_h = cpp_repo_fixture["files"]["tree.h"]
    
    chunks = chunk_file(
        file_path=tree_h,
        root_ref=root_ref,
        resolved_commit=resolved_commit,
        repo_root=repo_path,
        window_lines=10,
        overlap_lines=2,
    )
    
    # Read file lines
    lines = tree_h.read_text().splitlines()
    
    for chunk in chunks:
        # Extract expected content from file (convert to 0-indexed)
        start_idx = chunk.start_line - 1
        end_idx = chunk.end_line  # Inclusive, so don't subtract 1
        expected_content = "\n".join(lines[start_idx:end_idx])
        
        assert chunk.content == expected_content, \
            f"Chunk content mismatch for lines {chunk.start_line}-{chunk.end_line}"


def test_chunk_has_required_fields(cpp_repo_fixture):
    """Test: All required fields are present and non-empty.
    
    Given: C++ repo and chunking function
    When: chunk_file is called
    Then: Each chunk has all 10 required fields with valid values
    """
    repo_path = cpp_repo_fixture["path"]
    resolved_commit = cpp_repo_fixture["resolved_commit"]
    root_ref = cpp_repo_fixture["root_ref"]
    
    tree_h = cpp_repo_fixture["files"]["tree.h"]
    
    chunks = chunk_file(
        file_path=tree_h,
        root_ref=root_ref,
        resolved_commit=resolved_commit,
        repo_root=repo_path,
    )
    
    required_fields = [
        "chunk_id",
        "root_ref",
        "resolved_commit",
        "file_path",
        "language",
        "start_line",
        "end_line",
        "content",
        "doc_origin",
        "index_schema_version",
    ]
    
    for chunk in chunks:
        for field in required_fields:
            value = getattr(chunk, field, None)
            assert value is not None, f"Field {field} is None"
            assert str(value).strip() != "", f"Field {field} is empty string"


def test_chunk_id_is_deterministic(cpp_repo_fixture):
    """Test: Chunk IDs are deterministic (same input → same ID).
    
    Given: C++ repo
    When: Same file is chunked twice
    Then: Generated chunk IDs are identical
    """
    repo_path = cpp_repo_fixture["path"]
    resolved_commit = cpp_repo_fixture["resolved_commit"]
    root_ref = cpp_repo_fixture["root_ref"]
    
    tree_h = cpp_repo_fixture["files"]["tree.h"]
    
    # First chunking
    chunks1 = chunk_file(
        file_path=tree_h,
        root_ref=root_ref,
        resolved_commit=resolved_commit,
        repo_root=repo_path,
        window_lines=20,
        overlap_lines=5,
    )
    
    # Second chunking (identical parameters)
    chunks2 = chunk_file(
        file_path=tree_h,
        root_ref=root_ref,
        resolved_commit=resolved_commit,
        repo_root=repo_path,
        window_lines=20,
        overlap_lines=5,
    )
    
    assert len(chunks1) == len(chunks2), "Chunk count changed"
    
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.chunk_id == c2.chunk_id, \
            f"Chunk IDs differ: {c1.chunk_id} vs {c2.chunk_id}"


def test_chunk_respects_version_metadata(cpp_repo_fixture):
    """Test: Every chunk carries version metadata.
    
    Given: C++ repo with manifest data
    When: chunk_corpus is called
    Then: Every chunk has root_ref and resolved_commit matching manifest
    """
    from root_rag.corpus import Manifest
    
    repo_path = cpp_repo_fixture["path"]
    resolved_commit = cpp_repo_fixture["resolved_commit"]
    root_ref = cpp_repo_fixture["root_ref"]
    
    # Create a mock manifest
    manifest = Manifest(
        repo_url="file://test",
        root_ref=root_ref,
        resolved_commit=resolved_commit,
        local_path=str(repo_path),
        fetched_at="2026-02-27T00:00:00Z",
        dirty=False,
        tool_version="0.0.1",
    )
    
    chunks = chunk_corpus(manifest, repo_path)
    
    assert len(chunks) > 0, "Should produce chunks"
    
    for chunk in chunks:
        assert chunk.root_ref == root_ref, \
            f"Root ref mismatch: {chunk.root_ref} vs {root_ref}"
        assert chunk.resolved_commit == resolved_commit, \
            f"Commit mismatch: {chunk.resolved_commit} vs {resolved_commit}"


def test_file_discovery_sorted_order(cpp_repo_fixture):
    """Test: File discovery returns sorted paths (deterministic).
    
    Given: C++ repo with multiple files
    When: discover_text_files is called twice
    Then: File order is identical both times (sorted)
    """
    from root_rag.parser.files import discover_text_files
    
    repo_path = cpp_repo_fixture["path"]
    
    files1 = discover_text_files(repo_path)
    files2 = discover_text_files(repo_path)
    
    assert files1 == files2, "File discovery order changed"
    
    # Verify sorted
    files_str = [str(f) for f in files1]
    assert files_str == sorted(files_str), "Files are not sorted"


def test_chunking_respects_window_size(cpp_repo_fixture):
    """Test: Chunks respect specified window size.
    
    Given: C++ file and explicit window_lines=10
    When: chunk_file is called
    Then: Most chunks are ~10 lines (last may be smaller)
    """
    repo_path = cpp_repo_fixture["path"]
    resolved_commit = cpp_repo_fixture["resolved_commit"]
    root_ref = cpp_repo_fixture["root_ref"]
    
    tree_cxx = cpp_repo_fixture["files"]["tree.cxx"]
    
    window_lines = 10
    overlap_lines = 2
    
    chunks = chunk_file(
        file_path=tree_cxx,
        root_ref=root_ref,
        resolved_commit=resolved_commit,
        repo_root=repo_path,
        window_lines=window_lines,
        overlap_lines=overlap_lines,
    )
    
    assert len(chunks) > 0, "Should produce chunks"
    
    # Check chunk sizes
    for i, chunk in enumerate(chunks):
        chunk_size = chunk.end_line - chunk.start_line + 1
        # Most should be window_lines, last may be smaller
        if i < len(chunks) - 1:
            assert chunk_size <= window_lines, \
                f"Chunk {i} exceeds window: {chunk_size} > {window_lines}"
