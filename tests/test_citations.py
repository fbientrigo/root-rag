"""Tests for citation and provenance invariants (test_citations.py)."""
from pathlib import Path

import pytest

from root_rag.parser.chunks import chunk_file, chunk_corpus


def test_chunk_path_is_relative_posix(cpp_repo_fixture):
    """Test: file_path is relative and uses POSIX separators.
    
    Given: C++ repo with file at tree/inc/TTree.h
    When: file is chunked
    Then: chunk.file_path is relative (no /absolute), uses / not \
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
    
    for chunk in chunks:
        # Must not start with / or \ (not absolute)
        assert not chunk.file_path.startswith(("/", "\\")), \
            f"file_path must be relative, got: {chunk.file_path}"
        # Must use / not \
        assert "\\" not in chunk.file_path, \
            f"file_path must use POSIX separators (/) not backslash, got: {chunk.file_path}"
        # Must not have drive letter (Windows)
        assert ":" not in chunk.file_path or chunk.file_path.index(":") == 1, \
            f"file_path must not have Windows absolute path, got: {chunk.file_path}"


def test_chunk_content_is_nonempty_and_bounded(cpp_repo_fixture):
    """Test: Chunk content is not empty and within bounds.
    
    Given: C++ repo
    When: files are chunked
    Then: Every chunk has non-empty content, length < 1MB
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
    
    for chunk in chunks:
        assert len(chunk.content) > 0, "Content must not be empty"
        assert chunk.content.strip() != "", "Content must not be whitespace-only"
        assert len(chunk.content) < 1_000_000, \
            f"Content exceeds 1MB: {len(chunk.content)} bytes"


def test_chunks_do_not_mix_files(cpp_repo_fixture):
    """Test: Each chunk comes from exactly one file.
    
    Given: C++ repo with multiple files
    When: single file is chunked
    Then: All chunks from that call have same file_path
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
    
    expected_path = tree_h.relative_to(repo_path).as_posix()
    
    for chunk in chunks:
        assert chunk.file_path == expected_path, \
            f"Chunk file_path mismatch: {chunk.file_path} vs {expected_path}"


def test_chunks_do_not_mix_versions(cpp_repo_fixture):
    """Test: All chunks in one build have same root_ref + resolved_commit.
    
    Given: C++ repo with manifest
    When: chunk_corpus is called
    Then: Every chunk has root_ref and resolved_commit matching manifest
    """
    from root_rag.corpus import Manifest
    
    repo_path = cpp_repo_fixture["path"]
    resolved_commit = cpp_repo_fixture["resolved_commit"]
    root_ref = cpp_repo_fixture["root_ref"]
    
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
    
    for chunk in chunks:
        assert chunk.root_ref == root_ref, \
            f"Version mismatch in chunk: {chunk.root_ref} vs {root_ref}"
        assert chunk.resolved_commit == resolved_commit, \
            f"Commit mismatch in chunk: {chunk.resolved_commit} vs {resolved_commit}"


def test_doxygen_detection_sets_flag(cpp_repo_fixture):
    """Test: Chunks containing Doxygen markers have has_doxygen=True.
    
    Given: C++ file with Doxygen comments
    When: file is chunked
    Then: Chunks containing /** or //! have has_doxygen=True
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
        window_lines=50,  # Large window to likely catch Doxygen
        overlap_lines=5,
    )
    
    # tree.h has Doxygen comments at the start
    doxygen_chunks = [c for c in chunks if c.has_doxygen]
    
    assert len(doxygen_chunks) > 0, \
        "Should detect at least one chunk with Doxygen markers"
    
    # Verify those chunks actually contain Doxygen markers
    for chunk in doxygen_chunks:
        assert "/**" in chunk.content or "//!" in chunk.content, \
            f"Chunk marked has_doxygen=True but contains no markers: {chunk.content[:50]}"


def test_chunk_fields_no_cross_contamination(cpp_repo_fixture):
    """Test: Chunk metadata is correct per file (no cross-contamination).
    
    Given: Multiple files with different extensions (.h and .cxx)
    When: chunk_corpus is called
    Then: Language and doc_origin are correct for each
    """
    from root_rag.corpus import Manifest
    
    repo_path = cpp_repo_fixture["path"]
    resolved_commit = cpp_repo_fixture["resolved_commit"]
    root_ref = cpp_repo_fixture["root_ref"]
    
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
    
    # Separate by file
    header_chunks = [c for c in chunks if c.file_path.endswith(".h")]
    impl_chunks = [c for c in chunks if c.file_path.endswith(".cxx")]
    
    # Headers should have doc_origin=source_header
    for chunk in header_chunks:
        assert chunk.doc_origin == "source_header", \
            f"Header file should have doc_origin=source_header, got {chunk.doc_origin}"
        assert chunk.language == "cpp"
    
    # Implementation should have doc_origin=source_impl
    for chunk in impl_chunks:
        assert chunk.doc_origin == "source_impl", \
            f"Impl file should have doc_origin=source_impl, got {chunk.doc_origin}"
        assert chunk.language == "cpp"
