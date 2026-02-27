"""Tests for corpus fetcher (fetch command)."""
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from root_rag.corpus import fetch_corpus, InvalidRefError, Manifest


def test_fetch_writes_manifest_for_tag(tmp_path, git_repo_fixture):
    """Test: fetch resolves tag and writes manifest.
    
    Given: repo fixture with tag v0.1
    When: fetch_corpus is called with root_ref="v0.1"
    Then:
      - Manifest file is created
      - root_ref matches requested tag
      - resolved_commit matches tag's SHA
      - local_path exists and contains .git
    """
    repo_url = str(git_repo_fixture["path"])
    cache_dir = tmp_path / "cache"
    
    # Act
    manifest = fetch_corpus(
        repo_url=repo_url,
        root_ref="v0.1",
        cache_dir=cache_dir,
    )
    
    # Assert
    assert manifest.root_ref == "v0.1"
    assert manifest.resolved_commit == git_repo_fixture["tag_sha"]
    assert manifest.dirty is False
    
    local_path = Path(manifest.local_path)
    assert local_path.exists()
    assert (local_path / ".git").exists()
    assert (local_path / "A.h").exists()
    
    # Check manifest was persisted
    corpus_id = f"{git_repo_fixture['path'].name}__v0.1__{manifest.resolved_commit[:12]}"
    manifest_file = cache_dir / corpus_id / "manifest.json"
    # Note: actual corpus_id format might differ, but file should exist
    assert any((cache_dir / d / "manifest.json").exists() for d in cache_dir.iterdir() if d.is_dir())


def test_fetch_invalid_ref_raises_error(tmp_path, git_repo_fixture):
    """Test: fetch with invalid ref raises InvalidRefError.
    
    Given: repo fixture without ref "DOES_NOT_EXIST"
    When: fetch_corpus is called with root_ref="DOES_NOT_EXIST"
    Then: InvalidRefError is raised
    """
    repo_url = str(git_repo_fixture["path"])
    cache_dir = tmp_path / "cache"
    
    # Act & Assert
    with pytest.raises(InvalidRefError):
        fetch_corpus(
            repo_url=repo_url,
            root_ref="DOES_NOT_EXIST",
            cache_dir=cache_dir,
        )


def test_fetch_invalid_ref_exit_code_3(tmp_path, git_repo_fixture):
    """Test: CLI returns exit code 3 for invalid ref.
    
    Given: repo fixture without ref "DOES_NOT_EXIST"
    When: root-rag fetch is called with invalid ref
    Then: CLI exits with code 3
    """
    repo_url = str(git_repo_fixture["path"])
    cache_dir = tmp_path / "cache"
    
    result = subprocess.run(
        [
            "root-rag",
            "fetch",
            "--root-ref", "DOES_NOT_EXIST",
            "--repo-url", repo_url,
            "--cache-dir", str(cache_dir),
        ],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 3


def test_fetch_idempotent_same_ref(tmp_path, git_repo_fixture):
    """Test: fetching same ref twice is idempotent.
    
    Given: repo fixture with tag v0.1
    When: fetch_corpus is called twice with root_ref="v0.1"
    Then:
      - Second call returns cached manifest
      - resolved_commit is identical
      - fetched_at times may differ (depends on timestamp)
    """
    repo_url = str(git_repo_fixture["path"])
    cache_dir = tmp_path / "cache"
    
    # First fetch
    manifest1 = fetch_corpus(
        repo_url=repo_url,
        root_ref="v0.1",
        cache_dir=cache_dir,
    )
    fetched_at_1 = manifest1.fetched_at
    resolved_commit_1 = manifest1.resolved_commit
    
    # Second fetch (should use cache)
    manifest2 = fetch_corpus(
        repo_url=repo_url,
        root_ref="v0.1",
        cache_dir=cache_dir,
    )
    fetched_at_2 = manifest2.fetched_at
    resolved_commit_2 = manifest2.resolved_commit
    
    # Assert
    assert resolved_commit_1 == resolved_commit_2
    # For idempotent cache, fetched_at should be the same (loaded from disk)
    assert fetched_at_1 == fetched_at_2


def test_manifest_pydantic_roundtrip(tmp_path):
    """Test: Manifest can be serialized and deserialized.
    
    Given: a Manifest with valid fields
    When: save then load the manifest
    Then: roundtrip preserves all fields exactly
    """
    manifest_path = tmp_path / "manifest.json"
    
    original = Manifest(
        repo_url="https://github.com/test/repo.git",
        root_ref="v1.0.0",
        resolved_commit="abc123def456abc123def456abc123def456abc1",
        local_path="/tmp/test/repo",
        fetched_at="2026-02-27T10:30:00+00:00",
        dirty=False,
        tool_version="0.0.1",
    )
    
    # Save
    original.save(manifest_path)
    
    # Load
    loaded = Manifest.load(manifest_path)
    
    # Assert roundtrip
    assert loaded.repo_url == original.repo_url
    assert loaded.root_ref == original.root_ref
    assert loaded.resolved_commit == original.resolved_commit
    assert loaded.local_path == original.local_path
    assert loaded.fetched_at == original.fetched_at
    assert loaded.dirty == original.dirty
    assert loaded.tool_version == original.tool_version


def test_fetch_branch_ref(tmp_path, git_repo_fixture):
    """Test: fetch resolves branch reference correctly.
    
    Given: repo fixture with branch dev
    When: fetch_corpus is called with root_ref="dev"
    Then:
      - resolved_commit matches dev branch SHA
      - Manifest has dev branch's files
    """
    repo_url = str(git_repo_fixture["path"])
    cache_dir = tmp_path / "cache"
    
    manifest = fetch_corpus(
        repo_url=repo_url,
        root_ref="dev",
        cache_dir=cache_dir,
    )
    
    assert manifest.root_ref == "dev"
    assert manifest.resolved_commit == git_repo_fixture["dev_sha"]
    
    local_path = Path(manifest.local_path)
    assert (local_path / "B.h").exists()  # dev-specific file


def test_fetch_success_exit_code_0(tmp_path, git_repo_fixture):
    """Test: successful fetch returns CLI exit code 0.
    
    Given: repo fixture with tag v0.1
    When: root-rag fetch is called successfully
    Then: CLI exits with code 0
    """
    repo_url = str(git_repo_fixture["path"])
    cache_dir = tmp_path / "cache"
    
    result = subprocess.run(
        [
            "root-rag",
            "fetch",
            "--root-ref", "v0.1",
            "--repo-url", repo_url,
            "--cache-dir", str(cache_dir),
        ],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Corpus fetched" in result.stdout or "✓" in result.stdout


def test_manifest_validates_commit_sha():
    """Test: Manifest validates resolved_commit format.
    
    Given: invalid commit SHA (too short, non-hex)
    When: Manifest is created
    Then: ValidationError is raised
    """
    from pydantic import ValidationError
    
    # Too short
    with pytest.raises(ValidationError):
        Manifest(
            repo_url="https://example.com/repo.git",
            root_ref="v1",
            resolved_commit="abc",
            local_path="/tmp",
            fetched_at="2026-02-27T10:30:00+00:00",
            dirty=False,
            tool_version="0.0.1",
        )
    
    # Non-hex
    with pytest.raises(ValidationError):
        Manifest(
            repo_url="https://example.com/repo.git",
            root_ref="v1",
            resolved_commit="ghijklmnopqrstuvwxyz0123456789012345",
            local_path="/tmp",
            fetched_at="2026-02-27T10:30:00+00:00",
            dirty=False,
            tool_version="0.0.1",
        )
