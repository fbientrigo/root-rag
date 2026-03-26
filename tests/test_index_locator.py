"""Tests for index locator (resolve index by ID or root_ref)."""
import pytest
from datetime import datetime

from root_rag.index.locator import resolve_index
from root_rag.core.errors import IndexNotFoundError


class TestResolveIndexById:
    """Tests for resolving index by explicit index_id."""

    def test_resolve_index_by_id_success(self, tmp_path):
        """Resolve index by explicit index_id."""
        from root_rag.index.schemas import IndexManifest
        
        # Create fake index directory structure
        indexes_root = tmp_path / "indexes"
        indexes_root.mkdir()
        
        index_dir = indexes_root / "v0.1__abc123de__20260228T000000Z"
        index_dir.mkdir()
        
        manifest = IndexManifest(
            index_id="v0.1__abc123de__20260228T000000Z",
            corpus_id="v0.1__abc123de",
            root_ref="v0.1",
            resolved_commit="abc123de" + "0" * 32,
            corpus_url="https://example.com/repo.git",
            chunks_path="data/processed/chunks/v0.1__abc123de/chunks.jsonl",
            fts_db_path=str(index_dir / "fts.sqlite"),
            chunk_count=10,
            file_count=2,
            created_at="2026-02-28T00:00:00Z",
        )
        manifest.save(index_dir / "index_manifest.json")
        
        # Resolve by ID
        resolved = resolve_index(
            indexes_root=indexes_root,
            root_ref=None,
            index_id="v0.1__abc123de__20260228T000000Z",
        )
        
        assert resolved.index_id == "v0.1__abc123de__20260228T000000Z"
        assert resolved.root_ref == "v0.1"

    def test_resolve_index_by_id_not_found(self, tmp_path):
        """Index ID not found raises IndexNotFoundError."""
        indexes_root = tmp_path / "indexes"
        indexes_root.mkdir()
        
        with pytest.raises(IndexNotFoundError):
            resolve_index(
                indexes_root=indexes_root,
                root_ref=None,
                index_id="nonexistent__index__id",
            )


class TestResolveIndexByRootRef:
    """Tests for resolving index by root_ref (pick latest)."""

    def test_resolve_index_by_root_ref_picks_latest(self, tmp_path):
        """When multiple indices exist for same corpus, pick newest by timestamp."""
        from root_rag.index.schemas import IndexManifest
        
        indexes_root = tmp_path / "indexes"
        indexes_root.mkdir()
        
        # Create two indices for same corpus_id but different timestamps
        commit = "abc123de" + "0" * 32
        
        # Older index: 2026-02-27T10:00:00Z
        index1_dir = indexes_root / "v0.1__abc123de__20260227T100000Z"
        index1_dir.mkdir()
        manifest1 = IndexManifest(
            index_id="v0.1__abc123de__20260227T100000Z",
            corpus_id="v0.1__abc123de",
            root_ref="v0.1",
            resolved_commit=commit,
            corpus_url="https://example.com/repo.git",
            chunks_path="data/processed/chunks/v0.1__abc123de/chunks.jsonl",
            fts_db_path=str(index1_dir / "fts.sqlite"),
            chunk_count=10,
            file_count=2,
            created_at="2026-02-27T10:00:00Z",
        )
        manifest1.save(index1_dir / "index_manifest.json")
        
        # Newer index: 2026-02-28T15:30:00Z
        index2_dir = indexes_root / "v0.1__abc123de__20260228T153000Z"
        index2_dir.mkdir()
        manifest2 = IndexManifest(
            index_id="v0.1__abc123de__20260228T153000Z",
            corpus_id="v0.1__abc123de",
            root_ref="v0.1",
            resolved_commit=commit,
            corpus_url="https://example.com/repo.git",
            chunks_path="data/processed/chunks/v0.1__abc123de/chunks.jsonl",
            fts_db_path=str(index2_dir / "fts.sqlite"),
            chunk_count=15,
            file_count=3,
            created_at="2026-02-28T15:30:00Z",
        )
        manifest2.save(index2_dir / "index_manifest.json")
        
        # Resolve by root_ref
        resolved = resolve_index(
            indexes_root=indexes_root,
            root_ref="v0.1",
            index_id=None,
        )
        
        # Should pick the newer one
        assert resolved.index_id == "v0.1__abc123de__20260228T153000Z"
        assert resolved.chunk_count == 15

    def test_resolve_index_by_root_ref_not_found(self, tmp_path):
        """No index for given root_ref raises IndexNotFoundError."""
        indexes_root = tmp_path / "indexes"
        indexes_root.mkdir()
        
        with pytest.raises(IndexNotFoundError):
            resolve_index(
                indexes_root=indexes_root,
                root_ref="nonexistent_version",
                index_id=None,
            )

    def test_resolve_index_prefers_explicit_id_over_root_ref(self, tmp_path):
        """When both index_id and root_ref provided, index_id takes priority."""
        from root_rag.index.schemas import IndexManifest
        
        indexes_root = tmp_path / "indexes"
        indexes_root.mkdir()
        
        commit = "abc123de" + "0" * 32
        
        # Create index with specific ID
        index_dir = indexes_root / "v0.1__abc123de__20260228T000000Z"
        index_dir.mkdir()
        manifest = IndexManifest(
            index_id="v0.1__abc123de__20260228T000000Z",
            corpus_id="v0.1__abc123de",
            root_ref="v0.1",
            resolved_commit=commit,
            corpus_url="https://example.com/repo.git",
            chunks_path="data/processed/chunks/v0.1__abc123de/chunks.jsonl",
            fts_db_path=str(index_dir / "fts.sqlite"),
            chunk_count=10,
            file_count=2,
            created_at="2026-02-28T00:00:00Z",
        )
        manifest.save(index_dir / "index_manifest.json")
        
        # Resolve with both params (explicit ID should win)
        resolved = resolve_index(
            indexes_root=indexes_root,
            root_ref="v0.1",
            index_id="v0.1__abc123de__20260228T000000Z",
        )
        
        assert resolved.index_id == "v0.1__abc123de__20260228T000000Z"
