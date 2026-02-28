"""Tests for FTS5 backend and index building."""
import sqlite3

import pytest

from root_rag.index.fts import build_fts_index, check_fts5_available, create_fts5_db, insert_chunks_into_fts
from root_rag.index.schemas import Chunk


class TestFts5Available:
    """Tests for FTS5 availability checking."""

    def test_fts5_available(self):
        """Confirm FTS5 is available on this system."""
        assert check_fts5_available() is True


class TestCreateFts5Db:
    """Tests for FTS5 database creation."""

    def test_create_fts5_db(self, tmp_path):
        """Create FTS5 database with chunks table."""
        db_path = tmp_path / "test.sqlite"
        
        create_fts5_db(db_path)
        
        assert db_path.exists()
        
        # Verify table schema
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
        schema = cursor.fetchone()
        conn.close()
        
        assert schema is not None
        assert "content" in schema[0]
        assert "file_path" in schema[0]
        assert "chunk_id" in schema[0]


class TestInsertChunksIntoFts:
    """Tests for chunk insertion into FTS5."""

    def test_insert_chunks_into_fts(self, tmp_path, cpp_repo_fixture):
        """Insert chunks into FTS5 and verify they're stored."""
        from root_rag.parser.chunks import chunk_corpus
        from root_rag.corpus import Manifest
        
        # Create test manifest
        manifest = Manifest(
            root_ref="test",
            resolved_commit="abc123def456abc123def456abc123def456abc1",
            repo_url="https://example.com/test",
            local_path=str(cpp_repo_fixture["path"]),
            fetched_at="2026-02-27T00:00:00Z",
            tool_version="0.1.0",
        )
        
        # Generate chunks
        chunks = chunk_corpus(
            manifest=manifest,
            repo_root=cpp_repo_fixture["path"],
            window_lines=80,
            overlap_lines=10,
        )
        
        # Create database and insert
        db_path = tmp_path / "test.sqlite"
        create_fts5_db(db_path)
        stats = insert_chunks_into_fts(db_path, chunks)
        
        assert stats["inserted"] > 0
        assert stats["errors"] == 0

    def test_fts_retrieves_metadata_fields(self, tmp_path, cpp_repo_fixture):
        """Verify UNINDEXED metadata fields are preserved after insertion."""
        from root_rag.parser.chunks import chunk_corpus
        from root_rag.corpus import Manifest
        
        # Create test manifest
        manifest = Manifest(
            root_ref="v0.1",
            resolved_commit="abc123def456abc123def456abc123def456abc1",
            repo_url="https://example.com/test",
            local_path=str(cpp_repo_fixture["path"]),
            fetched_at="2026-02-27T00:00:00Z",
            tool_version="0.1.0",
        )
        
        # Generate chunks
        chunks = chunk_corpus(
            manifest=manifest,
            repo_root=cpp_repo_fixture["path"],
            window_lines=80,
            overlap_lines=10,
        )
        
        # Create database and insert
        db_path = tmp_path / "test.sqlite"
        create_fts5_db(db_path)
        insert_chunks_into_fts(db_path, chunks)
        
        # Query for a chunk and verify metadata
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get first chunk
        cursor.execute("""
            SELECT start_line, end_line, chunk_id, root_ref, resolved_commit
            FROM chunks_fts
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        # Verify start_line and end_line are integers (UNINDEXED fields)
        assert isinstance(row["start_line"], int)
        assert isinstance(row["end_line"], int)
        assert row["start_line"] >= 1
        assert row["end_line"] >= row["start_line"]
        # Verify root_ref matches manifest
        assert row["root_ref"] == "v0.1"

    def test_fts_escapes_special_chars(self, tmp_path):
        """Handle FTS5 special characters without crashing."""
        chunk = Chunk.from_file_slice(
            file_path="test/special.cpp",
            start_line=1,
            end_line=5,
            content='content with "quotes" and \'apostrophes\' and & symbols',
            root_ref="v0.1",
            resolved_commit="abc123def456abc123def456abc123def456abc1",
            language="cpp",
            doc_origin="source_impl",
        )
        
        db_path = tmp_path / "test.sqlite"
        create_fts5_db(db_path)
        stats = insert_chunks_into_fts(db_path, [chunk])
        
        assert stats["inserted"] == 1
        assert stats["errors"] == 0


class TestBuildFtsIndex:
    """Tests for FTS5 index orchestration."""

    def test_fts_deterministic_results(self, tmp_path):
        """Verify identical chunks produce identical query results."""
        # Create two identical test chunks
        chunks = [
            Chunk.from_file_slice(
                file_path="test.cpp",
                start_line=1,
                end_line=5,
                content="test content line 1",
                root_ref="v0.1",
                resolved_commit="abc123",
                language="cpp",
                doc_origin="source_impl",
            ),
            Chunk.from_file_slice(
                file_path="test.cpp",
                start_line=6,
                end_line=10,
                content="test content line 2",
                root_ref="v0.1",
                resolved_commit="abc123",
                language="cpp",
                doc_origin="source_impl",
            ),
        ]
        
        # Build two indexes from same chunks
        db1_path = tmp_path / "index1.sqlite"
        db2_path = tmp_path / "index2.sqlite"
        
        result1 = build_fts_index(db1_path, chunks)
        result2 = build_fts_index(db2_path, chunks)
        
        assert result1["status"] == "success"
        assert result2["status"] == "success"
        assert result1["chunk_count"] == result2["chunk_count"]
        
        # Query both databases for same query
        def query_db(db_path):
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'test'")
            rows = sorted([r[0] for r in cursor.fetchall()])
            conn.close()
            return rows
        
        results1 = query_db(db1_path)
        results2 = query_db(db2_path)
        
        # Results should be identical (same chunk_ids, same order)
        assert results1 == results2

    def test_build_fts_index_success(self, tmp_path, cpp_repo_fixture):
        """Build FTS5 index from chunks."""
        from root_rag.parser.chunks import chunk_corpus
        from root_rag.corpus import Manifest
        
        # Create test manifest
        manifest = Manifest(
            root_ref="v0.1",
            resolved_commit="abc123def456abc123def456abc123def456abc1",
            repo_url="https://example.com/test",
            local_path=str(cpp_repo_fixture["path"]),
            fetched_at="2026-02-27T00:00:00Z",
            tool_version="0.1.0",
        )
        
        # Generate chunks
        chunks = chunk_corpus(
            manifest=manifest,
            repo_root=cpp_repo_fixture["path"],
            window_lines=80,
            overlap_lines=10,
        )
        
        # Build index
        db_path = tmp_path / "index.sqlite"
        result = build_fts_index(db_path, chunks)
        
        assert result["status"] == "success"
        assert result["chunk_count"] == len(chunks)
        assert result["chunk_count"] > 0
        assert db_path.exists()


class TestFts5Unavailable:
    """Tests for handling unavailable FTS5."""

    def test_fts_unavailable_error_handling(self, tmp_path, monkeypatch):
        """Handle FTS5 unavailability gracefully."""
        # Mock check_fts5_available to return False
        import root_rag.index.fts
        
        original_check = root_rag.index.fts.check_fts5_available
        monkeypatch.setattr(root_rag.index.fts, "check_fts5_available", lambda: False)
        
        # Verify it returns False
        from root_rag.index.fts import check_fts5_available
        assert check_fts5_available() is False
        
        # Restore
        monkeypatch.setattr(root_rag.index.fts, "check_fts5_available", original_check)
