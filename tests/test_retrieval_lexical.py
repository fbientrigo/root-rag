"""Tests for lexical retrieval from FTS5 index."""
import pytest

from root_rag.retrieval.lexical import lexical_search
from root_rag.index.fts import create_fts5_db, insert_chunks_into_fts
from root_rag.index.schemas import Chunk


class TestLexicalSearchReturnsRankedResults:
    """Tests for basic lexical search retrieval."""

    def test_lexical_search_returns_ranked_results(self, tmp_path):
        """Query returns ranked results with required fields."""
        # Create test chunks
        chunks = [
            Chunk.from_file_slice(
                file_path="tree/inc/TTree.h",
                start_line=10,
                end_line=15,
                content="virtual Long64_t Draw(const char* expression);",
                root_ref="v0.1",
                resolved_commit="abc123" + "0" * 34,
                language="cpp",
                doc_origin="source_header",
                symbol_path="TTree::Draw",
                has_doxygen=True,
            ),
            Chunk.from_file_slice(
                file_path="tree/src/TTree.cxx",
                start_line=100,
                end_line=110,
                content="Long64_t TTree::Draw(const char* expr) { return 0; }",
                root_ref="v0.1",
                resolved_commit="abc123" + "0" * 34,
                language="cpp",
                doc_origin="source_impl",
            ),
        ]
        
        # Create FTS5 database
        db_path = tmp_path / "test.sqlite"
        create_fts5_db(db_path)
        insert_chunks_into_fts(db_path, chunks)
        
        # Search
        results = lexical_search(db_path, "TTree Draw", top_k=10)
        
        # Verify results
        assert len(results) >= 1
        
        # Check required fields
        for result in results:
            assert hasattr(result, "file_path")
            assert hasattr(result, "start_line")
            assert hasattr(result, "end_line")
            assert hasattr(result, "score")
            assert hasattr(result, "root_ref")
            assert hasattr(result, "resolved_commit")
            assert hasattr(result, "language")
            assert hasattr(result, "doc_origin")
            assert result.root_ref == "v0.1"

    def test_lexical_search_includes_version_metadata(self, tmp_path):
        """Results include root_ref and resolved_commit."""
        chunk = Chunk.from_file_slice(
            file_path="test.cpp",
            start_line=1,
            end_line=5,
            content="class MyClass {};",
            root_ref="v6-32-00",
            resolved_commit="def456" + "0" * 34,
            language="cpp",
            doc_origin="source_impl",
        )
        
        db_path = tmp_path / "test.sqlite"
        create_fts5_db(db_path)
        insert_chunks_into_fts(db_path, [chunk])
        
        results = lexical_search(db_path, "MyClass", top_k=10)
        
        assert len(results) >= 1
        assert results[0].root_ref == "v6-32-00"
        assert results[0].resolved_commit == "def456" + "0" * 34

    def test_lexical_search_returns_empty_for_unknown_query(self, tmp_path):
        """Unknown query returns empty list."""
        chunk = Chunk.from_file_slice(
            file_path="test.cpp",
            start_line=1,
            end_line=5,
            content="class MyClass {};",
            root_ref="v0.1",
            resolved_commit="abc123" + "0" * 34,
            language="cpp",
            doc_origin="source_impl",
        )
        
        db_path = tmp_path / "test.sqlite"
        create_fts5_db(db_path)
        insert_chunks_into_fts(db_path, [chunk])
        
        results = lexical_search(db_path, "TotallyFakeROOTClass", top_k=10)
        
        assert len(results) == 0

    def test_lexical_search_respects_top_k(self, tmp_path):
        """Results limited by top_k parameter."""
        chunks = [
            Chunk.from_file_slice(
                file_path=f"file{i}.cpp",
                start_line=(i + 1) * 10,
                end_line=(i + 1) * 10 + 5,
                content="test content " + "x" * i,
                root_ref="v0.1",
                resolved_commit="abc123" + "0" * 34,
                language="cpp",
                doc_origin="source_impl",
            )
            for i in range(5)
        ]
        
        db_path = tmp_path / "test.sqlite"
        create_fts5_db(db_path)
        insert_chunks_into_fts(db_path, chunks)
        
        results = lexical_search(db_path, "test", top_k=2)
        
        assert len(results) <= 2

    def test_lexical_search_ranked_by_bm25(self, tmp_path):
        """Results ordered by BM25 score."""
        # Create chunks with varying relevance to query
        chunks = [
            Chunk.from_file_slice(
                file_path="file1.cpp",
                start_line=1,
                end_line=5,
                content="TTree TTree TTree TTree",  # Multiple TTree mentions
                root_ref="v0.1",
                resolved_commit="abc123" + "0" * 34,
                language="cpp",
                doc_origin="source_impl",
            ),
            Chunk.from_file_slice(
                file_path="file2.cpp",
                start_line=10,
                end_line=15,
                content="class TTree { };",  # Single mention
                root_ref="v0.1",
                resolved_commit="abc123" + "0" * 34,
                language="cpp",
                doc_origin="source_impl",
            ),
        ]
        
        db_path = tmp_path / "test.sqlite"
        create_fts5_db(db_path)
        insert_chunks_into_fts(db_path, chunks)
        
        results = lexical_search(db_path, "TTree", top_k=10)
        
        # Should return both results
        assert len(results) == 2
        # First result should have better score (more TTree mentions)
        assert results[0].score <= results[1].score  # BM25 may be "lower is better"

    def test_lexical_search_lexnorm_mode_expands_aliases(self, tmp_path):
        """lexnorm mode expands query aliases for lexical mismatch."""
        chunk = Chunk.from_file_slice(
            file_path="shipdata/ShipStack.cxx",
            start_line=10,
            end_line=20,
            content=(
                "TClonesArray* particles = new TClonesArray(\"ShipMCTrack\"); "
                "stack->PushTrack(1, 2, 3, 4);"
            ),
            root_ref="v0.1",
            resolved_commit="abc123" + "0" * 34,
            language="cpp",
            doc_origin="source_impl",
        )

        db_path = tmp_path / "test.sqlite"
        create_fts5_db(db_path)
        insert_chunks_into_fts(db_path, [chunk])

        baseline_results = lexical_search(db_path, "object storage", top_k=10, query_mode="baseline")
        lexnorm_results = lexical_search(db_path, "object storage", top_k=10, query_mode="lexnorm")

        assert len(baseline_results) == 0
        assert len(lexnorm_results) >= 1
        assert lexnorm_results[0].chunk_id == chunk.chunk_id
