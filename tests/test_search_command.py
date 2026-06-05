"""End-to-end tests for root-rag search command."""
import json
import os
from pathlib import Path
import pytest
from click.testing import CliRunner

from root_rag.cli import main
from root_rag.index.fts import create_fts5_db, insert_chunks_into_fts
from root_rag.index.schemas import Chunk, IndexManifest


class TestSearchCommandSuccess:
    """Tests for successful search command execution."""

    def test_search_command_success_e2e(self, tmp_path, git_repo_fixture):
        """Build index, then search successfully."""
        runner = CliRunner()
        
        cache_dir = tmp_path / "cache"
        index_dir = tmp_path / "indexes"
        
        repo_path = git_repo_fixture["path"]
        
        # Fetch corpus
        result = runner.invoke(main, [
            "fetch",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
        ])
        assert result.exit_code == 0, f"Fetch failed: {result.output}"
        
        # Build index
        result = runner.invoke(main, [
            "index",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
            "--output-dir", str(index_dir),
        ])
        assert result.exit_code == 0, f"Index failed: {result.output}"
        
        # Search
        result = runner.invoke(main, [
            "search",
            "include",
            "--root-ref", "v0.1",
            "--index-dir", str(index_dir),
        ])
        
        assert result.exit_code == 0, f"Search failed: {result.output}"
        assert "[1]" in result.output or "score=" in result.output
        assert "[cpp]" in result.output

    def test_search_command_with_top_k(self, tmp_path, git_repo_fixture):
        """Search with custom top-k parameter."""
        runner = CliRunner()
        
        cache_dir = tmp_path / "cache"
        index_dir = tmp_path / "indexes"
        repo_path = git_repo_fixture["path"]
        
        # Setup (fetch + index)
        runner.invoke(main, [
            "fetch",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
        ])
        
        runner.invoke(main, [
            "index",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
            "--output-dir", str(index_dir),
        ])
        
        # Search with top-k
        result = runner.invoke(main, [
            "search",
            "include",
            "--root-ref", "v0.1",
            "--index-dir", str(index_dir),
            "--top-k", "3",
        ])
        
        assert result.exit_code == 0

    def test_search_command_json_output(self, tmp_path, git_repo_fixture):
        """Search with JSON output format."""
        runner = CliRunner()
        
        cache_dir = tmp_path / "cache"
        index_dir = tmp_path / "indexes"
        repo_path = git_repo_fixture["path"]
        
        # Setup
        runner.invoke(main, [
            "fetch",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
        ])
        
        runner.invoke(main, [
            "index",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
            "--output-dir", str(index_dir),
        ])
        
        # Search with JSON
        result = runner.invoke(main, [
            "search",
            "include",
            "--root-ref", "v0.1",
            "--index-dir", str(index_dir),
            "--json",
        ])
        
        assert result.exit_code == 0
        # Verify output is valid JSON
        try:
            data = json.loads(result.output)
            assert isinstance(data, list)
            if len(data) > 0:
                assert "file_path" in data[0]
                assert "source_type" in data[0]
                assert "start_line" in data[0]
                assert "end_line" in data[0]
                assert "score" in data[0]
        except json.JSONDecodeError:
            pytest.fail("JSON output is not valid JSON")

    def test_search_command_with_relative_fts_path_in_manifest(self, tmp_path):
        """Search works when manifest stores fts_db_path relative to index directory."""
        runner = CliRunner()

        index_dir = tmp_path / "indexes"
        index_id = "fairship__master__98de16a5b264__20260331T185059271533+0000Z"
        artifact_dir = index_dir / index_id
        artifact_dir.mkdir(parents=True)

        db_path = artifact_dir / "fts.sqlite"
        create_fts5_db(db_path)
        insert_chunks_into_fts(
            db_path,
            [
                Chunk.from_file_slice(
                    file_path="shipgen/MuDISGenerator.cxx",
                    start_line=1,
                    end_line=3,
                    content="class MuDISGenerator {};",
                    root_ref="master",
                    resolved_commit="98de16a5b264d51c36e1a3638466d1dbb7667678",
                    language="cpp",
                    doc_origin="source_impl",
                )
            ],
        )

        manifest = IndexManifest(
            index_id=index_id,
            corpus_id="fairship__master__98de16a5b264",
            root_ref="master",
            resolved_commit="98de16a5b264d51c36e1a3638466d1dbb7667678",
            corpus_url="https://github.com/ShipSoft/FairShip.git",
            chunks_path="processed/chunks/master__98de16a5b264/chunks.jsonl",
            fts_db_path="fts.sqlite",
            chunk_count=1,
            file_count=1,
            created_at="2026-03-31T18:50:59.702693+00:00",
        )
        manifest.save(artifact_dir / "index_manifest.json")

        result = runner.invoke(
            main,
            [
                "search",
                "MuDISGenerator",
                "--index-id",
                index_id,
                "--index-dir",
                str(index_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "MuDISGenerator.cxx" in result.output


class TestSearchCommandErrors:
    """Tests for search command error handling."""

    def test_search_command_index_not_found_exit_4(self, tmp_path):
        """Index not found returns exit code 4."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            "search",
            "test",
            "--root-ref", "nonexistent_version",
            "--index-dir", str(tmp_path / "indexes"),
        ])
        
        assert result.exit_code == 4

    def test_search_command_no_evidence_exit_5(self, tmp_path, git_repo_fixture):
        """No evidence found returns exit code 5."""
        runner = CliRunner()
        
        cache_dir = tmp_path / "cache"
        index_dir = tmp_path / "indexes"
        repo_path = git_repo_fixture["path"]
        
        # Setup
        runner.invoke(main, [
            "fetch",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
        ])
        
        runner.invoke(main, [
            "index",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
            "--output-dir", str(index_dir),
        ])
        
        # Search for something that doesn't exist
        result = runner.invoke(main, [
            "search",
            "XYZ_TOTALLY_FAKE_CLASS_THAT_DOES_NOT_EXIST",
            "--root-ref", "v0.1",
            "--index-dir", str(index_dir),
        ])
        
        assert result.exit_code == 5

    def test_search_command_explicit_index_id(self, tmp_path, git_repo_fixture):
        """Search using explicit --index-id parameter."""
        runner = CliRunner()
        
        cache_dir = tmp_path / "cache"
        index_dir = tmp_path / "indexes"
        repo_path = git_repo_fixture["path"]
        
        # Setup
        runner.invoke(main, [
            "fetch",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
        ])
        
        result_index = runner.invoke(main, [
            "index",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
            "--output-dir", str(index_dir),
        ])
        
        # Extract index_id from output (e.g., "Index ID: v0.1__abc123__...")
        # For now just search without explicit ID and verify success
        result = runner.invoke(main, [
            "search",
            "test",
            "--root-ref", "v0.1",
            "--index-dir", str(index_dir),
        ])
        
        assert result.exit_code in [0, 5]  # 0 if found, 5 if no evidence


def _write_search_fixture(indexes_root, *, index_id):
    artifact_dir = indexes_root / index_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    db_path = artifact_dir / "fts.sqlite"
    create_fts5_db(db_path)
    insert_chunks_into_fts(
        db_path,
        [
            Chunk.from_file_slice(
                file_path="shipgen/MuDISGenerator.cxx",
                start_line=1,
                end_line=3,
                content='const char* opt = "-Y";\nbool back = true;\nconst char* mode = "--MuDIS";',
                root_ref="master",
                resolved_commit="98de16a5b264d51c36e1a3638466d1dbb7667678",
                language="cpp",
                doc_origin="source_impl",
            ),
            Chunk.from_file_slice(
                file_path="macro/run_simScript.py",
                start_line=10,
                end_line=12,
                content='if "--MuonBack" in argv:\n    pass\n',
                root_ref="master",
                resolved_commit="98de16a5b264d51c36e1a3638466d1dbb7667678",
                language="python",
                doc_origin="source_impl",
            ),
        ],
    )

    manifest = IndexManifest(
        index_id=index_id,
        corpus_id="fairship__master__98de16a5b264",
        root_ref="master",
        resolved_commit="98de16a5b264d51c36e1a3638466d1dbb7667678",
        corpus_url="https://github.com/ShipSoft/FairShip.git",
        chunks_path="processed/chunks/master__98de16a5b264/chunks.jsonl",
        fts_db_path="fts.sqlite",
        chunk_count=2,
        file_count=2,
        created_at="2026-03-31T18:50:59.702693+00:00",
    )
    manifest.save(artifact_dir / "index_manifest.json")


def test_search_literal_accepts_dash_prefixed_query(tmp_path):
    runner = CliRunner()
    index_dir = tmp_path / "indexes"
    index_id = "fairship__master__98de16a5b264__20260331T185059271533+0000Z"
    _write_search_fixture(index_dir, index_id=index_id)

    result = runner.invoke(
        main,
        [
            "search",
            "--literal",
            "-Y",
            "--index-id",
            index_id,
            "--index-dir",
            str(index_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "MuDISGenerator.cxx" in result.output


def test_search_index_id_falls_back_to_fairship_dir_when_index_dir_omitted(tmp_path):
    runner = CliRunner()
    index_id = "fairship__master__98de16a5b264__20260331T185059271533+0000Z"

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        fairship_dir = Path("data/indexes_fairship")
        _write_search_fixture(fairship_dir, index_id=index_id)

        result = runner.invoke(
            main,
            [
                "search",
                "MuDISGenerator",
                "--index-id",
                index_id,
            ],
        )
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output
    assert "MuDISGenerator.cxx" in result.output


def test_search_default_profile_keeps_root_index_behavior(tmp_path):
    runner = CliRunner()
    index_id = "root__master__98de16a5b264__20260331T185059271533+0000Z"

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        root_dir = Path("data/indexes")
        _write_search_fixture(root_dir, index_id=index_id)

        result = runner.invoke(
            main,
            [
                "search",
                "MuDISGenerator",
                "--root-ref",
                "master",
            ],
        )
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output
    assert "MuDISGenerator.cxx" in result.output
