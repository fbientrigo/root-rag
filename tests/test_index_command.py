"""End-to-end tests for root-rag index command."""
import json

import pytest
from click.testing import CliRunner

from root_rag.cli import main
from root_rag.index.schemas import IndexManifest


class TestIndexCommandSuccess:
    """Tests for successful index command execution."""

    def test_index_command_success_e2e(self, tmp_path, git_repo_fixture):
        """Build index successfully end-to-end."""
        runner = CliRunner()
        
        cache_dir = tmp_path / "cache"
        output_dir = tmp_path / "indexes"
        repo_path = git_repo_fixture["path"]
        
        # First fetch the corpus using the local repo path (use v0.1 tag that exists in fixture)
        result = runner.invoke(main, [
            "fetch",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
        ])
        assert result.exit_code == 0, f"Fetch failed: {result.output}\nLast exit: {result.exit_code}"
        
        # Then build the index
        result = runner.invoke(main, [
            "index",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
            "--output-dir", str(output_dir),
        ])
        
        assert result.exit_code == 0, f"Index command failed: {result.output}"
        assert "Index created:" in result.output
        assert "Corpus ID:" in result.output
        assert "Chunks:" in result.output
        
        # Verify output files exist
        output_items = list(output_dir.glob("*"))
        assert len(output_items) > 0
        
        # Find the index directory
        index_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        assert len(index_dirs) > 0
        
        index_dir = index_dirs[0]
        assert (index_dir / "fts.sqlite").exists()
        assert (index_dir / "index_manifest.json").exists()

    def test_index_command_output_format(self, tmp_path, git_repo_fixture):
        """Verify index command output format."""
        runner = CliRunner()
        
        cache_dir = tmp_path / "cache"
        output_dir = tmp_path / "indexes"
        repo_path = git_repo_fixture["path"]
        
        # Fetch corpus
        runner.invoke(main, [
            "fetch",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
        ])
        
        # Build index
        result = runner.invoke(main, [
            "index",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
            "--output-dir", str(output_dir),
        ])
        
        # Check output contains expected sections
        assert "Index created:" in result.output
        assert "Corpus ID:" in result.output
        assert "Root Ref: v0.1" in result.output
        assert "Commit:" in result.output
        assert "Schema Version:" in result.output
        assert "Chunks:" in result.output
        assert "Files:" in result.output
        assert "Retrieval Modes:" in result.output
        assert "FTS DB:" in result.output
        assert "Manifest:" in result.output

    def test_index_manifest_structure(self, tmp_path, git_repo_fixture):
        """Verify index manifest has all required fields."""
        runner = CliRunner()
        
        cache_dir = tmp_path / "cache"
        output_dir = tmp_path / "indexes"
        repo_path = git_repo_fixture["path"]
        
        # Fetch and index
        runner.invoke(main, [
            "fetch",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
        ])
        
        result = runner.invoke(main, [
            "index",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
            "--output-dir", str(output_dir),
        ])
        
        assert result.exit_code == 0
        
        # Find and load manifest
        index_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        index_dir = index_dirs[0]
        manifest_file = index_dir / "index_manifest.json"
        
        assert manifest_file.exists()
        
        # Load and verify structure
        manifest = IndexManifest.load(manifest_file)
        
        assert manifest.index_id
        assert manifest.corpus_id
        assert manifest.root_ref == "v0.1"
        assert manifest.resolved_commit
        assert manifest.corpus_url
        assert manifest.chunks_path
        assert manifest.fts_db_path
        assert manifest.schema_version
        assert manifest.index_schema_version
        assert manifest.chunk_count > 0
        assert manifest.file_count > 0
        assert "lexical" in manifest.retrieval_modes
        assert manifest.created_at
        assert manifest.tool_version

    def test_index_persists_version_metadata(self, tmp_path, git_repo_fixture):
        """Verify version metadata in index manifest and FTS database."""
        import sqlite3
        
        runner = CliRunner()
        
        cache_dir = tmp_path / "cache"
        output_dir = tmp_path / "indexes"
        repo_path = git_repo_fixture["path"]
        
        # Fetch and index
        runner.invoke(main, [
            "fetch",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
        ])
        
        result = runner.invoke(main, [
            "index",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
            "--output-dir", str(output_dir),
        ])
        
        assert result.exit_code == 0
        
        # Verify manifest metadata
        index_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        index_dir = index_dirs[0]
        manifest_file = index_dir / "index_manifest.json"
        
        manifest = IndexManifest.load(manifest_file)
        assert manifest.root_ref == "v0.1"
        assert len(manifest.resolved_commit) == 40  # Full SHA-1 hash
        
        # Verify FTS5 database has matching root_ref
        fts_db = index_dir / "fts.sqlite"
        conn = sqlite3.connect(str(fts_db))
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT root_ref FROM chunks_fts")
        root_refs = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        assert len(root_refs) == 1
        assert root_refs[0] == "v0.1"


class TestIndexCommandErrors:
    """Tests for index command error handling."""

    def test_index_command_invalid_ref_exit_code_3(self, tmp_path):
        """Index with non-existent ref returns exit code 3."""
        runner = CliRunner()
        
        cache_dir = tmp_path / "cache"
        output_dir = tmp_path / "indexes"
        
        # Attempt to index non-existent ref
        result = runner.invoke(main, [
            "index",
            "--root-ref", "DOES_NOT_EXIST_REF",
            "--cache-dir", str(cache_dir),
            "--output-dir", str(output_dir),
        ])
        
        assert result.exit_code == 3


class TestIndexCommandConfig:
    """Tests for configuration handling."""

    def test_index_command_config_file_validation(self, tmp_path, git_repo_fixture):
        """Validate configuration file if provided."""
        runner = CliRunner()
        
        cache_dir = tmp_path / "cache"
        output_dir = tmp_path / "indexes"
        repo_path = git_repo_fixture["path"]
        
        # Create invalid config file
        bad_config = tmp_path / "bad.json"
        bad_config.write_text("not valid json {")
        
        # Fetch corpus first
        runner.invoke(main, [
            "fetch",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
        ])
        
        # Try index with bad config
        result = runner.invoke(main, [
            "index",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
            "--output-dir", str(output_dir),
            "--config", str(bad_config),
        ])
        
        assert result.exit_code == 7

    def test_index_command_missing_config_file(self, tmp_path, git_repo_fixture):
        """Missing configuration file returns exit code 7."""
        runner = CliRunner()
        
        cache_dir = tmp_path / "cache"
        output_dir = tmp_path / "indexes"
        repo_path = git_repo_fixture["path"]
        
        # Fetch corpus first
        runner.invoke(main, [
            "fetch",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
        ])
        
        # Try index with non-existent config
        non_existent_config = tmp_path / "does_not_exist.json"
        result = runner.invoke(main, [
            "index",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
            "--output-dir", str(output_dir),
            "--config", str(non_existent_config),
        ])
        
        assert result.exit_code == 7

    def test_index_command_valid_config_file(self, tmp_path, git_repo_fixture):
        """Valid configuration file allows index building."""
        runner = CliRunner()
        
        cache_dir = tmp_path / "cache"
        output_dir = tmp_path / "indexes"
        repo_path = git_repo_fixture["path"]
        
        # Create valid config file
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"feature": "enabled"}))
        
        # Fetch corpus
        runner.invoke(main, [
            "fetch",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
        ])
        
        # Index with valid config
        result = runner.invoke(main, [
            "index",
            "--root-ref", "v0.1",
            "--repo-url", str(repo_path),
            "--cache-dir", str(cache_dir),
            "--output-dir", str(output_dir),
            "--config", str(config),
        ])
        
        assert result.exit_code == 0
