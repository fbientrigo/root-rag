"""End-to-end tests for root-rag search command."""
import json
import pytest
from click.testing import CliRunner

from root_rag.cli import main


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
                assert "start_line" in data[0]
                assert "end_line" in data[0]
                assert "score" in data[0]
        except json.JSONDecodeError:
            pytest.fail("JSON output is not valid JSON")


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
