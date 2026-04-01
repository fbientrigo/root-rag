"""Tests for SOFIE indexing fixes and validation.

Tests the fix for P5: .hxx extension support and SOFIE corpus indexing.
"""
import logging
from pathlib import Path

import pytest

from root_rag.parser.files import INCLUDED_EXTENSIONS, discover_text_files
from root_rag.parser.seed_filter import (
    filter_files_by_seed_corpus,
    get_seed_corpus_paths,
    load_seed_corpus_config,
)

logger = logging.getLogger(__name__)


class TestHxxExtensionSupport:
    """Tests for .hxx file extension support."""

    def test_hxx_extension_included(self):
        """Verify .hxx is included in INCLUDED_EXTENSIONS."""
        assert ".hxx" in INCLUDED_EXTENSIONS, (
            "P5 Fix: .hxx extension must be included for SOFIE and ROOT indexing"
        )

    def test_hxx_files_discovered(self, tmp_path):
        """Verify .hxx files are discovered by file enumeration."""
        # Create test .hxx file
        test_file = tmp_path / "TestClass.hxx"
        test_file.write_text("class TestClass {};", encoding="utf-8")

        # Discover files
        files = discover_text_files(tmp_path)

        # Should include .hxx file
        assert len(files) == 1
        assert files[0].name == "TestClass.hxx"

    def test_multiple_header_extensions_discovered(self, tmp_path):
        """Verify all header extensions are discovered."""
        # Create various header files
        extensions = [".h", ".hpp", ".hh", ".hxx"]
        for ext in extensions:
            test_file = tmp_path / f"Test{ext}"
            test_file.write_text(f"// Header with {ext} extension", encoding="utf-8")

        # Discover files
        files = discover_text_files(tmp_path)

        # All should be discovered
        assert len(files) == 4
        discovered_exts = {f.suffix for f in files}
        assert discovered_exts == set(extensions)


class TestSofieCorpusConfig:
    """Tests for SOFIE corpus configuration validation."""

    @pytest.mark.skipif(
        not Path("data/raw/corpora").exists(),
        reason="ROOT corpus not available",
    )
    def test_sofie_config_paths_valid(self):
        """Validate SOFIE config references existing paths (where possible)."""
        config_path = Path("configs/sofie_corpus_root_636.yaml")
        if not config_path.exists():
            pytest.skip("SOFIE config not found")

        # Find ROOT corpus
        corpora_dir = Path("data/raw/corpora")
        root_dirs = list(corpora_dir.glob("root-project__root__*"))
        if not root_dirs:
            pytest.skip("ROOT corpus not cached")

        repo_root = root_dirs[0] / "repo"

        # Load config and check paths
        config = load_seed_corpus_config(config_path)
        seed_paths = get_seed_corpus_paths(config, repo_root)

        # Should have resolved some paths
        assert len(seed_paths) > 0, "SOFIE config should resolve at least some paths"

        # Log warnings for missing paths (non-fatal, may be version-specific)
        classes = config.get("corpus", {}).get("classes", [])
        total_headers = sum(len(c.get("headers", [])) for c in classes)
        existing_count = len(seed_paths)

        logger.info(f"SOFIE config: {existing_count}/{total_headers} paths exist")

        # At least 75% should exist (some may be version-specific)
        coverage = existing_count / total_headers if total_headers > 0 else 0
        assert coverage >= 0.75, (
            f"Too many missing paths: {existing_count}/{total_headers} "
            f"({coverage:.1%} coverage, expected >=75%)"
        )

    @pytest.mark.skipif(
        not Path("data/raw/corpora").exists(),
        reason="ROOT corpus not available",
    )
    def test_sofie_seed_filter_matches_files(self):
        """Verify seed corpus filter matches discovered .hxx files."""
        config_path = Path("configs/sofie_corpus_root_636.yaml")
        if not config_path.exists():
            pytest.skip("SOFIE config not found")

        # Find ROOT corpus
        corpora_dir = Path("data/raw/corpora")
        root_dirs = list(corpora_dir.glob("root-project__root__*"))
        if not root_dirs:
            pytest.skip("ROOT corpus not cached")

        repo_root = root_dirs[0] / "repo"

        # Get seed paths
        config = load_seed_corpus_config(config_path)
        seed_paths = get_seed_corpus_paths(config, repo_root)

        # Discover all files (should now include .hxx)
        all_files = discover_text_files(repo_root)
        sofie_discovered = [f for f in all_files if "sofie" in str(f).lower()]

        logger.info(f"Discovered {len(sofie_discovered)} SOFIE files")

        # Filter by seed corpus
        filtered = filter_files_by_seed_corpus(all_files, seed_paths)
        sofie_filtered = [f for f in filtered if "sofie" in str(f).lower()]

        logger.info(f"Filtered to {len(sofie_filtered)} SOFIE seed corpus files")

        # Should have matched some SOFIE files
        assert len(sofie_filtered) > 0, (
            "Filter should match SOFIE .hxx files (P5 fix verification)"
        )

        # Should be a reasonable match rate
        if len(sofie_discovered) > 0:
            match_rate = len(sofie_filtered) / len(sofie_discovered)
            logger.info(f"SOFIE filter match rate: {match_rate:.1%}")


class TestSofieIndexing:
    """Integration tests for SOFIE indexing."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("data/raw/corpora").exists(),
        reason="ROOT corpus not available",
    )
    def test_sofie_produces_chunks(self):
        """Integration test: SOFIE indexing produces expected chunk count."""
        from root_rag.corpus.manifest import Manifest
        from root_rag.index.builder import build_index

        config_path = Path("configs/sofie_corpus_root_636.yaml")
        if not config_path.exists():
            pytest.skip("SOFIE config not found")

        # Find ROOT corpus
        corpora_dir = Path("data/raw/corpora")
        root_dirs = list(corpora_dir.glob("root-project__root__*"))
        if not root_dirs:
            pytest.skip("ROOT corpus not cached")

        corpus_dir = root_dirs[0]
        repo_root = corpus_dir / "repo"
        manifest_path = corpus_dir / "manifest.json"

        if not manifest_path.exists():
            pytest.skip("ROOT manifest not found")

        # Load manifest
        manifest = Manifest.load(manifest_path)

        # Build index to temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_index(
                manifest=manifest,
                output_dir=Path(tmpdir),
                window_lines=80,
                overlap_lines=10,
                seed_corpus_config=config_path,
            )

            chunk_count = result.get("chunk_count", 0)
            file_count = result.get("file_count", 0)

            logger.info(f"SOFIE indexing: {chunk_count} chunks from {file_count} files")

            # After P5 fix, should have substantial chunks
            assert chunk_count >= 100, (
                f"SOFIE should produce >=100 chunks (got {chunk_count}). "
                "P5 fix may not be working."
            )

            assert file_count >= 20, (
                f"SOFIE should index >=20 files (got {file_count}). "
                "P5 fix may not be working."
            )

    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("data/indexes_sofie").exists(),
        reason="SOFIE index not available",
    )
    def test_sofie_index_contains_operators(self):
        """Verify SOFIE index contains operator definitions."""
        from root_rag.retrieval.lexical import lexical_search

        # Find latest SOFIE index
        indexes_dir = Path("data/indexes_sofie")
        index_dirs = sorted(indexes_dir.glob("v6-36-08__*"))
        if not index_dirs:
            pytest.skip("No SOFIE index found")

        index_dir = index_dirs[-1]  # Latest
        fts_db = index_dir / "fts.sqlite"

        if not fts_db.exists():
            pytest.skip("SOFIE FTS database not found")

        # Search for SOFIE operators
        results = lexical_search(fts_db, "ROperator_Conv", top_k=10)

        # Should find operator definitions
        assert len(results) > 0, (
            "SOFIE index should contain ROperator_Conv after P5 fix"
        )

        # Verify results are from .hxx files
        hxx_results = [r for r in results if r.file_path.endswith(".hxx")]
        assert len(hxx_results) > 0, (
            "At least some results should be from .hxx files"
        )


class TestRegressionProtection:
    """Tests to ensure the fix doesn't break existing functionality."""

    def test_original_extensions_still_work(self, tmp_path):
        """Verify original extensions (.h, .cpp, .cxx) still work."""
        # Create files with original extensions
        (tmp_path / "test.h").write_text("// header", encoding="utf-8")
        (tmp_path / "test.cpp").write_text("// impl", encoding="utf-8")
        (tmp_path / "test.cxx").write_text("// impl2", encoding="utf-8")

        files = discover_text_files(tmp_path)

        assert len(files) == 3
        exts = {f.suffix for f in files}
        assert exts == {".h", ".cpp", ".cxx"}

    @pytest.mark.skipif(
        not Path("data/indexes_tier1").exists(),
        reason="Tier 1 index not available",
    )
    def test_tier1_index_still_valid(self):
        """Verify Tier 1 index still works after adding .hxx support."""
        from root_rag.index.locator import resolve_index

        # Resolve Tier 1 index
        index_manifest = resolve_index(
            indexes_root=Path("data/indexes_tier1"),
            root_ref="v6-36-08",
        )

        # Should resolve successfully
        assert index_manifest is not None
        assert index_manifest.chunk_count > 0
        assert index_manifest.file_count > 0

        # Tier 1 should have more files now (includes .hxx)
        # Original: ~53 files, After fix: should be higher
        logger.info(
            f"Tier 1 after .hxx fix: {index_manifest.file_count} files, "
            f"{index_manifest.chunk_count} chunks"
        )
