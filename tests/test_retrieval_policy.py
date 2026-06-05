import pytest
from pathlib import Path
import json
from unittest.mock import MagicMock, patch
from root_rag.cli import _resolve_backend_and_results
from root_rag.index.schemas import IndexManifest

@pytest.fixture
def mock_manifest():
    manifest = MagicMock(spec=IndexManifest)
    manifest.fts_db_path = "data/fts.sqlite"
    manifest.resolved_commit = "98de16a5b264" + "0" * 28
    manifest.root_ref = "master"
    return manifest

@pytest.fixture
def forest_config_content():
    return {
        "default_profiles": ["small"],
        "profiles": [
            {
                "profile_id": "small",
                "source_commit": "98de16a5b264",
                "index_output_path": "data/forest/small"
            }
        ]
    }

def test_default_root_uses_lexical(mock_manifest):
    with patch("root_rag.cli.lexical_search") as mock_lex:
        mock_lex.return_value = ([], "lexical")
        results, backend, fallback = _resolve_backend_and_results(
            query="test",
            manifest=mock_manifest,
            top_k=5,
            retrieval_backend="lexical",
            retrieval_forest=None,
            fusion="rrf",
            dedup="line_overlap",
            tie_breaker="stable",
            baseline=False,
            profile="root"
        )
        assert backend == "lexical"
        assert fallback is None

def test_fairship_default_uses_forest_when_valid(mock_manifest, forest_config_content, tmp_path):
    with patch("root_rag.cli.Path") as mock_path_class, \
         patch("builtins.open", MagicMock()), \
         patch("json.load") as mock_json_load, \
         patch("root_rag.cli.build_retrieval_backend") as mock_build:
        
        # Setup mock Path objects
        mock_config_path = MagicMock()
        mock_config_path.exists.return_value = True
        
        mock_db_path = MagicMock()
        mock_db_path.exists.return_value = True
        
        def path_side_effect(arg):
            if "retrieval_forest_profiles.json" in str(arg):
                return mock_config_path
            if "fts.sqlite" in str(arg):
                return mock_db_path
            return MagicMock()
            
        mock_path_class.side_effect = path_side_effect
        mock_json_load.return_value = forest_config_content
        
        mock_backend = MagicMock()
        mock_backend.search.return_value = []
        mock_build.return_value = mock_backend
        
        results, backend, fallback = _resolve_backend_and_results(
            query="test",
            manifest=mock_manifest,
            top_k=5,
            retrieval_backend="lexical",
            retrieval_forest=None,
            fusion="rrf",
            dedup="line_overlap",
            tie_breaker="stable",
            baseline=False,
            profile="fairship"
        )
        assert backend == "forest"
        assert fallback is None

def test_fairship_baseline_forces_lexical(mock_manifest):
    with patch("root_rag.cli.lexical_search") as mock_lex:
        mock_lex.return_value = ([], "lexical")
        results, backend, fallback = _resolve_backend_and_results(
            query="test",
            manifest=mock_manifest,
            top_k=5,
            retrieval_backend="lexical",
            retrieval_forest=None,
            fusion="rrf",
            dedup="line_overlap",
            tie_breaker="stable",
            baseline=True,
            profile="fairship"
        )
        assert backend == "lexical"

def test_explicit_forest_fails_loudly_if_missing(mock_manifest):
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = False
        import click
        with pytest.raises(click.UsageError) as exc:
            _resolve_backend_and_results(
                query="test",
                manifest=mock_manifest,
                top_k=5,
                retrieval_backend="forest",
                retrieval_forest=None,
                fusion="rrf",
                dedup="line_overlap",
                tie_breaker="stable",
                baseline=False,
                profile="root"
            )
        assert "Forest config not found" in str(exc.value)

def test_fairship_forest_fallback_on_commit_mismatch(mock_manifest, forest_config_content):
    # Change manifest commit to something else
    mock_manifest.resolved_commit = "deadbeef1234" + "0" * 28
    
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("builtins.open", MagicMock()), \
         patch("json.load") as mock_json_load, \
         patch("root_rag.cli.lexical_search") as mock_lex:
        
        mock_exists.return_value = True
        mock_json_load.return_value = forest_config_content
        mock_lex.return_value = ([], "lexical")
        
        results, backend, fallback = _resolve_backend_and_results(
            query="test",
            manifest=mock_manifest,
            top_k=5,
            retrieval_backend="lexical",
            retrieval_forest=None,
            fusion="rrf",
            dedup="line_overlap",
            tie_breaker="stable",
            baseline=False,
            profile="fairship"
        )
        assert backend == "lexical"
        assert "commit mismatch" in fallback.lower()
