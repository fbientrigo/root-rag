"""Metadata checks for the frozen official baseline definition."""

from __future__ import annotations

import json
from pathlib import Path


def test_official_baseline_metadata_contract_is_machine_readable():
    manifest_path = Path("configs/baseline_manifest.json")
    assert manifest_path.exists(), f"Missing {manifest_path}"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["official_backend"] == "lexical_bm25_memory"
    assert manifest["official_query_mode"] == "baseline"
    assert manifest["official_corpus_profile"] == "fairship_only_valid"
    assert manifest["official_top_k"] == 10
    assert manifest["output_artifact_directory"] == "artifacts/baseline_official"
    assert manifest["semantic_retrieval"]["enabled_by_default"] is False
