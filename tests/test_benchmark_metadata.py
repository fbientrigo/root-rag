"""Ensure baseline benchmark metadata stays frozen."""

import json
from pathlib import Path


def test_baseline_benchmark_metadata():
    """Metadata must explicitly reflect the frozen baseline contract."""
    path = Path("artifacts/benchmark_retrieval_baseline_refactor.json")
    assert path.exists(), f"Run baseline benchmark to produce {path}"

    report = json.loads(path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    operational = report.get("operational", {})
    backend_metrics = operational.get("backend_metrics", {})

    assert metadata.get("backend") == "lexical_bm25_memory"
    assert metadata.get("query_mode") == "baseline"
    assert metadata.get("top_k") == 10
    assert metadata.get("bm25", {}).get("k1") == 1.5
    assert metadata.get("bm25", {}).get("b") == 0.75

    assert operational.get("backend_id") == "lexical_bm25_memory"
    assert backend_metrics.get("docs") is not None
    assert operational.get("query_latency_ms", {}).get("count", 0) > 0
