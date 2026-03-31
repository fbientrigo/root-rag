"""Interfaces for pluggable retrieval components."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Dict, List, Protocol, Union

from root_rag.retrieval.models import EvidenceCandidate

OperationalMetricValue = Union[int, float, str, None]
OperationalMetrics = Dict[str, OperationalMetricValue]


class QueryTransformer(Protocol):
    """Transforms user query text before backend search."""

    def transform(self, query: str) -> str:
        """Return transformed query text."""


class RetrievalBackend(Protocol):
    """Backend contract for retrieval engines (lexical, embedding, hybrid).

    Contract guarantees:
    - `search` returns at most `top_k` ranked `EvidenceCandidate` rows.
    - `search` returns an empty list for invalid/non-positive `top_k`.
    - recoverable runtime failures should fail closed (empty list), not crash callers.
    - `operational_metrics` returns JSON-safe scalar values for benchmark reports.
    """

    backend_id: str

    def search(self, query: str, top_k: int) -> List[EvidenceCandidate]:
        """Return ranked evidence candidates for transformed query text."""

    def operational_metrics(self) -> OperationalMetrics:
        """Return backend operational metrics for benchmark/reporting."""


class BaseRetrievalBackend(ABC):
    """Small concrete base class for retrieval backends."""

    backend_id = "unknown"

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[EvidenceCandidate]:
        """Return ranked evidence candidates for transformed query text."""

    @staticmethod
    def normalize_top_k(top_k: int) -> int:
        """Coerce top_k to a safe positive integer."""
        try:
            value = int(top_k)
        except (TypeError, ValueError):
            return 0
        return value if value > 0 else 0

    @staticmethod
    def normalize_operational_metrics(metrics: OperationalMetrics) -> OperationalMetrics:
        """Validate and normalize operational metrics payload shape."""
        normalized: OperationalMetrics = {}
        for key, value in metrics.items():
            if not isinstance(key, str):
                raise TypeError("operational metric keys must be strings")
            if isinstance(value, float) and not math.isfinite(value):
                normalized[key] = None
                continue
            if value is not None and not isinstance(value, (int, float, str)):
                raise TypeError(f"operational metric '{key}' has unsupported type: {type(value).__name__}")
            normalized[key] = value
        return normalized

    def operational_metrics(self) -> OperationalMetrics:
        """Return backend operational metrics for benchmark/reporting."""
        return {}
