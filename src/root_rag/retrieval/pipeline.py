"""Composable retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from root_rag.retrieval.interfaces import BaseRetrievalBackend, QueryTransformer, RetrievalBackend
from root_rag.retrieval.models import EvidenceCandidate


@dataclass
class RetrievalPipeline:
    """Compose query transformation with backend retrieval."""

    backend: RetrievalBackend
    query_transformer: QueryTransformer

    def search(self, query: str, top_k: int = 10) -> List[EvidenceCandidate]:
        top_k = BaseRetrievalBackend.normalize_top_k(top_k)
        if top_k == 0:
            return []

        transformed_query = self.query_transformer.transform(query)
        results = self.backend.search(transformed_query, top_k=top_k)
        if len(results) <= top_k:
            return results
        return results[:top_k]
