"""Retrieval module: lexical and semantic search over indexed evidence."""

from root_rag.retrieval.models import EvidenceCandidate
from root_rag.retrieval.lexical import lexical_search

__all__ = [
    "EvidenceCandidate",
    "lexical_search",
]
