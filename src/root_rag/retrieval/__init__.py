"""Retrieval module: lexical and semantic search over indexed evidence."""

from root_rag.retrieval.models import EvidenceCandidate
from root_rag.retrieval.backends import build_retrieval_backend
from root_rag.retrieval.lexical import lexical_search
from root_rag.retrieval.pipeline import RetrievalPipeline
from root_rag.retrieval.transformers import (
    IdentityQueryTransformer,
    RootLexicalQueryTransformer,
    build_query_transformer,
)

__all__ = [
    "EvidenceCandidate",
    "build_retrieval_backend",
    "lexical_search",
    "RetrievalPipeline",
    "IdentityQueryTransformer",
    "RootLexicalQueryTransformer",
    "build_query_transformer",
]
