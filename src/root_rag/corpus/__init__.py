"""Corpus management: fetching, caching, and manifest handling."""

from root_rag.core.errors import InvalidRefError
from root_rag.corpus.fetcher import fetch_corpus, resolve_git_ref
from root_rag.corpus.manifest import Manifest

__all__ = [
    "Manifest",
    "fetch_corpus",
    "resolve_git_ref",
    "InvalidRefError",
]
