"""Shared text-tokenization helpers for retrieval backends.

Centralizes the token regex and tokenizer that were previously duplicated across
``backends.py``, ``transformers.py``, and ``s1_semantic.py``. Keeping a single
canonical implementation avoids the helpers drifting apart over time.
"""

from __future__ import annotations

import re

# Matches runs of identifier characters (letters, digits, underscore).
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercased identifier tokens."""
    return [token.lower() for token in TOKEN_RE.findall(text)]
