"""Query transformers for retrieval pipelines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

from root_rag.retrieval.interfaces import QueryTransformer

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


LOW_SIGNAL_QUERY_TERMS: Set[str] = {
    "and",
    "in",
    "through",
    "with",
    "files",
    "file",
    "fairship",
    "root",
    "usage",
    "pattern",
    "implementation",
    "implementations",
    "overrides",
    "override",
    "modules",
    "module",
    "detectors",
    "detector",
    "definition",
    "test",
    "validation",
    "global",
    "local",
    "object",
}


QUERY_ALIAS_EXPANSIONS: Dict[str, List[str]] = {
    "tgeomanager": ["ggeomanager", "gettopvolume", "tgeonavigator", "findnode"],
    "navigation": ["findnode", "getcurrentnode", "tgeonavigator", "ggeomanager"],
    "open": ["tfile", "tfile_open"],
    "storage": ["tclonesarray", "pushtrack"],
    "assembly": ["addnode"],
    "constructgeometry": ["override"],
    "processhits": ["override"],
    "shipfieldmaker": ["defineglobalfield", "definelocalfield", "defineregionfield"],
    "setbranchaddress": ["getentry", "ttree"],
}


EXACT_SYMBOL_BOOST_TOKENS: Set[str] = {
    "setbranchaddress",
    "constructgeometry",
    "processhits",
}


def _tokenize(query: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(query)]


@dataclass
class IdentityQueryTransformer(QueryTransformer):
    """No-op query transformer."""

    def transform(self, query: str) -> str:
        return query


@dataclass
class RootLexicalQueryTransformer(QueryTransformer):
    """Rule-based lexical normalization for ROOT/FairShip query mismatch."""

    low_signal_terms: Set[str] = field(default_factory=lambda: set(LOW_SIGNAL_QUERY_TERMS))
    alias_expansions: Dict[str, List[str]] = field(
        default_factory=lambda: {token: aliases[:] for token, aliases in QUERY_ALIAS_EXPANSIONS.items()}
    )
    exact_symbol_boost_tokens: Set[str] = field(default_factory=lambda: set(EXACT_SYMBOL_BOOST_TOKENS))

    def transform(self, query: str) -> str:
        base_tokens = _tokenize(query)
        filtered_tokens = [token for token in base_tokens if token not in self.low_signal_terms]

        expanded_tokens: List[str] = []
        for token in filtered_tokens:
            aliases = self.alias_expansions.get(token)
            if aliases:
                # Keep critical symbols alongside aliases to preserve exact API matches.
                if token in self.exact_symbol_boost_tokens:
                    expanded_tokens.append(token)
                expanded_tokens.extend(aliases)
            else:
                expanded_tokens.append(token)

        # Preserve insertion order; remove duplicates.
        seen = set()
        deduped_tokens = []
        for token in expanded_tokens:
            if token not in seen:
                seen.add(token)
                deduped_tokens.append(token)

        if not deduped_tokens:
            return query
        return " ".join(deduped_tokens)


def build_query_transformer(mode: str) -> QueryTransformer:
    """Factory for retrieval query-transform modes."""
    if mode in {"identity", "baseline"}:
        return IdentityQueryTransformer()
    if mode == "lexnorm":
        return RootLexicalQueryTransformer()
    raise ValueError(f"Unknown query transformer mode: {mode}")
