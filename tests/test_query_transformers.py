"""Tests for retrieval query transformers."""

import pytest

from root_rag.retrieval.transformers import (
    IdentityQueryTransformer,
    RootLexicalQueryTransformer,
    build_query_transformer,
)


def test_identity_query_transformer_keeps_text():
    transformer = IdentityQueryTransformer()
    query = "TTree SetBranchAddress usage"
    assert transformer.transform(query) == query


def test_root_lexical_transformer_drops_low_signal_and_expands_aliases():
    transformer = RootLexicalQueryTransformer()
    transformed = transformer.transform("TGeoManager top volume and navigation in FairShip")
    tokens = transformed.split()

    assert "and" not in tokens
    assert "in" not in tokens
    assert "fairship" not in tokens
    assert "tgeomanager" not in tokens
    assert "ggeomanager" in tokens
    assert "gettopvolume" in tokens
    assert "findnode" in tokens


def test_build_query_transformer_factory_modes():
    assert isinstance(build_query_transformer("baseline"), IdentityQueryTransformer)
    assert isinstance(build_query_transformer("identity"), IdentityQueryTransformer)
    assert isinstance(build_query_transformer("lexnorm"), RootLexicalQueryTransformer)


def test_build_query_transformer_rejects_unknown_mode():
    with pytest.raises(ValueError):
        build_query_transformer("unknown")
