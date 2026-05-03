"""Tests for scripts/lint_wiki_claims.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "lint_wiki_claims.py"
    spec = importlib.util.spec_from_file_location("lint_wiki_claims", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_md(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_confirmed_claim(tmp_path: Path) -> None:
    module = _load_module()
    _write_md(
        tmp_path,
        "docs/wiki/a.md",
        "<!-- CLAIM: CONFIRMED -->\nClaim.\n<!-- SOURCE: macro/run.py:10-20 -->\n",
    )
    assert module.lint_wiki_claims(tmp_path / "docs/wiki") == []


def test_invalid_confirmed_claim_without_source(tmp_path: Path) -> None:
    module = _load_module()
    _write_md(tmp_path, "docs/wiki/a.md", "<!-- CLAIM: CONFIRMED -->\nClaim only.\n")
    errors = module.lint_wiki_claims(tmp_path / "docs/wiki")
    assert len(errors) == 1
    assert "CONFIRMED claim requires at least one SOURCE" in errors[0]


def test_valid_provisional_claim_with_todo(tmp_path: Path) -> None:
    module = _load_module()
    _write_md(tmp_path, "docs/wiki/a.md", "<!-- CLAIM: PROVISIONAL -->\nTODO: verify\n")
    assert module.lint_wiki_claims(tmp_path / "docs/wiki") == []


def test_valid_unresolved_claim_with_next_action(tmp_path: Path) -> None:
    module = _load_module()
    _write_md(tmp_path, "docs/wiki/a.md", "<!-- CLAIM: UNRESOLVED -->\nNext action: run query\n")
    assert module.lint_wiki_claims(tmp_path / "docs/wiki") == []


def test_invalid_unresolved_claim_without_next_action(tmp_path: Path) -> None:
    module = _load_module()
    _write_md(tmp_path, "docs/wiki/a.md", "<!-- CLAIM: UNRESOLVED -->\nMissing follow-up.\n")
    errors = module.lint_wiki_claims(tmp_path / "docs/wiki")
    assert len(errors) == 1
    assert "UNRESOLVED claim requires 'Next action:'" in errors[0]


def test_valid_superseded_claim(tmp_path: Path) -> None:
    module = _load_module()
    _write_md(tmp_path, "docs/wiki/a.md", "<!-- CLAIM: SUPERSEDED -->\nSuperseded by: claim.v2\n")
    assert module.lint_wiki_claims(tmp_path / "docs/wiki") == []


def test_invalid_superseded_claim_without_replacement(tmp_path: Path) -> None:
    module = _load_module()
    _write_md(tmp_path, "docs/wiki/a.md", "<!-- CLAIM: SUPERSEDED -->\nOld claim.\n")
    errors = module.lint_wiki_claims(tmp_path / "docs/wiki")
    assert len(errors) == 1
    assert "SUPERSEDED claim requires 'Superseded by:'" in errors[0]
