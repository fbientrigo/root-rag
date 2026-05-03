from pathlib import Path

from root_rag.corpus.manifest import Manifest
from root_rag.parser.chunks import chunk_corpus
from root_rag.parser.files import discover_text_files


def _mock_manifest(repo_root: Path) -> Manifest:
    return Manifest(
        repo_url="https://github.com/ShipSoft/FairShip.git",
        root_ref="master",
        resolved_commit="0123456789abcdef0123456789abcdef01234567",
        local_path=str(repo_root),
        fetched_at="2026-04-16T00:00:00+00:00",
        dirty=False,
        tool_version="0.1.0",
    )


def test_fairship_workflow_discovery_includes_target_files(tmp_path: Path):
    (tmp_path / "README.md").write_text("ShipReco workflow", encoding="utf-8")
    (tmp_path / "macro").mkdir()
    (tmp_path / "macro" / "ShipReco.py").write_text("def ShipReco():\n    pass\n", encoding="utf-8")
    (tmp_path / "macro" / "run_simScript.py").write_text("def run_simScript():\n    pass\n", encoding="utf-8")
    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "shipDigiReco.py").write_text("def shipDigiReco():\n    pass\n", encoding="utf-8")
    (tmp_path / "muonDIS").mkdir()
    (tmp_path / "muonDIS" / "makeMuonDIS.py").write_text("def makeMuonDIS():\n    pass\n", encoding="utf-8")

    files = discover_text_files(tmp_path, discovery_profile="fairship_workflow")
    rel = {p.relative_to(tmp_path).as_posix() for p in files}

    assert "README.md" in rel
    assert "macro/ShipReco.py" in rel
    assert "macro/run_simScript.py" in rel
    assert "python/shipDigiReco.py" in rel
    assert "muonDIS/makeMuonDIS.py" in rel


def test_default_discovery_remains_cpp_only(tmp_path: Path):
    (tmp_path / "README.md").write_text("docs", encoding="utf-8")
    (tmp_path / "macro").mkdir()
    (tmp_path / "macro" / "ShipReco.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "ship.cxx").write_text("class Ship {};\n", encoding="utf-8")

    files = discover_text_files(tmp_path)
    rel = {p.relative_to(tmp_path).as_posix() for p in files}

    assert "ship.cxx" in rel
    assert "README.md" not in rel
    assert "macro/ShipReco.py" not in rel


def test_chunk_corpus_with_fairship_workflow_profile_preserves_cpp_and_adds_scripts(tmp_path: Path):
    (tmp_path / "passive").mkdir()
    (tmp_path / "passive" / "ShipMuonShield.cxx").write_text("class ShipMuonShield {};\n", encoding="utf-8")
    (tmp_path / "macro").mkdir()
    (tmp_path / "macro" / "ShipReco.py").write_text("def ShipReco():\n    return 0\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("run_simScript and shipDigiReco", encoding="utf-8")

    manifest = _mock_manifest(tmp_path)
    chunks = chunk_corpus(manifest, tmp_path, discovery_profile="fairship_workflow")
    paths = {c.file_path for c in chunks}

    assert "passive/ShipMuonShield.cxx" in paths
    assert "macro/ShipReco.py" in paths
    assert "README.md" in paths
