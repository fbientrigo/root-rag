from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "lxplus_muondis_oracle_probe.py"
    spec = importlib.util.spec_from_file_location("lxplus_muondis_oracle_probe", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_file_returns_no_file(tmp_path: Path) -> None:
    module = _load_module()
    target = tmp_path / "nope.root"
    rec = module.probe_file(target)
    assert rec["transport_status"] == "NO_FILE"
    assert rec["output_valid"] is False
    assert rec["dis_tree_exists"] is False
    assert rec["error_reason"] is not None


def test_root_import_failure_handled(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    existing = tmp_path / "dummy.root"
    existing.write_text("x", encoding="utf-8")

    def fake_import(name: str):
        if name == "ROOT":
            raise ImportError("ROOT missing")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    rec = module.probe_file(existing)
    assert rec["transport_status"] == "ROOT_IMPORT_FAILED"
    assert rec["output_valid"] is False
    assert "ROOT" in rec["error_reason"]


class _FakeBranch:
    def __init__(self, name: str):
        self._name = name

    def GetName(self) -> str:
        return self._name


class _FakeKey:
    def __init__(self, name: str):
        self._name = name

    def GetName(self) -> str:
        return self._name


class _FakeCollection:
    def __init__(self, n: int):
        self._n = n

    def GetEntriesFast(self) -> int:
        return self._n


class _FakeTree:
    def __init__(self, name: str, branches: dict[str, object], entries: int = 1):
        self._name = name
        self._branches = branches
        self._entries = entries

    def GetListOfBranches(self):
        return [_FakeBranch(k) for k in self._branches]

    def GetEntries(self) -> int:
        return self._entries

    def GetEntry(self, _idx: int) -> int:
        for key, value in self._branches.items():
            setattr(self, key, value)
        return 1


class _FakeFile:
    def __init__(self, trees: dict[str, _FakeTree]):
        self._trees = trees

    def IsZombie(self) -> bool:
        return False

    def GetListOfKeys(self):
        return [_FakeKey(name) for name in self._trees]

    def Get(self, name: str):
        return self._trees.get(name)

    def Close(self) -> None:
        return None


def test_missing_branches_return_null(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    existing = tmp_path / "fake.root"
    existing.write_text("fake", encoding="utf-8")

    fake_dis = _FakeTree("DIS", {}, entries=1)
    fake_file = _FakeFile({"DIS": fake_dis})

    class FakeTFile:
        @staticmethod
        def Open(_path: str):
            return fake_file

    class FakeROOT:
        TFile = FakeTFile

    monkeypatch.setattr(module.importlib, "import_module", lambda name: FakeROOT if name == "ROOT" else None)
    rec = module.probe_file(existing)
    assert rec["dis_tree_exists"] is True
    assert rec["n_DISParticles"] is None
    assert rec["n_SoftParticles"] is None
    assert rec["n_SBT_hits"] is None
    assert rec["n_UBT_hits"] is None
    assert rec["n_veto_hits"] is None


def test_json_schema_fields_exact(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_module()
    existing = tmp_path / "fake.root"
    existing.write_text("fake", encoding="utf-8")

    dis_tree = _FakeTree(
        "DIS",
        {
            "DISParticles": _FakeCollection(2),
            "SoftParticles": _FakeCollection(3),
            "muon_vetoPoints": _FakeCollection(4),
            "muon_UpstreamTaggerPoints": _FakeCollection(5),
        },
        entries=1,
    )
    cbm_tree = _FakeTree("cbmsim", {"vetoPoint": _FakeCollection(6)}, entries=1)
    fake_file = _FakeFile({"DIS": dis_tree, "cbmsim": cbm_tree})

    class FakeTFile:
        @staticmethod
        def Open(_path: str):
            return fake_file

    class FakeROOT:
        TFile = FakeTFile

    monkeypatch.setattr(module.importlib, "import_module", lambda name: FakeROOT if name == "ROOT" else None)
    monkeypatch.setattr(sys, "argv", ["probe", "--input", str(existing)])

    rc = module.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert list(payload.keys()) == module.TOP_LEVEL_FIELDS
    assert payload["n_DISParticles"] == 2
    assert payload["n_SoftParticles"] == 3
    assert payload["n_SBT_hits"] == 4
    assert payload["n_UBT_hits"] == 5
    assert payload["n_veto_hits"] == 6
