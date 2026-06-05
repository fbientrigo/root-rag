from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "ingest_lxplus_muondis_result.py"
    spec = importlib.util.spec_from_file_location("ingest_lxplus_muondis_result", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_preflight(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_oracle(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _base_oracle() -> dict:
    return {
        "transport_status": "DIS_AND_CBMSIM_FOUND",
        "dis_tree_exists": True,
        "n_DISParticles": 1,
        "n_SoftParticles": 1,
        "n_SBT_hits": 0,
        "n_UBT_hits": 0,
        "n_veto_hits": 0,
        "fiducial_fail": None,
        "wall_like_fail": None,
        "output_valid": True,
        "error_reason": None,
    }


def test_missing_oracle_json(tmp_path: Path) -> None:
    module = _load_module()
    missing = tmp_path / "missing_oracle.json"
    oracle, err = module.load_oracle_json(missing)
    assert oracle is None
    assert err is not None


def test_oracle_json_no_file_status(tmp_path: Path) -> None:
    module = _load_module()
    preflight = _write_preflight(tmp_path / "preflight.log", ["[OK] FAIRSHIP directory exists"])
    oracle_path = _write_oracle(
        tmp_path / "oracle.json",
        {
            **_base_oracle(),
            "transport_status": "NO_FILE",
            "output_valid": False,
            "error_reason": "input file not found",
        },
    )
    p = module.summarize_preflight_log(preflight)
    o, err = module.load_oracle_json(oracle_path)
    status = module.classify_runtime_status(p, o, err)
    assert status == "ROOT_PROBE_FAILED"


def test_oracle_json_root_import_failed_status(tmp_path: Path) -> None:
    module = _load_module()
    preflight = _write_preflight(tmp_path / "preflight.log", ["[OK] FAIRSHIP directory exists"])
    oracle_path = _write_oracle(
        tmp_path / "oracle.json",
        {
            **_base_oracle(),
            "transport_status": "ROOT_IMPORT_FAILED",
            "output_valid": False,
            "error_reason": "failed to import ROOT",
        },
    )
    p = module.summarize_preflight_log(preflight)
    o, err = module.load_oracle_json(oracle_path)
    status = module.classify_runtime_status(p, o, err)
    assert status == "ROOT_PROBE_FAILED"


def test_oracle_json_output_valid_false(tmp_path: Path) -> None:
    module = _load_module()
    preflight = _write_preflight(tmp_path / "preflight.log", ["[OK] FAIRSHIP directory exists"])
    oracle_path = _write_oracle(
        tmp_path / "oracle.json",
        {
            **_base_oracle(),
            "transport_status": "NO_EXPECTED_TREE",
            "output_valid": False,
            "error_reason": "neither DIS nor cbmsim tree found",
        },
    )
    p = module.summarize_preflight_log(preflight)
    o, err = module.load_oracle_json(oracle_path)
    status = module.classify_runtime_status(p, o, err)
    assert status == "ROOT_PROBE_FAILED"


def test_oracle_json_output_valid_true(tmp_path: Path) -> None:
    module = _load_module()
    preflight = _write_preflight(tmp_path / "preflight.log", ["[OK] FAIRSHIP directory exists"])
    oracle_path = _write_oracle(tmp_path / "oracle.json", _base_oracle())
    p = module.summarize_preflight_log(preflight)
    o, err = module.load_oracle_json(oracle_path)
    status = module.classify_runtime_status(p, o, err)
    assert status == "ROOT_PROBE_PASS"


def test_preflight_log_with_fail_lines(tmp_path: Path) -> None:
    module = _load_module()
    preflight = _write_preflight(
        tmp_path / "preflight.log",
        [
            "[OK] FAIRSHIP directory exists",
            "[FAIL] SHIPSOFT is missing or not a directory",
            "[WARN] muonDis.root not found",
        ],
    )
    p = module.summarize_preflight_log(preflight)
    assert p["outcome"] == "FAIL"
    assert len(p["fail_lines"]) == 1


def test_preflight_log_with_warn_no_fail(tmp_path: Path) -> None:
    module = _load_module()
    preflight = _write_preflight(
        tmp_path / "preflight.log",
        [
            "[OK] FAIRSHIP directory exists",
            "[WARN] muonDis.root not found",
        ],
    )
    p = module.summarize_preflight_log(preflight)
    assert p["outcome"] == "WARN"
    assert len(p["fail_lines"]) == 0
    assert len(p["warn_lines"]) == 1
