"""Inspect existing ROOT outputs and emit a conservative MuonDIS oracle JSON record."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "muondis_oracle_probe_v1"
TOP_LEVEL_FIELDS = [
    "transport_status",
    "dis_tree_exists",
    "n_DISParticles",
    "n_SoftParticles",
    "n_SBT_hits",
    "n_UBT_hits",
    "n_veto_hits",
    "fiducial_fail",
    "wall_like_fail",
    "output_valid",
    "error_reason",
    "inspected_file",
    "trees_found",
    "branches_found",
    "schema_version",
]


def _empty_record(inspected_file: str) -> dict[str, Any]:
    return {
        "transport_status": "UNSET",
        "dis_tree_exists": False,
        "n_DISParticles": None,
        "n_SoftParticles": None,
        "n_SBT_hits": None,
        "n_UBT_hits": None,
        "n_veto_hits": None,
        "fiducial_fail": None,
        "wall_like_fail": None,
        "output_valid": False,
        "error_reason": None,
        "inspected_file": inspected_file,
        "trees_found": [],
        "branches_found": {},
        "schema_version": SCHEMA_VERSION,
    }


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _safe_len(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(len(value))
    except Exception:
        pass
    for method in ("GetEntriesFast", "GetEntries"):
        fn = getattr(value, method, None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:
                continue
    return None


def _tree_branch_names(tree: Any) -> list[str]:
    out: list[str] = []
    branches = tree.GetListOfBranches() if tree is not None else None
    if branches is None:
        return out
    for branch in branches:
        name = getattr(branch, "GetName", None)
        if callable(name):
            out.append(str(name()))
    return out


def _select_event_index(entries: int, requested: int) -> int | None:
    if entries <= 0:
        return None
    if requested < 0:
        return 0
    if requested >= entries:
        return entries - 1
    return requested


def _extract_tree_counts(tree: Any, event_index: int, branch_map: dict[str, str]) -> dict[str, int | None]:
    out: dict[str, int | None] = {k: None for k in branch_map}
    if tree is None:
        return out
    entries = int(tree.GetEntries()) if hasattr(tree, "GetEntries") else 0
    idx = _select_event_index(entries, event_index)
    if idx is None:
        return out
    tree.GetEntry(idx)
    for out_key, branch_name in branch_map.items():
        value = getattr(tree, branch_name, None)
        out[out_key] = _safe_len(value)
    return out


def _extract_explicit_bool_branch(tree: Any, event_index: int, branch_name: str) -> bool | None:
    if tree is None:
        return None
    branch_names = set(_tree_branch_names(tree))
    if branch_name not in branch_names:
        return None
    entries = int(tree.GetEntries()) if hasattr(tree, "GetEntries") else 0
    idx = _select_event_index(entries, event_index)
    if idx is None:
        return None
    tree.GetEntry(idx)
    return _coerce_bool(getattr(tree, branch_name, None))


def probe_file(input_path: Path, event_index: int = 0) -> dict[str, Any]:
    inspected = str(input_path)
    record = _empty_record(inspected)

    if not input_path.exists():
        record["transport_status"] = "NO_FILE"
        record["error_reason"] = f"input file not found: {input_path}"
        return record

    try:
        root_mod = importlib.import_module("ROOT")
    except Exception as exc:  # pragma: no cover - exercised in tests with monkeypatch
        record["transport_status"] = "ROOT_IMPORT_FAILED"
        record["error_reason"] = f"failed to import ROOT: {exc}"
        return record

    tf = root_mod.TFile.Open(str(input_path))
    if not tf or tf.IsZombie():
        record["transport_status"] = "ROOT_FILE_OPEN_FAILED"
        record["error_reason"] = f"failed to open ROOT file: {input_path}"
        return record

    key_names: list[str] = []
    keys = tf.GetListOfKeys()
    if keys is not None:
        for key in keys:
            key_names.append(str(key.GetName()))
    record["trees_found"] = key_names

    dis_tree = tf.Get("DIS")
    cbmsim_tree = tf.Get("cbmsim")
    record["dis_tree_exists"] = dis_tree is not None
    if dis_tree is not None:
        record["branches_found"]["DIS"] = _tree_branch_names(dis_tree)
    if cbmsim_tree is not None:
        record["branches_found"]["cbmsim"] = _tree_branch_names(cbmsim_tree)

    dis_counts = _extract_tree_counts(
        dis_tree,
        event_index,
        {
            "n_DISParticles": "DISParticles",
            "n_SoftParticles": "SoftParticles",
            "n_SBT_hits": "muon_vetoPoints",
            "n_UBT_hits": "muon_UpstreamTaggerPoints",
        },
    )
    for key, value in dis_counts.items():
        record[key] = value

    cbm_counts = _extract_tree_counts(cbmsim_tree, event_index, {"n_veto_hits": "vetoPoint"})
    record["n_veto_hits"] = cbm_counts["n_veto_hits"]

    # Conservative boolean extraction: only explicit same-name branches are accepted.
    record["fiducial_fail"] = _extract_explicit_bool_branch(dis_tree, event_index, "fiducial_fail")
    if record["fiducial_fail"] is None:
        record["fiducial_fail"] = _extract_explicit_bool_branch(cbmsim_tree, event_index, "fiducial_fail")

    record["wall_like_fail"] = _extract_explicit_bool_branch(dis_tree, event_index, "wall_like_fail")
    if record["wall_like_fail"] is None:
        record["wall_like_fail"] = _extract_explicit_bool_branch(cbmsim_tree, event_index, "wall_like_fail")

    if dis_tree is not None and cbmsim_tree is not None:
        record["transport_status"] = "DIS_AND_CBMSIM_FOUND"
    elif dis_tree is not None:
        record["transport_status"] = "DIS_TREE_ONLY"
        record["error_reason"] = "cbmsim tree not found"
    elif cbmsim_tree is not None:
        record["transport_status"] = "CBMSIM_ONLY"
        record["error_reason"] = "DIS tree not found"
    else:
        record["transport_status"] = "NO_EXPECTED_TREE"
        record["error_reason"] = "neither DIS nor cbmsim tree found"

    record["output_valid"] = True
    tf.Close()
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect existing ROOT file and emit MuonDIS oracle JSON.")
    parser.add_argument("--input", required=True, type=Path, help="Input ROOT file to inspect")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output file path")
    parser.add_argument("--event-index", type=int, default=0, help="Event index for branch-length inspection")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = probe_file(args.input, event_index=args.event_index)

    # Guard exact top-level schema in script output.
    ordered = {k: record[k] for k in TOP_LEVEL_FIELDS}
    payload = json.dumps(ordered, indent=2, sort_keys=False)
    print(payload)

    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
