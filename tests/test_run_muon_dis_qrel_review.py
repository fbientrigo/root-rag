"""Tests for scripts/run_muon_dis_qrel_review.py."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import yaml


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_muon_dis_qrel_review.py"
    spec = importlib.util.spec_from_file_location("run_muon_dis_qrel_review", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_qrel_review_creates_candidates_and_report(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        evidence_dir = tmp_path / "evidence" / "run1"
        evidence_dir.mkdir(parents=True)
        manifest = {
            "pack_id": "muon_dis_workflow_v1",
            "queries": [
                {"id": "q01_muondis_anchor", "query": "muonDIS", "status": "HIT_OR_TEXT_EVIDENCE", "output_file": str(evidence_dir / "q01_muondis_anchor.json")},
                {"id": "q09_inactivate_muon_processes", "query": "InactivateMuonProcesses", "status": "ZERO_HIT", "output_file": str(evidence_dir / "q09_inactivate_muon_processes.json")},
            ],
        }
        (evidence_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (evidence_dir / "q01_muondis_anchor.json").write_text(
            json.dumps({"hits": [{"file": "muonDIS/makeMuonDIS.py", "start_line": 351, "end_line": 355}]}),
            encoding="utf-8",
        )
        (evidence_dir / "q09_inactivate_muon_processes.json").write_text(json.dumps({"hits": []}), encoding="utf-8")

        summary = module.generate_qrel_review(
            evidence_dir=evidence_dir,
            run_id="run1",
            candidates_path=Path("benchmarks/muon_dis/qrels_candidates.yaml"),
            overwrite=False,
        )

        candidate_path = Path(summary["candidate_file"])
        payload = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
        assert summary["candidate_count"] == 1
        assert summary["not_found_in_index_count"] == 1
        assert summary["confirmed_qrels_modified"] is False
        assert payload["review_required"] is True
        assert payload["candidates"][0]["review_required"] is True
        assert payload["candidates"][0]["qrels"][0]["review_required"] is True
        assert payload["candidates"][1]["query_id"] == "q09_inactivate_muon_processes"
        assert payload["candidates"][1]["review_status"] == "NOT_FOUND_IN_INDEX"
        assert Path(summary["report_file"]).exists()
    finally:
        os.chdir(cwd_before)


def test_generate_qrel_review_uses_timestamped_output_when_candidates_exist(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        evidence_dir = tmp_path / "evidence" / "run2"
        evidence_dir.mkdir(parents=True)
        (evidence_dir / "manifest.json").write_text(
            json.dumps({"pack_id": "muon_dis_workflow_v1", "queries": [{"id": "q1", "query": "x", "status": "ZERO_HIT"}]}),
            encoding="utf-8",
        )

        candidates_path = tmp_path / "benchmarks" / "muon_dis" / "qrels_candidates.yaml"
        candidates_path.parent.mkdir(parents=True)
        candidates_path.write_text("existing: true\n", encoding="utf-8")

        summary = module.generate_qrel_review(
            evidence_dir=evidence_dir,
            run_id="run2",
            candidates_path=candidates_path,
            overwrite=False,
        )
        assert Path(summary["candidate_file"]) != candidates_path
        assert Path(summary["candidate_file"]).name.startswith("qrels_candidates_")
    finally:
        os.chdir(cwd_before)
