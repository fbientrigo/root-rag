"""Tests for scripts/freeze_muon_dis_benchmark.py."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import yaml


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "freeze_muon_dis_benchmark.py"
    spec = importlib.util.spec_from_file_location("freeze_muon_dis_benchmark", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_summary(path: Path, run_id: str = "muon_dis_emv_reconciled_01") -> None:
    payload = {
        "run_id": run_id,
        "acceptance_gate_status": "PASS",
        "index_id": "fairship_index",
        "index_dir": "data/indexes_fairship",
        "query_count": 9,
        "hit_count": 8,
        "zero_hit_count": 1,
        "error_count": 0,
        "qrels_state": "NO_CONFIRMED_QRELS",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_qrels(path: Path, confirmed_count: int) -> None:
    confirmed_rows = []
    if confirmed_count > 0:
        confirmed_rows = [
            {
                "query_id": "q01",
                "qrels": [
                    {
                        "file_path": f"muonDIS/makeMuonDIS_{idx}.py",
                        "start_line": 10,
                        "end_line": 20,
                        "relevance": 1,
                    }
                    for idx in range(confirmed_count)
                ],
            }
        ]
    payload = {"pack_id": "muon_dis_workflow_v1", "confirmed_qrels": confirmed_rows, "pending_qrels": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_plain_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_refuses_freeze_when_confirmed_qrels_below_threshold(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        summary = Path("reports/run_vertical_slice_summary.json")
        qrels = Path("benchmarks/muon_dis/qrels.yaml")
        golden = Path("benchmarks/muon_dis/golden_queries.yaml")
        query_pack = Path("query_packs/muon_dis_workflow.yaml")
        output = Path("benchmarks/muon_dis/V0_FREEZE.md")
        json_output = Path("artifacts/benchmarks/muon_dis_v0_freeze.json")

        _write_summary(summary)
        _write_qrels(qrels, confirmed_count=0)
        _write_plain_file(golden, "golden_queries: []\n")
        _write_plain_file(query_pack, "queries: []\n")
        before_qrels = qrels.read_text(encoding="utf-8")

        result = module.freeze_benchmark(
            summary_path=summary,
            qrels_path=qrels,
            golden_path=golden,
            query_pack_path=query_pack,
            output_path=output,
            json_output_path=json_output,
            min_confirmed_qrels=5,
            draft=False,
        )
        after_qrels = qrels.read_text(encoding="utf-8")

        assert result["freeze_status"] == "BLOCKED_INSUFFICIENT_QRELS"
        assert not output.exists()
        assert json_output.exists()
        assert before_qrels == after_qrels
    finally:
        os.chdir(cwd_before)


def test_draft_mode_writes_draft_without_claiming_final_freeze(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        summary = Path("reports/run_vertical_slice_summary.json")
        qrels = Path("benchmarks/muon_dis/qrels.yaml")
        golden = Path("benchmarks/muon_dis/golden_queries.yaml")
        query_pack = Path("query_packs/muon_dis_workflow.yaml")
        output = Path("benchmarks/muon_dis/V0_FREEZE.md")
        json_output = Path("artifacts/benchmarks/muon_dis_v0_freeze.json")

        _write_summary(summary)
        _write_qrels(qrels, confirmed_count=0)
        _write_plain_file(golden, "golden_queries: []\n")
        _write_plain_file(query_pack, "queries: []\n")

        result = module.freeze_benchmark(
            summary_path=summary,
            qrels_path=qrels,
            golden_path=golden,
            query_pack_path=query_pack,
            output_path=output,
            json_output_path=json_output,
            min_confirmed_qrels=5,
            draft=True,
        )

        draft_path = output.with_name("V0_FREEZE_DRAFT.md")
        text = draft_path.read_text(encoding="utf-8")
        assert result["freeze_status"] == "DRAFT_NOT_FINAL"
        assert draft_path.exists()
        assert "DRAFT / NOT A BENCHMARK FREEZE" in text
        assert not output.exists()
    finally:
        os.chdir(cwd_before)


def test_writes_final_freeze_when_confirmed_qrels_meet_threshold(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        summary = Path("reports/run_vertical_slice_summary.json")
        qrels = Path("benchmarks/muon_dis/qrels.yaml")
        golden = Path("benchmarks/muon_dis/golden_queries.yaml")
        query_pack = Path("query_packs/muon_dis_workflow.yaml")
        output = Path("benchmarks/muon_dis/V0_FREEZE.md")
        json_output = Path("artifacts/benchmarks/muon_dis_v0_freeze.json")

        _write_summary(summary)
        _write_qrels(qrels, confirmed_count=6)
        _write_plain_file(golden, "golden_queries: []\n")
        _write_plain_file(query_pack, "queries: []\n")
        before_qrels = qrels.read_text(encoding="utf-8")

        result = module.freeze_benchmark(
            summary_path=summary,
            qrels_path=qrels,
            golden_path=golden,
            query_pack_path=query_pack,
            output_path=output,
            json_output_path=json_output,
            min_confirmed_qrels=5,
            draft=False,
        )
        after_qrels = qrels.read_text(encoding="utf-8")

        assert result["freeze_status"] == "READY_FOR_FREEZE"
        assert output.exists()
        text = output.read_text(encoding="utf-8")
        assert "Muon DIS V0 Benchmark Freeze" in text
        assert "DRAFT / NOT A BENCHMARK FREEZE" not in text
        assert before_qrels == after_qrels
    finally:
        os.chdir(cwd_before)


def test_json_artifact_contains_required_metadata(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        summary = Path("reports/run_vertical_slice_summary.json")
        qrels = Path("benchmarks/muon_dis/qrels.yaml")
        golden = Path("benchmarks/muon_dis/golden_queries.yaml")
        query_pack = Path("query_packs/muon_dis_workflow.yaml")
        output = Path("benchmarks/muon_dis/V0_FREEZE.md")
        json_output = Path("artifacts/benchmarks/muon_dis_v0_freeze.json")

        _write_summary(summary)
        _write_qrels(qrels, confirmed_count=5)
        _write_plain_file(golden, "golden_queries: []\n")
        _write_plain_file(query_pack, "queries: []\n")

        module.freeze_benchmark(
            summary_path=summary,
            qrels_path=qrels,
            golden_path=golden,
            query_pack_path=query_pack,
            output_path=output,
            json_output_path=json_output,
            min_confirmed_qrels=5,
            draft=False,
        )

        payload = json.loads(json_output.read_text(encoding="utf-8"))
        required_keys = {
            "index_id",
            "index_dir",
            "run_id",
            "query_pack_path",
            "golden_queries_path",
            "qrels_path",
            "confirmed_qrel_count",
            "query_count",
            "hit_count",
            "zero_hit_count",
            "error_count",
            "qrels_state",
            "metrics_status",
            "freeze_status",
        }
        assert required_keys.issubset(payload.keys())
    finally:
        os.chdir(cwd_before)

