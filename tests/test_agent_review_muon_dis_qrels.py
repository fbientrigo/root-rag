from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import yaml


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "agent_review_muon_dis_qrels.py"
    spec = importlib.util.spec_from_file_location("agent_review_muon_dis_qrels", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_candidates(path: Path) -> None:
    payload = {
        "candidates": [
            {
                "query_id": "q02_make_muon_dis",
                "query_text": "makeMuonDIS",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [{"file_path": "muonDIS/makeMuonDIS.py", "start_line": 2, "end_line": 3}],
            },
            {
                "query_id": "q03_run_simscript",
                "query_text": "run_simScript",
                "review_status": "REVIEW_REQUIRED",
                "qrels": [{"file_path": "macro/run_simScript.py", "start_line": 1, "end_line": 2}],
            },
            {
                "query_id": "q09_inactivate_muon_processes",
                "query_text": "InactivateMuonProcesses",
                "review_status": "NOT_FOUND_IN_INDEX",
                "qrels": [],
            },
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_decisions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"decisions": []}, sort_keys=False), encoding="utf-8")


def test_agent_review_writes_yaml_and_markdown_with_snippet(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        fairship = Path("FairShip")
        output = Path("benchmarks/muon_dis/qrels_agent_review_proposals.yaml")
        report = Path("reports/muon_dis_agent_qrel_review.md")
        _write_candidates(candidates)
        _write_decisions(decisions)
        (fairship / "muonDIS").mkdir(parents=True, exist_ok=True)
        (fairship / "macro").mkdir(parents=True, exist_ok=True)
        (fairship / "muonDIS/makeMuonDIS.py").write_text("a\nmakeMuonDIS()\nmakeMuonDIS\nz\n", encoding="utf-8")
        (fairship / "macro/run_simScript.py").write_text("run_simScript\nx\n", encoding="utf-8")
        (Path("benchmarks/muon_dis/qrels.yaml")).write_text("confirmed_qrels: []\n", encoding="utf-8")
        before_qrels = Path("benchmarks/muon_dis/qrels.yaml").read_text(encoding="utf-8")

        rc = module.main(
            [
                "--candidates",
                str(candidates),
                "--decisions",
                str(decisions),
                "--fairship-path",
                str(fairship),
                "--output",
                str(output),
                "--report",
                str(report),
                "--preset",
                "critical-path",
                "--top-per-area",
                "1",
                "--limit",
                "10",
            ]
        )
        assert rc == 0
        assert output.exists()
        assert report.exists()
        payload = yaml.safe_load(output.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert len(payload["proposals"]) >= 2
        first = payload["proposals"][0]
        assert "snippet_excerpt" in first
        assert first["proposed_decision"].startswith("PROPOSE_")
        assert Path("benchmarks/muon_dis/qrels.yaml").read_text(encoding="utf-8") == before_qrels
    finally:
        os.chdir(cwd_before)


def test_agent_review_missing_file_is_not_approved(tmp_path: Path) -> None:
    module = _load_module()
    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        candidates = Path("benchmarks/muon_dis/qrels_candidates.yaml")
        decisions = Path("benchmarks/muon_dis/qrels_review_decisions.yaml")
        _write_candidates(candidates)
        _write_decisions(decisions)
        output = Path("benchmarks/muon_dis/qrels_agent_review_proposals.yaml")
        report = Path("reports/muon_dis_agent_qrel_review.md")
        rc = module.main(
            [
                "--candidates",
                str(candidates),
                "--decisions",
                str(decisions),
                "--fairship-path",
                "MissingFairShip",
                "--output",
                str(output),
                "--report",
                str(report),
            ]
        )
        assert rc == 0
        payload = yaml.safe_load(output.read_text(encoding="utf-8"))
        decisions_out = [row["proposed_decision"] for row in payload["proposals"]]
        assert "PROPOSE_APPROVED" not in decisions_out
    finally:
        os.chdir(cwd_before)


def test_agent_review_not_found_row_handled(tmp_path: Path) -> None:
    module = _load_module()
    rows = [
        module.CandidateRow(
            query_id="q09_inactivate_muon_processes",
            query_text="InactivateMuonProcesses",
            rank=0,
            file_path="NOT_FOUND_IN_INDEX",
            start_line=None,
            end_line=None,
            is_not_found=True,
        )
    ]
    proposals = module._review_rows(rows, tmp_path / "FairShip")
    assert proposals[0]["proposed_decision"] == "PROPOSE_NOT_FOUND_IN_INDEX"


def test_critical_path_selection_balanced() -> None:
    module = _load_module()
    rows = []
    for qid in module.PRESET_CRITICAL_PATH:
        rows.append(module.CandidateRow(qid, qid, 1, f"{qid}.py", 1, 2, False))
    rows.extend(
        [
            module.CandidateRow("q02_make_muon_dis", "x", 2, "a.py", 1, 2, False),
            module.CandidateRow("q02_make_muon_dis", "x", 3, "b.py", 1, 2, False),
        ]
    )
    selected = module._apply_ordering(rows, "critical-path", 1, 10)
    first_ids = [row.query_id for row in selected[:8]]
    assert set(first_ids) == set(module.PRESET_CRITICAL_PATH[:8])


def test_filename_only_match_not_approved() -> None:
    module = _load_module()
    row = module.CandidateRow("qx", "run_simScript", 1, "macro/run_simScript.py", 1, 1, False)
    decision, *_ = module._propose_from_snippet(row, context="1: x", exact="1: print('x')")
    assert decision != "PROPOSE_APPROVED"


def test_short_snippet_defaults_to_needs_context() -> None:
    module = _load_module()
    row = module.CandidateRow("qx", "makeMuonDIS", 1, "muonDIS/makeMuonDIS.py", 1, 2, False)
    decision, *_ = module._propose_from_snippet(
        row,
        context="1: def makeMuonDIS():\n2:     pass",
        exact="1: def makeMuonDIS():\n2:     pass",
    )
    assert decision == "PROPOSE_NEEDS_CONTEXT"


def test_doc_hit_not_automatically_approved() -> None:
    module = _load_module()
    row = module.CandidateRow("qx", "run_simScript", 1, "README.md", 1, 12, False)
    decision, *_ = module._propose_from_snippet(
        row,
        context="1: run_simScript",
        exact="\n".join([f"{i}: run_simScript" for i in range(1, 12)]),
    )
    assert decision == "PROPOSE_REJECTED"


def test_not_found_cannot_become_qrel() -> None:
    module = _load_module()
    rows = [
        module.CandidateRow(
            query_id="q09_inactivate_muon_processes",
            query_text="InactivateMuonProcesses",
            rank=0,
            file_path="NOT_FOUND_IN_INDEX",
            start_line=None,
            end_line=None,
            is_not_found=True,
        )
    ]
    proposals = module._review_rows(rows, Path("FairShip"))
    assert proposals[0]["proposed_decision"] == "PROPOSE_NOT_FOUND_IN_INDEX"
