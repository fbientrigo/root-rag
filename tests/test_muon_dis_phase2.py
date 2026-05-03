"""Phase 2 Muon DIS scaffold and evaluation tests."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


def _load_eval_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_muon_dis_retrieval.py"
    spec = importlib.util.spec_from_file_location("evaluate_muon_dis_retrieval", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_muon_dis_scaffold_keeps_pending_qrels_and_consistent_ids() -> None:
    pack = yaml.safe_load(Path("query_packs/muon_dis_workflow.yaml").read_text(encoding="utf-8"))
    golden = yaml.safe_load(Path("benchmarks/muon_dis/golden_queries.yaml").read_text(encoding="utf-8"))
    qrels = yaml.safe_load(Path("benchmarks/muon_dis/qrels.yaml").read_text(encoding="utf-8"))

    pack_ids = {row["id"] for row in pack["queries"]}
    golden_rows = golden["golden_queries"]
    golden_ids = {row["query_id"] for row in golden_rows}
    pending_rows = qrels["pending_qrels"]
    pending_ids = {row["query_id"] for row in pending_rows}

    assert golden_ids == pack_ids
    assert pending_ids == pack_ids
    assert qrels["confirmed_qrels"] == []
    assert all(row["pending_label"] is True for row in golden_rows)
    assert all(row["qrels"] == [] for row in golden_rows)
    assert all(row["pending_label"] is True for row in pending_rows)
    assert all(row["qrels"] == [] for row in pending_rows)
    assert all(row["review_status"] == "NOT FOUND IN INDEX" for row in pending_rows)


def test_evaluate_muon_dis_marks_pending_without_fabricating_metrics(tmp_path: Path) -> None:
    module = _load_eval_module()
    evidence_dir = tmp_path / "evidence" / "run1"
    evidence_dir.mkdir(parents=True)

    (evidence_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pack_id": "muon_dis_workflow_v1",
                "pack_path": "query_packs/muon_dis_workflow.yaml",
                "timestamp": "2026-04-27T00:00:00+00:00",
                "top_k": 10,
                "queries": [
                    {
                        "id": "q01_muondis_anchor",
                        "query": "muonDIS",
                        "output_file": str(evidence_dir / "q1.json"),
                        "return_code": 0,
                        "stderr": "",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (evidence_dir / "q1.json").write_text(
        json.dumps([{"file_path": "macro/make_nTuple.py", "start_line": 10, "end_line": 20, "score": 1.0}]),
        encoding="utf-8",
    )

    golden_path = tmp_path / "golden.yaml"
    golden_path.write_text(
        yaml.safe_dump(
            {
                "golden_queries": [
                    {
                        "query_id": "q01_muondis_anchor",
                        "query_text": "muonDIS",
                        "pack": "muon_dis_workflow_v1",
                        "qrels": [],
                        "pending_label": True,
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    qrels_path = tmp_path / "qrels.yaml"
    qrels_path.write_text(
        yaml.safe_dump(
            {
                "confirmed_qrels": [],
                "pending_qrels": [
                    {
                        "query_id": "q01_muondis_anchor",
                        "pending_label": True,
                        "qrels": [],
                        "review_status": "NOT FOUND IN INDEX",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = module.evaluate_run(
        evidence_dir=evidence_dir,
        golden_path=golden_path,
        qrels_path=qrels_path,
        top_k_override=None,
    )

    assert result["summary"]["scored_query_count"] == 0
    assert result["summary"]["pending_query_count"] == 1
    assert result["summary"]["macro_precision_at_k"] == 0.0
    assert result["summary"]["macro_recall_at_k"] == 0.0
    assert result["summary"]["qrels_state"] == "NO_CONFIRMED_QRELS"
    assert result["pending_queries"] == ["q01_muondis_anchor"]
    assert result["per_query"][0]["evaluation_state"] == "pending_qrels"


def test_evaluate_muon_dis_scores_confirmed_qrels(tmp_path: Path) -> None:
    module = _load_eval_module()
    evidence_dir = tmp_path / "evidence" / "run2"
    evidence_dir.mkdir(parents=True)

    (evidence_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pack_id": "muon_dis_workflow_v1",
                "pack_path": "query_packs/muon_dis_workflow.yaml",
                "timestamp": "2026-04-27T00:00:00+00:00",
                "top_k": 2,
                "queries": [
                    {
                        "id": "q02_make_muon_dis",
                        "query": "makeMuonDIS",
                        "output_file": str(evidence_dir / "q2.json"),
                        "return_code": 0,
                        "stderr": "",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (evidence_dir / "q2.json").write_text(
        json.dumps(
            [
                {"file_path": "muonDIS/makeMuonDIS.py", "start_line": 5, "end_line": 11, "score": 0.9},
                {"file_path": "muonDIS/other.py", "start_line": 1, "end_line": 3, "score": 0.2},
            ]
        ),
        encoding="utf-8",
    )

    golden_path = tmp_path / "golden.yaml"
    golden_path.write_text(
        yaml.safe_dump(
            {
                "golden_queries": [
                    {
                        "query_id": "q02_make_muon_dis",
                        "query_text": "makeMuonDIS",
                        "pack": "muon_dis_workflow_v1",
                        "qrels": [],
                        "pending_label": False,
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    qrels_path = tmp_path / "qrels.yaml"
    qrels_path.write_text(
        yaml.safe_dump(
            {
                "confirmed_qrels": [
                    {
                        "query_id": "q02_make_muon_dis",
                        "qrels": [
                            {
                                "file_path": "muonDIS/makeMuonDIS.py",
                                "start_line": 5,
                                "end_line": 11,
                                "relevance": 2,
                            }
                        ],
                    }
                ],
                "pending_qrels": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = module.evaluate_run(
        evidence_dir=evidence_dir,
        golden_path=golden_path,
        qrels_path=qrels_path,
        top_k_override=2,
    )

    assert result["summary"]["scored_query_count"] == 1
    assert result["summary"]["pending_query_count"] == 0
    assert result["summary"]["macro_precision_at_k"] == 0.5
    assert result["summary"]["macro_recall_at_k"] == 1.0
    row = result["per_query"][0]
    assert row["scored"] is True
    assert row["retrieved_positive_count"] == 1
    assert row["qrels_positive_count"] == 1


def test_evaluate_muon_dis_text_wrapper_scores_when_hits_are_parsed(tmp_path: Path) -> None:
    """Text-wrapper evidence with parsed hits should be scored like structured retrieval output."""
    module = _load_eval_module()
    evidence_dir = tmp_path / "evidence" / "text_wrapper"
    evidence_dir.mkdir(parents=True)

    (evidence_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pack_id": "muon_dis_workflow_v1",
                "pack_path": "query_packs/muon_dis_workflow.yaml",
                "timestamp": "2026-04-27T00:00:00+00:00",
                "top_k": 10,
                "evidence_format": "text-wrapper",
                "queries": [
                    {
                        "id": "q01_muondis_anchor",
                        "query": "muonDIS",
                        "output_file": str(evidence_dir / "q1.json"),
                        "evidence_format": "text-wrapper",
                        "return_code": 0,
                        "stderr": "",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (evidence_dir / "q1.json").write_text(
        json.dumps(
            {
                "query_id": "q01_muondis_anchor",
                "query": "muonDIS",
                "command": ["root-rag", "ask", "muonDIS", "--top-k", "10"],
                "evidence_format": "text-wrapper",
                "return_code": 0,
                "stdout": "Evidence (ROOT v6-36-08):\n[1] macro/make_nTuple.py:10-20\n",
                "stderr": "",
                "hits": [{"rank": 1, "file": "macro/make_nTuple.py", "start_line": 10, "end_line": 20}],
                "notes": "Raw CLI text output; not structured JSON.",
            }
        ),
        encoding="utf-8",
    )

    golden_path = tmp_path / "golden.yaml"
    golden_path.write_text(
        yaml.safe_dump(
            {
                "golden_queries": [
                    {
                        "query_id": "q01_muondis_anchor",
                        "query_text": "muonDIS",
                        "pack": "muon_dis_workflow_v1",
                        "qrels": [],
                        "pending_label": True,
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    qrels_path = tmp_path / "qrels.yaml"
    qrels_path.write_text(
        yaml.safe_dump(
            {
                "confirmed_qrels": [],
                "pending_qrels": [
                    {
                        "query_id": "q01_muondis_anchor",
                        "pending_label": True,
                        "qrels": [],
                        "review_status": "NOT FOUND IN INDEX",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = module.evaluate_run(
        evidence_dir=evidence_dir,
        golden_path=golden_path,
        qrels_path=qrels_path,
        top_k_override=None,
    )

    assert result["summary"]["scored_query_count"] == 0
    assert result["summary"]["pending_query_count"] == 1
    assert result["summary"]["text_evidence_unsupported_count"] == 0
    assert result["summary"]["macro_precision_at_k"] == 0.0
    assert result["summary"]["macro_recall_at_k"] == 0.0
    assert result["text_evidence_unsupported_queries"] == []
    row = result["per_query"][0]
    assert row["status"] == "HIT"
    assert "metrics_state" not in row
    assert row["evaluation_state"] == "pending_qrels"
    assert row["scored"] is False
