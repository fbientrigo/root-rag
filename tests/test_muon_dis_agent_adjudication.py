from __future__ import annotations

from pathlib import Path

import yaml


def _load_yaml(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_agent_adjudication_created_from_safe_shortlist() -> None:
    safe = _load_yaml("benchmarks/muon_dis/qrels_safe_to_review.yaml")
    adjudicated = _load_yaml("benchmarks/muon_dis/qrels_agent_adjudicated.yaml")

    assert isinstance(adjudicated, dict)
    assert "adjudicated" in adjudicated
    safe_keys = {
        (r["query_id"], r["file_path"], r["start_line"], r["end_line"], r["rank"])
        for r in safe["safe_to_review"]
    }
    adjudicated_keys = {
        (r["query_id"], r["file_path"], r["start_line"], r["end_line"], r["rank"])
        for r in adjudicated["adjudicated"]
    }
    assert adjudicated_keys == safe_keys


def test_agent_approved_requires_proposal_redteam_agreement() -> None:
    adjudicated = _load_yaml("benchmarks/muon_dis/qrels_agent_adjudicated.yaml")
    proposals = _load_yaml("benchmarks/muon_dis/qrels_agent_review_proposals.yaml")
    redteam = _load_yaml("benchmarks/muon_dis/qrels_agent_review_redteam.yaml")

    p_index = {
        (r["query_id"], r["file_path"], r["start_line"], r["end_line"], r["rank"]): r
        for r in proposals["proposals"]
    }
    r_index = {
        (r["query_id"], r["file_path"], r["start_line"], r["end_line"]): r
        for r in redteam["reviews"]
    }

    for row in adjudicated["adjudicated"]:
        if row["agent_decision"] != "AGENT_APPROVED":
            continue
        p = p_index[(row["query_id"], row["file_path"], row["start_line"], row["end_line"], row["rank"])]
        rt = r_index[(row["query_id"], row["file_path"], row["start_line"], row["end_line"])]
        assert p["proposed_decision"] == "PROPOSE_APPROVED"
        assert p["proposed_relevance"] == 3
        assert p["confidence"] in {"HIGH", "MEDIUM"}
        assert rt["redteam_decision"] == "KEEP_APPROVED"
        assert rt["redteam_confidence"] in {"HIGH", "MEDIUM"}
        assert row["file_path"] != "NOT_FOUND_IN_INDEX"
        assert row["start_line"] is not None
        assert row["end_line"] is not None
        assert row["human_review_required"] is True


def test_not_found_in_index_cannot_be_agent_approved() -> None:
    adjudicated = _load_yaml("benchmarks/muon_dis/qrels_agent_adjudicated.yaml")
    for row in adjudicated["adjudicated"]:
        assert not (row["agent_decision"] == "AGENT_APPROVED" and row["file_path"] == "NOT_FOUND_IN_INDEX")


def test_qrels_yaml_not_modified() -> None:
    qrels = _load_yaml("benchmarks/muon_dis/qrels.yaml")
    assert qrels.get("confirmed_qrels", []) == []


def test_qrels_review_decisions_not_modified() -> None:
    decisions = _load_yaml("benchmarks/muon_dis/qrels_review_decisions.yaml")
    assert decisions.get("review_id") == "muon_dis_qrel_review_2026_04_30_guarded_scaffold"
