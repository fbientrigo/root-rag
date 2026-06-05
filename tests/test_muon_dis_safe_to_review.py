from __future__ import annotations

from pathlib import Path

import yaml


def _load_yaml(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_safe_shortlist_requires_proposal_redteam_agreement() -> None:
    safe = _load_yaml("benchmarks/muon_dis/qrels_safe_to_review.yaml")
    props = _load_yaml("benchmarks/muon_dis/qrels_agent_review_proposals.yaml")
    red = _load_yaml("benchmarks/muon_dis/qrels_agent_review_redteam.yaml")

    p_index = {
        (r["query_id"], r["file_path"], r["start_line"], r["end_line"], r["rank"]): r
        for r in props["proposals"]
    }
    r_index = {
        (r["query_id"], r["file_path"], r["start_line"], r["end_line"]): r
        for r in red["reviews"]
    }

    for row in safe["safe_to_review"]:
        k_prop = (row["query_id"], row["file_path"], row["start_line"], row["end_line"], row["rank"])
        k_red = (row["query_id"], row["file_path"], row["start_line"], row["end_line"])
        assert k_prop in p_index
        assert k_red in r_index
        p = p_index[k_prop]
        rr = r_index[k_red]
        assert p["proposed_decision"] == "PROPOSE_APPROVED"
        assert p["confidence"] == "HIGH"
        assert p["proposed_relevance"] == 3
        assert rr["redteam_decision"] == "KEEP_APPROVED"
        assert rr["redteam_confidence"] in {"HIGH", "MEDIUM"}
        assert row["file_path"] != "NOT_FOUND_IN_INDEX"
        assert row["start_line"] is not None
        assert row["end_line"] is not None


def test_downgraded_proposal_excluded() -> None:
    safe = _load_yaml("benchmarks/muon_dis/qrels_safe_to_review.yaml")
    excluded = {
        (r["query_id"], r["file_path"], r["start_line"], r["end_line"], r["rank"]): r["exclusion_reason"]
        for r in safe["excluded"]
    }
    assert excluded[("q03_run_simscript", "muonShieldOptimization/run_prod.py", 1, 80, 1)] == "downgraded by red-team"


def test_not_found_excluded() -> None:
    safe = _load_yaml("benchmarks/muon_dis/qrels_safe_to_review.yaml")
    not_found = [r for r in safe["excluded"] if r["file_path"] == "NOT_FOUND_IN_INDEX"]
    assert len(not_found) == 1
    assert not_found[0]["exclusion_reason"] == "not found"


def test_qrels_files_not_modified() -> None:
    qrels = _load_yaml("benchmarks/muon_dis/qrels.yaml")
    decisions = _load_yaml("benchmarks/muon_dis/qrels_review_decisions.yaml")
    assert qrels.get("confirmed_qrels", []) == []
    pending = qrels.get("pending_qrels", [])
    assert len(pending) == 9
    assert all(item.get("review_status") == "NOT FOUND IN INDEX" for item in pending)
    assert decisions.get("review_id") == "muon_dis_qrel_review_2026_04_30_guarded_scaffold"
