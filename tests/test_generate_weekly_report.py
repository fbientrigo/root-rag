"""Tests for scripts/generate_weekly_report.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    """Load generate_weekly_report.py for direct function tests."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_weekly_report.py"
    spec = importlib.util.spec_from_file_location("generate_weekly_report", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pack(path: Path) -> None:
    """Write a minimal query pack with research question."""
    path.write_text(
        "\n".join(
            [
                "pack_id: muon_dis_workflow_v0",
                "rq: FairShip Muon DIS workflow tracing",
                "created: 2026-04-27",
                "tags:",
                "  - fairship",
                "queries:",
                "  - id: q_hit",
                "    natural_language: \"hit\"",
                "    bm25_tokens: [makeMuonDIS, sigmaDIS]",
                "    expected_files: []",
                "    tier: mvp",
                "    golden: true",
                "  - id: q_zero",
                "    natural_language: \"zero\"",
                "    bm25_tokens: [unknown, token]",
                "    expected_files: []",
                "    tier: mvp",
                "    golden: true",
            ]
        ),
        encoding="utf-8",
    )


def _write_manifest(evidence_dir: Path, pack_path: Path) -> None:
    """Write deterministic manifest fixture."""
    manifest = {
        "pack_path": str(pack_path),
        "pack_id": "muon_dis_workflow_v0",
        "timestamp": "2026-04-27T12:00:00+00:00",
        "queries": [
            {
                "id": "q_hit",
                "query": "makeMuonDIS sigmaDIS",
                "command": ["root-rag", "ask", "makeMuonDIS sigmaDIS", "--top-k", "10", "--json"],
                "output_file": str(evidence_dir / "q_hit.json"),
                "return_code": 0,
                "stderr": "",
            },
            {
                "id": "q_zero",
                "query": "unknown token",
                "command": ["root-rag", "ask", "unknown token", "--top-k", "10", "--json"],
                "output_file": str(evidence_dir / "q_zero.json"),
                "return_code": 0,
                "stderr": "",
            },
        ],
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_generate_report_handles_hit_and_zero_hit(tmp_path: Path) -> None:
    """Report should include HIT and ZERO_HIT rows deterministically."""
    module = _load_module()
    evidence_dir = tmp_path / "evidence" / "run1"
    evidence_dir.mkdir(parents=True)
    pack_path = tmp_path / "query_packs" / "pack.yaml"
    pack_path.parent.mkdir(parents=True)

    _write_pack(pack_path)
    _write_manifest(evidence_dir, pack_path)

    (evidence_dir / "q_hit.json").write_text(
        json.dumps([
            {
                "file_path": "shipgen/muonDIS.py",
                "start_line": 12,
                "end_line": 24,
                "score": 0.88,
            }
        ]),
        encoding="utf-8",
    )
    (evidence_dir / "q_zero.json").write_text("[]", encoding="utf-8")

    output_path = tmp_path / "reports" / "run1.md"
    module.generate_report(evidence_dir, output_path, title="Weekly Muon DIS")

    report = output_path.read_text(encoding="utf-8")
    assert "# Weekly Muon DIS" in report
    assert "## Evidence Summary" in report
    assert "## Retrieval Table" in report
    assert "## Candidate Workflow Nodes" in report
    assert "## Missing or Zero-Hit Queries" in report
    assert "## Errors" in report
    assert "| q_hit | makeMuonDIS sigmaDIS | 1 | shipgen/muonDIS.py | 12-24 | 0.880000 | HIT |" in report
    assert "| q_zero | unknown token | 0 | N/A | N/A | N/A | ZERO_HIT |" in report
    assert "TODO `q_zero`: no hits for query `unknown token`." in report


def test_generate_report_missing_query_json_marks_error(tmp_path: Path) -> None:
    """Missing per-query JSON should be reported as ERROR."""
    module = _load_module()
    evidence_dir = tmp_path / "evidence" / "run_missing"
    evidence_dir.mkdir(parents=True)
    pack_path = tmp_path / "query_packs" / "pack.yaml"
    pack_path.parent.mkdir(parents=True)

    _write_pack(pack_path)
    _write_manifest(evidence_dir, pack_path)

    (evidence_dir / "q_zero.json").write_text("[]", encoding="utf-8")
    # q_hit.json intentionally missing

    output_path = tmp_path / "reports" / "missing.md"
    module.generate_report(evidence_dir, output_path)

    report = output_path.read_text(encoding="utf-8")
    assert "| q_hit | makeMuonDIS sigmaDIS | 0 | N/A | N/A | N/A | ERROR |" in report
    assert "execution/evidence error (`missing-file:q_hit.json`)" in report


def test_generate_report_accepts_object_evidence_shape(tmp_path: Path) -> None:
    """Object payload with `evidence` list should be parsed as HIT."""
    module = _load_module()
    evidence_dir = tmp_path / "evidence" / "obj_shape"
    evidence_dir.mkdir(parents=True)
    pack_path = tmp_path / "query_packs" / "pack.yaml"
    pack_path.parent.mkdir(parents=True)
    _write_pack(pack_path)
    _write_manifest(evidence_dir, pack_path)

    (evidence_dir / "q_hit.json").write_text(
        json.dumps(
            {
                "answer": "stub",
                "evidence": [
                    {
                        "file_path": "macro/run_simScript.py",
                        "start_line": 30,
                        "end_line": 42,
                        "score": 0.77,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (evidence_dir / "q_zero.json").write_text(json.dumps({"evidence": []}), encoding="utf-8")

    output_path = tmp_path / "reports" / "obj_shape.md"
    module.generate_report(evidence_dir, output_path)
    report = output_path.read_text(encoding="utf-8")

    assert "| q_hit | makeMuonDIS sigmaDIS | 1 | macro/run_simScript.py | 30-42 | 0.770000 | HIT |" in report
    assert "| q_zero | unknown token | 0 | N/A | N/A | N/A | ZERO_HIT |" in report


def test_generate_report_handles_text_wrapper_success(tmp_path: Path) -> None:
    """Text-wrapper evidence should be reported without inventing structured hits."""
    module = _load_module()
    evidence_dir = tmp_path / "evidence" / "text_success"
    evidence_dir.mkdir(parents=True)
    pack_path = tmp_path / "query_packs" / "pack.yaml"
    pack_path.parent.mkdir(parents=True)
    _write_pack(pack_path)

    manifest = {
        "pack_path": str(pack_path),
        "pack_id": "muon_dis_workflow_v0",
        "timestamp": "2026-04-27T12:00:00+00:00",
        "evidence_format": "text-wrapper",
        "queries": [
            {
                "id": "q_hit",
                "query": "makeMuonDIS sigmaDIS",
                "command": ["root-rag", "ask", "makeMuonDIS sigmaDIS", "--top-k", "10"],
                "output_file": str(evidence_dir / "q_hit.json"),
                "evidence_format": "text-wrapper",
                "return_code": 0,
                "stderr": "",
            }
        ],
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (evidence_dir / "q_hit.json").write_text(
        json.dumps(
            {
                "query_id": "q_hit",
                "query": "makeMuonDIS sigmaDIS",
                "command": ["root-rag", "ask", "makeMuonDIS sigmaDIS", "--top-k", "10"],
                "evidence_format": "text-wrapper",
                "return_code": 0,
                "stdout": "Evidence (ROOT v6-36-08):\n[1] file.py:1-2\n",
                "stderr": "",
                "hits": [{"rank": 1, "file": "file.py", "start_line": 1, "end_line": 2}],
                "notes": "Raw CLI text output; not structured JSON.",
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "reports" / "text_success.md"
    module.generate_report(evidence_dir, output_path)
    report = output_path.read_text(encoding="utf-8")

    assert "| q_hit | makeMuonDIS sigmaDIS | 1 | file.py | 1-2 | N/A | HIT |" in report
    assert "- `file.py:1-2` from query_ids: q_hit." in report


def test_generate_report_handles_text_wrapper_zero_hit(tmp_path: Path) -> None:
    """Text-wrapper return code 5 should render as ZERO_HIT with zero hits."""
    module = _load_module()
    evidence_dir = tmp_path / "evidence" / "text_zero"
    evidence_dir.mkdir(parents=True)
    pack_path = tmp_path / "query_packs" / "pack.yaml"
    pack_path.parent.mkdir(parents=True)
    _write_pack(pack_path)

    manifest = {
        "pack_path": str(pack_path),
        "pack_id": "muon_dis_workflow_v0",
        "timestamp": "2026-04-27T12:00:00+00:00",
        "evidence_format": "text-wrapper",
        "queries": [
            {
                "id": "q_zero",
                "query": "unknown token",
                "output_file": str(evidence_dir / "q_zero.json"),
                "evidence_format": "text-wrapper",
                "return_code": 5,
                "stderr": "",
            }
        ],
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (evidence_dir / "q_zero.json").write_text(
        json.dumps(
            {
                "query_id": "q_zero",
                "query": "unknown token",
                "command": ["root-rag", "ask", "unknown token", "--top-k", "10"],
                "evidence_format": "text-wrapper",
                "return_code": 5,
                "stdout": "No evidence found",
                "stderr": "",
                "hits": None,
                "notes": "Raw CLI text output; not structured JSON.",
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "reports" / "text_zero.md"
    module.generate_report(evidence_dir, output_path)
    report = output_path.read_text(encoding="utf-8")

    assert "| q_zero | unknown token | 0 | N/A | N/A | N/A | ZERO_HIT |" in report
    assert "TODO `q_zero`: no hits for query `unknown token`." in report


def test_generate_report_handles_text_wrapper_error(tmp_path: Path) -> None:
    """Text-wrapper return code 2 should render as ERROR without crashing."""
    module = _load_module()
    evidence_dir = tmp_path / "evidence" / "text_error"
    evidence_dir.mkdir(parents=True)
    pack_path = tmp_path / "query_packs" / "pack.yaml"
    pack_path.parent.mkdir(parents=True)
    _write_pack(pack_path)

    manifest = {
        "pack_path": str(pack_path),
        "pack_id": "muon_dis_workflow_v0",
        "timestamp": "2026-04-27T12:00:00+00:00",
        "evidence_format": "text-wrapper",
        "queries": [
            {
                "id": "q_hit",
                "query": "makeMuonDIS sigmaDIS",
                "output_file": str(evidence_dir / "q_hit.json"),
                "evidence_format": "text-wrapper",
                "return_code": 2,
                "stderr": "Error: No such option: --json",
            }
        ],
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (evidence_dir / "q_hit.json").write_text(
        json.dumps(
            {
                "query_id": "q_hit",
                "query": "makeMuonDIS sigmaDIS",
                "command": ["root-rag", "ask", "makeMuonDIS sigmaDIS", "--top-k", "10"],
                "evidence_format": "text-wrapper",
                "return_code": 2,
                "stdout": "",
                "stderr": "Error: No such option: --json",
                "hits": None,
                "notes": "Raw CLI text output; not structured JSON.",
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "reports" / "text_error.md"
    module.generate_report(evidence_dir, output_path)
    report = output_path.read_text(encoding="utf-8")

    assert "| q_hit | makeMuonDIS sigmaDIS | 0 | N/A | N/A | N/A | ERROR |" in report
    assert "execution/evidence error (`return-code:2:q_hit.json`)" in report


def test_generate_report_handles_legacy_empty_evidence_file(tmp_path: Path) -> None:
    """A legacy zero-byte evidence file should be an ERROR row, not a crash."""
    module = _load_module()
    evidence_dir = tmp_path / "evidence" / "empty"
    evidence_dir.mkdir(parents=True)
    pack_path = tmp_path / "query_packs" / "pack.yaml"
    pack_path.parent.mkdir(parents=True)
    _write_pack(pack_path)

    manifest = {
        "pack_path": str(pack_path),
        "pack_id": "muon_dis_workflow_v0",
        "timestamp": "2026-04-27T12:00:00+00:00",
        "queries": [
            {
                "id": "q_hit",
                "query": "makeMuonDIS sigmaDIS",
                "output_file": str(evidence_dir / "q_hit.json"),
                "return_code": 2,
                "stderr": "Error: No such option: --json",
            }
        ],
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (evidence_dir / "q_hit.json").write_text("", encoding="utf-8")

    output_path = tmp_path / "reports" / "empty.md"
    module.generate_report(evidence_dir, output_path)
    report = output_path.read_text(encoding="utf-8")

    assert "| q_hit | makeMuonDIS sigmaDIS | 0 | N/A | N/A | N/A | ERROR |" in report
    assert "execution/evidence error (`empty-file:q_hit.json`)" in report


def test_generate_report_resolves_manifest_output_paths_from_repo_root(tmp_path: Path) -> None:
    """Manifest output_file may already be relative to repository root."""
    module = _load_module()
    root = tmp_path
    evidence_dir = root / "evidence" / "run_rel"
    evidence_dir.mkdir(parents=True)
    pack_path = root / "query_packs" / "pack.yaml"
    pack_path.parent.mkdir(parents=True)
    _write_pack(pack_path)

    manifest = {
        "pack_path": str(pack_path),
        "pack_id": "muon_dis_workflow_v0",
        "timestamp": "2026-04-27T12:00:00+00:00",
        "queries": [
            {
                "id": "q_hit",
                "query": "makeMuonDIS sigmaDIS",
                "command": ["root-rag", "ask", "makeMuonDIS sigmaDIS", "--top-k", "10", "--json"],
                "output_file": "evidence/run_rel/q_hit.json",
                "return_code": 0,
                "stderr": "",
            }
        ],
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (evidence_dir / "q_hit.json").write_text(
        json.dumps([{"file_path": "muonDIS/makeMuonDIS.py", "start_line": 1, "end_line": 8, "score": 0.91}]),
        encoding="utf-8",
    )

    output_path = root / "reports" / "run_rel.md"
    cwd_before = Path.cwd()
    try:
        import os
        os.chdir(root)
        module.generate_report(evidence_dir, output_path)
    finally:
        os.chdir(cwd_before)

    report = output_path.read_text(encoding="utf-8")
    assert "| q_hit | makeMuonDIS sigmaDIS | 1 | muonDIS/makeMuonDIS.py | 1-8 | 0.910000 | HIT |" in report


def test_main_returns_error_on_bad_manifest(tmp_path: Path) -> None:
    """Main should return non-zero when manifest is invalid."""
    module = _load_module()
    evidence_dir = tmp_path / "evidence" / "bad_manifest"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "manifest.json").write_text("{}", encoding="utf-8")

    output_path = tmp_path / "reports" / "bad.md"
    exit_code = module.main([
        "--evidence-dir",
        str(evidence_dir),
        "--output",
        str(output_path),
    ])

    assert exit_code == 1
    assert not output_path.exists()
