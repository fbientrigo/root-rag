"""Tests for scripts/run_query_pack.py."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_run_query_pack_module():
    """Load run_query_pack.py as a module for direct function testing."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_query_pack.py"
    spec = importlib.util.spec_from_file_location("run_query_pack", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pack(path: Path) -> None:
    """Write minimal deterministic query pack fixture."""
    path.write_text(
        "\n".join(
            [
                "pack_id: test_pack",
                "rq: test",
                "created: 2026-04-27",
                "tags:",
                "  - test",
                "queries:",
                "  - id: q1",
                "    natural_language: \"first\"",
                "    bm25_tokens:",
                "      - alpha",
                "      - beta",
                "    expected_files: []",
                "    tier: mvp",
                "    golden: true",
                "  - id: q2",
                "    natural_language: \"second\"",
                "    bm25_tokens:",
                "      - gamma",
                "    expected_files: []",
                "    tier: mvp",
                "    golden: false",
            ]
        ),
        encoding="utf-8",
    )


def test_load_query_pack_parses_yaml(tmp_path: Path) -> None:
    """YAML parsing should load expected pack id and query count."""
    module = _load_run_query_pack_module()
    pack_path = tmp_path / "pack.yaml"
    _write_pack(pack_path)

    payload = module.load_query_pack(pack_path)

    assert payload["pack_id"] == "test_pack"
    assert len(payload["queries"]) == 2


def test_run_query_pack_writes_manifest_and_outputs_with_mocked_subprocess(tmp_path: Path, monkeypatch) -> None:
    """Runner should save per-query outputs and manifest metadata."""
    module = _load_run_query_pack_module()
    pack_path = tmp_path / "pack.yaml"
    output_dir = tmp_path / "evidence" / "run1"
    _write_pack(pack_path)

    calls = []
    monkeypatch.setattr(module.shutil, "which", lambda name: "C:/bin/root-rag" if name == "root-rag" else None)

    def fake_run(command, capture_output, text, check):
        calls.append(command)
        stdout = "Evidence (ROOT v6-36-08):\n\n[1] x.cpp:1-2\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    config = module.RunConfig(
        pack_path=pack_path,
        output_dir=output_dir,
        top_k=7,
        index_dir=None,
        index_id=None,
        root_ref=None,
        evidence_format="text-wrapper",
        dry_run=False,
        fail_fast=False,
        root_rag_cmd=None,
    )

    manifest, exit_code = module.run_query_pack(config)

    assert exit_code == 0
    assert len(calls) == 2
    assert calls[0] == ["root-rag", "ask", "alpha beta", "--top-k", "7"]
    assert calls[1] == ["root-rag", "ask", "gamma", "--top-k", "7"]
    assert "--json" not in calls[0]

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest["pack_id"] == "test_pack"
    assert loaded_manifest["evidence_format"] == "text-wrapper"
    assert len(loaded_manifest["queries"]) == 2
    assert loaded_manifest["queries"][0]["return_code"] == 0
    assert loaded_manifest["queries"][0]["evidence_format"] == "text-wrapper"

    q1_output = output_dir / "q1.json"
    q2_output = output_dir / "q2.json"
    assert q1_output.exists()
    assert q2_output.exists()
    assert q1_output.stat().st_size > 0
    q1_wrapper = json.loads(q1_output.read_text(encoding="utf-8"))
    assert q1_wrapper["evidence_format"] == "text-wrapper"
    assert q1_wrapper["return_code"] == 0
    assert q1_wrapper["hits"] == [{"rank": 1, "file": "x.cpp", "start_line": 1, "end_line": 2}]
    assert q1_wrapper["stdout"].startswith("Evidence (ROOT v6-36-08):")
    assert manifest == loaded_manifest


def test_run_query_pack_dry_run_skips_subprocess(tmp_path: Path, monkeypatch) -> None:
    """Dry-run mode should not execute subprocess and still emit artifacts."""
    module = _load_run_query_pack_module()
    pack_path = tmp_path / "pack.yaml"
    output_dir = tmp_path / "evidence" / "dry"
    _write_pack(pack_path)
    monkeypatch.setattr(module.shutil, "which", lambda name: "C:/bin/root-rag" if name == "root-rag" else None)

    def raise_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called in dry-run mode")

    monkeypatch.setattr(module.subprocess, "run", raise_if_called)

    config = module.RunConfig(
        pack_path=pack_path,
        output_dir=output_dir,
        top_k=10,
        index_dir=None,
        index_id=None,
        root_ref=None,
        evidence_format="text-wrapper",
        dry_run=True,
        fail_fast=False,
        root_rag_cmd=None,
    )

    manifest, exit_code = module.run_query_pack(config)

    assert exit_code == 0
    assert len(manifest["queries"]) == 2
    wrapper = json.loads((output_dir / "q1.json").read_text(encoding="utf-8"))
    assert wrapper["evidence_format"] == "text-wrapper"
    assert json.loads(wrapper["stdout"])["dry_run"] is True
    assert wrapper["command"] == ["root-rag", "ask", "alpha beta", "--top-k", "10"]


def test_run_query_pack_passes_explicit_index_options(tmp_path: Path, monkeypatch) -> None:
    """Runner should pass explicit index options through to root-rag ask."""
    module = _load_run_query_pack_module()
    pack_path = tmp_path / "pack.yaml"
    output_dir = tmp_path / "evidence" / "indexed"
    index_dir = tmp_path / "indexes_fairship"
    _write_pack(pack_path)

    calls = []
    monkeypatch.setattr(module.shutil, "which", lambda name: "C:/bin/root-rag" if name == "root-rag" else None)

    def fake_run(command, capture_output, text, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="Evidence", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    config = module.RunConfig(
        pack_path=pack_path,
        output_dir=output_dir,
        top_k=3,
        index_dir=index_dir,
        index_id="idx_fairship_latest",
        root_ref=None,
        evidence_format="text-wrapper",
        dry_run=False,
        fail_fast=False,
        root_rag_cmd=None,
    )

    manifest, exit_code = module.run_query_pack(config)

    assert exit_code == 0
    assert calls[0] == [
        "root-rag",
        "ask",
        "alpha beta",
        "--top-k",
        "3",
        "--index-dir",
        str(index_dir),
        "--index-id",
        "idx_fairship_latest",
    ]
    assert manifest["index_dir"] == str(index_dir)
    assert manifest["index_id"] == "idx_fairship_latest"
    wrapper = json.loads((output_dir / "q1.json").read_text(encoding="utf-8"))
    assert wrapper["command"] == calls[0]


def test_run_query_pack_fail_fast_stops_after_first_error(tmp_path: Path, monkeypatch) -> None:
    """Fail-fast should stop processing after first command failure."""
    module = _load_run_query_pack_module()
    pack_path = tmp_path / "pack.yaml"
    output_dir = tmp_path / "evidence" / "fail_fast"
    _write_pack(pack_path)
    monkeypatch.setattr(module.shutil, "which", lambda name: "C:/bin/root-rag" if name == "root-rag" else None)

    call_count = {"count": 0}

    def fake_run(command, capture_output, text, check):
        call_count["count"] += 1
        return subprocess.CompletedProcess(command, 3, stdout="", stderr="boom")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    config = module.RunConfig(
        pack_path=pack_path,
        output_dir=output_dir,
        top_k=10,
        index_dir=None,
        index_id=None,
        root_ref=None,
        evidence_format="text-wrapper",
        dry_run=False,
        fail_fast=True,
        root_rag_cmd=None,
    )

    manifest, exit_code = module.run_query_pack(config)

    assert exit_code == 1
    assert call_count["count"] == 1
    assert len(manifest["queries"]) == 1
    assert manifest["queries"][0]["stderr"] == "boom"
    assert json.loads((output_dir / "q1.json").read_text(encoding="utf-8"))["return_code"] == 3


def test_run_query_pack_return_code_5_writes_zero_hit_wrapper(tmp_path: Path, monkeypatch) -> None:
    """No-evidence return code should emit a non-empty wrapper and not fail the run."""
    module = _load_run_query_pack_module()
    pack_path = tmp_path / "pack.yaml"
    output_dir = tmp_path / "evidence" / "zero"
    _write_pack(pack_path)
    monkeypatch.setattr(module.shutil, "which", lambda name: "C:/bin/root-rag" if name == "root-rag" else None)

    def fake_run(command, capture_output, text, check):
        return subprocess.CompletedProcess(command, 5, stdout="No evidence found", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    config = module.RunConfig(
        pack_path=pack_path,
        output_dir=output_dir,
        top_k=10,
        index_dir=None,
        index_id=None,
        root_ref=None,
        evidence_format="text-wrapper",
        dry_run=False,
        fail_fast=False,
        root_rag_cmd=None,
    )

    manifest, exit_code = module.run_query_pack(config)

    assert exit_code == 0
    assert manifest["queries"][0]["status"] == "ZERO_HIT"
    wrapper = json.loads((output_dir / "q1.json").read_text(encoding="utf-8"))
    assert wrapper["return_code"] == 5
    assert wrapper["stdout"] == "No evidence found"
    assert wrapper["hits"] == []
    assert (output_dir / "q1.json").stat().st_size > 0


def test_run_query_pack_return_code_2_writes_error_wrapper(tmp_path: Path, monkeypatch) -> None:
    """CLI usage errors should still leave inspectable non-empty evidence wrappers."""
    module = _load_run_query_pack_module()
    pack_path = tmp_path / "pack.yaml"
    output_dir = tmp_path / "evidence" / "error"
    _write_pack(pack_path)
    monkeypatch.setattr(module.shutil, "which", lambda name: "C:/bin/root-rag" if name == "root-rag" else None)

    def fake_run(command, capture_output, text, check):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="Error: No such option: --json")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    config = module.RunConfig(
        pack_path=pack_path,
        output_dir=output_dir,
        top_k=10,
        index_dir=None,
        index_id=None,
        root_ref=None,
        evidence_format="text-wrapper",
        dry_run=False,
        fail_fast=False,
        root_rag_cmd=None,
    )

    manifest, exit_code = module.run_query_pack(config)

    assert exit_code == 1
    assert manifest["queries"][0]["status"] == "ERROR"
    wrapper = json.loads((output_dir / "q1.json").read_text(encoding="utf-8"))
    assert wrapper["return_code"] == 2
    assert wrapper["stderr"] == "Error: No such option: --json"
    assert wrapper["hits"] == []
    assert (output_dir / "q1.json").stat().st_size > 0


def test_parse_text_wrapper_hits_extracts_rank_file_and_range() -> None:
    """Parser should extract ranked file ranges from text-wrapper stdout lines."""
    module = _load_run_query_pack_module()
    stdout = (
        "Evidence (ROOT master, commit 98de16a5b264):\n\n"
        "[1] muonDIS/makeMuonDIS.py:351-355\n"
        "[2] macro/run_simScript.py:841-902\n"
    )

    hits = module._parse_text_wrapper_hits(stdout)

    assert hits == [
        {"rank": 1, "file": "muonDIS/makeMuonDIS.py", "start_line": 351, "end_line": 355},
        {"rank": 2, "file": "macro/run_simScript.py", "start_line": 841, "end_line": 902},
    ]


def test_run_query_pack_warns_when_output_dir_contains_manifest(tmp_path: Path, monkeypatch, capsys) -> None:
    """Reusing an evidence directory should emit a clear warning and persist it in manifest."""
    module = _load_run_query_pack_module()
    pack_path = tmp_path / "pack.yaml"
    output_dir = tmp_path / "evidence" / "existing"
    _write_pack(pack_path)
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module.shutil, "which", lambda name: "C:/bin/root-rag" if name == "root-rag" else None)

    def fake_run(command, capture_output, text, check):
        return subprocess.CompletedProcess(command, 0, stdout="Evidence", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    config = module.RunConfig(
        pack_path=pack_path,
        output_dir=output_dir,
        top_k=10,
        index_dir=None,
        index_id=None,
        root_ref=None,
        evidence_format="text-wrapper",
        dry_run=False,
        fail_fast=False,
        root_rag_cmd=None,
    )

    manifest, exit_code = module.run_query_pack(config)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "output-dir already contains manifest.json" in captured.out
    assert manifest["output_dir_reused"] is True
    assert len(manifest["warnings"]) == 1
    assert "Prefer a fresh evidence directory." in manifest["warnings"][0]


def test_run_query_pack_fallbacks_to_python_module_when_root_rag_missing(tmp_path: Path, monkeypatch) -> None:
    module = _load_run_query_pack_module()
    pack_path = tmp_path / "pack.yaml"
    output_dir = tmp_path / "evidence" / "fallback"
    _write_pack(pack_path)
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    monkeypatch.setattr(module.sys, "executable", "C:/venv/python.exe")

    calls = []

    def fake_run(command, capture_output, text, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="Evidence", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    config = module.RunConfig(
        pack_path=pack_path,
        output_dir=output_dir,
        top_k=10,
        index_dir=None,
        index_id=None,
        root_ref=None,
        evidence_format="text-wrapper",
        dry_run=False,
        fail_fast=False,
        root_rag_cmd=None,
    )
    manifest, exit_code = module.run_query_pack(config)
    assert exit_code == 0
    assert calls[0][:5] == ["C:/venv/python.exe", "-m", "root_rag.cli", "ask", "alpha beta"]
    assert manifest["command_resolution_mode"] == "python_module_fallback"
    assert manifest["python_executable"] == "C:/venv/python.exe"
    wrapper = json.loads((output_dir / "q1.json").read_text(encoding="utf-8"))
    assert wrapper["command"] == calls[0]


def test_run_query_pack_honors_explicit_root_rag_cmd(tmp_path: Path, monkeypatch) -> None:
    module = _load_run_query_pack_module()
    pack_path = tmp_path / "pack.yaml"
    output_dir = tmp_path / "evidence" / "explicit"
    _write_pack(pack_path)
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    calls = []

    def fake_run(command, capture_output, text, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="Evidence", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    config = module.RunConfig(
        pack_path=pack_path,
        output_dir=output_dir,
        top_k=10,
        index_dir=None,
        index_id=None,
        root_ref=None,
        evidence_format="text-wrapper",
        dry_run=False,
        fail_fast=False,
        root_rag_cmd="root-rag-custom",
    )
    manifest, exit_code = module.run_query_pack(config)
    assert exit_code == 0
    assert calls[0][0] == "root-rag-custom"
    assert manifest["command_resolution_mode"] == "path_executable"
    assert manifest["root_rag_cmd"] == "root-rag-custom"
