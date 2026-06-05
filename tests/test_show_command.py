"""Tests for root-rag show command."""
import json
import os
from pathlib import Path

from click.testing import CliRunner

from root_rag.cli import main
from root_rag.index.schemas import IndexManifest


def _write_show_fixture(indexes_root, *, index_id="fairship__master__abc123def456__20260504T120000Z"):
    index_dir = indexes_root / index_id
    index_dir.mkdir(parents=True)

    chunks_path = index_dir / "chunks.jsonl"
    rows = [
        {
            "chunk_id": "c1",
            "root_ref": "master",
            "resolved_commit": "abc123def456abc123def456abc123def456abcd",
            "file_path": "shipgen/MuDISGenerator.cxx",
            "language": "cpp",
            "start_line": 70,
            "end_line": 74,
            "content": "\n".join(
                [
                    "line70",
                    "line71",
                    "line72",
                    "line73",
                    "line74",
                ]
            ),
            "doc_origin": "source_impl",
            "index_schema_version": "1.0.0",
            "symbol_path": None,
            "has_doxygen": False,
        }
    ]
    with open(chunks_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    manifest = IndexManifest(
        index_id=index_id,
        corpus_id="fairship__master__abc123def456",
        root_ref="master",
        resolved_commit="abc123def456abc123def456abc123def456abcd",
        corpus_url="https://github.com/ShipSoft/FairShip.git",
        chunks_path=str(chunks_path),
        fts_db_path=str(index_dir / "fts.sqlite"),
        chunk_count=1,
        file_count=1,
        created_at="2026-05-04T12:00:00Z",
    )
    manifest.save(index_dir / "index_manifest.json")
    return index_id


def test_show_parses_file_range_and_prints_lines(tmp_path):
    runner = CliRunner()
    indexes_root = tmp_path / "indexes"
    index_id = _write_show_fixture(indexes_root)

    result = runner.invoke(
        main,
        [
            "show",
            "shipgen/MuDISGenerator.cxx:71-73",
            "--index-id",
            index_id,
            "--index-dir",
            str(indexes_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "shipgen/MuDISGenerator.cxx:71-73" in result.output
    assert "71 | line71" in result.output
    assert "72 | line72" in result.output
    assert "73 | line73" in result.output


def test_show_supports_context(tmp_path):
    runner = CliRunner()
    indexes_root = tmp_path / "indexes"
    index_id = _write_show_fixture(indexes_root)

    result = runner.invoke(
        main,
        [
            "show",
            "shipgen/MuDISGenerator.cxx:72-72",
            "--index-id",
            index_id,
            "--index-dir",
            str(indexes_root),
            "--context",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "71 | line71" in result.output
    assert "72 | line72" in result.output
    assert "73 | line73" in result.output


def test_show_reports_when_chunk_coverage_is_incomplete(tmp_path):
    runner = CliRunner()
    indexes_root = tmp_path / "indexes"
    index_id = _write_show_fixture(indexes_root)

    result = runner.invoke(
        main,
        [
            "show",
            "shipgen/MuDISGenerator.cxx:71-78",
            "--index-id",
            index_id,
            "--index-dir",
            str(indexes_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "78 | [line not available in indexed chunks]" in result.output
    assert "Indexed chunks do not cover all requested lines" in result.output


def test_show_resolves_legacy_processed_chunks_path_under_data(tmp_path):
    runner = CliRunner()
    index_dir = tmp_path / "indexes"
    index_id = "fairship__master__abc123def456__20260504T120000Z"
    artifact_dir = index_dir / index_id
    artifact_dir.mkdir(parents=True)

    chunks_path = tmp_path / "data" / "processed" / "chunks" / "master__abc123def456" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True)
    row = {
        "chunk_id": "c1",
        "root_ref": "master",
        "resolved_commit": "abc123def456abc123def456abc123def456abcd",
        "file_path": "shipgen/MuDISGenerator.cxx",
        "language": "cpp",
        "start_line": 70,
        "end_line": 71,
        "content": "line70\nline71",
        "doc_origin": "source_impl",
        "index_schema_version": "1.0.0",
        "symbol_path": None,
        "has_doxygen": False,
    }
    with open(chunks_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    manifest = IndexManifest(
        index_id=index_id,
        corpus_id="fairship__master__abc123def456",
        root_ref="master",
        resolved_commit="abc123def456abc123def456abc123def456abcd",
        corpus_url="https://github.com/ShipSoft/FairShip.git",
        chunks_path="processed/chunks/master__abc123def456/chunks.jsonl",
        fts_db_path="fts.sqlite",
        chunk_count=1,
        file_count=1,
        created_at="2026-05-04T12:00:00Z",
    )
    manifest.save(artifact_dir / "index_manifest.json")

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(
            main,
            [
                "show",
                "shipgen/MuDISGenerator.cxx:70-71",
                "--index-id",
                index_id,
                "--index-dir",
                str(index_dir),
            ],
        )
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output
    assert "70 | line70" in result.output
    assert "71 | line71" in result.output
