#!/usr/bin/env python3
"""Build the frozen benchmark corpus (artifacts/corpus.jsonl) from a FairShip checkout.

Reads the curated file list in configs/benchmark_corpus_files.txt, chunks each
file with the canonical sliding-window chunker, and writes one JSON row per
chunk to the output corpus file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import root_rag.index.schemas  # noqa: F401  (avoid index/parser circular import)
from root_rag.parser.chunks import chunk_file


def _chunk_to_row(chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "root_ref": chunk.root_ref,
        "resolved_commit": chunk.resolved_commit,
        "file_path": chunk.file_path,
        "language": chunk.language,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "line_range": [chunk.start_line, chunk.end_line],
        "doc_origin": chunk.doc_origin,
        "text": chunk.content,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fairship-path",
        type=Path,
        required=True,
        help="Path to a local FairShip checkout.",
    )
    parser.add_argument(
        "--files-manifest",
        type=Path,
        default=Path("configs/benchmark_corpus_files.txt"),
        help="Text file listing repo-relative FairShip paths to include.",
    )
    parser.add_argument(
        "--root-ref",
        default="master",
        help="FairShip ref recorded as provenance for each chunk.",
    )
    parser.add_argument(
        "--resolved-commit",
        required=True,
        help="FairShip commit SHA recorded as provenance for each chunk.",
    )
    parser.add_argument("--window-lines", type=int, default=40)
    parser.add_argument("--overlap-lines", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/corpus.jsonl"),
    )
    args = parser.parse_args()

    file_paths = [
        line.strip()
        for line in args.files_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    rows: list[dict] = []
    for rel_path in file_paths:
        abs_path = args.fairship_path / rel_path
        chunks = chunk_file(
            file_path=abs_path,
            root_ref=args.root_ref,
            resolved_commit=args.resolved_commit,
            repo_root=args.fairship_path,
            window_lines=args.window_lines,
            overlap_lines=args.overlap_lines,
        )
        if not chunks:
            raise SystemExit(f"No chunks produced for {rel_path}")
        rows.extend(_chunk_to_row(chunk) for chunk in chunks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} chunks from {len(file_paths)} files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
