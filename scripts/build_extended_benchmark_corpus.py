#!/usr/bin/env python3
"""Build minimal extended benchmark corpus: FairShip corpus + external ROOT SOFIE docs."""

from __future__ import annotations

import argparse
from pathlib import Path


def _read_non_empty_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-corpus", type=Path, default=Path("artifacts/corpus.jsonl"))
    parser.add_argument(
        "--external-corpus",
        type=Path,
        default=Path("configs/root_sofie_minimal_corpus.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/corpus_extended_root_sofie.jsonl"),
    )
    args = parser.parse_args()

    base_lines = _read_non_empty_lines(args.base_corpus)
    external_lines = _read_non_empty_lines(args.external_corpus)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(base_lines + external_lines) + "\n", encoding="utf-8")

    print(
        f"Built extended corpus: {args.output} "
        f"(base={len(base_lines)} rows, external={len(external_lines)} rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
