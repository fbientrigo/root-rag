"""Lint wiki claim blocks under docs/wiki."""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


ALLOWED_STATUSES = {"CONFIRMED", "PROVISIONAL", "UNRESOLVED", "SUPERSEDED"}
CLAIM_RE = re.compile(r"<!--\s*CLAIM:\s*([A-Z_]+)\s*-->")
SOURCE_RE = re.compile(r"<!--\s*SOURCE:\s*([^>]+)\s*-->")
SOURCE_FORMAT_RE = re.compile(r"^[A-Za-z0-9_.\-/]+:[0-9]+-[0-9]+$")


@dataclass(frozen=True)
class ClaimBlock:
    file_path: Path
    line: int
    status: str
    body_lines: List[str]
    source_values: List[str]
    source_lines: List[int]


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.md")):
        if path.is_file():
            yield path


def _extract_claim_blocks(path: Path) -> List[ClaimBlock]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: List[ClaimBlock] = []

    active_status: str | None = None
    active_line = 0
    active_body: List[str] = []
    active_sources: List[str] = []
    active_source_lines: List[int] = []

    def flush() -> None:
        nonlocal active_status, active_line, active_body, active_sources, active_source_lines
        if active_status is None:
            return
        blocks.append(
            ClaimBlock(
                file_path=path,
                line=active_line,
                status=active_status,
                body_lines=active_body[:],
                source_values=active_sources[:],
                source_lines=active_source_lines[:],
            )
        )
        active_status = None
        active_line = 0
        active_body = []
        active_sources = []
        active_source_lines = []

    for idx, line in enumerate(lines, start=1):
        claim_match = CLAIM_RE.search(line)
        if claim_match:
            flush()
            active_status = claim_match.group(1).strip()
            active_line = idx
            continue

        if active_status is None:
            continue

        active_body.append(line)
        source_match = SOURCE_RE.search(line)
        if source_match:
            active_sources.append(source_match.group(1).strip())
            active_source_lines.append(idx)

    flush()
    return blocks


def lint_wiki_claims(root: Path) -> List[str]:
    """Return lint errors; empty list means success."""
    errors: List[str] = []

    if not root.exists():
        return [f"{root}:1: wiki path does not exist"]

    for path in _iter_markdown_files(root):
        blocks = _extract_claim_blocks(path)
        for block in blocks:
            status = block.status
            body_text = "\n".join(block.body_lines)
            src = str(path).replace("\\", "/")

            if status not in ALLOWED_STATUSES:
                errors.append(
                    f"{src}:{block.line}: invalid CLAIM status '{status}'. "
                    f"Allowed: {sorted(ALLOWED_STATUSES)}"
                )
                continue

            for source_value, source_line in zip(block.source_values, block.source_lines):
                if not SOURCE_FORMAT_RE.match(source_value):
                    errors.append(
                        f"{src}:{source_line}: invalid SOURCE format '{source_value}'. "
                        "Expected path/to/file.ext:start-end"
                    )

            if status == "CONFIRMED":
                if len(block.source_values) == 0:
                    errors.append(f"{src}:{block.line}: CONFIRMED claim requires at least one SOURCE.")
            elif status == "PROVISIONAL":
                has_todo = "TODO" in body_text
                if len(block.source_values) == 0 and not has_todo:
                    errors.append(f"{src}:{block.line}: PROVISIONAL claim requires SOURCE or TODO.")
            elif status == "UNRESOLVED":
                if "Next action:" not in body_text:
                    errors.append(f"{src}:{block.line}: UNRESOLVED claim requires 'Next action:'.")
            elif status == "SUPERSEDED":
                if "Superseded by:" not in body_text:
                    errors.append(f"{src}:{block.line}: SUPERSEDED claim requires 'Superseded by:'.")

    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lint wiki claim blocks.")
    parser.add_argument("wiki_path", nargs="?", default="docs/wiki", type=Path, help="Wiki root path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    errors = lint_wiki_claims(args.wiki_path)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"OK: wiki claims lint passed for {args.wiki_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
