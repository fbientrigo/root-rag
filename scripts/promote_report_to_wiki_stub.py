"""Print manual promotion checklist from a weekly report markdown file."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")


def _extract_section_lines(text: str, section_name: str) -> list[str]:
    lines = text.splitlines()
    active = False
    collected: list[str] = []
    for line in lines:
        heading = HEADING_RE.match(line)
        if heading:
            if active:
                break
            active = heading.group(1).strip().lower() == section_name.strip().lower()
            continue
        if active:
            collected.append(line)
    return collected


def build_checklist(report_path: Path) -> list[str]:
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")
    text = report_path.read_text(encoding="utf-8")

    candidate_nodes = [
        line.strip()
        for line in _extract_section_lines(text, "Candidate Workflow Nodes")
        if line.strip().startswith("- ")
    ]
    unresolved = [
        line.strip()
        for line in _extract_section_lines(text, "Unresolved Questions")
        if line.strip().startswith("- ")
    ]
    errors = [
        line.strip()
        for line in _extract_section_lines(text, "Errors")
        if line.strip().startswith("- ")
    ]

    checklist: list[str] = []
    checklist.append("Manual Wiki Promotion Checklist")
    checklist.append(f"- Report: {report_path}")
    checklist.append("")
    checklist.append("Candidate confirmed claims:")
    if candidate_nodes:
        checklist.extend(
            [
                f"- Review node evidence for claim promotion: {row[2:].strip()}"
                for row in candidate_nodes
            ]
        )
    else:
        checklist.append("- None detected; keep claims provisional/unresolved.")
    checklist.append("")
    checklist.append("Candidate unresolved questions:")
    if unresolved:
        checklist.extend(
            [f"- Add/update open question entry: {row[2:].strip()}" for row in unresolved]
        )
    else:
        checklist.append("- None detected.")
    checklist.append("")
    checklist.append("Candidate workflow graph updates:")
    if candidate_nodes:
        checklist.append("- Check whether any node or edge can move from PROVISIONAL to CONFIRMED.")
    else:
        checklist.append("- Keep graph as stage-level PROVISIONAL/UNRESOLVED.")
    if errors:
        checklist.append("- Resolve report errors before promoting any claim.")
    return checklist


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual checklist generator for wiki promotion.")
    parser.add_argument("--report", required=True, type=Path, help="Path to weekly report markdown")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        lines = build_checklist(args.report)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
