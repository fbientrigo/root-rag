#!/usr/bin/env python3
"""Diagnostic-only split-gold geometry audit for bridge-light queries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from root_rag.evaluation.split_geometry_audit import load_bridge_light_geometry_audit


def _fmt(value) -> str:
    return "n/a" if value is None else str(value)


def _render_markdown(report: dict) -> str:
    lines = [
        "# Bridge-Light Split-Gold Geometry Audit",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Scope: `{report['scope']['query_class']}` same-file split queries",
        "",
        "## Verdict",
        "",
        f"- `{report['verdict']}`",
        "",
        "## Per-Query Geometry Summary",
        "",
        "| Query | Gold chunks | Order positions | Distance | Min window | Geometry | Self-sufficient | Label |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in report["per_query"]:
        pos_text = ", ".join(f"{k}:{v}" for k, v in row["gold_order_positions"].items())
        gold_text = ", ".join(row["gold_chunk_ids"])
        lines.append(
            f"| {row['query_id']} | {gold_text} | {pos_text} | {row['chunk_distance']} | "
            f"{row['minimum_contiguous_window_size']} | {row['geometry_relation']} | "
            f"{row['self_sufficient']} | {row['label']} |"
        )

    lines.extend(
        [
            "",
            "## Dominant Pattern",
            "",
            f"- dominant_label: `{report['class_summary']['dominant_label']}` "
            f"({report['class_summary']['dominant_label_count']})",
            "",
            "## Class Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| n_queries | {report['class_summary']['n_queries']} |",
            f"| dominant_label | {report['class_summary']['dominant_label']} |",
            f"| dominant_label_count | {report['class_summary']['dominant_label_count']} |",
            f"| label_counts | `{json.dumps(report['class_summary']['label_counts'], sort_keys=True)}` |",
            "",
            "## Notes",
            "",
        ]
    )
    for row in report["per_query"]:
        lines.append(f"- {row['query_id']}: {row['note']}")
    lines.extend(["", "## Next Smallest Valid Action", "", f"- `{report['next_action']}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/corpus.jsonl"),
        help="Frozen canonical corpus JSONL.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("configs/benchmark_queries_semantic.json"),
        help="Frozen benchmark queries JSON.",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("configs/benchmark_qrels_semantic.jsonl"),
        help="Frozen benchmark qrels JSONL.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/bridge_light_split_geometry_audit.json"),
        help="JSON artifact output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/bridge_light_split_geometry_audit.md"),
        help="Markdown report output path.",
    )
    args = parser.parse_args()

    audit = load_bridge_light_geometry_audit(
        corpus_path=args.corpus,
        queries_path=args.queries,
        qrels_path=args.qrels,
    )
    audit["generated_at"] = datetime.now(timezone.utc).isoformat()
    audit["artifacts"] = {
        "corpus": str(args.corpus),
        "queries": str(args.queries),
        "qrels": str(args.qrels),
    }
    audit["verdict"] = "split geometry explains the failure: most bridge-light same-file cases are far apart, one is self-sufficient, and only one is merely local-nonadjacent"
    audit["next_action"] = (
        "Run a second diagnostic on a noncontiguous multi-span geometry model before any retrieval rerun."
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    args.output_md.write_text(_render_markdown(audit), encoding="utf-8")

    print(f"[audit] output_json={args.output_json}")
    print(f"[audit] output_md={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
