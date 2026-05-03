#!/usr/bin/env python3
"""Run a shadow focal chunk-granularity experiment on bridge-light queries."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from root_rag.evaluation.focal_chunk_experiment import (
    build_mode_pipelines,
    build_shadow_corpus_rows,
    build_shadow_qrels_rows,
    compare_rate_blocks,
    search_pipelines,
    summarize_class_rates,
    summarize_query,
)
from root_rag.evaluation.semantic_v1 import load_corpus, load_qrels, load_queries
from root_rag.retrieval.s1_semantic import SemanticIndexManifest, SentenceTransformerLocalEmbedder, build_semantic_index_artifacts


BRIDGE_LIGHT_QUERY_IDS = {f"br{i:03d}" for i in range(1, 9)}


def _log(message: str) -> None:
    print(message, flush=True)


def _load_baseline(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _query_id(row) -> str:
    return str(getattr(row, "query_id", None) or row["id"])


def _query_text(row) -> str:
    return str(getattr(row, "query", None) or row["query"])


def _query_class(row) -> str:
    return str(getattr(row, "query_class", None) or row["query_class"])


def _select_bridge_light_queries(queries: list) -> list[dict]:
    selected = [
        {"id": _query_id(row), "query": _query_text(row), "query_class": _query_class(row)}
        for row in queries
        if _query_id(row) in BRIDGE_LIGHT_QUERY_IDS
    ]
    selected.sort(key=lambda row: row["id"])
    return selected


def _index_by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["chunk_id"]: row for row in rows}


def _per_query_id(row: dict) -> str:
    return str(row.get("query_id") or row.get("id"))


def _baseline_query_lookup(baseline: dict, query_id: str) -> dict:
    for row in baseline["per_query"]:
        if _per_query_id(row) == query_id:
            return row
    raise KeyError(query_id)


def _format_rate(value: float) -> str:
    return f"{value:.3f}"


def _format_rank(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _build_verdict(
    *,
    baseline_summary: dict,
    shadow_summary: dict,
    br006_before: dict,
    br006_after: dict,
) -> tuple[str, str, str]:
    delta_both = shadow_summary["both_golds_found_rate"] - baseline_summary["both_golds_found_rate"]
    delta_late = baseline_summary["late_rank_rate"] - shadow_summary["late_rank_rate"]

    br006_improved = (
        br006_after["aggregate"]["best_mode_gold_count"] > br006_before["aggregate"]["best_mode_gold_count"]
        or (
            br006_before["aggregate"]["best_mode_best_rank"] is not None
            and br006_after["aggregate"]["best_mode_best_rank"] is not None
            and br006_after["aggregate"]["best_mode_best_rank"] < br006_before["aggregate"]["best_mode_best_rank"]
        )
    )

    if delta_both > 0 and delta_late >= 0 and br006_improved:
        return (
            "signal supports focal chunk granularity",
            "proceed",
            "Shadow adjacent-pair widening improved bridge-light gold coverage without worsening late-rank pressure, and br006 improved or held steady.",
        )

    if delta_both <= 0 and delta_late <= 0 and not br006_improved:
        return (
            "no useful signal",
            "reject",
            "The shadow corpus did not move bridge-light coverage or late-rank behavior enough to justify a granularity change.",
        )

    if delta_both == 0 and delta_late < 0:
        return (
            "no useful signal",
            "reject",
            "The shadow corpus kept gold coverage flat and made late-rank behavior worse, so the granularity change is not supported.",
        )

    return (
        "inconclusive",
        "iterate once with one smaller variant",
        "The shadow pairing changed some outcomes but not enough to isolate a clean granularity effect.",
    )


def _render_markdown(report: dict) -> str:
    baseline = report["baseline"]
    shadow = report["shadow"]
    comparison = report["comparison"]
    lines = [
        "# Bridge-Light Focal Chunk-Granularity Experiment",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Shadow rule: `{report['experiment']['shadow_rule']}`",
        f"- Depth: `{report['experiment']['top_k']}`",
        "",
        "## Verdict",
        "",
        f"- `{report['verdict']}`",
        "",
        "## Files Modified",
        "",
    ]
    for path in report["files_modified"]:
        lines.append(f"- `{path}`")

    lines.extend(
        [
            "",
            "## Commands Run",
            "",
        ]
    )
    for command in report["commands_run"]:
        lines.append(f"- `{command}`")

    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
        ]
    )
    for key, value in report["artifacts"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Class-Level Comparison",
            "",
            "| Metric | Baseline V1.2 | Shadow | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric in (
        "both_golds_found_rate",
        "one_gold_found_rate",
        "zero_gold_found_rate",
        "same_file_split_rate",
        "late_rank_rate",
    ):
        lines.append(
            f"| {metric} | {_format_rate(baseline[metric])} | {_format_rate(shadow[metric])} | "
            f"{_format_rate(comparison['rate_delta'][metric])} |"
        )

    lines.extend(
        [
            "",
            "## Per-Query Bridge-Light Comparison",
            "",
            "| Query | Baseline count | Shadow count | Baseline rank | Shadow rank | Late rank before/after |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["per_query"]:
        baseline_agg = row["baseline"]["aggregate"]
        shadow_agg = row["shadow"]["aggregate"]
        lines.append(
            f"| {row['query_id']} | {baseline_agg['best_mode_gold_count']} | "
            f"{shadow_agg['best_mode_gold_count']} | {_format_rank(baseline_agg['best_mode_best_rank'])} | "
            f"{_format_rank(shadow_agg['best_mode_best_rank'])} | {baseline_agg['late_rank']} / {shadow_agg['late_rank']} |"
        )

    br006 = report["br006"]
    br006_baseline = br006["baseline"]["aggregate"]
    br006_shadow = br006["shadow"]["aggregate"]
    lines.extend(
        [
            "",
            "## br006 Specific Result",
            "",
            f"- Baseline count/rank: `{br006_baseline['best_mode_gold_count']}` / `{_format_rank(br006_baseline['best_mode_best_rank'])}`",
            f"- Shadow count/rank: `{br006_shadow['best_mode_gold_count']}` / `{_format_rank(br006_shadow['best_mode_best_rank'])}`",
            f"- Baseline late rank: `{br006_baseline['late_rank']}`",
            f"- Shadow late rank: `{br006_shadow['late_rank']}`",
            f"- Note: {br006['note']}",
            "",
            "## Recommendation",
            "",
            f"- `{report['recommendation']}`",
        ]
    )
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
        help="Frozen bridge-light query set.",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("configs/benchmark_qrels_semantic.jsonl"),
        help="Frozen bridge-light qrels.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path("artifacts/diagnostics/bridge_light_br001_br008_depth200/bridge_light_br001_br008_depth200_competition_diagnostic.json"),
        help="Frozen V1.2 bridge-light depth-200 diagnostic JSON.",
    )
    parser.add_argument(
        "--semantic-model",
        default="intfloat/e5-base-v2",
        help="Local embedding model used for the shadow semantic index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Search depth for the diagnostic sweep.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/bridge_light_focal_chunk_experiment.json"),
        help="Experiment artifact JSON output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/bridge_light_focal_chunk_experiment.md"),
        help="Experiment markdown report output path.",
    )
    parser.add_argument(
        "--shadow-dir",
        type=Path,
        default=Path("artifacts/bridge_light_focal_chunk_experiment"),
        help="Directory for shadow corpus and semantic artifacts.",
    )
    args = parser.parse_args()

    start = datetime.now(timezone.utc)
    _log("[1/6] Load frozen inputs")
    baseline = _load_baseline(args.baseline_json)
    corpus_rows = load_corpus(args.corpus)
    query_rows = _select_bridge_light_queries(load_queries(args.queries))
    qrels_map = load_qrels(args.qrels)
    corpus_by_id = _index_by_id(corpus_rows)
    bridge_light_qrels = {qid: qrels_map[qid] for qid in BRIDGE_LIGHT_QUERY_IDS}

    _log("[2/6] Build shadow corpus")
    shadow_rows, shadow_membership = build_shadow_corpus_rows(corpus_rows)
    args.shadow_dir.mkdir(parents=True, exist_ok=True)
    shadow_corpus_path = args.shadow_dir / "shadow_corpus.jsonl"
    shadow_corpus_path.write_text("\n".join(json.dumps(row) for row in shadow_rows), encoding="utf-8")
    shadow_qrels_path = args.shadow_dir / "shadow_qrels.jsonl"
    shadow_qrels_rows = build_shadow_qrels_rows(qrels_map=bridge_light_qrels, shadow_membership=shadow_membership)
    shadow_qrels_path.write_text("\n".join(json.dumps(row) for row in shadow_qrels_rows), encoding="utf-8")

    _log("[3/6] Build shadow semantic index")
    embedder = SentenceTransformerLocalEmbedder(
        model_name=args.semantic_model,
        device="cpu",
        batch_size=64,
        local_files_only=True,
    )
    semantic_dir = args.shadow_dir / "semantic"
    semantic_manifest = build_semantic_index_artifacts(
        corpus_rows=shadow_rows,
        corpus_path=shadow_corpus_path,
        output_dir=semantic_dir,
        embedder=embedder,
        corpus_source_identifier="bridge-light-focal-adjacent-pairs",
    )
    semantic_manifest_path = semantic_dir / "semantic_manifest.json"
    semantic_manifest = SemanticIndexManifest.load(semantic_manifest_path)

    _log("[4/6] Run shadow sweep")
    pipelines = build_mode_pipelines(
        corpus_rows=shadow_rows,
        corpus_path=shadow_corpus_path,
        semantic_manifest_path=semantic_manifest_path,
        semantic_model_name=args.semantic_model,
        semantic_embedder=embedder,
    )
    runs = search_pipelines(pipelines=pipelines, queries=query_rows, top_k=args.top_k)

    _log("[5/6] Summarize results")
    per_query = []
    for query_row in query_rows:
        query_id = str(query_row["id"])
        gold_chunk_ids = sorted(bridge_light_qrels[query_id].keys())
        per_query.append(
            summarize_query(
                query_row=query_row,
                gold_chunk_ids=gold_chunk_ids,
                corpus_by_id=corpus_by_id,
                shadow_membership=shadow_membership,
                runs=runs,
            )
        )

    per_query.sort(key=lambda row: row["query_id"])
    shadow_summary = summarize_class_rates(per_query)
    baseline_summary = baseline["class_summary"]
    rate_delta = compare_rate_blocks(baseline_summary, shadow_summary)
    comparison = {
        "rate_delta": rate_delta,
        "shadow_minus_baseline": {
            "both_golds_found_rate": shadow_summary["both_golds_found_rate"] - baseline_summary["both_golds_found_rate"],
            "one_gold_found_rate": shadow_summary["one_gold_found_rate"] - baseline_summary["one_gold_found_rate"],
            "zero_gold_found_rate": shadow_summary["zero_gold_found_rate"] - baseline_summary["zero_gold_found_rate"],
            "same_file_split_rate": shadow_summary["same_file_split_rate"] - baseline_summary["same_file_split_rate"],
            "late_rank_rate": shadow_summary["late_rank_rate"] - baseline_summary["late_rank_rate"],
        },
    }
    query_map = {row["query_id"]: row for row in per_query}
    br006_before = _baseline_query_lookup(baseline, "br006")
    br006_after = query_map["br006"]
    verdict, recommendation, verdict_note = _build_verdict(
        baseline_summary=baseline_summary,
        shadow_summary=shadow_summary,
        br006_before=br006_before,
        br006_after=br006_after,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "name": "bridge-light focal chunk-granularity shadow experiment",
            "shadow_rule": "same-file adjacent chunk pairs",
            "top_k": args.top_k,
            "semantic_model": args.semantic_model,
            "baseline_json": str(args.baseline_json),
        },
        "artifacts": {
            "corpus": str(args.corpus),
            "queries": str(args.queries),
            "qrels": str(args.qrels),
            "baseline_json": str(args.baseline_json),
            "shadow_corpus": str(shadow_corpus_path),
            "shadow_qrels": str(shadow_qrels_path),
            "shadow_semantic_manifest": str(semantic_manifest_path),
            "shadow_semantic_index": str(semantic_manifest.index_path),
            "shadow_semantic_vectors": str(semantic_manifest.vectors_path),
        },
        "commands_run": [
            " ".join([sys.executable, *sys.argv]),
        ],
        "files_modified": [
            "src/root_rag/evaluation/focal_chunk_experiment.py",
            "scripts/run_bridge_light_focal_chunk_experiment.py",
            "tests/test_focal_chunk_experiment.py",
        ],
        "baseline": baseline_summary,
        "shadow": shadow_summary,
        "comparison": comparison,
        "per_query": [
            {
                "query_id": row["query_id"],
                "query_text": row["query_text"],
                "query_class": row["query_class"],
                "same_file_split": row["same_file_split"],
                "baseline": _baseline_query_lookup(baseline, row["query_id"]),
                "shadow": row,
                "delta": {
                    "best_mode_gold_count": row["aggregate"]["best_mode_gold_count"]
                    - _baseline_query_lookup(baseline, row["query_id"])["aggregate"]["best_mode_gold_count"],
                    "best_mode_best_rank": (
                        None
                        if row["aggregate"]["best_mode_best_rank"] is None
                        or _baseline_query_lookup(baseline, row["query_id"])["aggregate"]["best_mode_best_rank"] is None
                        else row["aggregate"]["best_mode_best_rank"]
                        - _baseline_query_lookup(baseline, row["query_id"])["aggregate"]["best_mode_best_rank"]
                    ),
                },
            }
            for row in per_query
        ],
        "br006": {
            "baseline": br006_before,
            "shadow": br006_after,
            "note": verdict_note,
        },
        "verdict": verdict,
        "recommendation": recommendation,
        "verdict_note": verdict_note,
    }

    _log("[6/6] Write artifacts")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(_render_markdown(report), encoding="utf-8")

    _log(f"[done] json={args.output_json}")
    _log(f"[done] md={args.output_md}")
    _log(f"[done] shadow_corpus={shadow_corpus_path}")
    _log(f"[done] shadow_semantic={semantic_dir}")
    _log(f"[done] elapsed={(datetime.now(timezone.utc) - start).total_seconds():.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
