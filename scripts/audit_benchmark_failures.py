#!/usr/bin/env python3
"""Audit benchmark failures query-by-query to diagnose WHY each failed query fails.

Outputs:
- artifacts/benchmark_failure_audit.json (machine-readable)
- artifacts/benchmark_failure_audit.md (human-readable report)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Set

from root_rag.evaluation.metrics import TopKMetrics, compute_topk_metrics
from root_rag.retrieval.backends import build_retrieval_backend
from root_rag.retrieval.pipeline import RetrievalPipeline
from root_rag.retrieval.transformers import build_query_transformer

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
FAILURE_MODE_ORDER = (
    "coverage_miss",
    "ranking_miss",
    "qrel_mismatch_suspected",
    "ambiguous_query",
    "large_rank_gap",
)


@dataclass
class QrelDiagnostic:
    """Per-qrel diagnostic metadata."""

    chunk_id: str
    relevance: int
    exists_in_corpus: bool
    rank: int | None  # None if not in top-k, else 1-indexed rank
    bm25_score: float | None
    token_overlap_count: int
    query_tokens_matched: List[str]
    chunk_preview: str  # First 100 chars


@dataclass
class QueryAudit:
    """Comprehensive failure diagnostic for one benchmark query."""

    query_id: str
    query_text: str
    query_class: str
    qrels_positive_count: int
    qrels_diagnostics: List[QrelDiagnostic]

    # Metrics
    mrr_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    retrieved_positive_count: int

    # Top-k diagnostics
    top_k_chunk_ids: List[str]
    top_k_scores: List[float]
    best_positive_rank: int | None  # None if zero-recall
    best_positive_score: float | None
    worst_top_k_score: float
    score_gap_vs_best_positive: float | None  # Gap between best positive and rank 1

    # Failure classification heuristics
    failure_modes: List[str] = field(default_factory=list)
    header_source_bias: str | None = None  # "header_dominated" | "source_dominated" | None
    declaration_vs_impl_bias: str | None = None
    qrel_mismatch_suspected: bool = False
    notes: List[str] = field(default_factory=list)


def load_corpus(corpus_path: Path) -> tuple[Dict[str, dict], List[dict]]:
    """Load corpus.jsonl into memory by chunk_id and as list of rows."""
    corpus_by_id = {}
    corpus_rows = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                corpus_by_id[chunk["chunk_id"]] = chunk
                corpus_rows.append(chunk)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Malformed corpus line {line_num}: {e}")
    return corpus_by_id, corpus_rows


def load_benchmark_queries(queries_path: Path) -> List[dict]:
    """Load benchmark_queries.json."""
    with queries_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_benchmark_qrels(qrels_path: Path) -> Dict[str, List[tuple[str, int]]]:
    """Load benchmark_qrels.jsonl into {query_id: [(chunk_id, relevance), ...]}."""
    qrels = defaultdict(list)
    with qrels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            qrels[entry["query_id"]].append((entry["chunk_id"], entry["relevance"]))
    return dict(qrels)


def extract_tokens(text: str) -> Set[str]:
    """Extract lowercase alphanumeric tokens."""
    return {t.lower() for t in TOKEN_RE.findall(text)}


def compute_token_overlap(query_tokens: Set[str], chunk_text: str) -> tuple[int, List[str]]:
    """Return (overlap_count, matched_tokens)."""
    chunk_tokens = extract_tokens(chunk_text)
    matched = sorted(query_tokens & chunk_tokens)
    return len(matched), matched


def classify_header_source_bias(top_k_chunk_ids: List[str], corpus: Dict[str, dict]) -> str | None:
    """Classify header/source bias in top-k results."""
    header_count = sum(
        1 for cid in top_k_chunk_ids if cid in corpus and corpus[cid]["file_path"].endswith(".h")
    )
    source_count = sum(
        1
        for cid in top_k_chunk_ids
        if cid in corpus and corpus[cid]["file_path"].endswith((".cxx", ".cpp", ".cc"))
    )
    total = len(top_k_chunk_ids)
    if header_count > 0.7 * total:
        return "header_dominated"
    if source_count > 0.7 * total:
        return "source_dominated"
    return None


def infer_failure_modes(audit: QueryAudit, corpus: Dict[str, dict]) -> List[str]:
    """Infer likely failure modes based on heuristics."""
    modes: List[str] = []

    # Coverage miss: positive qrels not in corpus
    missing_qrels = [qd for qd in audit.qrels_diagnostics if not qd.exists_in_corpus]
    if missing_qrels:
        modes.append("coverage_miss")

    # Ranking miss: positive qrels in corpus but not in top-k or ranked very low
    ranked_positives = [qd for qd in audit.qrels_diagnostics if qd.rank is not None]
    if audit.qrels_positive_count > 0 and len(ranked_positives) < audit.qrels_positive_count:
        if all(qd.exists_in_corpus for qd in audit.qrels_diagnostics):
            modes.append("ranking_miss")

    # Qrel mismatch: qrels exist in corpus but have extremely low token overlap
    low_overlap = [
        qd
        for qd in audit.qrels_diagnostics
        if qd.exists_in_corpus and qd.token_overlap_count < 2
    ]
    if len(low_overlap) >= audit.qrels_positive_count:
        modes.append("qrel_mismatch_suspected")

    # Ambiguous query: query has < 3 unique tokens
    query_tokens = extract_tokens(audit.query_text)
    if len(query_tokens) < 3:
        modes.append("ambiguous_query")

    # Score gap analysis: best positive exists but is far from top-1
    if audit.best_positive_rank and audit.best_positive_rank > 5:
        modes.append("large_rank_gap")

    # Keep classification deterministic for reproducible audit artifacts.
    deduped = sorted(set(modes))
    ordered: List[str] = [mode for mode in FAILURE_MODE_ORDER if mode in deduped]
    ordered.extend(mode for mode in deduped if mode not in FAILURE_MODE_ORDER)
    return ordered


def audit_query(
    query_entry: dict,
    qrels: List[tuple[str, int]],
    corpus: Dict[str, dict],
    retrieval_results: List[tuple[str, float]],
    top_k: int = 10,
) -> QueryAudit:
    """Build comprehensive audit for one query."""
    query_id = query_entry["id"]
    query_text = query_entry["query"]
    query_class = query_entry["query_class"]

    # Build qrel diagnostics
    query_tokens = extract_tokens(query_text)
    qrels_diagnostics = []

    ranked_chunk_ids = [cid for cid, _ in retrieval_results[:top_k]]
    ranked_scores = {cid: score for cid, score in retrieval_results[:top_k]}
    rank_map = {cid: idx + 1 for idx, cid in enumerate(ranked_chunk_ids)}

    for chunk_id, relevance in qrels:
        exists = chunk_id in corpus
        rank = rank_map.get(chunk_id)
        score = ranked_scores.get(chunk_id)
        
        if exists:
            chunk_text = corpus[chunk_id].get("text", corpus[chunk_id].get("content", ""))
            overlap_count, matched_tokens = compute_token_overlap(query_tokens, chunk_text)
            preview = chunk_text[:100].replace("\n", " ")
        else:
            overlap_count, matched_tokens = 0, []
            preview = "[NOT IN CORPUS]"

        qrels_diagnostics.append(
            QrelDiagnostic(
                chunk_id=chunk_id,
                relevance=relevance,
                exists_in_corpus=exists,
                rank=rank,
                bm25_score=score,
                token_overlap_count=overlap_count,
                query_tokens_matched=matched_tokens,
                chunk_preview=preview,
            )
        )

    # Compute metrics
    relevance_map = {cid: rel for cid, rel in qrels}
    metrics = compute_topk_metrics(
        ranked_chunk_ids=ranked_chunk_ids,
        relevance_by_chunk=relevance_map,
        top_k=top_k,
        qrels_positive_count=len(qrels),
    )

    # Compute top-k metadata
    top_k_scores = [score for _, score in retrieval_results[:top_k]]
    worst_top_k_score = min(top_k_scores) if top_k_scores else 0.0

    # Best positive metadata
    positives_ranked = [qd for qd in qrels_diagnostics if qd.rank is not None]
    if positives_ranked:
        best_positive = min(positives_ranked, key=lambda qd: qd.rank)
        best_positive_rank = best_positive.rank
        best_positive_score = best_positive.bm25_score
        # Gap between best positive and top-1
        top_1_score = retrieval_results[0][1] if retrieval_results else 0.0
        score_gap = top_1_score - best_positive_score if best_positive_score else None
    else:
        best_positive_rank = None
        best_positive_score = None
        score_gap = None

    audit = QueryAudit(
        query_id=query_id,
        query_text=query_text,
        query_class=query_class,
        qrels_positive_count=len(qrels),
        qrels_diagnostics=qrels_diagnostics,
        mrr_at_k=metrics.mrr_at_k,
        recall_at_k=metrics.recall_at_k,
        ndcg_at_k=metrics.ndcg_at_k,
        retrieved_positive_count=metrics.retrieved_positive_count,
        top_k_chunk_ids=ranked_chunk_ids,
        top_k_scores=top_k_scores,
        best_positive_rank=best_positive_rank,
        best_positive_score=best_positive_score,
        worst_top_k_score=worst_top_k_score,
        score_gap_vs_best_positive=score_gap,
    )

    # Infer failure modes
    audit.failure_modes = infer_failure_modes(audit, corpus)
    audit.qrel_mismatch_suspected = "qrel_mismatch_suspected" in audit.failure_modes
    audit.header_source_bias = classify_header_source_bias(ranked_chunk_ids, corpus)

    # Check if top results are declarations vs implementations
    top_5_chunk_ids = ranked_chunk_ids[:5]
    top_5_chunks = [corpus.get(cid) for cid in top_5_chunk_ids if cid in corpus]
    
    # Heuristic: count chunks with class/struct declarations
    declaration_keywords = ["class ", "struct ", "namespace ", "typedef "]
    decl_count = sum(
        1
        for chunk in top_5_chunks
        if any(kw in chunk.get("text", chunk.get("content", "")) for kw in declaration_keywords)
    )
    if decl_count > 3:
        audit.declaration_vs_impl_bias = "declaration_dominated"
    
    # Add contextual notes
    if audit.recall_at_k == 0.0:
        audit.notes.append("ZERO RECALL: No positive qrels found in top-k")
    if "coverage_miss" in audit.failure_modes:
        missing_count = sum(1 for qd in qrels_diagnostics if not qd.exists_in_corpus)
        audit.notes.append(f"{missing_count}/{len(qrels)} qrels missing from corpus")
    if "qrel_mismatch_suspected" in audit.failure_modes:
        audit.notes.append("Low token overlap suggests qrel/query mismatch")

    return audit


def generate_markdown_report(audits: List[QueryAudit], output_path: Path) -> None:
    """Generate human-readable markdown diagnostic report."""
    lines = [
        "# Benchmark Failure Audit Report",
        "",
        f"**Total Queries:** {len(audits)}",
        "",
        "Classification is deterministic (fixed thresholds and fixed failure-mode ordering).",
        "",
        "Deterministic failure-mode rules:",
        "- `coverage_miss`: at least one positive qrel chunk is missing from corpus.",
        "- `ranking_miss`: all positive qrels exist, but fewer than all positives are retrieved in top-k.",
        "- `qrel_mismatch_suspected`: all positive qrels have token overlap `< 2` with the query.",
        "- `ambiguous_query`: query has fewer than 3 unique alphanumeric tokens.",
        "- `large_rank_gap`: best positive rank exists and is greater than 5.",
        f"- Ordered output modes: {', '.join(FAILURE_MODE_ORDER)}",
        "",
        "---",
        "",
    ]

    # Summary statistics
    zero_recall_queries = [a for a in audits if a.recall_at_k == 0.0]
    partial_recall_queries = [a for a in audits if 0.0 < a.recall_at_k < 1.0]
    perfect_recall_queries = [a for a in audits if a.recall_at_k == 1.0]

    lines.extend([
        "## Summary Statistics",
        "",
        f"- **Zero Recall Queries:** {len(zero_recall_queries)} / {len(audits)}",
        f"- **Partial Recall Queries:** {len(partial_recall_queries)} / {len(audits)}",
        f"- **Perfect Recall Queries:** {len(perfect_recall_queries)} / {len(audits)}",
        "",
        "### Failure Mode Distribution",
        "",
    ])

    # Count failure modes
    failure_mode_counter = Counter()
    for audit in audits:
        for mode in audit.failure_modes:
            failure_mode_counter[mode] += 1

    for mode, count in failure_mode_counter.most_common():
        lines.append(f"- **{mode}:** {count} queries")

    lines.extend(["", "---", ""])

    # Per-query details (focus on failures)
    lines.extend(["## Query-by-Query Diagnostics", ""])

    # Sort: zero-recall first, then by recall ascending
    sorted_audits = sorted(audits, key=lambda a: (a.recall_at_k, a.mrr_at_k))

    for audit in sorted_audits:
        lines.extend([
            f"### {audit.query_id}: {audit.query_text}",
            "",
            f"- **Query Class:** `{audit.query_class}`",
            f"- **MRR@10:** {audit.mrr_at_k:.4f}",
            f"- **Recall@10:** {audit.recall_at_k:.4f}",
            f"- **nDCG@10:** {audit.ndcg_at_k:.4f}",
            f"- **Retrieved Positives:** {audit.retrieved_positive_count} / {audit.qrels_positive_count}",
            "",
        ])

        if audit.failure_modes:
            lines.append(f"**Failure Modes:** {', '.join(audit.failure_modes)}")
            lines.append("")

        if audit.notes:
            lines.append("**Notes:**")
            for note in audit.notes:
                lines.append(f"- {note}")
            lines.append("")

        # Qrel diagnostics
        lines.append("**Qrel Diagnostics:**")
        lines.append("")
        for qd in audit.qrels_diagnostics:
            lines.append(f"- `{qd.chunk_id}` (rel={qd.relevance})")
            lines.append(f"  - Exists in corpus: {qd.exists_in_corpus}")
            if qd.rank:
                lines.append(f"  - Rank: {qd.rank}")
                lines.append(f"  - BM25 Score: {qd.bm25_score:.4f}")
            else:
                lines.append(f"  - Rank: NOT IN TOP-10")
            lines.append(f"  - Token overlap: {qd.token_overlap_count}")
            if qd.query_tokens_matched:
                lines.append(f"  - Matched tokens: {', '.join(qd.query_tokens_matched[:10])}")
            lines.append(f"  - Preview: {qd.chunk_preview}")
            lines.append("")

        # Top-k analysis
        if audit.best_positive_rank:
            lines.append(f"**Best Positive:** Rank {audit.best_positive_rank}, Score {audit.best_positive_score:.4f}")
            if audit.score_gap_vs_best_positive is not None:
                lines.append(f"**Score Gap (Top-1 vs Best Positive):** {audit.score_gap_vs_best_positive:.4f}")
        else:
            lines.append("**Best Positive:** None in top-10")
        
        lines.append(f"**Worst Top-10 Score:** {audit.worst_top_k_score:.4f}")

        if audit.header_source_bias:
            lines.append(f"**Header/Source Bias:** {audit.header_source_bias}")

        if audit.declaration_vs_impl_bias:
            lines.append(f"**Declaration/Impl Bias:** {audit.declaration_vs_impl_bias}")

        # For zero-recall queries, show what's actually in top-10
        if audit.recall_at_k == 0.0 and audit.top_k_chunk_ids:
            lines.append("")
            lines.append("**Top-10 Retrieved (all non-relevant):**")
            for rank, (chunk_id, score) in enumerate(
                zip(audit.top_k_chunk_ids[:10], audit.top_k_scores[:10]), start=1
            ):
                lines.append(f"{rank}. `{chunk_id}` (score={score:.4f})")

        lines.extend(["", "---", ""])

    # Aggregated insights section
    lines.extend(["## Aggregated Insights", ""])

    # Zero-recall query analysis
    if zero_recall_queries:
        lines.extend([
            f"### Zero-Recall Queries ({len(zero_recall_queries)} total)",
            "",
        ])
        for audit in zero_recall_queries:
            lines.append(f"- **{audit.query_id}** ({audit.query_class}): {audit.query_text}")
            failure_summary = ", ".join(audit.failure_modes) if audit.failure_modes else "no specific failure mode"
            lines.append(f"  - Failure modes: {failure_summary}")
        lines.append("")

    # Query class performance
    by_class = defaultdict(list)
    for audit in audits:
        by_class[audit.query_class].append(audit)

    lines.extend(["### Performance by Query Class", ""])
    for query_class, class_audits in sorted(by_class.items()):
        avg_recall = sum(a.recall_at_k for a in class_audits) / len(class_audits)
        avg_mrr = sum(a.mrr_at_k for a in class_audits) / len(class_audits)
        zero_count = sum(1 for a in class_audits if a.recall_at_k == 0.0)
        lines.append(
            f"- **{query_class}** ({len(class_audits)} queries): "
            f"Avg Recall@10={avg_recall:.2f}, Avg MRR@10={avg_mrr:.2f}, "
            f"Zero-Recall={zero_count}/{len(class_audits)}"
        )

    lines.extend(["", "### Common Failure Patterns", ""])
    
    # Header/source bias analysis
    header_biased = [a for a in audits if a.header_source_bias == "header_dominated"]
    source_biased = [a for a in audits if a.header_source_bias == "source_dominated"]
    if header_biased:
        lines.append(f"- **Header-dominated top-10:** {len(header_biased)} queries")
        lines.append(f"  - Query IDs: {', '.join(a.query_id for a in header_biased)}")
    if source_biased:
        lines.append(f"- **Source-dominated top-10:** {len(source_biased)} queries")
        lines.append(f"  - Query IDs: {', '.join(a.query_id for a in source_biased)}")

    # Low token overlap pattern
    low_overlap_queries = []
    for audit in audits:
        avg_overlap = sum(qd.token_overlap_count for qd in audit.qrels_diagnostics) / len(audit.qrels_diagnostics)
        if avg_overlap < 2.0:
            low_overlap_queries.append((audit.query_id, avg_overlap))
    
    if low_overlap_queries:
        lines.append(f"- **Low token overlap (< 2 tokens avg):** {len(low_overlap_queries)} queries")
        lines.append(f"  - Query IDs: {', '.join(qid for qid, _ in low_overlap_queries)}")

    lines.extend(["", "---", ""])

    # Recommendations
    lines.extend([
        "## Diagnostic Recommendations",
        "",
        "Based on the failure patterns observed:",
        "",
    ])

    if len(zero_recall_queries) > 0:
        lines.append(f"1. **Coverage Issues:** {len(zero_recall_queries)} queries have zero recall")
        ranking_miss_count = sum(1 for a in zero_recall_queries if "ranking_miss" in a.failure_modes)
        if ranking_miss_count > 0:
            lines.append(f"   - {ranking_miss_count} are ranking misses (qrels exist but not retrieved)")
            lines.append("   - Consider: query expansion, term weighting, or BM25 parameter tuning")

    if len(low_overlap_queries) > len(zero_recall_queries):
        lines.append(f"2. **Query-Qrel Mismatch:** {len(low_overlap_queries)} queries have low token overlap")
        lines.append("   - Consider: reviewing qrel relevance, query reformulation, or semantic retrieval")

    if header_biased or source_biased:
        lines.append("3. **File Type Bias:** Header/source bias detected in top-k results")
        lines.append("   - Consider: file-type boosting or separate header/source indices")

    avg_partial_rank = sum(
        a.best_positive_rank for a in partial_recall_queries if a.best_positive_rank
    ) / len(partial_recall_queries) if partial_recall_queries else 0
    if avg_partial_rank > 5:
        lines.append(f"4. **Ranking Quality:** Partial-recall queries have avg best-positive rank {avg_partial_rank:.1f}")
        lines.append("   - Consider: improving lexical scoring, query processing, or adding dense retrieval")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/corpus.jsonl"),
        help="Path to corpus.jsonl",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("configs/benchmark_queries.json"),
        help="Path to benchmark_queries.json",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("configs/benchmark_qrels.jsonl"),
        help="Path to benchmark_qrels.jsonl",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/benchmark_failure_audit.json"),
        help="Output path for JSON audit",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/benchmark_failure_audit.md"),
        help="Output path for Markdown report",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k cutoff for retrieval")
    parser.add_argument(
        "--backend",
        type=str,
        default="lexical_bm25_memory",
        choices=["lexical_bm25_memory", "dense_hash_memory", "bm25", "dense_hash"],
        help="Retrieval backend.",
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        default="baseline",
        choices=["identity", "baseline", "lexnorm"],
        help="Query transformer mode.",
    )
    args = parser.parse_args()

    print("Loading corpus...")
    corpus, corpus_rows = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} chunks")

    print("Loading benchmark queries...")
    queries = load_benchmark_queries(args.queries)
    print(f"Loaded {len(queries)} queries")

    print("Loading benchmark qrels...")
    qrels_map = load_benchmark_qrels(args.qrels)
    print(f"Loaded qrels for {len(qrels_map)} queries")

    print(f"Building retrieval backend: {args.backend}...")
    backend = build_retrieval_backend(
        args.backend,
        corpus_rows=corpus_rows,
        corpus_artifact_path=args.corpus,
    )
    query_transformer = build_query_transformer(args.query_mode)
    pipeline = RetrievalPipeline(backend=backend, query_transformer=query_transformer)

    print("Running retrieval and auditing queries...")
    audits = []
    for query_entry in queries:
        query_id = query_entry["id"]
        query_text = query_entry["query"]
        qrels = qrels_map.get(query_id, [])

        if not qrels:
            print(f"Warning: No qrels found for query {query_id}, skipping")
            continue

        # Retrieve
        results_candidates = pipeline.search(query_text, top_k=100)  # Retrieve more to check ranks
        # Convert to (chunk_id, score) tuples
        results = [(cand.chunk_id, cand.score) for cand in results_candidates]
        
        # Audit
        audit = audit_query(
            query_entry=query_entry,
            qrels=qrels,
            corpus=corpus,
            retrieval_results=results,
            top_k=args.top_k,
        )
        audits.append(audit)
        print(f"  [{query_id}] Recall@{args.top_k}={audit.recall_at_k:.2f}, MRR@{args.top_k}={audit.mrr_at_k:.2f}")

    print(f"\nAudited {len(audits)} queries")

    # Serialize to JSON
    print(f"Writing JSON audit to {args.output_json}...")
    audit_dicts = [asdict(audit) for audit in audits]
    args.output_json.write_text(json.dumps(audit_dicts, indent=2), encoding="utf-8")

    # Generate markdown report
    print(f"Generating markdown report to {args.output_md}...")
    generate_markdown_report(audits, args.output_md)

    print("\nAudit complete!")


if __name__ == "__main__":
    main()
