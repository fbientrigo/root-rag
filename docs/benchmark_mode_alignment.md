# Benchmark Mode Alignment

This document defines the reproducible benchmark/audit comparisons for baseline query interpretation, lexnorm query interpretation, and the first local semantic-style retrieval baseline.

It is not the canonical frozen baseline definition. The official frozen baseline contract is defined in `docs/baseline_contract.md`, and the canonical baseline command remains `python scripts/run_official_bm25_baseline.py`.

## Goal

Ensure apples-to-apples diagnostics by fixing all variables except `query_mode`.

## Official Tracks

| Track | backend | query_mode | top_k |
| --- | --- | --- | --- |
| B0 | `lexical_bm25_memory` | `baseline` | `10` |
| B1 | `lexical_bm25_memory` | `lexnorm` | `10` |
| S0 | `semantic_hash_memory` | `baseline` | `10` |

## Local Run

```bash
python scripts/run_benchmark_mode_tracks.py
```

## Generated Artifacts

- `artifacts/benchmark_eval_results_B0.json`
- `artifacts/benchmark_eval_results_B1.json`
- `artifacts/benchmark_eval_results_S0.json`
- `artifacts/benchmark_failure_audit_B0.json`
- `artifacts/benchmark_failure_audit_B0.md`
- `artifacts/benchmark_failure_audit_B1.json`
- `artifacts/benchmark_failure_audit_B1.md`
- `artifacts/benchmark_failure_audit_S0.json`
- `artifacts/benchmark_failure_audit_S0.md`
- `artifacts/benchmark_mode_comparison.md`
- `artifacts/benchmark_semantic_comparison.md`
- `artifacts/manual_zero_recall_review_template.md`

## CI Run

Workflow file:
- `.github/workflows/benchmark_mode_alignment.yml`

Triggers:
- `workflow_dispatch`
- Relevant `push`/`pull_request` path changes in retrieval/evaluation scripts and benchmark inputs.

Uploaded workflow artifact:
- `benchmark-mode-alignment`

Manual GitHub run:
1. Open **Actions**.
2. Select **benchmark-mode-alignment**.
3. Click **Run workflow**.
4. Download artifact **benchmark-mode-alignment**.

## Reproducibility Contract

- `top_k` fixed to `10`.
- backend fixed to `lexical_bm25_memory`.
- Queries fixed from `configs/benchmark_queries.json`.
- Qrels fixed from `configs/benchmark_qrels.jsonl`.
- Failure-mode classification is deterministic and ordered.
- `qrel_mismatch_suspected` boolean is derived from `failure_modes` and must stay consistent.

## Semantic Retrieval Status (Verified 2026-04-02)

Code verification:
- `src/root_rag/retrieval/backends.py` currently exposes:
  - `lexical_fts5`
  - `lexical_bm25_memory`
  - `dense_hash_memory` (deterministic hashed vectors; not embedding-based semantic retrieval)
  - `semantic_hash_memory` (deterministic local semantic-style baseline)

Conclusion:
- `S0` semantic-style retrieval is implemented locally.
- True provider-backed embedding retrieval is **not implemented** yet.

## Accepted Next Step (Auto-Accepted)

Decision:
- `S0` is implemented as `semantic_hash_memory`.
- The next semantic step after `S0` is a provider-backed embedding backend.

Scope for S0:
1. Add an official semantic comparison track against B0 using identical queries/qrels/top_k.
2. Emit `benchmark_eval_results_S0.json` in the same schema.
3. Keep lexical B0/B1 contract unchanged.
4. Defer provider-backed embeddings to the next semantic milestone.

Minimum acceptance criteria:
- No regression in B0 contract outputs.
- Reproducible S0 run command documented.
- Side-by-side comparison report includes B0 vs S0 global/per-class/per-query deltas.

Tracking plan:
- `plans/semantic_s0_accepted_2026-04-02.md`
