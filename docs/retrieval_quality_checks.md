# Retrieval Quality Checks

This project now treats retrieval quality evaluation as first-class code.

## Metrics

Primary metrics at `k=10`:
- `MRR@10`
- `Recall@10`
- `nDCG@10` (graded relevance)

Derived comparison labels:
- `helped`
- `hurt`
- `unchanged`

Implementation source:
- `src/root_rag/evaluation/metrics.py`

## Official Baseline

Official reproducible baseline backend:
- `lexical_bm25_memory` (in-memory BM25 lexical backend)

Opt-in experimental dense backend (non-default):
- `dense_hash_memory` (deterministic hashed dense vectors, no external embedding dependency)

Frozen benchmark run contract:
- `query_mode=baseline`
- `top_k=10`
- backend: `lexical_bm25_memory`

Run command:
- `python scripts/run_retrieval_benchmark.py --backend lexical_bm25_memory --query-mode baseline --top-k 10 --output artifacts/benchmark_retrieval_baseline_refactor.json`
Always execute this command before reporting a baseline comparison so the metadata/test expectations stay accurate.

Dense side-by-side comparison command:
- `python scripts/run_retrieval_benchmark.py --backend dense_hash_memory --query-mode baseline --top-k 10 --side-by-side-lexical --output artifacts/benchmark_retrieval_dense_hash_baseline.json`
This keeps lexical frozen settings unchanged while adding `side_by_side_vs_lexical_baseline` in output.

Frozen benchmark inputs:
- `artifacts/corpus.jsonl` (retrieval corpus)
- `artifacts/benchmark_eval_results.json` (legacy benchmark artifact used to reconstruct fixed query/qrel set)
- `configs/benchmark_queries.json` (materialized query set snapshot)
- `configs/benchmark_qrels.jsonl` (materialized qrel snapshot)

Benchmark output now includes both:
- retrieval quality metrics (`summary`, `per_class`, `per_query`)
- operational metrics (`operational.backend_metrics`, `operational.query_latency_ms`)
- optional dense-vs-lexical comparison block (`side_by_side_vs_lexical_baseline`) for non-lexical runs

## Test Coverage

Core metric tests:
- `tests/test_evaluation_metrics.py`

Query preprocessing tests:
- `tests/test_query_transformers.py`

Pipeline composition tests:
- `tests/test_retrieval_pipeline.py`

Retrieval behavior tests (including lexnorm mode):
- `tests/test_retrieval_lexical.py`

Backend contract tests:
- `tests/test_retrieval_backend_contract.py`

## Extensibility Contract

Retrieval is now split into interchangeable components:
- query transformer: `src/root_rag/retrieval/transformers.py`
- backend interface: `src/root_rag/retrieval/interfaces.py`
- FTS5 backend: `src/root_rag/retrieval/backends.py`
- orchestration: `src/root_rag/retrieval/pipeline.py`

Embedding systems can be added as new `RetrievalBackend` implementations without changing CLI contracts.

Backend contract guarantees (frozen for comparisons):
- `search(query, top_k)` returns ranked `EvidenceCandidate` rows and never returns more than `top_k`
- non-positive/invalid `top_k` resolves to empty retrieval results
- recoverable backend runtime failures should fail closed as empty retrieval results
- `operational_metrics()` returns JSON-safe scalar values (`int|float|str|null`)

Dense-specific operational metrics (when backend is `dense_hash_memory`):
- `vector_dim`
- `similarity`
- `avg_nonzero_dims`

## Future Backend Comparison Rule

Any future backend must be compared against the official lexical baseline using the same frozen inputs and cutoffs:
1. Run baseline with `--backend lexical_bm25_memory --query-mode baseline --top-k 10`.
2. Run candidate backend with identical query/qrel set and `top_k`.
3. Compare `MRR@10`, `Recall@10`, `nDCG@10` plus operational metrics from the same report format.
4. Treat any unexplained metric drift in the lexical baseline as a contract break.
