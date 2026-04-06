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
- corpus profile: `fairship_only_valid`
- branch role: frozen evaluation floor

Formal source of truth:
- `docs/baseline_contract.md`
- `configs/baseline_manifest.json`

Run command:
- `python scripts/run_official_bm25_baseline.py`
This is the official one-command baseline run. It emits:
- `artifacts/baseline_official/benchmark_eval_results_baseline.json`
- `artifacts/baseline_official/benchmark_failure_audit_baseline.json`
- `artifacts/baseline_official/benchmark_failure_audit_baseline.md`
- `artifacts/baseline_official/baseline_run_manifest.json`
- `artifacts/baseline_official/baseline_summary.md`

Official mode-alignment command (B0 vs B1, plus audits and comparison report):
- `python scripts/run_benchmark_mode_tracks.py`
This generates:
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

S1 semantic artifact build command:
- `python scripts/build_semantic_index.py --corpus artifacts/corpus.jsonl --output-dir artifacts/semantic_s1 --model-name sentence-transformers/all-MiniLM-L6-v2`
- This produces `index.faiss`, `records.jsonl`, `vectors.npy`, and `semantic_manifest.json`.

S1 benchmark command:
- `python scripts/run_retrieval_benchmark.py --backend hybrid_s1 --query-mode baseline --top-k 10 --semantic-manifest artifacts/semantic_s1/semantic_manifest.json --output artifacts/benchmark_eval_results_S1.json`

Official mode-alignment command with S1 enabled:
- `python scripts/run_benchmark_mode_tracks.py --s1-semantic-manifest artifacts/semantic_s1/semantic_manifest.json`
- This adds:
- `artifacts/benchmark_eval_results_S1.json`
- `artifacts/benchmark_failure_audit_S1.json`
- `artifacts/benchmark_failure_audit_S1.md`
- `artifacts/benchmark_semantic_comparison_S1.md`

CI automation:
- GitHub Actions workflow: `.github/workflows/benchmark_mode_alignment.yml`
- Triggered automatically on relevant `push`/`pull_request` changes and manually via `workflow_dispatch`.
- Uploads the generated B0/B1 benchmark and audit artifacts as workflow artifacts.

Manual GitHub run:
1. Open Actions and choose `benchmark-mode-alignment`.
2. Click `Run workflow`.
3. Download artifact `benchmark-mode-alignment`.

Dense side-by-side comparison command:
- `python scripts/run_retrieval_benchmark.py --backend dense_hash_memory --query-mode baseline --top-k 10 --side-by-side-lexical --output artifacts/benchmark_retrieval_dense_hash_baseline.json`
This keeps lexical frozen settings unchanged while adding `side_by_side_vs_lexical_baseline` in output.

Semantic-hash S0 benchmark command:
- `python scripts/run_retrieval_benchmark.py --backend semantic_hash_memory --query-mode baseline --top-k 10 --output artifacts/benchmark_eval_results_S0.json`
- This is a deterministic local semantic-style baseline based on expanded identifier parts, alias features, and character trigrams.

Frozen benchmark inputs for the official baseline:
- `artifacts/corpus.jsonl` (retrieval corpus)
- `configs/benchmark_queries.json` (materialized query set snapshot)
- `configs/benchmark_qrels.jsonl` (materialized qrel snapshot)
- `configs/benchmark_query_subsets.json` (official category subsets)

Official category subsets tracked by the baseline contract:
- `root_basic`
- `sofie_absence_control`
- `root_sofie_integration`
- `repo_specific`
- `critical_queries`
- `fairship_only_valid`
- `extended_corpus_valid`

Current corpus note:
- `sofie` and `root_sofie_integration` subsets are intentionally empty on the current frozen corpus (`artifacts/corpus.jsonl`), so future SOFIE coverage can be added without changing artifact schema.

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

## Semantic Status

Verified in `src/root_rag/retrieval/backends.py`:
- Available backends: `lexical_fts5`, `lexical_bm25_memory`, `dense_hash_memory`, `semantic_hash_memory`, `semantic_faiss`, `hybrid_s1`.
- `dense_hash_memory` is deterministic hashed-vector retrieval and is not embedding-based semantic retrieval.
- `semantic_hash_memory` is a deterministic local semantic-style baseline, not a remote embedding provider.
- `semantic_faiss` is the opt-in S1 local-embedding exact-FAISS backend.
- `hybrid_s1` is the opt-in S1 lexical-plus-semantic backend with weighted RRF and symbol-safe lexical pinning.

Current status:
- True remote embedding retrieval is not available and is intentionally out of scope for S1-v1.

Accepted next step (already approved):
1. `S0` is now implemented as `semantic_hash_memory` and compared against frozen lexical `B0`.
2. The next semantic milestone is a provider-backed embedding backend beyond `S0`.
3. Keep B0/B1 benchmark contract unchanged.

## Future Backend Comparison Rule

Any future backend must be compared against the official lexical baseline using the same frozen inputs and cutoffs:
1. Run baseline with `--backend lexical_bm25_memory --query-mode baseline --top-k 10`.
2. Run candidate backend with identical query/qrel set and `top_k`.
3. Compare `MRR@10`, `Recall@10`, `nDCG@10` plus operational metrics from the same report format.
4. Treat any unexplained metric drift in the lexical baseline as a contract break.
