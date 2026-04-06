# Baseline Contract

## Purpose

The `baseline` branch is the frozen evaluation floor for ROOT-RAG retrieval comparisons.
It is not the place for iterative retrieval tuning, semantic-default experiments, UI work, or changing benchmark semantics.
Future work on `development` and later `main` must compare against this branch rather than redefining the floor.

## Branch Semantics

- `baseline`: frozen evaluation floor and comparison anchor.
- `development`: active retrieval experimentation and integration branch.
- `main`: stable usable branch for the current project state.

These branch roles are intentionally narrow.
`baseline` is scientific reference material first, not the most capable branch.

## Frozen Scope

The following are frozen for the official baseline definition version `1.0.0`:

- Official backend: `lexical_bm25_memory`
- Official query mode: `baseline`
- Official corpus profile: `fairship_only_valid`
- Official benchmark queries: `configs/benchmark_queries.json`
- Official query subsets: `configs/benchmark_query_subsets.json`
- Official qrels: `configs/benchmark_qrels.jsonl`
- Official top-k: `10`
- Official output artifact directory: `artifacts/baseline_official`
- Official runner command: `python scripts/run_official_bm25_baseline.py`

The machine-readable source of truth is `configs/baseline_manifest.json`.
If the contract in this document and the manifest diverge, the manifest is authoritative for machine checks and the divergence should be treated as a defect.

## Official Inputs

Official corpus profile:
- `fairship_only_valid`

Official backend(s):
- `lexical_bm25_memory` only

Official query mode(s):
- `baseline` only

Official query subsets:
- `root_basic`
- `sofie_absence_control`
- `root_sofie_integration`
- `repo_specific`
- `critical_queries`
- `fairship_only_valid`
- `extended_corpus_valid`

Official qrels files:
- Official baseline qrels: `configs/benchmark_qrels.jsonl`
- Extended qrels kept for non-baseline benchmark support: `configs/benchmark_qrels_extended.jsonl`

Official corpus profiles:
- Official baseline profile: `fairship_only_valid`
- Non-baseline support profile retained for explicit future comparison work: `extended_corpus_valid`

`extended_corpus_valid` remains in the repository because it is part of the benchmark assets, but it is not the official frozen baseline run.
Using it does not count as running the canonical baseline.

## Official Outputs

The official baseline runner must write exactly these artifacts into `artifacts/baseline_official/`:

- `benchmark_eval_results_baseline.json`
- `benchmark_failure_audit_baseline.json`
- `benchmark_failure_audit_baseline.md`
- `baseline_run_manifest.json`
- `baseline_summary.md`

These names are frozen and are the stable comparison surface for future branches.

## Allowed Changes On `baseline`

Allowed changes are exceptional and should be minimal:

- Documentation clarifications that do not change behavior
- Corrections to broken baseline execution when the intended frozen contract is preserved
- Tests that enforce the frozen contract
- Artifact regeneration using the official command when inputs and code are unchanged

Not allowed:

- Retrieval-quality optimization
- Silent changes to benchmark inputs
- Silent changes to artifact names or schema
- Changing the default backend or query mode
- Broadening metrics or benchmark scope without an explicit new baseline definition version

Any behavioral change that affects retrieval outputs, benchmark inputs, or comparison semantics should happen on `development`, not on `baseline`.

## Anti-Goals

- No dense retrieval by default
- No semantic retrieval by default
- No UI work
- No moving-target metrics inside baseline
- No candidate patch mixing inside the official baseline path

## Determinism Rules

The official baseline must remain boring and reproducible.

- The runner is manifest-driven.
- The runner must fail loudly if required files are missing.
- The runner must not select `lexnorm`, `semantic_hash_memory`, `semantic_faiss`, `hybrid_s1`, or other candidate backends.
- The frozen evaluation report must not include volatile timing or timestamp fields.
- Semantic retrieval is disabled by default and disallowed in the official baseline path.

## Separation From Experiments

The repository may contain other benchmark helpers and comparison scripts.
Those are not the canonical baseline runner.

Baseline-only:
- `scripts/run_official_bm25_baseline.py`
- `configs/baseline_manifest.json`
- `artifacts/baseline_official/`

Experimental or comparison-oriented helpers:
- `scripts/run_benchmark_mode_tracks.py`
- `scripts/run_retrieval_benchmark.py`
- `scripts/build_semantic_index.py`

Experimental helpers may compare B0, B1, S0, or later candidates.
They must not be treated as the frozen baseline definition.

## Comparison Policy For Future Branches

Future branches must compare against the official baseline artifacts and frozen settings.

Minimum comparison discipline:
- Run `python scripts/run_official_bm25_baseline.py` on `baseline` or use the frozen baseline artifacts produced by that exact command.
- Run the candidate branch with the same benchmark queries, same qrels, same corpus profile, and same top-k unless the purpose is explicitly to evaluate a different corpus profile.
- Report global deltas, per-class deltas, zero-recall deltas, helped queries, hurt queries, and unchanged queries.
- Make the verdict explicit: `ACCEPT`, `REJECT`, or `INCONCLUSIVE`.
- If the candidate changes the benchmark surface itself, treat that as a contract break rather than a retrieval win.

## Official Command

```bash
python scripts/run_official_bm25_baseline.py
```
