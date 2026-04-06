# Semantic S0 Next Step (Accepted)

Date: 2026-04-02

Status: Accepted and implemented as local `semantic_hash_memory`

## Why this exists

B0/B1 mode alignment is now reproducible and audited.
The next retrieval milestone is introducing a true semantic baseline without breaking the lexical contract.

## Decision

Implement `S0` as an opt-in semantic retrieval track, compared directly against lexical `B0`.
Current implementation uses deterministic local semantic-hash features.

## Scope

1. Add a deterministic local semantic-style retrieval backend (non-default).
2. Add a reproducible benchmark command for `S0`.
3. Generate `artifacts/benchmark_eval_results_S0.json` with current evaluation schema.
4. Generate B0-vs-S0 comparison report in the same style as B0-vs-B1.
5. Keep B0/B1 scripts, outputs, and CI behavior unchanged.

## Non-goals

- No removal or behavior change of lexical B0/B1 tracks.
- No large refactor of retrieval pipeline.
- No qrel/query set changes.

## Acceptance Criteria

- B0 metrics remain stable under unchanged inputs.
- S0 run is deterministic/reproducible.
- B0-vs-S0 report includes global, per-class, and per-query deltas.
- CI still passes benchmark mode alignment workflow for B0/B1.

## Follow-on step

After this implementation, the next semantic milestone is a provider-backed embedding backend.
