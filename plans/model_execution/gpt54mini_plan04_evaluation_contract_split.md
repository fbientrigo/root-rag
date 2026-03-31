# Plan 04 - Evaluation Contract Split

Model: `gpt54mini`  
Timebox: `<= 2.5 hours`

## SMART Goal
Separate retrieval evaluation flow from legacy artifact reconstruction flow, while preserving existing benchmark outputs.

Success is measured by:
1. benchmark runner has clear boundaries (load inputs, run backend, compute metrics, compare)
2. reconstruction-specific logic is isolated and non-blocking for future backends
3. output JSON schema remains backward-compatible

## Operating Regime
- Refactor for clarity with minimal file movement
- No change in retrieval math or ranking

## Primary Files To Touch
- `scripts/run_retrieval_benchmark.py`
- `src/root_rag/evaluation/` (if extraction of helpers is needed)
- `tests/` benchmark runner tests

## Hard Restrictions
- Preserve current benchmark output fields used by existing docs/artifacts
- Do not modify qrels semantics
- Stop if per-query rankings drift

## Steps
1. Identify mixed concerns in benchmark script.
2. Extract pure evaluation helpers from reconstruction helpers.
3. Keep CLI arguments stable or backward-compatible.
4. Validate old artifacts can still be compared with new output.

## Verification
- Run baseline and lexnorm benchmark modes
- Compare summaries and per-query outputs with existing artifacts
- Run targeted benchmark script tests

## Definition Of Done
- Benchmark code paths are cleanly separated
- Existing benchmark comparisons still work without drift

