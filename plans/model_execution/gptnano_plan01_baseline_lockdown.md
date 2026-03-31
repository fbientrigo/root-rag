# Plan 01 - Baseline Lockdown

Model: `gptnano`  
Timebox: `<= 90 minutes`

## SMART Goal
By the end of this plan, baseline benchmark execution must be unambiguous and frozen:
- one canonical command
- frozen backend/mode/top-k values documented and asserted
- no retrieval metric drift

Success is measured by:
1. baseline command in docs and script help text alignment
2. metadata assertions added in tests
3. baseline metrics exactly unchanged vs current artifact

## Operating Regime
- Low-risk docs/tests/CLI-surface updates only
- No ranking algorithm changes
- No extraction/provenance/tiering/corpus logic changes

## Primary Files To Touch
- `docs/retrieval_quality_checks.md`
- `scripts/run_retrieval_benchmark.py`
- `tests/` (new targeted benchmark metadata test)

## Hard Restrictions
- Do not change BM25/FTS retrieval behavior
- Do not change query transformer behavior
- If any summary metric drifts, stop and report drift source

## Steps
1. Normalize benchmark command examples so they always include backend, query mode, and top-k explicitly.
2. Add a small test that validates required metadata keys and frozen values for baseline mode.
3. Run benchmark once and compare with existing baseline artifact summary/per-query.
4. Record comparison result in a short note in the PR/body or commit message context.

## Verification
- Run: benchmark baseline command
- Run: targeted test for benchmark metadata contract
- Confirm: `mrr@10`, `recall@10`, `ndcg@10` unchanged

## Definition Of Done
- Clear baseline command exists
- Baseline metadata contract is test-enforced
- Metrics are confirmed unchanged

