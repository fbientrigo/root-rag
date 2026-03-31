# Plan 05 - Operational Metrics Schema

Model: `gptnano`  
Timebox: `<= 75 minutes`

## SMART Goal
Standardize benchmark operational metrics schema so all backends report comparable operational signals.

Success is measured by:
1. explicit schema keys in docs and code (`latency`, `size`, `backend_id`, nullability rules)
2. lexical benchmark output includes the standardized keys
3. a test validates required operational fields

## Operating Regime
- Schema consistency and docs/tests only
- No ranking/retrieval behavior changes

## Primary Files To Touch
- `scripts/run_retrieval_benchmark.py`
- `docs/retrieval_quality_checks.md`
- `tests/` benchmark output schema test

## Hard Restrictions
- Do not invent backend-specific semantics in shared keys
- Use `null`/`None` explicitly when a metric is unavailable
- Keep baseline metrics unchanged

## Steps
1. Define required operational keys and allowed types.
2. Normalize output serialization to that schema.
3. Add output schema test and update docs.
4. Re-run baseline benchmark and compare quality metrics.

## Verification
- Schema test passes
- Baseline quality metrics unchanged
- JSON output remains machine-consumable

## Definition Of Done
- Operational metrics are consistent and test-enforced

