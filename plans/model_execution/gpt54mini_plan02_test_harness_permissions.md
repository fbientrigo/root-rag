# Plan 02 - Test Harness Permissions

Model: `gpt54mini`  
Timebox: `<= 2 hours`

## SMART Goal
Within this plan, retrieval/evaluation tests must run without temp/cache permission failures in this environment.

Success is measured by:
1. reproducible pytest command using repo-local writable temp/cache dirs
2. no `PermissionError` from temp/cache setup
3. retrieval and evaluation targeted suites complete

## Operating Regime
- Environment and test harness stabilization only
- Functional retrieval logic must remain unchanged

## Primary Files To Touch
- `pyproject.toml` (if pytest opts are centralized)
- `tests/conftest.py` (if fixtures need local tmp root)
- optional: `scripts/` helper for canonical test command

## Hard Restrictions
- Do not silence failing tests with skips unless explicitly justified
- Do not reduce assertion strength in existing retrieval tests
- No destructive filesystem actions

## Steps
1. Reproduce the current temp/cache permission failure.
2. Add deterministic repo-local temp/cache strategy for pytest runtime.
3. Ensure the strategy works in CI-like non-interactive shell context.
4. Run retrieval/evaluation focused test set and confirm pass/fail signals are real.

## Verification
- Run: targeted pytest command for retrieval/evaluation modules
- Confirm: no temp/cache permission errors
- Confirm: test results are stable across two runs

## Definition Of Done
- A single documented command runs targeted tests without permission errors
- No functional retrieval changes were needed

