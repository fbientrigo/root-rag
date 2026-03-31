# Plan 03 - Backend Contract Hardening

Model: `gpt53codexhigh`  
Reasoning level: `high`  
Timebox: `<= 3 hours`

## SMART Goal
Stabilize the retrieval backend contract so any future backend can plug in without changing benchmark semantics or shared result schema.

Success is measured by:
1. explicit contract guarantees documented in code and docs
2. contract-level tests for required backend behavior
3. lexical backend still reproduces identical baseline metrics

## Operating Regime
- Interface and contract hardening
- Backward-compatible by default
- No retrieval ranking logic changes

## Primary Files To Touch
- `src/root_rag/retrieval/interfaces.py`
- `src/root_rag/retrieval/backends.py`
- `src/root_rag/retrieval/pipeline.py`
- `tests/` contract tests
- `docs/retrieval_quality_checks.md`

## Hard Restrictions
- Keep shared result object explicit (`EvidenceCandidate` unless a migration is fully justified)
- No embeddings in this plan
- Stop and report if lexical benchmark drift appears

## Steps
1. Review current interface surface and identify implicit assumptions.
2. Convert assumptions to explicit contract points (required methods, required metadata behavior, error behavior).
3. Add contract tests using stub backends and lexical backend.
4. Re-run lexical benchmark to confirm unchanged outputs.

## Verification
- Contract tests pass
- Benchmark comparison shows no drift
- CLI search path still resolves and returns expected evidence fields

## Definition Of Done
- Retrieval contract is explicit, tested, and backward-compatible
- Lexical baseline remains unchanged

