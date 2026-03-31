# Plan 06 - Dense Readiness Design

Model: `gpt53codexhigh`  
Reasoning level: `high`  
Timebox: `<= 3.5 hours`

## SMART Goal
Produce a concrete, code-adjacent dense-readiness design that can be implemented without altering baseline benchmark rules.

Success is measured by:
1. clear backend loading/registration design
2. required dense config and artifact contract defined
3. comparison protocol against lexical baseline specified and testable

## Operating Regime
- Design + minimal scaffolding only
- No embedding model integration in this plan

## Primary Files To Touch
- `docs/` design note (new or existing retrieval docs)
- `src/root_rag/retrieval/` (only if tiny registry seam is needed)
- `tests/` for protocol-level assertions

## Hard Restrictions
- No dependency additions for embeddings/vector DB
- No change to lexical benchmark default behavior
- Dense path must be opt-in and isolated

## Steps
1. Map required dense backend lifecycle: config -> index artifact -> query -> scored evidence.
2. Define strict input/output contract matching current evaluation path.
3. Specify mandatory benchmark comparison procedure (same queries, qrels, top-k).
4. Add lightweight protocol tests or TODO-guard rails.

## Verification
- Design doc is implementable and references real code entry points
- Existing lexical runs unaffected
- Contract tests/docs eliminate ambiguity for next implementation step

## Definition Of Done
- Dense backend can be implemented later without changing evaluation rules

