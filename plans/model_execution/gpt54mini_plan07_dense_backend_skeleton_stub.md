# Plan 07 - Dense Backend Skeleton Stub (Optional)

Model: `gpt54mini`  
Timebox: `<= 2 hours`

## SMART Goal
Add a non-functional dense backend skeleton that compiles, is discoverable, and is explicitly disabled for production use.

Success is measured by:
1. stub backend class exists behind opt-in flag/config
2. clear `NotImplementedError` boundaries for unimplemented sections
3. skip-marked tests verify wiring without pretending dense retrieval works

## Operating Regime
- Scaffolding only
- No embeddings, no external vector service, no retrieval behavior claims

## Primary Files To Touch
- `src/root_rag/retrieval/` (new stub backend module)
- backend factory/registry entrypoint
- `tests/` stub wiring tests
- optional docs note for stub status

## Hard Restrictions
- Do not change lexical default path
- Do not alter benchmark baseline outputs
- Do not add heavyweight dependencies

## Steps
1. Add stub class implementing backend contract.
2. Register stub in backend factory under explicit dense key.
3. Add tests proving registration/wiring and explicit non-implementation behavior.
4. Document that dense backend is scaffold-only.

## Verification
- Stub imports and wiring tests pass
- Lexical benchmark path remains unchanged
- Dense path fails fast with explicit message

## Definition Of Done
- Project has safe scaffold for dense backend without any functional dense retrieval

