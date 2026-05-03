# Prompt: Query Pack Iteration

Read `AGENTS.md` first. Then read `agents/codex_emv/runbooks/03_query_pack_iteration.md`.

Goal: improve the Muon DIS query pack while keeping queries atomic, identifier-first, and evidence-grounded.

Rules:

- Use real console commands.
- If a command fails, produce top-3 hypotheses with priors summing to 100%.
- Classify each query as `HIT_OR_TEXT_EVIDENCE`, `ZERO_HIT`, or `ERROR`.
- Split long conceptual queries into atomic identifier-first queries.
- Do not invent FairShip claims.
- Do not promote query hits to qrels or wiki claims without manual review.
- Keep missing evidence as pending with `NOT FOUND IN INDEX`.

Required final response:

- Verdict: `ACCEPT`, `ACCEPT WITH NOTES`, or `BLOCKED`.
- Files modified.
- Tests run.
- Smoke result.
- Query classifications.
- Blockers.
- Exact single-line PowerShell commands run.
