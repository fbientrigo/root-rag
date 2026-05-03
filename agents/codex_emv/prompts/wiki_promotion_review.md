# Prompt: Wiki Promotion Review

Read `AGENTS.md` first. Then read `docs/wiki/CLAIM_FORMAT.md` and `agents/codex_emv/templates/acceptance_gates.md`.

Goal: review whether evidence is strong enough for wiki promotion.

Rules:

- Use real console commands.
- If a command fails, produce top-3 hypotheses with priors summing to 100%.
- Do not mark any wiki claim `CONFIRMED` unless the required source anchors are present.
- Every function, class, or macro claim needs file path and line range from root-rag output.
- Every call-order or data-flow claim needs at least two independent evidence records.
- If required evidence is missing, write `NOT FOUND IN INDEX`.
- Do not use web knowledge or model memory for FairShip code gaps.

Required final response:

- Verdict: `ACCEPT`, `ACCEPT WITH NOTES`, or `BLOCKED`.
- Files modified.
- Tests run.
- Wiki linter result.
- Smoke result, if retrieval artifacts were generated.
- Claims accepted, rejected, or left unresolved.
- Exact single-line PowerShell commands run.
