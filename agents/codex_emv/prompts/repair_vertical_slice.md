# Prompt: Repair Vertical Slice

Read `AGENTS.md` first. Then read the EMV harness at `agents/codex_emv/README.md`.

Goal: repair the Muon DIS vertical slice without inventing FairShip claims.

Rules:

- Use real console commands.
- If any command fails, produce top-3 hypotheses with priors summing to 100%.
- For each hypothesis, include evidence, falsification command, result, and updated probability.
- Do not infer FairShip behavior from memory.
- Do not mark wiki claims `CONFIRMED`.
- Do not promote text-wrapper evidence without manual review.
- Do not claim success until manifest, report, and evaluator output are inspected.

Required final response:

- Verdict: `ACCEPT`, `ACCEPT WITH NOTES`, or `BLOCKED`.
- Files modified.
- Tests run.
- Smoke result.
- Manifest/report/evaluator inspection result.
- Blockers.
- Exact single-line PowerShell commands run.
