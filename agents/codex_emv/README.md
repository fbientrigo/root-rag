# Codex EMV Harness

EMV means Evidence, Manifest, Verification. This harness is the operating checklist for Codex sessions that repair or extend root-rag workflow tracing without turning failed smoke runs into progress claims.

## Start Here

1. Read `AGENTS.md`.
2. Read `docs/planning/boulder_muondis_workflow_tracing.json`.
3. Run `agents/codex_emv/runbooks/00_environment.md`.
4. For Muon DIS vertical-slice work, run `agents/codex_emv/runbooks/01_vertical_slice_muondis.md`.
5. If a command fails, stop feature work and use `agents/codex_emv/runbooks/02_failure_triage.md`.
6. Before any success claim, fill `agents/codex_emv/templates/acceptance_gates.md`.

## Hard Rules

- Do not invent FairShip claims.
- Do not mark wiki claims `CONFIRMED`.
- Do not promote text-wrapper evidence into qrels or wiki claims without manual review.
- Do not present a report as successful when every query is `ERROR`.
- Use real console commands, then inspect manifest, report, and evaluator output.
- If evidence is missing, write `NOT FOUND IN INDEX`.

## Heartbeat Protocol

### How to Start a Heartbeat

1. Read `AGENTS.md`.
2. Read this file.
3. Read `agents/codex_emv/heartbeat/current.md`.
4. Read `agents/codex_emv/heartbeat/next_prompt.md` if it exists and is non-empty.
5. Execute the operational prompt before starting unrelated work.

### How to Close a Heartbeat

1. Fill `agents/codex_emv/templates/acceptance_gates.md`.
2. Fill `agents/codex_emv/templates/final_report.md`.
3. Update `agents/codex_emv/heartbeat/state.json`.
4. If needed, write the next operational prompt to `agents/codex_emv/heartbeat/next_prompt.md`.

### What to Do with `ACCEPT WITH NOTES`

- List all notes and classify each one.
- Write a concrete continuation prompt into `agents/codex_emv/heartbeat/next_prompt.md`.
- Set `Next prompt written: YES` in the final report.

### What to Do with `BLOCKED`

- Reproduce the failure and record top hypotheses using `agents/codex_emv/runbooks/02_failure_triage.md`.
- Record a minimal unblock command.
- Write a concrete unblock prompt into `agents/codex_emv/heartbeat/next_prompt.md`.
- Set `Next prompt written: YES` in the final report.

## Required Final Shape

Every Codex final response for harness-driven work must include:

- Verdict: `ACCEPT`, `ACCEPT WITH NOTES`, or `BLOCKED`.
- Files created or modified.
- Tests run.
- Smoke command result.
- Manifest/report/evaluator inspection result.
- Blockers or deferred work.
- Exact single-line PowerShell commands run.
