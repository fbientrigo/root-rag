# Acceptance Gates

A task is not `ACCEPT` unless every applicable hard gate is satisfied.

## Hard Gates

- [ ] Targeted tests pass.
- [ ] Real smoke command ran.
- [ ] Manifest inspected.
- [ ] Report inspected.
- [ ] Evaluator inspected, if relevant.
- [ ] No stale evidence directory was reused silently.
- [ ] No report is presented as success if all queries are `ERROR`.
- [ ] Wiki linter passes if wiki was touched.
- [ ] Workflow graph validator passes if graph was touched.
- [ ] Final response includes exact commands run.
- [ ] Heartbeat next-prompt policy is satisfied for verdict (`ACCEPT WITH NOTES`/`BLOCKED` require next prompt file).
- [ ] Heartbeat can be updated via short safe command mode (`--preset` or `--from-json`).
- [ ] Wiki `CONFIRMED` claim gate enforced:
  - `SOURCE` file:line evidence required,
  - qrel/review evidence required for workflow relevance claims,
  - no unresolved contradiction.
- [ ] PROVISIONAL claims are not written using confirmed language.

## Muon DIS Gate

- [ ] `python scripts/emv_status.py` inspected before verdict.
- [ ] `python scripts/print_muon_dis_review_sheet.py` generated reviewer-facing sheet.
- [ ] `run_query_pack.py` ran with FairShip `--index-dir` and `--index-id`.
- [ ] Report shows `Error queries < Query Count`.
- [ ] Preferably, report shows `Queries with hits > 0`.
- [ ] qrels may remain pending.
- [ ] Claims were not promoted from text-wrapper evidence without manual review.
- [ ] V0 freeze is blocked unless confirmed/promoted qrels satisfy threshold.

## Non-clean Closure Gate

`ACCEPT WITH NOTES` is valid only if:

- [ ] All notes are listed.
- [ ] Each note is classified as one of:
  - `deferred_by_design`
  - `needs_followup`
  - `blocked_external`
  - `informational`
- [ ] Every `needs_followup` note has a concrete action.
- [ ] `agents/codex_emv/heartbeat/next_prompt.md` is written.
- [ ] `agents/codex_emv/heartbeat/state.json` points to that next prompt path.

`BLOCKED` is valid only if:

- [ ] Failure is reproduced.
- [ ] Top hypotheses are listed.
- [ ] Minimal next unblock command is specified.
- [ ] `agents/codex_emv/heartbeat/next_prompt.md` is written.
- [ ] `agents/codex_emv/heartbeat/state.json` points to that next prompt path.

## Verdict Rules

- `ACCEPT`: all hard gates pass and no relevant blocker remains.
- `ACCEPT WITH NOTES`: core task is complete, but qrels, optional tooling, or manual review remains pending.
- `BLOCKED`: any hard gate fails, any smoke command cannot run, or required evidence is missing.

## Short-Command Preferred Modes

- Primary status command:
  - `python scripts/emv_status.py`
- Primary review-sheet command:
  - `python scripts/print_muon_dis_review_sheet.py`
- Primary vertical-slice command:
  - `python scripts/run_muon_dis_vertical_slice.py --run-id <short_run_id>`
- Primary heartbeat update commands:
  - `python scripts/update_heartbeat.py --preset qrel_review_pending`
  - `python scripts/update_heartbeat.py --from-json reports/<file>.json`
