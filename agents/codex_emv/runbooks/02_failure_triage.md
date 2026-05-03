# 02 Failure Triage

Use this runbook for any failing command, suspicious manifest, all-error report, stale evidence risk, or evaluator mismatch.

## Protocol

For any failure:

1. Produce top 3 hypotheses.
2. Assign priors summing to 100%.
3. List evidence for and against each hypothesis.
4. Define one falsification command for each hypothesis.
5. Run the minimal command that distinguishes hypotheses.
6. Update probabilities after evidence.
7. If the same hypothesis fails twice, demote it and switch.

Use this exact format:

```markdown
H1 — 70%: ...
Evidence:
Falsification:
Result:
Updated probability:

H2 — 20%: ...
Evidence:
Falsification:
Result:
Updated probability:

H3 — 10%: ...
Evidence:
Falsification:
Result:
Updated probability:
```

## Minimal Commands

```powershell
python scripts/run_query_pack.py --help
```

```powershell
root-rag ask --help
```

```powershell
Get-ChildItem data/indexes_fairship -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 3 Name,FullName
```

```powershell
Get-Content evidence/<run-id>/manifest.json
```

```powershell
Get-Content reports/<run-id>_muon_dis_workflow.md
```

```powershell
Get-Content reports/<run-id>_muon_dis_eval.json
```

## Common Failure Classes

- CLI contract mismatch: command uses an option the target CLI does not accept.
- Wrong index: `index_dir` or `index_id` is missing, stale, or points to the wrong index root.
- Stale artifacts: report or evaluator reads an old evidence directory.
- Evidence shape mismatch: report or evaluator expects structured hits but receives text-wrapper output.
- Query mismatch: query is too broad, too conceptual, or not identifier-first.

## Stop Conditions

- Stop before reporting success if all manifest query statuses are `ERROR`.
- Stop before reporting success if the report has `Error queries` equal to `Query Count`.
- Stop before wiki promotion if the evidence is text-wrapper only.
- Stop before modifying retrieval internals unless a falsification command proves the failure is in retrieval behavior.
