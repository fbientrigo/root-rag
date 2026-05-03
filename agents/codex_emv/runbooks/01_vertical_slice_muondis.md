# 01 Vertical Slice: Muon DIS

Run from repository root in PowerShell. Every command is single-line. Use a fresh run id so stale evidence is not reused silently.

## Primary Path

```powershell
python scripts/run_muon_dis_vertical_slice.py --run-id muon_dis_emv_$(Get-Date -Format yyyyMMdd_HHmmss)
```

Then generate review-only qrel candidates:

```powershell
python scripts/run_muon_dis_qrel_review.py --evidence-dir evidence/<run_id> --run-id <run_id> --overwrite
```

Heartbeat close command (short, safe modes):

```powershell
python scripts/update_heartbeat.py --preset qrel_review_pending
```

or

```powershell
python scripts/update_heartbeat.py --from-json reports/<heartbeat_update>.json
```

## Manual Fallback Bundle

```powershell
python -m pytest tests/test_run_query_pack.py tests/test_generate_weekly_report.py tests/test_muon_dis_phase2.py tests/test_lint_wiki_claims.py tests/test_validate_workflow_graph.py -q
```

```powershell
python scripts/run_muon_dis_vertical_slice.py --run-id muon_dis_emv_$(Get-Date -Format yyyyMMdd_HHmmss) --index-dir data/indexes_fairship
```

```powershell
python scripts/run_muon_dis_vertical_slice.py --run-id muon_dis_emv_$(Get-Date -Format yyyyMMdd_HHmmss) --index-dir data/indexes_fairship --index-id <index_id>
```

```powershell
python scripts/run_muon_dis_vertical_slice.py --run-id muon_dis_emv_$(Get-Date -Format yyyyMMdd_HHmmss) --index-dir data/indexes_fairship --skip-tests
```

```powershell
python scripts/lint_wiki_claims.py docs/wiki
```

```powershell
python scripts/validate_workflow_graph.py workflow_graphs/muon_dis_workflow.json
```

## Required Inspection

- Manifest: confirm `index_dir`/`index_id` are valid and query statuses are not all `ERROR`.
- Report: confirm `Error queries < Query Count`.
- Report: prefer `Queries with hits > 0`.
- Evaluator: confirm pending qrels remain pending unless manually reviewed.
- Do not promote claims from text-wrapper evidence without manual review.
