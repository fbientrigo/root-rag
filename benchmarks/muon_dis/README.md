# Muon DIS Benchmark Scaffold

This folder contains the initial benchmark structure for Muon DIS workflow retrieval.

## Files
- `golden_queries.yaml`: query definitions for the benchmark set.
- `qrels.yaml`: relevance labels (`qrels`) split into confirmed vs pending.

## Current State
- All six initial entries are `pending_label: true`.
- No qrel line ranges are marked as confirmed yet.
- `confirmed_qrels` is intentionally empty until manual evidence review is complete.

## How To Promote Pending To Confirmed
1. Retrieve evidence records from root-rag outputs for the target query.
2. Manually verify each candidate against source evidence and capture:
   - `file`
   - `start_line`
   - `end_line`
   - `relevance`
3. Add validated qrel entries to:
   - `golden_queries.yaml` under that query's `qrels` list
   - `qrels.yaml` under `confirmed_qrels`
4. Set `pending_label: false` for the query after at least one qrel is confirmed.
5. Remove or update the matching record under `pending_qrels`.

## Guardrails
- Do not invent line ranges.
- Keep `pending_label: true` when exact evidence is not verified.
- If evidence is missing, keep unresolved state and use `NOT FOUND IN INDEX`.

## Local Deterministic Workflow (Phase 2)
1. Run query pack and emit evidence artifacts:
   - `python scripts/run_query_pack.py --pack query_packs/muon_dis_workflow.yaml --output-dir evidence/<run-id> --dry-run`
2. Generate weekly markdown report:
   - `python scripts/generate_weekly_report.py --evidence-dir evidence/<run-id> --output reports/<run-id>_muon_dis_workflow.md`
3. Evaluate retrieval run against Muon DIS benchmark scaffolding:
   - `python scripts/evaluate_muon_dis_retrieval.py --evidence-dir evidence/<run-id> --output reports/<run-id>_muon_dis_eval.json`

Notes:
- The evaluation script scores only `confirmed_qrels`.
- Queries with `pending_label: true` remain pending and are never auto-promoted.
