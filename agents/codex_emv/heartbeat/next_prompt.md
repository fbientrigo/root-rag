# Next Heartbeat Operational Prompt

Task: perform manual qrel decisions editing with reviewer support artifacts, then run guarded promotion checks.

Constraints:
- Do not auto-confirm qrels.
- Do not promote wiki claims.
- Keep `benchmarks/muon_dis/qrels.yaml` unchanged unless explicit `APPROVED` decisions are present with valid reviewer rationale.
- Keep workflow graph claims unconfirmed.
- Do not freeze V0 benchmark until confirmed/promoted qrels satisfy threshold.

Required outputs:
- Fresh status snapshot from `python scripts/emv_status.py`.
- Review sheet from `python scripts/print_muon_dis_review_sheet.py --output reports/muon_dis_qrel_review_sheet.md`.
- Updated `benchmarks/muon_dis/qrels_review_decisions.yaml` with explicit reviewer decisions.
- Dry-run promotion summary from `python scripts/promote_muon_dis_qrels.py --dry-run`.
- If approvals exist, run guarded promotion in normal mode and capture summary; otherwise keep qrels pending.

Validation requirements:
- Run targeted tests and validators before closing the heartbeat.
- Include exact single-line PowerShell commands and results in the final report.
- Use this targeted test command:
  - `python -m pytest tests/test_run_muon_dis_qrel_review.py tests/test_promote_muon_dis_qrels.py tests/test_emv_status.py tests/test_update_heartbeat.py -q`
- Run validators:
  - `python scripts/lint_wiki_claims.py docs/wiki`
  - `python scripts/validate_workflow_graph.py workflow_graphs/muon_dis_workflow.json`
