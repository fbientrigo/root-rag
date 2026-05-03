# 03 Query Pack Iteration

BM25 query packs must stay atomic and identifier-first. Long conceptual queries are expected to fail more often because they dilute lexical matching.

## Primary Operational Commands

```powershell
python scripts/emv_status.py
```

```powershell
python scripts/print_muon_dis_review_sheet.py
```

## Query Rules

- Prefer one identifier or distinctive token per query.
- Split combined workflow questions into multiple query ids.
- Keep expected evidence pending until manually reviewed.
- Do not add a claim because a query name sounds domain-relevant.

## Examples

Bad:

```yaml
bm25_tokens: [makeMuonDIS, pythia6, sigmaDIS]
```

Good:

```yaml
bm25_tokens: [makeMuonDIS]
```

Good:

```yaml
bm25_tokens: [muonDIS]
```

Good:

```yaml
bm25_tokens: [ShipReco]
```

## Classification

- `HIT_OR_TEXT_EVIDENCE`: command returned success and produced text-wrapper evidence. Keep pending until manually reviewed for source anchors.
- `ZERO_HIT`: command ran and found no evidence. Split, rename, or keep pending with `NOT FOUND IN INDEX`.
- `ERROR`: command failed or artifact is missing/invalid. Triage the command or environment before editing query terms.

## Decision Rules

- Split a query when it contains more than one independent identifier or stage token.
- Remove a query only when it is out of scope for the current workflow target.
- Keep a query pending when it is relevant but evidence is missing or not manually verified.
- Add a new query only when it tests a concrete identifier, file stem, macro name, class name, or command token.

## Iteration Loop

1. Change one query-pack concern at a time.
2. Run targeted query-pack tests.
3. Rerun the real Muon DIS vertical slice with a fresh run id:

```powershell
python scripts/run_muon_dis_vertical_slice.py --run-id muon_dis_iter_$(Get-Date -Format yyyyMMdd_HHmmss)
```

4. If query changes were tiny and targeted tests already ran in the same heartbeat, you may skip duplicate tests:

```powershell
python scripts/run_muon_dis_vertical_slice.py --run-id muon_dis_iter_$(Get-Date -Format yyyyMMdd_HHmmss) --skip-tests
```

5. Use explicit `--index-dir` or `--index-id` only when diagnosing index selection issues.
6. Review `reports/<run-id>_vertical_slice_summary.json` first, then inspect detailed artifacts.
7. Inspect manifest statuses.
8. Inspect report summary.
9. Inspect evaluator pending/scored state.
10. Record unresolved evidence gaps instead of promoting claims.

## Guarded Qrel Promotion

1. Generate or refresh candidates with `python scripts/run_muon_dis_qrel_review.py ...`.
2. Generate compact reviewer-facing sheet with `python scripts/print_muon_dis_review_sheet.py`.
3. Review `benchmarks/muon_dis/qrels_review_decisions.yaml` manually.
4. Keep default decisions as `NEEDS_CONTEXT` until a reviewer writes explicit decisions.
5. Promote only reviewed approvals using `python scripts/promote_muon_dis_qrels.py --dry-run` first, then normal mode.
6. Never auto-confirm qrels; `NEEDS_CONTEXT` and `REJECTED` entries must not be promoted.
7. Keep wiki claim promotion blocked until manual qrel review is complete.
8. V0 benchmark freeze is blocked until enough reviewed qrels are approved and promoted.
