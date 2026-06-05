# Muon DIS V0 Readiness Policy

## Purpose

This policy defines when Muon DIS can move from draft benchmarking to a final V0 freeze.
Readiness is coverage-based and is not decided by total qrel count alone.

## Mandatory V0 Coverage Areas

- `makeMuonDIS` (`q02_make_muon_dis`)
- `run_simScript` (`q03_run_simscript`)
- `ShipReco` (`q04_shipreco`)
- `DOCA` (`q05_doca`)
- `SBT` (`q06_sbt`)
- `UBT` (`q07_ubt`)
- `muIoni` (`q08_muioni`)
- `InactivateMuonProcesses` (`q09_inactivate_muon_processes`)

## Coverage Rules

1. Ordinary retrieval targets (`q02`..`q08`) require at least one area-backed anchor through:
   - an `APPROVED` review decision in `qrels_review_decisions.yaml`, or
   - a confirmed qrel in `qrels.yaml`.
2. `InactivateMuonProcesses` can be satisfied in either way:
   - approved/confirmed anchor evidence, or
   - explicit reviewed `NOT_FOUND_IN_INDEX` status in `qrels_candidates.yaml`.
3. `NOT_FOUND_IN_INDEX` for `InactivateMuonProcesses` satisfies coverage for that area only.
   - It does not create a confirmed qrel.
   - It does not increase `confirmed_qrel_count`.
4. Final V0 freeze remains blocked until all mandatory areas are covered.
5. Draft freeze (`--draft`) may run before full readiness and must remain clearly marked non-final.

## Status Fields (from `scripts/emv_status.py`)

- `v0_coverage_ready`: `true` only when all mandatory areas are covered.
- `missing_v0_coverage_areas`: uncovered mandatory areas.
- `reviewed_not_found_areas`: areas covered via reviewed `NOT_FOUND_IN_INDEX`.
- `v0_readiness_state`:
  - `NO_QRELS_CONFIRMED`
  - `PARTIAL_COVERAGE`
  - `COVERAGE_READY_FOR_FREEZE`
  - `V0_FROZEN`

## What This Policy Measures

- Coverage across mandatory Muon DIS workflow areas.
- Reviewer decision progress (`APPROVED` vs pending).
- Final freeze eligibility coupling to coverage.

## What This Policy Does Not Measure

- Semantic correctness of evidence without human review.
- Workflow graph truth promotion.
- Wiki claim confirmation quality.
- SOTA retrieval or Graphify experiment quality.
