# MuDIS_transport_candidate

Tags: #FairShip #runtime #MuDIS

## Status
RUNTIME_UNVALIDATED

## Summary
Transport chain candidate exists but is not runtime validated.

## What this note adds
Captures runtime chain candidate without overclaiming.

## Claims
<!-- CLAIM: PROVISIONAL -->
MuonBack to MuDIS runtime chain remains candidate-only.
<!-- SOURCE: muonShieldOptimization/run_prod.py:10-21 -->
<!-- SOURCE: muonDIS/make_nTuple_SBT.py:211-224 -->

## Evidence anchors
- `muonShieldOptimization/run_prod.py:10-21` - pipeline call-site pattern.
- `muonDIS/make_nTuple_SBT.py:211-224` - ntuple step anchor.

## Connections
| from | relation | to | status | evidence | does_not_prove |
|---|---|---|---|---|---|
| `[[fairship/runtime/MuonBack_smoke]]` | may feed | `[[fairship/runtime/MuDIS_transport_candidate]]` | `RUNTIME_UNVALIDATED` | `muonShieldOptimization/run_prod.py:10-21`; `muonDIS/make_nTuple_SBT.py:211-224` | reproducible lineage |

## Operational use
Use to centralize runtime-chain unresolved status.

## What this does NOT prove
- Runtime-valid canonical sequence.

## Open questions
- What file lineage proves this chain end-to-end?


