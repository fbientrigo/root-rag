# Oracle_schema

Tags: #FairShip #oracle #MuDIS

## Status
PROVISIONAL

## Summary
Current oracle fields are candidate observables, not final truth labels.

## What this note adds
Defines bounded observable schema and non-proofs.

## Claims
<!-- CLAIM: PROVISIONAL -->
Candidate observable fields include dis_tree_exists and detector/count fields.
<!-- SOURCE: reports/fairship_muon_dis_oracle_observable_schema.md:5-13 -->

## Evidence anchors
- `reports/fairship_muon_dis_oracle_observable_schema.md:5-13` - candidate field list.

## Connections
| from | relation | to | status | evidence | does_not_prove |
|---|---|---|---|---|---|
| `[[fairship/trees/DIS_tree]]` | provides candidate observables to | `[[fairship/oracle/Oracle_schema]]` | `CONFIRMED_BY_CODE` | `muonDIS/makeMuonDIS.py:152-165` | final truth semantics |
| `[[fairship/oracle/Oracle_schema]]` | depends on | `[[fairship/runtime/oracle_probe_runtime]]` | `RUNTIME_UNVALIDATED` | `docs/wiki/workflows/fairship_muon_dis_lxplus_validation.md:1-48` | thesis validation |

## Operational use
Use as schema contract for runtime probe interpretation.

## What this does NOT prove
- Final oracle truth labels.

## Open questions
- Which downstream predicate code defines final labels?


