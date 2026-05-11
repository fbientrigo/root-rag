# oracle_probe_runtime

Tags: #FairShip #runtime #oracle

## Status
RUNTIME_UNVALIDATED

## Summary
Oracle probe is defined as runtime verification step; validated outputs are not yet promoted.

## What this note adds
Connects runtime artifacts to oracle candidate fields.

## Claims
<!-- CLAIM: PROVISIONAL -->
Oracle runtime probe remains required for validation and not yet promoted.
<!-- SOURCE: docs/wiki/workflows/fairship_muon_dis_lxplus_validation.md:1-48 -->

## Evidence anchors
- `docs/wiki/workflows/fairship_muon_dis_lxplus_validation.md:1-48` - probe workflow.

## Connections
| from | relation | to | status | evidence | does_not_prove |
|---|---|---|---|---|---|
| `[[fairship/runtime/oracle_probe_runtime]]` | validates candidates in | `[[fairship/oracle/Oracle_schema]]` | `RUNTIME_UNVALIDATED` | `docs/wiki/workflows/fairship_muon_dis_lxplus_validation.md:1-48` | final truth labels |

## Operational use
Use as evidence sink for future runtime json/log artifacts.

## What this does NOT prove
- Physics correctness of labels.

## Open questions
- Which runtime artifact set is minimal for promotion?


