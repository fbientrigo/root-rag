# cbmsim_tree

Tags: #FairShip #tree #runtime

## Status
PROVISIONAL

## Summary
cbmsim appears as runtime output inspection target in validation workflows.

## What this note adds
Separates runtime tree inspection from DIS tree construction.

## Claims
<!-- CLAIM: PROVISIONAL -->
cbmsim is treated as runtime-inspection surface in validation workflow materials.
<!-- SOURCE: docs/wiki/workflows/fairship_muon_dis_lxplus_validation.md:1-48 -->

## Evidence anchors
- `docs/wiki/workflows/fairship_muon_dis_lxplus_validation.md:1-48` - runtime checklist mention.

## Connections
| from | relation | to | status | evidence | does_not_prove |
|---|---|---|---|---|---|
| `[[fairship/runtime/oracle_probe_runtime]]` | inspects | `[[fairship/trees/cbmsim_tree]]` | `RUNTIME_UNVALIDATED` | `docs/wiki/workflows/fairship_muon_dis_lxplus_validation.md:1-48` | successful run |

## Operational use
Use as placeholder runtime-inspection target.

## What this does NOT prove
- Existence of cbmsim in produced outputs.

## Open questions
- Which exact producer path guarantees cbmsim emission?


