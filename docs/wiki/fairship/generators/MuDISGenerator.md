# MuDISGenerator

Tags: #FairShip #MuDIS #generator

## Status
CONFIRMED_BY_CODE

## Summary
Generator reads DIS branches and injects tracks into primary generator.

## What this note adds
Reusable code-backed generator contract.

## Claims
<!-- CLAIM: PROVISIONAL -->
`MuDISGenerator::Init` binds `InMuon`, `DISParticles`, `SoftParticles`.
<!-- SOURCE: shipgen/MuDISGenerator.cxx:34-53 -->

## Evidence anchors
- `shipgen/MuDISGenerator.cxx:34-53` - tree binding.
- `shipgen/MuDISGenerator.cxx:191-221` - `AddTrack` use.

## Connections
| from | relation | to | status | evidence | does_not_prove |
|---|---|---|---|---|---|
| `[[fairship/generators/MuDISGenerator]]` | consumes | `[[fairship/trees/DIS_tree]]` | `CONFIRMED_BY_CODE` | `shipgen/MuDISGenerator.cxx:34-53` | runtime branch availability |
| `[[fairship/generators/MuDISGenerator]]` | injects into | `[[fairship/generators/FairPrimaryGenerator]]` | `CONFIRMED_BY_CODE` | `shipgen/MuDISGenerator.cxx:191-221` | physics correctness |

## Operational use
Use for code-level transport expectations and branch dependency checks.

## What this does NOT prove
- Runtime LXPLUS success.

## Open questions
- Which module maps outputs into final oracle truth labels?


