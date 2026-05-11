# DIS_tree

Tags: #FairShip #MuDIS #tree

## Status
CONFIRMED_BY_CODE

## Summary
DIS tree contains MuDIS input and observable candidate branches.

## What this note adds
Canonical tree concept linking scripts, generator, and oracle.

## Claims
<!-- CLAIM: PROVISIONAL -->
`DIS` tree includes `InMuon`, `DISParticles`, `SoftParticles`.
<!-- SOURCE: muonDIS/makeMuonDIS.py:152-160 -->

## Evidence anchors
- `muonDIS/makeMuonDIS.py:152-165` - branch declarations.
- `shipgen/MuDISGenerator.cxx:34-53` - branch consumption.

## Connections
| from | relation | to | status | evidence | does_not_prove |
|---|---|---|---|---|---|
| `[[fairship/scripts/makeMuonDIS]]` | produces | `[[fairship/trees/DIS_tree]]` | `CONFIRMED_BY_CODE` | `muonDIS/makeMuonDIS.py:147-150` | runtime integrity |
| `[[fairship/trees/DIS_tree]]` | feeds | `[[fairship/generators/MuDISGenerator]]` | `CONFIRMED_BY_CODE` | `shipgen/MuDISGenerator.cxx:34-53` | full chain validation |

## Operational use
Use for schema inspection and oracle candidate mapping.

## What this does NOT prove
- Final truth semantics.

## Open questions
- Which runtime schema snapshot is accepted?


