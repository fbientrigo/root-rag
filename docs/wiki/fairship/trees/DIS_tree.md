# DIS_tree

Tags: #FairShip #MuDIS #tree

## Status
CONFIRMED_BY_CODE

## Summary
The `DIS` tree is the primary data structure for the Muon DIS pipeline. It contains input muon state, generated interaction products, and original detector hits.

## What this note adds
Canonical tree concept linking scripts, generator, and reconstruction.

## Claims
<!-- CLAIM: CONFIRMED code-local -->
The `DIS` tree contains `InMuon`, `DISParticles`, `SoftParticles`, `muon_vetoPoints`, and `muon_UpstreamTaggerPoints`.
<!-- SOURCE: muonDIS/makeMuonDIS.py:152-165 -->

## Evidence anchors
- `muonDIS/makeMuonDIS.py:152-165`: Branch declarations.
- `shipgen/MuDISGenerator.cxx:34-53`: Branch binding and consumption.
- `muonDIS/add_muonresponse.py:141-204`: Merging original hits into the simulation tree.

## Connections
| from | relation | to | status | evidence |
|---|---|---|---|---|
| [[fairship/scripts/makeMuonDIS]] | produces | [[fairship/trees/DIS_tree]] | CONFIRMED_BY_CODE | `muonDIS/makeMuonDIS.py:152` |
| [[fairship/trees/DIS_tree]] | schema defined in | [[fairship/trees/DIS_tree_schema]] | CONFIRMED_BY_CODE | N/A |
| [[fairship/trees/DIS_tree]] | feeds | [[fairship/generators/MuDISGenerator]] | CONFIRMED_BY_CODE | `shipgen/MuDISGenerator.cxx:34` |

## Operational use
The `DIS` tree is the "contract" between the pre-selection and simulation stages of the Muon DIS workflow.

## What this does NOT prove
- Final truth semantics (requires oracle validation).

## Open questions
- None regarding the tree structure.
