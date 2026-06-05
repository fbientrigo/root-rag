# MuonBack

Tags: #FairShip #MuonBack #Generator

## Summary
`MuonBackGenerator` code defines input reading behavior for muon-back style sources.

## Status
CONFIRMED_BY_CODE

## What this node adds
Anchors MuonBack behavior to implementation details without claiming MuDIS bridge runtime truth.

## Claims
<!-- CLAIM: CONFIRMED code-local -->
MuonBack generator code handles branch/address setup for input ingestion paths.
<!-- SOURCE: shipgen/MuonBackGenerator.cxx:71-150 -->

## Connections
| from | relation | to | status | evidence |
|---|---|---|---|---|
| `[[MuonBack]]` | interacts with | `[[vetoPoint]]` | `PROVISIONAL` | `shipgen/MuonBackGenerator.cxx:71-150` |
| `[[run_simScript_MuonBack]]` | configures | `[[MuonBack]]` | `PROVISIONAL` | `macro/run_simScript.py:571-640` |

## Evidence
- `shipgen/MuonBackGenerator.cxx:71-150`: Ingestion/branch behavior anchor.
- `shipgen/MuonBackGenerator.h:1-69`: Class API definition.

## What this does NOT prove
- Canonical MuonBack->MuDIS runtime handoff.
- LXPLUS execution success.

## Open questions
- Which runtime artifact proves handoff into MuDIS ntuple generation?
