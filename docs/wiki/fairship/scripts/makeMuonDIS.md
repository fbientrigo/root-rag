# makeMuonDIS

Tags: #FairShip #MuDIS #script #tree

## Status
CONFIRMED_BY_CODE

## Summary
Ntuple builder defines and fills `DIS` tree branches.

## What this note adds
Pins branch declarations used by transport/oracle notes.

## Claims
<!-- CLAIM: PROVISIONAL -->
`DIS` tree and branch containers are declared in `makeMuonDIS.py`.
<!-- SOURCE: muonDIS/makeMuonDIS.py:147-165 -->

## Evidence anchors
- `muonDIS/makeMuonDIS.py:147-150` - tree creation.
- `muonDIS/makeMuonDIS.py:152-165` - branch declarations.

## Connections
| from | relation | to | status | evidence | does_not_prove |
|---|---|---|---|---|---|
| `[[fairship/scripts/makeMuonDIS]]` | produces | `[[fairship/trees/DIS_tree]]` | `CONFIRMED_BY_CODE` | `muonDIS/makeMuonDIS.py:147-150` | runtime file validity |

## Operational use
Use as schema source for branch-level checks.

## What this does NOT prove
- Runtime transport success.

## Open questions
- Is there a versioned schema artifact beyond implementation code?


