# run_simScript MuonBack

Tags: #FairShip #MuonBack #script #runtime

## Status
PROVISIONAL

## Summary
MuonBack steering branch is code-visible in `run_simScript.py`.

## What this note adds
Anchors MuonBack steering without implying runtime success.

## Claims
<!-- CLAIM: PROVISIONAL -->
MuonBack steering branch appears in `macro/run_simScript.py` after shared setup.
<!-- SOURCE: macro/run_simScript.py:571-640 -->

## Evidence anchors
- `macro/run_simScript.py:571-640` - MuonBack branch.
- `macro/run_simScript.py:360-369` - shared DY/Yheight setup.

## Connections
| from | relation | to | status | evidence | does_not_prove |
|---|---|---|---|---|---|
| `[[fairship/config/DY_Yheight]]` | configures shared setup for | `[[fairship/scripts/run_simScript_MuonBack]]` | `CONFIRMED_BY_CODE` | `macro/run_simScript.py:360-369` | safe DY value |
| `[[fairship/scripts/run_simScript_MuonBack]]` | configures | `[[MuonBack]]` | `PROVISIONAL` | `macro/run_simScript.py:571-640` | successful execution |

## Operational use
Use as entry node for MuonBack runtime-preflight planning.

## What this does NOT prove
- LXPLUS execution success.
- Canonical MuonBack->MuDIS runtime handoff.

## Open questions
- What runtime artifacts prove successful MuonBack execution?


