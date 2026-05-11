# run_simScript MuDIS

Tags: #FairShip #MuDIS #script #runtime

## Status
PROVISIONAL

## Summary
MuDIS steering branch wires generator setup in `run_simScript.py`.

## What this note adds
Bounded steering knowledge for runtime packet preparation.

## Claims
<!-- CLAIM: PROVISIONAL -->
MuDIS mode configures `MuDISGenerator` via steering script.
<!-- SOURCE: macro/run_simScript.py:544-553 -->

## Evidence anchors
- `macro/run_simScript.py:544-553` - MuDIS generator wiring.
- `macro/run_simScript.py:407-409` - `FairPrimaryGenerator` setup.

## Connections
| from | relation | to | status | evidence | does_not_prove |
|---|---|---|---|---|---|
| `[[fairship/scripts/run_simScript_MuDIS]]` | configures | `[[fairship/generators/MuDISGenerator]]` | `CONFIRMED_BY_CODE` | `macro/run_simScript.py:544-553` | runtime success |
| `[[fairship/scripts/run_simScript_MuDIS]]` | configures | `[[fairship/generators/FairPrimaryGenerator]]` | `CONFIRMED_BY_CODE` | `macro/run_simScript.py:407-409` | execution ordering correctness |

## Operational use
Use for command/path validation before runtime smoke.

## What this does NOT prove
- LXPLUS success.
- Truth-label correctness.

## Open questions
- Which exact staged command set is validated?


