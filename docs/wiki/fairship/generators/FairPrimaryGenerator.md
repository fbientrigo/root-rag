# FairPrimaryGenerator

Tags: #FairShip #generator #transport

## Status
CONFIRMED_BY_CODE

## Summary
Primary generator receives configured generators in steering flow.

## What this note adds
Defines where generator composition occurs.

## Claims
<!-- CLAIM: PROVISIONAL -->
`FairPrimaryGenerator` is created before branch-specific generator setup.
<!-- SOURCE: macro/run_simScript.py:407-409 -->

## Evidence anchors
- `macro/run_simScript.py:407-409` - instantiation site.

## Connections
| from | relation | to | status | evidence | does_not_prove |
|---|---|---|---|---|---|
| `[[fairship/scripts/run_simScript_MuDIS]]` | configures | `[[fairship/generators/FairPrimaryGenerator]]` | `CONFIRMED_BY_CODE` | `macro/run_simScript.py:407-409` | full command correctness |

## Operational use
Use as parent node for generator wiring interpretation.

## What this does NOT prove
- Runtime event correctness.

## Open questions
- How does ordering behave in complete runtime chain?


