# DY_Yheight

Tags: #FairShip #config #geometry

## Status
PROVISIONAL

## Summary
DY parser/type/default and Yheight conversion are code-anchored; safe thesis value remains unresolved.

## What this note adds
Keeps DY semantics bounded and reusable.

## Claims
<!-- CLAIM: PROVISIONAL -->
`-Y` is parsed as float `dy` default `6.0`; propagated to `Yheight` with `u.m` conversion.
<!-- SOURCE: macro/run_simScript.py:192-193 -->
<!-- SOURCE: macro/run_simScript.py:360-369 -->
<!-- SOURCE: python/geometry_config.py:147-149 -->
<!-- SOURCE: python/geometry_config.py:182-184 -->

## Evidence anchors
- `macro/run_simScript.py:192-193` - parser.
- `macro/run_simScript.py:360-369` - propagation.
- `python/geometry_config.py:147-149,182-184` - unit conversion.

## Connections
| from | relation | to | status | evidence | does_not_prove |
|---|---|---|---|---|---|
| `[[fairship/config/DY_Yheight]]` | configures shared setup for | `[[fairship/scripts/run_simScript_MuonBack]]` | `CONFIRMED_BY_CODE` | `macro/run_simScript.py:360-369` | safe DY range |
| `[[fairship/config/DY_Yheight]]` | configures shared setup for | `[[fairship/scripts/run_simScript_MuDIS]]` | `CONFIRMED_BY_CODE` | `macro/run_simScript.py:360-369` | thesis impact certainty |

## Operational use
Use as mandatory check before any smoke command using explicit DY.

## What this does NOT prove
- Safe numeric thesis DY range/value.

## Open questions
- NOT FOUND IN INDEX


