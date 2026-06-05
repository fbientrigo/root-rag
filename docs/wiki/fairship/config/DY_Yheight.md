# DY_Yheight

Tags: #FairShip #config #geometry

## Status
CONFIRMED_BY_CODE

## Summary
The `DY` parameter (shield height) is parsed from the `-Y` or `--Dy` flag in steering scripts. It is converted to meters and used as the `Yheight` parameter in the geometry configuration.

## What this note adds
Canonical configuration anchor with verified unit conversion and propagation.

## Claims
<!-- CLAIM: CONFIRMED code-local -->
`-Y` is parsed as float `dy` (default `6.0`) and converted to `Yheight` in meters.
<!-- SOURCE: macro/run_simScript.py:141-260 -->

## Evidence anchors
- `macro/run_simScript.py:192-193`: Parser argument definition.
- `macro/run_simScript.py:360`: `Yheight = options.dy` propagation.
- `python/geometry_config.py:147-149`: `Yheight` used in geometry setup with `* u.m` conversion.

## Connections
| from | relation | to | status | evidence |
|---|---|---|---|---|
| [[fairship/config/DY_Yheight]] | configures | [[fairship/scripts/run_simScript]] | CONFIRMED_BY_CODE | `macro/run_simScript.py:192` |
| [[fairship/config/DY_Yheight]] | converted in | [[fairship/scripts/geometry_config]] | CONFIRMED_BY_CODE | `python/geometry_config.py:147` |

## Operational use
DY is a critical parameter for defining the acceptance and shielding effectiveness in Muon DIS studies.

## What this does NOT prove
- Safe numeric thesis DY range/value (requires physics optimization).

## Open questions
- None regarding the code-local parsing and propagation.
