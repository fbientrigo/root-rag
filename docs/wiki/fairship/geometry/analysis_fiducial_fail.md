# analysis_fiducial_fail

Tags: #FairShip #geometry #oracle

## Status
PROVISIONAL

## Summary
Analysis fiducial fail channel exists and must remain separate from wall-hit semantics.

## What this note adds
Prevents semantic collapse in oracle labels.

## Claims
<!-- CLAIM: PROVISIONAL -->
analysis fiducial fail logic exists in analysis tooling.
<!-- SOURCE: macro/ShipAna.py:221-250 -->
<!-- SOURCE: python/experimental/analysis_toolkit.py:169-184 -->

## Evidence anchors
- `macro/ShipAna.py:221-250` - analysis-side fiducial logic.
- `python/experimental/analysis_toolkit.py:169-184` - toolkit reference.

## Connections
| from | relation | to | status | evidence | does_not_prove |
|---|---|---|---|---|---|
| `[[fairship/geometry/analysis_fiducial_fail]]` | is not equivalent to | `[[fairship/geometry/geant4_volume_wall_hit]]` | `UNRESOLVED` | `macro/ShipAna.py:221-250`; `python/experimental/analysis_toolkit.py:169-184` | canonical wall truth |

## Operational use
Use as distinct channel in oracle interpretation notes.

## What this does NOT prove
- Geant4 boundary-hit truth.

## Open questions
- Which direct geant4 boundary observable proves wall interaction?


