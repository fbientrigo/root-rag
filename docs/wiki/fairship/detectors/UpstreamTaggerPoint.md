# UpstreamTaggerPoint

Tags: #FairShip #detector #oracle #UBT

## Status
CONFIRMED_BY_CODE

## Summary
`UpstreamTaggerPoint` is the detector-point class for the Upstream Tagger (UBT), storing MC hits during simulation.

## What this note adds
Detector-point class anchor for the UBT.

## Claims
<!-- CLAIM: CONFIRMED code-local -->
`UpstreamTaggerPoint` inherits from `SHiP::DetectorPoint`.
<!-- SOURCE: UpstreamTagger/UpstreamTaggerPoint.h:16 -->

## Evidence anchors
- `UpstreamTagger/UpstreamTaggerPoint.h:16-31`: Class definition.
- `UpstreamTagger/UpstreamTaggerPoint.cxx:1-21`: Implementation.

## Connections
| from | relation | to | status | evidence |
|---|---|---|---|---|
| [[fairship/branches/UpstreamTaggerPoint]] | stores | [[fairship/detectors/UpstreamTaggerPoint]] | CONFIRMED_BY_CODE | `UpstreamTagger/UpstreamTaggerPoint.h` |
| [[fairship/trees/DIS_tree]] | contains | [[fairship/branches/muon_UpstreamTaggerPoints]] | CONFIRMED_BY_CODE | `muonDIS/makeMuonDIS.py:164` |

## Operational use
Used for branch dependency checks in the Muon DIS pipeline.

## What this does NOT prove
- Final physical truth labels.
