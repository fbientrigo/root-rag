# UBT

Tags: #FairShip #detector #oracle #UBT

## Status
CONFIRMED_BY_CODE

## Summary
The Upstream Tagger (UBT) is used for vetoing upstream backgrounds. It employs a 0.1 GeV/c momentum threshold for hits contributing to the veto decision.

## What this note adds
Canonical detector anchor with verified momentum threshold logic.

## Claims
<!-- CLAIM: CONFIRMED code-local -->
UBT logic employs a 0.1 GeV/c momentum threshold for contributing hits.
<!-- SOURCE: python/shipVeto.py:77 -->

## Evidence anchors
- `python/shipVeto.py:77`: `if mom.Mag() > 0.1: nHits += 1`
- `UpstreamTagger/UpstreamTagger.h:39`: Default dimensions 4.4m x 6.4m x 16cm.

## Connections
| from | relation | to | status | evidence |
|---|---|---|---|---|
| [[fairship/detectors/UBT]] | implements threshold in | [[fairship/scripts/shipVeto]] | CONFIRMED_BY_CODE | `python/shipVeto.py:77` |
| [[fairship/detectors/UBT]] | defined in | [[fairship/detectors/UpstreamTagger]] | CONFIRMED_BY_CODE | `UpstreamTagger/UpstreamTagger.h:39` |

## Operational use
Used to verify the UBT veto logic and its simplified geometry in the simulation.

## What this does NOT prove
- Physics rationale for the 0.1 GeV/c threshold (UNRESOLVED).
- Final physical truth labels.

## Open questions
- What is the physical basis for the 0.1 GeV/c threshold?
