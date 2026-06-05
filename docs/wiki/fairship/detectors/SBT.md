# SBT

Tags: #FairShip #detector #oracle #SBT

## Status
CONFIRMED_BY_CODE

## Summary
The Scintillating Beauty Tagger (SBT) is a liquid scintillator detector used for vetoing backgrounds. In the digitization layer, it implements a 45 MeV energy deposition threshold.

## What this note adds
Canonical detector anchor with verified threshold logic.

## Claims
<!-- CLAIM: CONFIRMED code-local -->
SBT implements a 45 MeV (0.045 GeV) energy threshold in the digitization layer.
<!-- SOURCE: python/detectors/SBTDetector.py:52-53 -->

## Evidence anchors
- `python/detectors/SBTDetector.py:52-53`: `if ElossPerDetId[seg] < 0.045: aHit.setInvalid()`
- `python/shipVeto.py:61`: `hitSegments += 1  # threshold of 45 MeV per segment`

## Connections
| from | relation | to | status | evidence |
|---|---|---|---|---|
| [[fairship/detectors/SBT]] | implements threshold in | [[fairship/scripts/SBTDetector]] | CONFIRMED_BY_CODE | `python/detectors/SBTDetector.py:52-53` |
| [[fairship/detectors/SBT]] | used by | [[fairship/scripts/shipVeto]] | CONFIRMED_BY_CODE | `python/shipVeto.py:60-65` |

## Operational use
The 45 MeV threshold is critical for grounding the veto efficiency claims in the thesis.

## What this does NOT prove
- Final physical truth labels (requires oracle mapping).
- Physics validation of the 99.9% efficiency claim.

## Open questions
- None regarding the code-local implementation.
