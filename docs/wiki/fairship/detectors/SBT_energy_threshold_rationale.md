---
type: node
domain: fairship
topic: veto
status: CONFIRMED external-doc
claim_ids: [CLM-010]
tags: [SBT, threshold, energy-loss, MIP]
evidence_type: external-doc
---

# SBT Energy Threshold Rationale

## Role
The Surrounding Background Tagger (SBT) uses an energy loss threshold to distinguish between signal-like particle traversals and low-energy noise or environmental background.

## 45 MeV Threshold (Physics Rationale)
- **Value**: 0.045 GeV (45 MeV).
- **Primary Rationale**: 45 MeV corresponds to the energy deposition of a minimum-ionising particle passing through about 30 cm of liquid scintillator.
- **Verified Passage**: "45 MeV corresponds to a minimum-ionising particle passing about 30 cm of liquid scintillator, with detection efficiency close to 99.9%." (arXiv:2112.01487, Section 5.3.2).
- **Efficiency Goal**: Tuned to achieve a detection efficiency close to **99.9%** for charged particles.

## Implementation
- **Digitization**: In `SBTDetector.py`, any cell with a cumulative energy loss less than 45 MeV is marked as "invalid" (`aHit.setInvalid()`).
- **Origin**: Implementation based on technical specifications provided by the SHiP SBT group.

## Source Anchors
- **External Ref**: [[EXTERNAL_EVIDENCE_REGISTRY#EXT-002]] (arXiv:2112.01487 / EPJ C 82:486, Section 5.3.2).
- **Code Ref**: `python/detectors/SBTDetector.py:53`: `if ElossPerDetId[seg] < 0.045: aHit.setInvalid()`.
- **Code Ref**: `python/shipVeto.py:61`: `hitSegments += 1  # threshold of 45 MeV per segment`.

## Links
- [[fairship/detectors/SBT]]
- [[indexes/EXTERNAL_EVIDENCE_REGISTRY]]
