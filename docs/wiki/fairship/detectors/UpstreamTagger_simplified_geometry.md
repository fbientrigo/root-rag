---
type: node
domain: fairship
topic: veto
status: CONFIRMED
claim_ids: [CLM-009]
tags: [UpstreamTagger, geometry, UBT, simulation]
---

# UpstreamTagger Simplified Geometry

## Role
The Upstream Background Tagger (UBT) is a detector positioned upstream of the decay vessel to tag background particles originating from the muon shield.

## Simplified Design
- **Evolution**: Originally implemented as a detailed RPC with multiple material layers, the UBT was simplified to a single vacuum box scoring plane to reduce geometry overlaps and simulation overhead.
- **Dimensions**:
    - **X**: 4.4 meters
    - **Y**: 6.4 meters
    - **Z (Thickness)**: 16 centimeters
- **Threshold**: Particles with momentum $P < 0.1$ GeV/c are typically excluded from tagging counts to focus on trackable background.

## Source Anchors
- `UpstreamTagger/UpstreamTagger.h:30-42`: Documentation of the simplification history and default dimensions.
- `python/shipVeto.py:67-81`: Enforcement of the 0.1 GeV/c momentum threshold.

## Links
- [[fairship/detectors/UBT]]
- [[fairship/geometry/geometry_config]]
