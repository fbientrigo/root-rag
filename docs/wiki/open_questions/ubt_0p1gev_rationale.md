# Open Question: UBT 0.1 GeV/c Rationale

**Status**: OPEN
**Category**: Physics Validation
**Context**: Code implements a 0.1 GeV/c momentum threshold for UBT counting (`shipVeto.py`). External documentation ([[EXTERNAL_EVIDENCE_REGISTRY#EXT-001]]) suggests this is a simulation baseline.

## Description
Why is 0.1 GeV/c the choice for the UBT threshold?

## Questions
1. Is this threshold related to the RPC (Resistive Plate Chamber) efficiency at low momentum?
2. Does this threshold significantly impact the veto rate for soft muons?
3. Is this threshold applied at the particle generation level or only in the analysis/veto logic?

## Search Targets
- `UpstreamTagger` scoring logic.
- SHiP Technical Proposal (SPSC-2015-016).
