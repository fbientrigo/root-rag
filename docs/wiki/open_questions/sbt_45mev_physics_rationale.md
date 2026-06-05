# Open Question: SBT 45 MeV Physics Rationale

**Status**: OPEN
**Category**: Physics Validation
**Context**: Code implements a 45 MeV threshold (`SBTDetector.py`). External documentation ([[EXTERNAL_EVIDENCE_REGISTRY#EXT-002]]) suggests this is a MIP energy loss target.

## Description
Does the 45 MeV threshold account for digitization efficiency, light yield, and quenching (Birks' Law)?

## Questions
1. Is the 45 MeV a "true" energy deposit or a "reconstructed" energy after digitization?
2. Does the threshold change for the Plastic Scintillator technology option?
3. What is the expected "Random Veto" rate at this threshold due to environmental background?

## Search Targets
- `SBTDetector.py` digitization methods.
- SHiP SBT design notes on Indico.
