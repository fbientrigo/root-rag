# Open Questions: Muon DIS Workflow

## 1. Event Classification (Front/Side/Cavern)
- **Question**: Where are the "front", "side", and "cavern" regions precisely defined for Muon DIS events?
- **Context**: These terms appear in documentation and task descriptions but are not found as explicit labels or variables in the core FairShip simulation code (`run_simScript.py`, `MuDISGenerator.cxx`).
- **Hypothesis**: They might be defined in the initial muon background generation step (pre-ntuple) or as z-region cuts in external analysis scripts not currently indexed.

## 2. Process Control Consolidation
- **Question**: Is there a single canonical script or routine to "Inactivate Muon Processes" during background studies?
- **Context**: `run_prod.py` mentions switching off `muIoni`, `muBrems`, and `muPair` for specific production runs, but the implementation is fragmented across G4 macros and environment variables.
- **Hypothesis**: A consolidation might exist in a private or project-specific macro file like `g4config.in` (which is not currently indexed).

## 3. "DIS Occurred" Truth Label
- **Question**: Is there a dedicated boolean branch in the final simulation output to indicate "DIS occurred"?
- **Context**: Currently, the presence of DIS is inferred from the particle stack or by checking the cross-section weight in `MCTrack`.
- **Hypothesis**: Downstream analysis tasks might use a derived branch or a specific flag in the `ShipEventHeader`.

## 4. IP10 vs IP250
- **Question**: Why do some scripts use an IP cut of 10 cm while others use 250 cm?
- **Context**: `ShipAna.py` plots IP up to 10 cm, while `analysis_toolkit.py` defaults to 250 cm.
- **Hypothesis**: 10 cm might be a tight signal selection cut, while 250 cm is a loose pre-selection cut for background rejection.
