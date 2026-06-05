---
type: workflow-trace
domain: fairship
topic: muon_dis
status: CONFIRMED code-local
fairship_commit: 98de16a5b264
tags: [veto, SBT, reconstruction, digitization, MuonDIS]
---

# Digitization & Reconstruction Trace: MuonDIS/SBT Veto Flow

## 1. Scope and Non-Goals
This trace maps the data flow from low-level energy deposition in the Surrounding Background Tagger (SBT) through digitization and reconstruction to the high-level veto decision logic.

- **Goals**: Map class interactions, branch names, and threshold enforcement points.
- **Non-Goals**: LXPLUS runtime validation, physics validation of veto efficiency, or modification of retrieval logic.

## 2. Code-Local Trace

| Step | Layer | File | Logic / Method | Line Range |
| :--- | :--- | :--- | :--- | :--- |
| **Energy Loss** | Simulation | `veto/veto.cxx` | `ProcessHits` records energy deposition in `vetoPoint`. | (Derived) |
| **Digitization** | Digi | `python/detectors/SBTDetector.py` | `digitize`: Aggregates `Eloss` per segment. | 40-58 |
| **Thresholding** | Digi | `python/detectors/SBTDetector.py` | `if Eloss < 0.045: aHit.setInvalid()` (45 MeV cut). | 52-53 |
| **Hit Creation** | Digi | `python/detectors/SBTDetector.py` | Appends `vetoHit` objects to `det` vector. | 54 |
| **Extrapolation**| Reco | `python/shipDigiReco.py` | `findVetoHitOnTrack`: Extrapolates tracks to `vetoHitPos` (via `GetXYZ()`). | 420-433 |
| **Geometry** | Reco | `veto/vetoHit.cxx` | `GetXYZ()`: Resolves spatial coordinates via `TGeoManager` paths. | 33-41 |
| **Linking** | Reco | `python/shipDigiReco.py` | `linkVetoOnTracks`: Saves min distance to `VetoHitOnTrack` branch. | 439-443 |
| **Veto Logic** | Analysis | `python/shipVeto.py` | `SBT_decision`: Uses `aDigi.isValid()` to count `hitSegments`. | 60-61 |
| **Probabilistic Veto** | Analysis | `python/shipVeto.py` | Calculates veto probability `w = (1 - eff) ** hits`. | 62-63 |
| **Consumption** | Analysis | `macro/ShipAna.py` | Calls `SBT_decision()` and fills `nrSBT` histograms. | 697-701 |

## 3. Data Objects & Branches

- **Input**: `vetoPoint` (MC points in `cbmsim` tree).
- **Intermediate (Digi)**: `Digi_vetoHits` branch containing `vetoHit` objects.
- **Intermediate (Reco)**: `VetoHitOnTrack` branch containing `vetoHitOnTrack` objects (Hit ID + Distance).
- **Veto Hit Class**: `veto/vetoHit.h` (inherits from `SHiP::DetectorHit`).
- **Linking Class**: `veto/vetoHitOnTrack.h` (stores min distance to track).

## 4. Layer Attribution

1. **Digitization**: Responsible for hardware-mimicry (energy thresholds, timing jitter).
2. **Reconstruction**: Responsible for geometric association (track-to-hit proximity).
3. **Veto/Analysis**: Responsible for the logical "Veto" decision (hit counting, efficiency scaling).

## 5. Known Unresolved Gaps
- **Acceptance**: The relationship between the `vetoHit` status and the `Acceptance` term in the normalization formula remains `UNRESOLVED` and is not explicitly calculated in the indexed code.
- **Hard Rejection**: While `ShipAna.py` records the veto result, the specific analysis macro that applies a hard event rejection (e.g., `if veto: continue`) for the final thesis MuonDIS sample was not uniquely identified in the current code profile.

## 6. Thesis-Safe Wording
The SBT veto flow is grounded in code-local evidence for FairShip commit 98de16a5b264. The 45 MeV energy threshold is enforced at the digitization layer (`SBTDetector.py`), ensuring that only hits above this value contribute to the subsequent reconstruction linking and veto decisions. Track-to-veto association is performed via Genfit extrapolation in the reconstruction layer (`shipDigiReco.py`), utilizing dynamically resolved segment coordinates from the FairShip geometry ([[fairship/detectors/vetoHit_GetXYZ_geometry]]).

Machine-verifiable claims are documented in:
- [[reports/digitization_reconstruction_trace_claim_audit.claims.json]]
