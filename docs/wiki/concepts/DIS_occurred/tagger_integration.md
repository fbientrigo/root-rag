# Tagger Integration: SBT and UBT

This document details how the Surrounding Background Tagger (SBT) and Upstream Background Tagger (UBT) are used to select and augment Muon DIS events.

## 1. Muon Selection (`make_nTuple_SBT.py`)

Muons are selected from the massive background files if they satisfy specific kinematic and geometric criteria.

### Selection Criteria
- **Particle ID**: Must be a muon (`abs(pid) == 13`).
- **Momentum Threshold**: `P_threshold / u.GeV < P`. (Default is often **3 GeV/c**).
- **SBT Hit**: The muon must have a hit in the `vetoPoint` container with a `detID` in the range `1000 < detID < 999999`.

### Extraction Logic
For every selected muon, the script saves:
1.  **Muon Kinematics**: `imuondata` (PID, Px, Py, Pz, X, Y, Z, Weight, Time).
2.  **Soft Tracks**: All MCTracks where `MotherId == selected_muon` (excluding nuclear interactions).
3.  **Veto Hits**: All `vetoPoint` entries associated with the muon.
4.  **UBT Hits**: All `UpstreamTaggerPoint` entries associated with the muon.

## 2. Hit Merging (`add_muonresponse.py`)

Because the DIS simulation starts at the interaction vertex, it does not naturally contain the hits produced by the muon *before* the interaction. The `add_muonresponse.py` script restores these hits.

### Merging Logic
1.  **Interaction Point**: Retrieve the DIS interaction vertex `z_int` from `MCTrack[0]`.
2.  **Filter and Merge**:
    - Iterate through the hits saved in `muonDis.root` (`muon_vetoPoints`).
    - If `hit.GetZ() < z_int`, the hit is added to the simulation's `vetoPoint` array.
    - The `TrackID` of these added hits is reset to **0** to match the incoming muon in the DIS simulation.

### Purpose
This ensures that the final simulation file contains the full history of the muon as it passed through the upstream detectors, which is critical for background rejection studies (veto efficiency).

## 3. Operational Constants

| Constant | Value | Source |
|---|---|---|
| `P_threshold` | 3 GeV/c | `make_nTuple_SBT.py` |
| `SBT detID Range` | 1000 - 999,999 | `make_nTuple_SBT.py` |
| `Merged TrackID` | 0 | `add_muonresponse.py` |

## Source Anchors
- `muonDIS/make_nTuple_SBT.py:220-250`: Muon selection criteria.
- `muonDIS/add_muonresponse.py:155-170`: Hit filtering by interaction Z.
- `muonDIS/add_muonresponse.py:164`: `hit.SetTrackID(0)` override.
