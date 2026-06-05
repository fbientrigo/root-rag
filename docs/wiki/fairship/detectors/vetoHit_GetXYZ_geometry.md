---
type: node
domain: fairship
topic: geometry
status: CONFIRMED code-local
fairship_commit: 98de16a5b264
tags: [vetoHit, GetXYZ, TGeoManager, geometry-path, LocalToMaster]
---

# vetoHit::GetXYZ() Geometry Trace

## Role
The `vetoHit::GetXYZ()` method provides the 3D spatial coordinates (global/master frame) of a digitized veto hit. This is critical for track-to-veto linking during reconstruction, as it defines the target point for track extrapolation.

## Implementation Trace
The implementation in `veto/vetoHit.cxx` follows a systematic path through the FairShip geometry.

1.  **ID Parsing**: The `fDetectorID` (inherited from `SHiP::DetectorHit`) is parsed into geometric indices: `ShapeType`, `blockNr`, `Zlayer`, and `number`.
    *   *Evidence*: `veto/vetoHit.cxx:63-66`
2.  **Path Construction**: A `TString` geometry path is constructed.
    *   **Base Path**: `cave/DecayVolume_1/T2_1/VetoLiSc_0/`
    *   **Prefixes**: Appends `LiScX_`, `LiScY_`, `LiSc_S3_`, etc., based on `ShapeType`.
    *   *Evidence*: `veto/vetoHit.cxx:60-88`
3.  **Geometry Navigation**: Uses the `TGeoNavigator` to change the current node to the constructed path.
    *   *Evidence*: `veto/vetoHit.cxx:90-91`
4.  **Local to Master Transformation**:
    *   Retrieves the `TGeoBBox` shape from the volume of the current node.
    *   Takes the local origin: `shape->GetOrigin()`.
    *   Transforms the origin to the Master (Global) coordinate system using `nav->LocalToMaster(origin, master)`.
    *   *Evidence*: `veto/vetoHit.cxx:33-41`

## Usage in Reconstruction
The reconstructed position is used in `shipDigiReco.py` to calculate the distance of closest approach between a fitted track and the veto segment.
*   *Evidence*: `python/shipDigiReco.py:424`: `vetoHitPos = vetoHit.GetXYZ()`

## Data Convention
*   **Coordinate System**: Master (Global) frame.
*   **Precision**: Determined by the `TGeoManager` resolution and the placement of volumes in the simulation geometry.

## Robustness & Technical Debt
*   **Hardcoded Path**: The base geometry path `cave/DecayVolume_1/T2_1/VetoLiSc_0/` is hardcoded. While the base components (`cave`, `DecayVolume`, `T2`, `VetoLiSc`) are traceable to `ShipGeoCave.cxx`, `shipDet_conf.py`, and `veto.cxx`, the specific instantiation indices (e.g., `_1`, `_0`) depend entirely on ROOT's runtime volume naming during geometry construction.
*   **Missing Safeguards**: The `GetNode()` function (`veto/vetoHit.cxx:90-93`) uses `nav->cd(path)` followed by `nav->GetCurrentNode()`. There is no code-local fallback or null-pointer check if `cd()` fails. A change in the geometry hierarchy (e.g., via `geometry_config.yaml`) will result in silent failures or runtime crashes during track extrapolation.

## Source Anchors
- `veto/vetoHit.h:29`: Declaration of `GetXYZ()`.
- `veto/vetoHit.cxx:32-43`: Implementation of `GetXYZ()`.
- `veto/vetoHit.cxx:57-93`: Implementation of `GetNode()` (Geometry Path construction).
- `Detector/DetectorHit.h:40`: Definition of `fDetectorID`.

## Links
- [[fairship/detectors/SBT]]
- [[fairship/reconstruction/vetoHit_track_linking]]
- [[workflows/muon_dis_digitization_reconstruction_trace]]
