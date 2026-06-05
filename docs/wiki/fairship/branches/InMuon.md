---
type: branch-note
subsystem: MuonDIS
claim_state: CONFIRMED code-local
evidence_type: code-local
runtime_validated: false
physics_validated: false
fairship_commit: 98de16a5b264
aliases:
  - DIS tree
  - TVectorD
---

# Branch: InMuon

## Role
The `InMuon` branch serves as the primary input container for the `MuDISGenerator`. It encapsulates the state of the incoming muon that undergoes a Deep Inelastic Scattering (DIS) interaction, along with metadata required for normalization and vertex placement.

## What it is
`InMuon` is the primary branch in the `DIS` tree (within `muonDis.root`) that stores the global kinematics and normalization metadata for a generated interaction.

## Where it appears
- **Producer**: `muonDIS/makeMuonDIS.py`
- **Storage**: `muonDis.root`
- **Consumer**: `MuDISGenerator.cxx` and analysis macros.

## Producer / storage / consumer
Data is stored as a `TClonesArray` of `TVectorD`. Each vector contains 14 fields (vertex, momentum, cross-section, weight factors, etc.).

## Correction Note
> [!IMPORTANT]
> Previous Wiki versions incorrectly described this as TVectorD(13). Current code-local evidence (`muonDIS/makeMuonDIS.py:231`) confirms `TVectorD(14)`.

## Minimal code evidence
```python
# muonDIS/makeMuonDIS.py:212-231
212 |         mu = array(
213 |             "d",
214 |             [
215 |                 pid,        # 0
216 |                 px,         # 1
217 |                 py,         # 2
218 |                 pz,         # 3
219 |                 E,          # 4
220 |                 x,          # 5
221 |                 y,          # 6
222 |                 z,          # 7
223 |                 w,          # 8
224 |                 isProton,   # 9
225 |                 xsec,       # 10
226 |                 time_muon,  # 11
227 |                 args.nDIS,  # 12
228 |                 nmuons,     # 13
229 |             ],
230 |         )
231 |         muPart = r.TVectorD(14, mu)
```

## Index Mapping (Hardened)
- `[0]`: **PID** (Particle ID)
- `[1-3]`: **Momentum** (PX, PY, PZ) in GeV/c.
- `[4]`: **Energy** (E) in GeV.
- `[5-7]`: **Position** (X, Y, Z) in meters (Note: `MuDISGenerator` scales these by 100 to convert to cm).
- `[8]`: **Weight** (W) of the original muon event.
- `[9]`: **isProton** flag (1.0 for proton target, 0.0 for neutron target).
- `[10]`: **Cross-section** (mb) extracted from Pythia PARI(1).
- `[11]`: **Time** (ns).
- `[12]`: **nDIS** (Multiplicity/Normalization factor, see [[fairship/normalization/nDIS]]). `MuDISGenerator` uses `1/nDIS`.
- `[13]`: **nmuons** (Number of muons in the original MuBack event).

## Unresolved Semantics
- **Coordinate System**: While values are in meters, the global origin and alignment relative to the SHiP detector reference frame (e.g., target center vs. shield exit) require geometry-file verification.
- **Post-Shield Guarantee**: While the pipeline implies `InMuon` originates from muon shield leakage (MuonBack), the exact cut-off or "injection plane" depends on the upstream simulation's configuration.
- **Units Consistency**: `makeMuonDIS.py` writes meters; `MuDISGenerator.cxx` expects meters and converts to cm. Verification of input `imuondata` units is still needed (PROVISIONAL: meters).

## Interpretation
The use of `TVectorD` instead of a custom C++ class allows for easy storage of heterogeneous metadata without complex serialization logic, though it requires documented index mapping.

## Agent guidance
Avoid raw FTS5 bracket syntax like `InMuon[0][13]` in queries; use high-signal aliases such as `InMuon`, `nDIS`, or `nmuons`. Reference the `MuonDIS Variable Pointer Map` for the full mapping.

## Thesis use
Used to verify that the generation phase (Pythia6) and simulation phase (FairShip) are correctly synchronized.

## Source Anchors
- `muonDIS/makeMuonDIS.py:149-160`: Branch declaration (`TClonesArray("TVectorD")`).
- `muonDIS/makeMuonDIS.py:212-231`: `InMuon` array initialization and indexing.
- `shipgen/MuDISGenerator.cxx:38-53`: Tree and branch binding.
- `shipgen/MuDISGenerator.cxx:85-95`: Runtime extraction of indices 0, 5-8, 10-12.
- `macro/run_simScript.py:858-861`: Promotion of `InMuon[0][10]` to `CrossSection`.

## Next Query Pack
- `TGeoManager` and `ship_geo` references to find the Z-alignment of the MuDIS injection window.
- `MeanMaterialBudget` usage in `MuDISGenerator.cxx:117-150`.
