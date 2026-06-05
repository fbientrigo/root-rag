# FairShip Muon DIS Operational Pipeline

## Status Semantics
- **CONFIRMED**: Directly verified code-local fact.
- **PROVISIONAL**: Plausible staged workflow but not runtime-validated.
- **UNRESOLVED**: Not found or missing evidence.

## Boundary
<!-- CLAIM: CONFIRMED -->
Index `fairship__master__98de16a5b264__20260428T060008679936+0000Z`, commit `98de16a5b264`.
<!-- SOURCE: evidence/fairship_muon_dis_operational_20260504/manifest.json:1-220 -->

<!-- CLAIM: UNRESOLVED -->
Geometry for MuDIS wall/fiducial oracle incomplete.
NOT FOUND IN INDEX
Next: bind volume definitions to MuDIS selection.

<!-- CLAIM: UNRESOLVED -->
LXPLUS execution not validated end-to-end.
NOT FOUND IN INDEX
Next: staged validation on LXPLUS.

## Pipeline Claims

### 1. Muon Background Selection
<!-- CLAIM: PROVISIONAL -->
Code-supported staged Muon DIS workflow surface exists: selection ntuple -> DIS generation -> MuDIS transport -> optional response merge.
<!-- SOURCE: muonDIS/make_nTuple_SBT.py:1-80 -->
<!-- SOURCE: muonDIS/makeMuonDIS.py:141-220 -->
<!-- SOURCE: macro/run_simScript.py:491-570 -->
<!-- SOURCE: muonDIS/add_muonresponse.py:71-150 -->

<!-- CLAIM: CONFIRMED -->
Muons hitting the Surrounding Background Tagger (SBT) with momentum > 3 GeV are collected into a dedicated ntuple.
<!-- SOURCE: muonDIS/make_nTuple_SBT.py:191-248 -->
- **Input**: `ship.conical.MuonBack-TGeant4.root`
- **Output**: `muonsProduction_wsoft_SBT.root` (Tree: `MuonAndSoftInteractions`)

### 2. DIS Event Production (`makeMuonDIS.py`)
<!-- CLAIM: CONFIRMED -->
`makeMuonDIS.py` branches: `InMuon`, `DISParticles`, `SoftParticles`, `muon_vetoPoints`, `muon_UpstreamTaggerPoints` in tree `DIS`.
<!-- SOURCE: muonDIS/makeMuonDIS.py:141-220 -->

<!-- CLAIM: CONFIRMED -->
DIS interactions are generated for the selected muons using Pythia6 (FIXT mode, `gamma/mu+` or `gamma/mu-` on `p+` or `n0` targets).
<!-- SOURCE: muonDIS/makeMuonDIS.py:167-245 -->

### 3. Simulation Transport (`run_simScript.py --MuDIS`)
<!-- CLAIM: CONFIRMED -->
`run_simScript.py --MuDIS` instantiates `MuDISGenerator`, sets z-window, and adds it to the `FairPrimaryGenerator`.
<!-- SOURCE: macro/run_simScript.py:491-570 -->
- **Default Z-Window**: From `Chamber1` (front of UVT) up to `TrackStation1`.
<!-- SOURCE: macro/run_simScript.py:541-555 -->

<!-- CLAIM: CONFIRMED -->
`MuDISGenerator` reads `InMuon`, `DISParticles`, and `SoftParticles`. It determines vertex placement using the material budget (via `shipgen::MeanMaterialBudget`) and injects particles into `FairPrimaryGenerator`.
<!-- SOURCE: shipgen/MuDISGenerator.cxx:71-150 -->
<!-- SOURCE: shipgen/MuDISGenerator.cxx:151-227 -->

### 4. Physics Metadata & Response Merge
<!-- CLAIM: CONFIRMED -->
`run_simScript.py` extracts the DIS cross-section from `InMuon[0][10]` and persists it as a float branch `CrossSection` in the output `cbmsim` tree.
<!-- SOURCE: macro/run_simScript.py:841-902 -->

<!-- CLAIM: PROVISIONAL -->
SBT/UBT response in `make_nTuple_SBT.py`, merge in `add_muonresponse.py` (z-filter).
<!-- SOURCE: muonDIS/make_nTuple_SBT.py:281-360 -->
<!-- SOURCE: muonDIS/add_muonresponse.py:141-204 -->

## Known Limits & Unresolved Items

<!-- CLAIM: PROVISIONAL -->
**Event Classification (Front/Side/Cavern)**: Standard SHiP terminology for background regions. 
- **Front**: Upstream (UBT region).
- **Side**: SBT/Vacuum vessel wall region.
- **Cavern**: Hall floor/ceiling/walls.
<!-- SOURCE: [[EXTERNAL_EVIDENCE_REGISTRY#EXT-003]] -->
Next: investigate if these are implemented in initial MuonBack production macros.

<!-- CLAIM: UNRESOLVED -->
No single-command LXPLUS recipe.
NOT FOUND IN INDEX
Next: record minimal staged recipe.

<!-- CLAIM: UNRESOLVED -->
`InactivateMuonProcesses` not found.
NOT FOUND IN INDEX
Next: search Geant4 process controls.

<!-- CLAIM: UNRESOLVED -->
`DIS occurred` truth label not found.
NOT FOUND IN INDEX
Next: trace analysis consumers.
