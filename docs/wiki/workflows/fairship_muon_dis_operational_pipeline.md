# FairShip Muon DIS Operational Pipeline

## Boundary Conditions

<!-- CLAIM: PROVISIONAL -->
This page is scoped to FairShip evidence indexed at commit `98de16a5b264` through root-rag index `fairship__master__98de16a5b264__20260428T060008679936+0000Z`.
<!-- SOURCE: evidence/fairship_muon_dis_operational_20260504/manifest.json:1-220 -->

<!-- CLAIM: UNRESOLVED -->
Geometry assumptions needed for a thesis-grade MuDIS wall/fiducial oracle are incomplete in this snapshot.
NOT FOUND IN INDEX
Next action: bind detector-volume/geometry definitions to MuDIS-specific selection semantics with additional retrieval.

<!-- CLAIM: UNRESOLVED -->
LXPLUS execution has not been validated end-to-end for this exact workflow chain in this evidence pass.
NOT FOUND IN INDEX
Next action: execute staged validation on LXPLUS and archive command/output transcript as evidence.

<!-- CLAIM: UNRESOLVED -->
Direct custom post-shield muon-state injection as a native one-step FairShip path remains unresolved.
NOT FOUND IN INDEX
Next action: extend retrieval for generator interfaces and run-script entry points for direct custom state loading.

## Pipeline Claims

<!-- CLAIM: PROVISIONAL -->
Operationally, retrieved code supports a staged flow: muon-background selection/ntuple (`MuonAndSoftInteractions`) -> DIS generation (`muonDis.root` / `DIS`) -> transport with `run_simScript.py --MuDIS` -> optional muon-response merge.
<!-- SOURCE: muonDIS/make_nTuple_SBT.py:1-80 -->
<!-- SOURCE: muonDIS/makeMuonDIS.py:141-220 -->
<!-- SOURCE: macro/run_simScript.py:491-570 -->
<!-- SOURCE: muonDIS/add_muonresponse.py:71-150 -->

<!-- CLAIM: PROVISIONAL -->
`makeMuonDIS.py` defines output branches `InMuon`, `DISParticles`, `SoftParticles`, `muon_vetoPoints`, and `muon_UpstreamTaggerPoints` in tree `DIS`.
<!-- SOURCE: muonDIS/makeMuonDIS.py:141-220 -->

<!-- CLAIM: PROVISIONAL -->
`run_simScript.py --MuDIS` instantiates `ROOT.MuDISGenerator()`, sets MuDIS z-window positions, initializes from input file/first event, and adds it to the primary generator chain.
<!-- SOURCE: macro/run_simScript.py:491-570 -->
<!-- SOURCE: shipgen/MuDISGenerator.h:1-51 -->

<!-- CLAIM: PROVISIONAL -->
`MuDISGenerator` initializes branch readers for `InMuon`, `DISParticles`, and `SoftParticles`, then injects incoming muon, DIS particles, and filtered soft particles into `FairPrimaryGenerator`.
<!-- SOURCE: shipgen/MuDISGenerator.cxx:1-80 -->
<!-- SOURCE: shipgen/MuDISGenerator.cxx:141-220 -->
<!-- SOURCE: shipgen/MuDISGenerator.cxx:211-227 -->

<!-- CLAIM: PROVISIONAL -->
In mudis mode, `run_simScript.py` adds `CrossSection` to output `cbmsim` using `muondis_event.InMuon[0][10]`.
<!-- SOURCE: macro/run_simScript.py:841-902 -->
<!-- SOURCE: muonDIS/makeMuonDIS.py:71-150 -->

<!-- CLAIM: PROVISIONAL -->
SBT/UBT response persistence appears in `make_nTuple_SBT.py`, and optional upstream-hit merge appears in `add_muonresponse.py` with z-filter `hit.GetZ() < interaction_point.Z()`.
<!-- SOURCE: muonDIS/make_nTuple_SBT.py:281-360 -->
<!-- SOURCE: muonDIS/add_muonresponse.py:141-204 -->

## Known Limits

<!-- CLAIM: UNRESOLVED -->
A canonical single-command LXPLUS recipe spanning all MuDIS stages is not established in retrieved evidence.
NOT FOUND IN INDEX
Next action: execute and record a minimal staged recipe on LXPLUS, then convert only verified steps to CONFIRMED.

<!-- CLAIM: UNRESOLVED -->
A named routine/token `InactivateMuonProcesses` was not retrieved in this index snapshot.
NOT FOUND IN INDEX
Next action: continue synonym-based retrieval around Geant4 process controls and validate runtime toggles.

<!-- CLAIM: UNRESOLVED -->
A canonical downstream boolean `DIS occurred` truth label branch/API was not retrieved.
NOT FOUND IN INDEX
Next action: trace downstream analysis/reco consumers for explicit truth labeling conventions.
