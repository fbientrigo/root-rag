# Muon DIS Claim Registry

This registry tracks the status and evidence for all technical claims related to the FairShip Muon DIS workflow.

| claim_id | claim | status | source file(s) | source line(s) | validation needed | linked wiki page |
|----------|-------|--------|----------------|----------------|-------------------|------------------|
| CLM-001 | Code-supported staged Muon DIS workflow surface exists: selection ntuple -> DIS generation -> MuDIS transport -> optional response merge. | PROVISIONAL | `muonDIS/make_nTuple_SBT.py`, `muonDIS/makeMuonDIS.py`, `macro/run_simScript.py` | N/A | Requires staged LXPLUS runtime execution before end-to-end workflow confirmation. | `fairship_muon_dis_operational_pipeline.md` |
| CLM-002 | `makeMuonDIS.py` produces `InMuon`, `DISParticles`, `SoftParticles` branches in tree `DIS`. `InMuon` uses a 14-element `TVectorD` schema. | CONFIRMED | `muonDIS/makeMuonDIS.py` | 141-231 | Schema verification | `docs/wiki/nodes/InMuon.md` |
| CLM-003 | `run_simScript.py --MuDIS` instantiates `MuDISGenerator` and adds it to `FairPrimaryGenerator`. | CONFIRMED | `macro/run_simScript.py` | 541-555 | Runtime check | `fairship_muon_dis_operational_pipeline.md` |
| CLM-004 | `MuDISGenerator` reads `InMuon`, `DISParticles`, `SoftParticles` and injects tracks into simulation. | CONFIRMED | `shipgen/MuDISGenerator.cxx` | 71-222 | Track injection audit | `fairship_muon_dis_operational_pipeline.md` |
| CLM-005 | `MuDISGenerator` picks interaction vertex along muon trajectory based on material budget. | CONFIRMED | `shipgen/MuDISGenerator.cxx` | 117-150 | Geometry navigation check | `fairship_muon_dis_operational_pipeline.md` |
| CLM-006 | DIS cross-section is persisted in `cbmsim` tree from `InMuon[0][10]` (value in mb). | CONFIRMED | `macro/run_simScript.py` | 841-866 | Tree schema audit | `docs/wiki/nodes/InMuon.md` |
| CLM-007 | Events classified as 'front', 'side', or 'cavern' in simulation workflow. | CONFIRMED external-doc | [[EXTERNAL_EVIDENCE_REGISTRY#EXT-003]] | N/A | Verified terminology exists in external Indico talk; no local code implementation found in indexed FairShip. | `fairship_muon_dis_operational_pipeline.md` |
| CLM-008 | SBT/UBT response merge involves z-filtering. | PROVISIONAL | `muonDIS/add_muonresponse.py` | 141-204 | Logic verification | `fairship_muon_dis_operational_pipeline.md` |
| CLM-009 | Veto system architecture uses 36-step radial scan for fiducial check. | CONFIRMED | `reports/veto_system_deep_dive.md` | 66 | Geometry check | `veto_system_architecture.md` |
| CLM-010 | SBT 45 MeV threshold represents MIP energy loss in 30cm liquid scintillator. | PROVISIONAL | [[EXTERNAL_EVIDENCE_REGISTRY#EXT-002]] | N/A | Physics validation | `veto_system_architecture.md` |
| CLM-011 | UBT 0.1 GeV/c threshold is a standard simulation tracking cut-off. | PROVISIONAL | [[EXTERNAL_EVIDENCE_REGISTRY#EXT-001]] | N/A | Physics validation | `veto_system_architecture.md` |

## Registry Status Definitions
- **CONFIRMED**: Directly verified in source code or definitive research artifacts.
- **PROVISIONAL**: Based on strong indirect evidence or unverified script fragments.
- **UNRESOLVED**: Claim made in documentation but source evidence not found in index.
- **CONTRADICTED**: Source evidence directly refutes the claim.
