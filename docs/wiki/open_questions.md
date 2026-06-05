# Open Questions

## LXPLUS execution
- status: `UNRESOLVED`
- links: [[fairship/runtime/LXPLUS_preflight]], [[fairship/runtime/MuonBack_smoke]], [[fairship/runtime/oracle_probe_runtime]]

## DY/Yheight
- status: `CLOSED`
- Resolution: Confirmed parsing in `run_simScript.py` and unit conversion to meters in `geometry_config.py`.
- links: [[fairship/config/DY_Yheight]], [[fairship/scripts/run_simScript_MuonBack]], [[fairship/scripts/run_simScript_MuDIS]]

## custom muon state injection
- status: `UNRESOLVED`
- links: [[fairship/generators/MuDISGenerator]], [[fairship/branches/InMuon]]

## MuonBack to MuDIS handoff
- status: `RUNTIME_UNVALIDATED`
- links: [[maps/MuonBack_to_MuDIS_map]], [[fairship/runtime/MuDIS_transport_candidate]]

## oracle truth labels
- status: `CLOSED`
- Resolution: `CrossSection` branch in `cbmsim` is the definitive proxy. See [[reports/fairship_muon_dis_resolution_20260511]].
- links: [[fairship/oracle/Oracle_schema]]

## wall/fiducial/Geant4 volume ambiguity
- status: `UNRESOLVED`
- links: [[fairship/geometry/analysis_fiducial_fail]], [[fairship/geometry/wall_like_fail]], [[fairship/geometry/geant4_volume_wall_hit]]

## ROOT schema validation
- status: `CLOSED`
- Resolution: Confirmed schemas for `DISParticles` and `SoftParticles` in [[reports/fairship_muon_dis_resolution_20260511]].
- links: [[fairship/trees/DIS_tree]], [[fairship/branches/DISParticles]], [[fairship/branches/SoftParticles]]

## IP cuts (10 vs 250)
- status: `CLOSED`
- Resolution: 10cm is signal cut, 250cm is pre-selection. See [[reports/fairship_muon_dis_resolution_20260511]].
- links: [[fairship/scripts/run_simScript_MuDIS]]

## qrels and benchmark status
- status: `UNRESOLVED`
- links: `scripts/emv_status.py`, `reports/*vertical_slice_summary.json`

## thesis risk integration
- status: `SCAFFOLD`
- links: [[thesis_risks/R1_NF_ESS_collapse]], [[thesis_risks/R2_proxy_tail_calibration]], [[thesis_risks/R3_support_OOD]], [[thesis_risks/R4_cost_benchmark_contamination]]
