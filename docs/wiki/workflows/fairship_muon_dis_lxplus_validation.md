# FairShip Muon DIS LXPLUS Oracle Validation

## Boundary
<!-- CLAIM: PROVISIONAL -->
Index `fairship__master__98de16a5b264__20260428T060008679936+0000Z`, commit `98de16a5b264`.
<!-- SOURCE: reports/fairship_muon_dis_operational_research_audit.md:11-12 -->

<!-- CLAIM: PROVISIONAL -->
Operational packet: `reports/fairship_muon_dis_lxplus_execution_packet.md`.
<!-- SOURCE: reports/fairship_muon_dis_lxplus_execution_packet.md:1-153 -->

## Command Surfaces
<!-- CLAIM: PROVISIONAL -->
MuonBack: `python "$FAIRSHIP/macro/run_simScript.py" --MuonBack -f "$SHIPSOFT/data/pythia8_Geant4_onlyMuons.root" -Y <DY_IN_METERS>`.
`DY` default `6.0 m` not thesis-safe.
<!-- SOURCE: muonShieldOptimization/run_prod.py:1-80 -->

<!-- CLAIM: PROVISIONAL -->
MuDIS transport: `run_simScript.py --MuDIS -f <muonDis.root>`. No canonical chain.
<!-- SOURCE: macro/run_simScript.py:491-570 -->

<!-- CLAIM: PROVISIONAL -->
Oracle extraction: `python scripts/lxplus_muondis_oracle_probe.py --input <ROOT_FILE> --output oracle_probe.json`.
<!-- SOURCE: reports/fairship_muon_dis_lxplus_execution_packet.md:47-53 -->

## Oracle Scope
<!-- CLAIM: PROVISIONAL -->
Counters: `dis_tree_exists`, `n_DISParticles`, `n_SoftParticles`, `n_SBT_hits`, `n_UBT_hits`. Unresolved: `fiducial_fail`, `wall_like_fail`.
<!-- SOURCE: reports/fairship_muon_dis_oracle_observable_schema.md:1-61 -->

<!-- CLAIM: UNRESOLVED -->
`wall_like_fail` not Geant4 wall-hit truth.
NOT FOUND IN INDEX
Next: retain split channels until API evidence retrieved.

## Known Limits
<!-- CLAIM: UNRESOLVED -->
No MuonBack->MuDIS LXPLUS recipe.
NOT FOUND IN INDEX
Next: run staged commands on LXPLUS.

<!-- CLAIM: UNRESOLVED -->
No native post-shield muon injection.
NOT FOUND IN INDEX
Next: probe generator interfaces.
