# Oracle Observables Map

## Purpose
Map code-visible observables to oracle-facing fields while keeping truth-label gaps explicit.

## Reading order
1. [[fairship/trees/DIS_tree]]
2. [[fairship/branches/DISParticles]]
3. [[fairship/branches/SoftParticles]]
4. [[fairship/detectors/SBT]] / [[fairship/detectors/UBT]]
5. [[fairship/oracle/Oracle_schema]]

## Edge table
| from | relation | to | status | evidence | does_not_prove | next_validation |
|---|---|---|---|---|---|---|
| `[[fairship/trees/DIS_tree]]` | provides candidate observables to | `[[fairship/oracle/Oracle_schema]]` | `CONFIRMED_BY_CODE` | `muonDIS/makeMuonDIS.py:152-165` | runtime branch availability | branch-list verification |
| `[[fairship/branches/DISParticles]]` | candidate count maps into | `[[fairship/oracle/Oracle_schema]]` | `PROVISIONAL` | `reports/fairship_muon_dis_oracle_observable_schema.md:5-13` | final truth label | trace downstream predicate code |
| `[[fairship/branches/SoftParticles]]` | candidate count maps into | `[[fairship/oracle/Oracle_schema]]` | `PROVISIONAL` | `reports/fairship_muon_dis_oracle_observable_schema.md:5-13` | final truth label | inspect runtime distributions |
| `[[fairship/detectors/SBT]]` | candidate count maps into | `[[fairship/oracle/Oracle_schema]]` | `PROVISIONAL` | `reports/fairship_muon_dis_oracle_observable_schema.md:5-13` | SBT threshold truth rule | validate threshold policy |
| `[[fairship/detectors/UBT]]` | candidate count maps into | `[[fairship/oracle/Oracle_schema]]` | `PROVISIONAL` | `reports/fairship_muon_dis_oracle_observable_schema.md:5-13` | UBT threshold truth rule | validate threshold policy |
| `[[fairship/detectors/vetoPoint]]` | contributes detector-hit channel to | `[[fairship/oracle/Oracle_schema]]` | `PROVISIONAL` | `veto/vetoPoint.h:14-20`; `reports/fairship_muon_dis_oracle_observable_schema.md:5-13` | geant4 wall-hit truth | explicit boundary-hit evidence |
| `[[fairship/detectors/UpstreamTaggerPoint]]` | contributes detector-hit channel to | `[[fairship/oracle/Oracle_schema]]` | `PROVISIONAL` | `UpstreamTagger/UpstreamTaggerPoint.h:16-22`; `reports/fairship_muon_dis_oracle_observable_schema.md:5-13` | final truth decision | runtime label correlation study |
| `[[fairship/scripts/run_simScript_MuDIS]]` | contributes output channels for | `[[fairship/oracle/Oracle_schema]]` | `CONFIRMED_BY_CODE` | `macro/run_simScript.py:841-861` | runtime-correct branch content | probe produced output |
| `[[fairship/geometry/analysis_fiducial_fail]]` | is not equivalent to | `[[fairship/geometry/geant4_volume_wall_hit]]` | `UNRESOLVED` | `macro/ShipAna.py:221-250`; `python/experimental/analysis_toolkit.py:169-184` | canonical wall truth | search geant4 boundary observables |
| `[[fairship/oracle/Oracle_schema]]` | depends on | `[[fairship/runtime/oracle_probe_runtime]]` | `RUNTIME_UNVALIDATED` | `docs/wiki/workflows/fairship_muon_dis_lxplus_validation.md:1-48` | validated thesis oracle | execute staged packet and archive logs |
