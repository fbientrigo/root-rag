# EDGE_REGISTRY

| from | relation | to | status | evidence | next validation |
|---|---|---|---|---|---|
| `[[fairship/scripts/makeMuonDIS]]` | produces | `[[fairship/trees/DIS_tree]]` | `CONFIRMED_BY_CODE` | `muonDIS/makeMuonDIS.py:147-150` | runtime branch-list check |
| `[[fairship/trees/DIS_tree]]` | feeds | `[[fairship/generators/MuDISGenerator]]` | `CONFIRMED_BY_CODE` | `shipgen/MuDISGenerator.cxx:34-53` | runtime probe on produced file |
| `[[fairship/scripts/run_simScript_MuDIS]]` | configures | `[[fairship/generators/MuDISGenerator]]` | `CONFIRMED_BY_CODE` | `macro/run_simScript.py:544-553` | LXPLUS transport logs |
| `[[fairship/scripts/run_simScript_MuonBack]]` | configures | `[[MuonBack]]` | `PROVISIONAL` | `macro/run_simScript.py:571-640` | staged runtime run logs |
| `[[MuonBack]]` | produces inputs consumed by | `[[fairship/scripts/makeMuonDIS]]` | `PROVISIONAL` | `muonShieldOptimization/run_prod.py:10-21`; `muonDIS/make_nTuple_SBT.py:211-224` | artifact lineage hashes |
| `[[fairship/oracle/Oracle_schema]]` | depends on | `[[fairship/runtime/oracle_probe_runtime]]` | `RUNTIME_UNVALIDATED` | `docs/wiki/workflows/fairship_muon_dis_lxplus_validation.md:1-48` | staged packet execution |
| `[[fairship/geometry/analysis_fiducial_fail]]` | is not equivalent to | `[[fairship/geometry/geant4_volume_wall_hit]]` | `UNRESOLVED` | `macro/ShipAna.py:221-250`; `python/experimental/analysis_toolkit.py:169-184` | explicit geant4 boundary evidence |
