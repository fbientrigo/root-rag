# MuonBack to MuDIS Map

## Purpose
Constrain MuonBack/MuDIS boundary knowledge without inventing canonical runtime chain truth.

## Reading order
1. [[fairship/scripts/run_simScript_MuonBack]]
2. [[fairship/config/DY_Yheight]]
3. [[fairship/scripts/makeMuonDIS]]
4. [[fairship/scripts/run_simScript_MuDIS]]

## Edge table
| from | relation | to | status | evidence | does_not_prove | next_validation |
|---|---|---|---|---|---|---|
| `[[fairship/config/DY_Yheight]]` | configures shared setup for | `[[fairship/scripts/run_simScript_MuonBack]]` | `CONFIRMED_BY_CODE` | `macro/run_simScript.py:360-369` | safe DY range/value | range guard + runtime checks |
| `[[fairship/config/DY_Yheight]]` | configures shared setup for | `[[fairship/scripts/run_simScript_MuDIS]]` | `CONFIRMED_BY_CODE` | `macro/run_simScript.py:360-369` | thesis impact certainty | controlled runtime comparison |
| `[[fairship/scripts/run_simScript_MuonBack]]` | configures | `[[MuonBack]]` | `PROVISIONAL` | `macro/run_simScript.py:571-640` | successful execution | LXPLUS staged run logs |
| `[[MuonBack]]` | produces inputs consumed by | `[[fairship/scripts/makeMuonDIS]]` | `PROVISIONAL` | `muonShieldOptimization/run_prod.py:10-21`; `muonDIS/make_nTuple_SBT.py:211-224` | canonical reproducible handoff | artifact lineage with hashes |
| `[[fairship/scripts/makeMuonDIS]]` | produces | `[[fairship/trees/DIS_tree]]` | `CONFIRMED_BY_CODE` | `muonDIS/makeMuonDIS.py:147-150` | runtime data validity | branch-list runtime check |
| `[[fairship/trees/DIS_tree]]` | feeds | `[[fairship/generators/MuDISGenerator]]` | `CONFIRMED_BY_CODE` | `shipgen/MuDISGenerator.cxx:34-53` | full chain correctness | runtime probe on muonDis.root |
| `[[fairship/scripts/run_simScript_MuDIS]]` | configures | `[[fairship/generators/MuDISGenerator]]` | `CONFIRMED_BY_CODE` | `macro/run_simScript.py:544-553` | execution success | LXPLUS transport logs |
| `[[fairship/runtime/MuonBack_smoke]]` | relationship to MuDIS chain | `[[fairship/runtime/MuDIS_transport_candidate]]` | `RUNTIME_UNVALIDATED` | `docs/wiki/workflows/fairship_muon_dis_lxplus_validation.md:1-48` | canonical chain truth | execute staged chain and archive logs |
