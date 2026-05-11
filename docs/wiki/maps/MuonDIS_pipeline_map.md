# MuonDIS Pipeline Map

## Purpose
Code-backed MuDIS preparation and transport relationships with runtime gaps isolated.

## Reading order
1. [[fairship/scripts/makeMuonDIS]]
2. [[fairship/trees/DIS_tree]]
3. [[fairship/generators/MuDISGenerator]]
4. [[fairship/scripts/run_simScript_MuDIS]]
5. [[fairship/oracle/Oracle_schema]]

## Edge table
| from | relation | to | status | evidence | does_not_prove | next_validation |
|---|---|---|---|---|---|---|
| `[[fairship/scripts/makeMuonDIS]]` | produces | `[[fairship/trees/DIS_tree]]` | `CONFIRMED_BY_CODE` | `muonDIS/makeMuonDIS.py:147-150` | runtime file integrity | branch-list check on runtime ROOT |
| `[[fairship/trees/DIS_tree]]` | contains | `[[fairship/branches/InMuon]]` | `CONFIRMED_BY_CODE` | `muonDIS/makeMuonDIS.py:152-154` | schema stability | runtime schema snapshot |
| `[[fairship/trees/DIS_tree]]` | contains | `[[fairship/branches/DISParticles]]` | `CONFIRMED_BY_CODE` | `muonDIS/makeMuonDIS.py:155-157` | truth-label semantics | downstream predicate trace |
| `[[fairship/trees/DIS_tree]]` | contains | `[[fairship/branches/SoftParticles]]` | `CONFIRMED_BY_CODE` | `muonDIS/makeMuonDIS.py:158-160` | threshold semantics | runtime oracle probe |
| `[[fairship/scripts/run_simScript_MuDIS]]` | configures | `[[fairship/generators/FairPrimaryGenerator]]` | `CONFIRMED_BY_CODE` | `macro/run_simScript.py:407-409` | runtime ordering correctness | inspect runtime logs |
| `[[fairship/scripts/run_simScript_MuDIS]]` | configures | `[[fairship/generators/MuDISGenerator]]` | `CONFIRMED_BY_CODE` | `macro/run_simScript.py:544-553` | successful execution | LXPLUS staged run |
| `[[fairship/generators/MuDISGenerator]]` | consumes | `[[fairship/trees/DIS_tree]]` | `CONFIRMED_BY_CODE` | `shipgen/MuDISGenerator.cxx:34-53` | runtime branch availability | runtime probe on produced file |
| `[[fairship/scripts/run_simScript_MuDIS]]` | contributes outputs to | `[[fairship/oracle/Oracle_schema]]` | `PROVISIONAL` | `macro/run_simScript.py:841-861` | validated oracle correctness | runtime output inspection |
| `[[fairship/oracle/Oracle_schema]]` | depends on | `[[fairship/runtime/oracle_probe_runtime]]` | `RUNTIME_UNVALIDATED` | `docs/wiki/workflows/fairship_muon_dis_lxplus_validation.md:1-48` | successful physics chain | archive runtime transcript |
