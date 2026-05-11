# FairShip Muon DIS LXPLUS Oracle Validation

## Boundary

<!-- CLAIM: PROVISIONAL -->
This workflow is bounded to root-rag index `fairship__master__98de16a5b264__20260428T060008679936+0000Z` and commit anchor `98de16a5b264`.
<!-- SOURCE: reports/fairship_muon_dis_operational_research_audit.md:11-12 -->

<!-- CLAIM: PROVISIONAL -->
The active operational packet for thesis-side oracle extraction is `reports/fairship_muon_dis_lxplus_execution_packet.md`, with explicit command classes and conservative oracle probe output.
<!-- SOURCE: reports/fairship_muon_dis_lxplus_execution_packet.md:1-153 -->

## Command Surfaces

<!-- CLAIM: PROVISIONAL -->
Code-anchored MuonBack steering surface remains `python "$FAIRSHIP/macro/run_simScript.py" --MuonBack -f "$SHIPSOFT/data/pythia8_Geant4_onlyMuons.root" -Y <DY_IN_METERS>`.
`DY` is `DY_PROVISIONAL`: parser/type/unit/default are code-backed, but thesis-safe numeric value/range remains unresolved.
Code-default mode (omit `-Y`) relies on implementation default `6.0 m` and is not thesis-safe by itself.
<!-- SOURCE: muonShieldOptimization/run_prod.py:1-80 -->
<!-- SOURCE: reports/fairship_muon_dis_DY_semantics.md:1-42 -->

<!-- CLAIM: PROVISIONAL -->
MuDIS transport surface remains candidate/provisional: `run_simScript.py --MuDIS -f <muonDis.root> ...`; no canonical end-to-end chain is established.
<!-- SOURCE: macro/run_simScript.py:491-570 -->
<!-- SOURCE: docs/wiki/open_questions.md:26-28 -->

<!-- CLAIM: PROVISIONAL -->
Oracle extraction command for existing ROOT files is:
`python scripts/lxplus_muondis_oracle_probe.py --input <ROOT_FILE> --output oracle_probe.json`
<!-- SOURCE: reports/fairship_muon_dis_lxplus_execution_packet.md:47-53 -->

## Oracle Scope

<!-- CLAIM: PROVISIONAL -->
Oracle fields are separated into code-confirmed counters/presence channels (`dis_tree_exists`, `n_DISParticles`, `n_SoftParticles`, `n_SBT_hits`, `n_UBT_hits`, `n_veto_hits`) and unresolved truth channels (`fiducial_fail`, `wall_like_fail`).
<!-- SOURCE: reports/fairship_muon_dis_oracle_observable_schema.md:1-61 -->

<!-- CLAIM: UNRESOLVED -->
`wall_like_fail` is not a canonical Geant4 wall-hit truth field in this evidence snapshot.
NOT FOUND IN INDEX
Next action: retain split channels (`analysis_fiducial_fail`, `detector_or_veto_hit`, `geant4_volume_wall_hit`) until canonical branch/API evidence is retrieved.

## Known Limits

<!-- CLAIM: UNRESOLVED -->
Canonical single-command MuonBack->MuDIS LXPLUS chain remains unresolved.
NOT FOUND IN INDEX
Next action: run staged commands on LXPLUS and archive outputs before promoting any canonical recipe.

<!-- CLAIM: UNRESOLVED -->
Native one-step custom post-shield muon-state injection remains unresolved.
NOT FOUND IN INDEX
Next action: continue focused retrieval and execution probes around generator/steering interfaces.
