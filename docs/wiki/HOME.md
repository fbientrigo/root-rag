# HOME

## Purpose
Navigation index for FairShip MuonBack/MuDIS/oracle LLMWiki.

## Primary indexes
- [[WIKI_STATE]]
- [[indexes/NODE_REGISTRY]]
- [[indexes/MAP_REGISTRY]]
- [[indexes/EDGE_REGISTRY]]
- [[open_questions]]
- [[STRUCTURE]]

## Task paths
- understand MuDIS code path:
  - `WIKI_STATE -> maps/MuonDIS_pipeline_map -> fairship/generators/MuDISGenerator -> fairship/trees/DIS_tree`
- understand MuonBack to MuDIS boundary:
  - `WIKI_STATE -> maps/MuonBack_to_MuDIS_map -> fairship/scripts/run_simScript_MuonBack -> fairship/scripts/makeMuonDIS -> fairship/scripts/run_simScript_MuDIS`
- understand oracle outputs:
  - `maps/Oracle_observables_map -> fairship/oracle/Oracle_schema -> guides/How_to_validate_MuonDIS_on_LXPLUS`
- prepare LXPLUS validation:
  - `guides/How_to_validate_MuonDIS_on_LXPLUS -> fairship/runtime/LXPLUS_preflight -> open_questions`
- extend the wiki:
  - `guides/How_to_extend_the_FairShip_LLMWiki -> indexes/NODE_REGISTRY -> indexes/MAP_REGISTRY`
- inspect thesis risk notes:
  - `thesis_risks/R1_NF_ESS_collapse -> thesis_risks/R2_proxy_tail_calibration -> thesis_risks/R3_support_OOD -> thesis_risks/R4_cost_benchmark_contamination`
