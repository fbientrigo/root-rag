# Open Question: Front/Side/Cavern Implementation

**Status**: RESOLVED (external-doc)
**Category**: Static Code Search
**Context**: External documentation ([[EXTERNAL_EVIDENCE_REGISTRY#EXT-003]]) defines these regions for Muon DIS background. Static audit of the local FairShip index (commit `98de16a5b264`) failed to find any local code implementation.

## Resolution
Terminology confirmed in external conference talk (EXT-003). No local code evidence found in indexed snippets or repository checkout. Claim moved to `CONFIRMED external-doc`.

## Evidence Table
| File / Artifact | Hit Status | Notes |
| :--- | :--- | :--- |
| `muonDIS/make_nTuple_SBT.py` | NOT FOUND | No classification logic in indexed lines 1-80, 191-248. |
| `muonDIS/makeMuonDIS.py` | NOT FOUND | No classification logic in indexed lines 1-80, 71-150, 211-231. |
| `muonShieldOptimization/ana_ShipMuon.py` | NOT FOUND | No classification logic found in snippets. |
| `artifacts/corpus.jsonl` | NOT FOUND | Grep for 'front', 'side', 'cavern' yielded 0 hits in indexed chunks. |
| `EXT-003` (Indico) | **CONFIRMED** | Primary source defining these regions for SHiP. |

## Conclusion
The 'front', 'side', and 'cavern' classifications are likely analysis-level labels used in external reports/talks but not explicitly hardcoded in the baseline production scripts indexed here.
