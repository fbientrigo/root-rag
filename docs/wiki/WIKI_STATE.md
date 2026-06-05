# WIKI_STATE

| area | status | anchor |
|---|---|---|
| FairShip index anchor | ACTIVE | `data/indexes_fairship/*` via `scripts/emv_status.py` |
| DY/Yheight | CONFIRMED_BY_CODE | [[fairship/config/DY_Yheight]] |
| SBT status | CONFIRMED_BY_CODE | [[fairship/detectors/SBT]] |
| UBT status | CONFIRMED_BY_CODE | [[fairship/detectors/UBT]] |
| MuonBack runtime | RUNTIME_UNVALIDATED | [[fairship/scripts/run_simScript_MuonBack]] |
| MuDIS runtime | RUNTIME_UNVALIDATED | [[fairship/scripts/run_simScript_MuDIS]] |
| Oracle status | PROVISIONAL | [[fairship/oracle/Oracle_schema]] |
| LXPLUS status | UNRESOLVED | [[fairship/runtime/LXPLUS_preflight]] |
| qrel status | UNRESOLVED | `scripts/emv_status.py` |
| thesis risk notes | SCAFFOLD | [[thesis_risks/R1_NF_ESS_collapse]] |

## Status conventions
- Node `Status` is node-level strongest reusable state.
- Claim-level `<!-- CLAIM: ... -->` is finer-grained and authoritative.
- Edge status is relationship confidence.
- `CONFIRMED_BY_CODE` does not imply runtime validation.

## Next 3 actions
1. Keep map edges aligned to canonical atomic notes only.
2. Resolve open evidence gaps in `open_questions` without promoting runtime claims.
3. Prepare runtime packet from guides/runtime notes before any LXPLUS execution claim.
