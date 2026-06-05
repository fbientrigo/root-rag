---
type: node
domain: fairship
topic: muon_dis
status: CONFIRMED
claim_ids: [CLM-DRT-RECO-BRANCH-010]
tags: [DIS_tree, schema, MuDIS, branches]
---

# DIS Tree Schema

## Role
The `DIS` tree (typically stored in `muonDis.root`) defines the data exchange format between the pre-selection stage and the Muon DIS simulation phase. It ensures all relevant information from the original muon background is available during generation.

## Canonical Branches
All branches are stored as `TClonesArray` objects.

| Branch Name | Data Type | Role | Details |
| :--- | :--- | :--- | :--- |
| **InMuon** | `TVectorD` (14) | Incoming muon state | [[fairship/branches/InMuon]] |
| **DISParticles** | `TVectorD` (5) | DIS products | PDG, PX, PY, PZ, E |
| **SoftParticles** | `TVectorD` (9) | Soft tracks | [[fairship/branches/SoftParticles]] |
| **muon_vetoPoints** | `vetoPoint` | SBT/UBT hits | Original MC points |
| **muon_UpstreamTaggerPoints** | `UpstreamTaggerPoint` | UBT hits | Original MC points |

## Code-Local Evidence
The branches are declared and initialized in `makeMuonDIS.py`:
- `InMuon`: `TClonesArray("TVectorD")` (`line 152`)
- `DISParticles`: `TClonesArray("TVectorD")` (`line 155`)
- `SoftParticles`: `TClonesArray("TVectorD")` (`line 158`)
- `muon_vetoPoints`: `TClonesArray("vetoPoint")` (`line 161`)
- `muon_UpstreamTaggerPoints`: `TClonesArray("UpstreamTaggerPoint")` (`line 164`)

## Propagation to cbmsim
During the simulation phase (`run_simScript.py --MuDIS`), the `InMuon[0][10]` cross-section is promoted to a dedicated `CrossSection` branch in the output `cbmsim` tree for analysis-level consumption.

## Source Anchors
- `muonDIS/makeMuonDIS.py:152-165`: Branch declarations in the `DIS` tree.
- `macro/run_simScript.py:841-861`: Cross-section promotion logic.

## Links
- [[fairship/scripts/makeMuonDIS]]
- [[fairship/generators/MuDISGenerator]]
- [[fairship/branches/CrossSection]]
