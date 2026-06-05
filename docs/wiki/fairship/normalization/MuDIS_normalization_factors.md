---
type: node
domain: fairship
topic: muon_dis
status: CONFIRMED
claim_ids: [CLM-002, CLM-006]
tags: [normalization, nDIS, nmuons, MuDIS]
---

# MuDIS Normalization Factors

## Role
The Muon DIS background generation pipeline uses several factors to normalize the rare background rate to the expected proton-on-target (POT) or muon-on-target equivalent.

## Key Factors
- **nmuons**: The number of muons in the original Muon Background event (from which the seed muon was selected).
- **nDIS**: A user-defined multiplicity factor (via `args.nDIS` in `makeMuonDIS.py`). It defines how many DIS interactions are generated per selected muon to increase the sample size.
- **Weight (W)**: The statistical weight of the original muon.
- **Cross-section ($\sigma_{DIS}$)**: The total DIS cross-section for the interaction, extracted from Pythia's `PARI(1)`.

## Normalization Logic
1. `makeMuonDIS.py` saves `nDIS` and `nmuons` into the `InMuon` branch.
2. `MuDISGenerator` reads these values and applies a per-event weight of `1/nDIS`.
3. The final physics normalization for a given observable $O$ is typically:
   $N_{phys} = \sum \frac{O \cdot W \cdot \sigma_{DIS}}{nDIS \cdot nmuons \cdot \text{Acceptance}}$

## Source Anchors
- `muonDIS/makeMuonDIS.py:200`: Extraction of `nmuons`.
- `muonDIS/makeMuonDIS.py:227`: Insertion of `nDIS` into `InMuon`.
- `shipgen/MuDISGenerator.cxx:94`: Consumption of `nDIS`.

## Links
- [[fairship/branches/InMuon]]
- [[fairship/branches/CrossSection]]
