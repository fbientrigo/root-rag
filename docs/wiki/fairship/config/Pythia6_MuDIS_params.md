---
type: node
domain: fairship
topic: muon_dis
status: CONFIRMED
claim_ids: [CLM-001]
tags: [Pythia6, config, MuDIS, makeMuonDIS]
---

# Pythia6 MuDIS Parameters

## Role
`makeMuonDIS.py` utilizes the legacy `TPythia6` interface to generate DIS interactions for muons. The configuration is tuned for fixed-target physics.

## Confirmed Configuration
- **Mode**: `FIXT` (Fixed Target).
- **Process Selection**: `MSEL = 2` (All QCD processes / Minimum bias).
- **Scale Parameter**: `PARP(2) = 2.0` (Lowest c.m. energy for hard processes).
- **Target Nucleons**:
    - `p+`: Proton target.
    - `n0`: Neutron target.
- **Decay Suppression**: Decays are explicitly disabled (`SetMDCY(kc, 1, 0)`) for long-lived and strange particles to allow FairShip/Geant4 to handle their transport:
    - Pions/Kaons: 211 ($\pi^\pm$), 321 ($K^\pm$), 130 ($K^0_L$), 310 ($K^0_S$), 311 ($K^0$).
    - Hyperons: 3112 ($\Sigma^-$), 3122 ($\Lambda^0$), 3222 ($\Sigma^+$), 3312 ($\Xi^-$), 3322 ($\Xi^0$), 3334 ($\Omega^-$).

## Source Anchors
- `muonDIS/makeMuonDIS.py:168-172`: `MSEL`, `PARP`, and `MDCY` settings.
- `muonDIS/makeMuonDIS.py:232`: Initialization with `FIXT` and `p+`.
- `muonDIS/makeMuonDIS.py:237`: Switch to `n0` target for neutron interactions.

## Links
- [[fairship/scripts/makeMuonDIS]]
- [[open_questions/pythia6_to_pythia8_muondis_mapping]]
