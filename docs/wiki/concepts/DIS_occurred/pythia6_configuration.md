# Pythia6 Configuration for Muon DIS

This document explores the specific Pythia6 settings used to generate Deep Inelastic Scattering (DIS) interactions in the FairShip pipeline.

## 1. Interaction Selection (`makeMuonDIS.py`)

Pythia6 is configured to focus on "Minimum Bias" like interactions but within the context of Deep Inelastic Scattering.

### Core Settings
- **`SetMSEL(2)`**: Sets the process selection to all QCD processes (Minimum Bias).
- **`SetPARP(2, 2)`**: Sets the minimum $Q$ cut-off for the hard scattering to 2 GeV. This ensures we are studying the "Inelastic" regime.
- **`Initialize("FIXT", mutype[pid], target, p)`**: Configures Pythia for **Fixed Target** mode (`FIXT`).
  - `mutype[pid]`: Either `gamma/mu+` or `gamma/mu-`. Note that Pythia6 models muon DIS via an equivalent photon approximation.
  - `target`: Iterates between `p+` (proton) and `n0` (neutron).
  - `p`: The laboratory momentum of the incoming muon.

## 2. Stability & Decays

Specific unstable particles are "switched off" to allow Geant4 to handle their decays in the detector geometry.

### Particle Stability
The script iterates through a list of PDG codes and calls `SetMDCY(kc, 1, 0)` to prevent Pythia from decaying them:
- **K-Mesons**: $K^{\pm}$ (211 - wait, 211 is pion), $K_L$ (130), $K_S$ (310), $K^{\pm}$ (321).
- **Hyperons**: $\Lambda$ (3122), $\Sigma$ (3112, 3222), $\Xi$ (3312, 3322), $\Omega$ (3334).

*Note: The script uses PDG 211 (Pion) in the decay-off list, which is an interesting operational detail as pions are usually allowed to decay in Geant4.*

## 3. Metadata Extraction

After `GenerateEvent()`, the script extracts physical observables:
- **`GetPARI(1)`**: The total cross-section of the generated process ($\sigma_{DIS}$).
- **`GetN()`**: Total number of particles produced in the interaction.
- **`Pyedit(1)`**: Used to filter the stack to only include primary products.

## 4. Operational Settings

| Parameter | Value | Description |
|---|---|---|
| `SetMSTU(11, 11)`| 11 | Sets the logical unit for output (suppressing verbose headers). |
| `SetMRPY(1, seed)`| time-based | Sets the random seed for reproducibility. |

## Source Anchors
- `muonDIS/makeMuonDIS.py:170-185`: Pythia6 initialization and stability settings.
- `muonDIS/makeMuonDIS.py:227`: `PARI(1)` extraction.
- `muonDIS/makeMuonDIS.py:233-245`: Target splitting logic.
