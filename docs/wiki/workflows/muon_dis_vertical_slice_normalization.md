---
type: vertical-slice
domain: fairship
topic: muon_dis
status: PROVISIONAL
evidence_type: code-local
fairship_commit: 98de16a5b264
tags: [normalization, weighting, vertical-slice, nDIS, CrossSection]
---

# Vertical Slice: Muon DIS Normalization & Weighting

## Purpose
This vertical slice traces the lifecycle of normalization metadata and statistical weights from initial event generation through simulation to the final output artifacts. It serves as a technical reference for the "Normalization and Weighting" chapter of the thesis, specifically for FairShip commit 98de16a5b264.

## 1. Generation Layer: `muonDIS/makeMuonDIS.py`
The generation phase extracts physics metadata and normalization factors to be carried through the pipeline.

- **nmuons**: The total number of muons in the source `muonBack` event is extracted to normalize the seed rate.
  - *Evidence*: `muonDIS/makeMuonDIS.py:200`
- **Cross-section ($\sigma_{DIS}$)**: The total DIS cross-section is extracted from Pythia 6's `PARI(1)` parameter.
  - *Evidence*: `muonDIS/makeMuonDIS.py:225`
- **nDIS**: A user-defined multiplicity factor (via `args.nDIS`) used to increase the statistical sample of DIS interactions per seed muon.
  - *Evidence*: `muonDIS/makeMuonDIS.py:227`
- **Storage**: These values are packed into the `InMuon` branch, stored as a `TVectorD(14)`.
  - *Evidence*: `muonDIS/makeMuonDIS.py:212-231`

## 2. Simulation Layer: `shipgen/MuDISGenerator.cxx`
The simulation generator reads the `InMuon` metadata and applies weights to the resulting `MCTrack` objects.

- **Weight Calculation**: The event weight is calculated as $W_{event} = W_{muon} \cdot (1 / nDIS)$.
  - *Evidence*: `shipgen/MuDISGenerator.cxx:94`: `Double_t DIS_multiplicity = 1 / (*mu)[12];`
- **MCTrack persistence**:
  - The calculated $W_{event}$ is passed to `FairPrimaryGenerator::AddTrack` for the incoming muon.
    - *Evidence*: `shipgen/MuDISGenerator.cxx:182`
  - **Overloading**: The DIS cross-section is explicitly saved in the weight field of the first outgoing DIS particle (MCTrack index 1).
    - *Evidence*: `shipgen/MuDISGenerator.cxx:203`: `cpg->AddTrack(..., cross_sec); // save DIS cross section in MCTrack[1]`

## 3. Post-Simulation Promotion: `macro/run_simScript.py`
To simplify analysis, the simulation macro promotes metadata from the input `DIS` tree to the output simulation file.

- **CrossSection Branch**: The cross-section stored at `InMuon[0][10]` is promoted to a top-level `CrossSection` branch in the output `cbmsim` tree.
  - *Evidence*: `macro/run_simScript.py:858-861`

## 4. Analysis Layer: Normalization Guidance (PROVISIONAL)
The total expected number of physics events $N_{phys}$ for a given observable $O$ is proposed as:

$$N_{phys} = \sum_{events} \frac{O \cdot W_{muon} \cdot \sigma_{DIS}}{nDIS \cdot nmuons \cdot \text{Acceptance}}$$

**Caveats**:
- **Interpretive Formula**: This formula is a synthesis based on individual metadata components found in code; it is NOT directly retrieved as a single expression from FairShip.
- **Acceptance**: The `Acceptance` term remains `UNRESOLVED` in the local code context and requires external validation.
- **Runtime Validation**: This weighting scheme has not been verified against a running LXPLUS simulation.

## Data Flow Map

| Variable | Generation | Simulation | Output |
| :--- | :--- | :--- | :--- |
| **nDIS** | `InMuon[0][12]` | `1/nDIS` applied to $W$ | N/A (implied in $W$) |
| **CrossSection** | `InMuon[0][10]` | `MCTrack[1].GetWeight()` | `CrossSection` branch |
| **nmuons** | `InMuon[0][13]` | Logged | N/A |
| **Original $W$** | `InMuon[0][8]` | Basis for $W_{event}$ | `MCTrack[0].GetWeight()` |

## Verification
- **Code-local**: Indices and mapping verified against FairShip master (commit 98de16a5b264).
- **Physics Rationale**: `PROVISIONAL`. Formula requires alignment with SHiP MuonDIS internal notes.
- **Runtime**: `NOT PERFORMED`.
