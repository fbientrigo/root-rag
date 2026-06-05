---
type: concept-note
topic: muon_dis
status: CONFIRMED code-local
fairship_commit: 98de16a5b264
tags: [nDIS, normalization, weight, DIS, simulation]
---

# Concept: nDIS Normalization Factor

## Definition
`nDIS` (often referenced in code as `args.nDIS` or `nDISPerMuon`) is a user-defined multiplicity factor that determines how many Deep Inelastic Scattering (DIS) interaction attempts are generated for every single incoming background muon.

## Role in the Pipeline
The `nDIS` factor is used to statistically enhance the sampling of DIS interactions, which are rare compared to the total number of background muons. This "oversampling" must be corrected later by a weight factor of `1/nDIS` to maintain the correct absolute normalization of the background.

### 1. Generation Phase (`makeMuonDIS.py`)
- **Parser**: Defined via `-nDISPerMuon` or `--nDIS` (default: **1000**).
- **Execution**: For each input muon from the background ntuple, `makeMuonDIS.py` enters a loop that runs `nDIS` times.
- **Targeting**: The first half of the interactions (`nDIS // 2`) are typically generated on a proton target (`p+`), and the second half on a neutron target (`n0`).
- **Storage**: The value of `nDIS` is stored in the `InMuon` branch of the `DIS` tree at index `[12]`.

### 2. Simulation Phase (`MuDISGenerator.cxx`)
- **Extraction**: The `MuDISGenerator` reads the `nDIS` value from `InMuon[12]`.
- **Weighting**: It calculates a `DIS_multiplicity` as:
  $$DIS\_multiplicity = \frac{1}{nDIS}$$
- **Propagation**: This factor is multiplied by the original muon weight ($w$) and applied to the incoming muon track in the `AddTrack` call.
- **Code Anchor**:
  ```cpp
  // shipgen/MuDISGenerator.cxx:94
  Double_t DIS_multiplicity = 1 / (*mu)[12]; // 1/nDIS
  
  // shipgen/MuDISGenerator.cxx:182
  cpg->AddTrack(..., w * DIS_multiplicity); 
  ```

## Advanced Normalization (Material Budget)
In addition to the `1/nDIS` factor, the `MuDISGenerator` modifies the weight of outgoing DIS particles based on the material encountered along the muon trajectory:
1.  **Material Weight**: `w = average_density * track_length` (using `mparam[0] * mparam[4]`).
2.  **Cross-Section**: The first outgoing DIS particle (index 0) carries the cross-section (`cross_sec`) as its weight in the `MCTrack` container for easy normalization in analysis.

## Official Repository Links
- [shipgen/MuDISGenerator.cxx](https://github.com/ShipSoft/FairShip/blob/master/shipgen/MuDISGenerator.cxx)
- [muonDIS/makeMuonDIS.py](https://github.com/ShipSoft/FairShip/blob/master/muonDIS/makeMuonDIS.py)

## Summary of Normalization
The final weight of a DIS event in the simulation output is a product of:
1.  **Original Muon Weight ($w$)**: Normalized to a spill (e.g., $10^{13}$ protons on target).
2.  **DIS Multiplicity ($1/nDIS$)**: Corrects for the oversampling during generation.
3.  **Cross-Section ($\sigma$ in mb)**: The physical interaction probability (stored in index `[10]`).
4.  **Material Density ($\rho$)**: Integrated along the muon trajectory to pick an interaction point.

## Why it matters
Without the `1/nDIS` factor, the simulated DIS background would be artificially inflated by a factor of 1000, leading to a catastrophic overestimation of the background rate in the SHiP detector.

## Source Anchors
- `muonDIS/makeMuonDIS.py:41-48`: Argument parser and default value.
- `muonDIS/makeMuonDIS.py:227`: Storage in `InMuon[12]`.
- `shipgen/MuDISGenerator.cxx:94`: Extraction and calculation of `1/nDIS`.
- `shipgen/MuDISGenerator.cxx:182`: Application to `AddTrack` weight.

## Links
- [[fairship/branches/InMuon]]
- [[fairship/generators/MuDISGenerator]]
- [[workflows/muon_dis_pipeline]]
