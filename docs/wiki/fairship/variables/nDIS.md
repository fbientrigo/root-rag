---
type: variable-note
subsystem: MuonDIS
claim_state: CONFIRMED code-local
evidence_type: code-local
runtime_validated: false
physics_validated: false
fairship_commit: 98de16a5b264
aliases:
  - nmuons
  - weight
  - normalization
---

# Variable: nDIS

## What it is
`nDIS` is a normalization variable representing the total number of Deep Inelastic Scattering (DIS) interactions generated in a specific production run. It is used to weight simulated events so they can be scaled to the total number of expected muons (POT).

## Where it appears
- **Producer**: `muonDIS/makeMuonDIS.py`
- **Consumer**: `shipgen/MuDISGenerator.cxx`
- **Storage**: ROOT file header/InMuon branch

## Producer / storage / consumer
In `makeMuonDIS.py`, the total number of interactions is calculated and inserted into the `InMuon` data structure. The `MuDISGenerator` reads this value to apply a $1/nDIS$ weight to each event during simulation.

## Minimal code evidence
```cpp
    // shipgen/MuDISGenerator.cxx:94
    Double_t DIS_multiplicity = 1 / (*mu)[12];  // nDIS stored at index 12
```

## Interpretation
The weighting mechanism ensures that the sum of weights across all simulated events correctly represents the physics cross-section, independent of the number of events actually generated.

## Agent guidance
When querying for normalization, always check for `nDIS` at index 12 of the `InMuon` branch. Do not confuse it with `nmuons` (total muons in source) at index 13.

## Thesis use
Critical for the "Normalization and Weighting" chapter.

## Open questions
Is `nDIS` updated if files are merged via `muDIS_mergeFiles.py`?
