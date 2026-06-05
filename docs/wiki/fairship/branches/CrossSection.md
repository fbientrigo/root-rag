---
type: branch-note
subsystem: MuonDIS
claim_state: CONFIRMED code-local
evidence_type: code-local
runtime_validated: false
physics_validated: false
fairship_commit: 98de16a5b264
aliases:
  - InMuon[0][10]
  - promotion
---

# Branch: CrossSection

## What it is
The `CrossSection` branch stores the differential cross-section (in mb) for each Muon DIS interaction. It is "promoted" from the input `muonDis.root` file to the main simulation output tree (`cbmsim`).

## Where it appears
- **Producer**: `muonDIS/makeMuonDIS.py` (as index 10 in `InMuon`)
- **Storage**: `ship.conestoga.root` (Simulation output)
- **Consumer**: Analysis macros for physics yield calculations.

## Producer / storage / consumer
During simulation, `run_simScript.py` extracts the cross-section value from the generator and writes it into a dedicated branch in the output `TTree`.

## Minimal code evidence
    # macro/run_simScript.py:859
    for n in range( nEvents ):
        # ...
        if options.MuDIS:
            CrossSection.Fill(sTree.InMuon[0][10])

## Interpretation
This promotion ensures that the cross-section information is preserved through the simulation chain without needing to re-read the input background file during analysis.

## Agent guidance
Note that `CrossSection` is filled only if the `--MuDIS` flag is active in `run_simScript.py`.

## Thesis use
Used to calculate the expected number of events: $Yield = L \cdot \sigma \cdot \epsilon$.

## Open questions
Are the units consistent (mb vs cm^2) throughout the pipeline?
