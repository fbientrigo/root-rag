---
type: config-note
subsystem: MuonDIS
claim_state: CONFIRMED code-local
evidence_type: code-local
runtime_validated: false
physics_validated: false
fairship_commit: 98de16a5b264
aliases:
  - z-window
  - Chamber1
---

# Config: mu_start

## What it is
`mu_start` is a geometric configuration parameter that defines the upstream boundary (in z) for the Muon DIS interaction window.

## Where it appears
- **Producer**: `macro/run_simScript.py`
- **Storage**: Memory (passed to `MuDISGenerator`)
- **Consumer**: `shipgen/MuDISGenerator.cxx` (for vertex sampling)

## Producer / storage / consumer
In `run_simScript.py`, `mu_start` is calculated relative to the z-position of `Chamber1`. This value is then passed to the generator to ensure interactions occur within the relevant detector volume.

## Minimal code evidence
    # macro/run_simScript.py:549
    if options.MuDIS:
        mu_start = ship_geo.Chamber1.z - ship_geo.chambers.Tub1length - 10.0*u.cm
        mu_end   = ship_geo.Chamber1.z

## Interpretation
The 10 cm buffer upstream of the tracking chamber is a safety margin to capture interactions that might produce tracks originating just outside the active volume.

## Agent guidance
Always trace `mu_start` back to `ship_geo.Chamber1.z`. Changes to the tracker geometry will automatically shift the MuonDIS interaction window.

## Thesis use
Essential for describing the "Fiducial Volume" of the simulation.

## Open questions
Is the 10 cm buffer a physics requirement or a technical optimization?
