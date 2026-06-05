# Veto System Architecture (LEGACY / MERGED)

> [!IMPORTANT]
> This page has been moved to the canonical [Veto System Architecture](../../workflows/veto_system_architecture.md). 
> This file is preserved for historical source anchors but should not be expanded.

## Overview
<!-- CLAIM: CONFIRMED -->
The FairShip veto system identifies and rejects background events using active tagging detectors (SBT, UBT), multiplicity cuts, and fiducial volume checks.
<!-- SOURCE: python/shipVeto.py:10-161 -->

## Detector Subsystems

### Surrounding Background Tagger (SBT)
<!-- CLAIM: CONFIRMED -->
The SBT encloses the decay vessel. It supports Liquid Scintillator (default) and Plastic technologies.
<!-- SOURCE: veto/veto.h:81-128 -->
- **Digitization**: Energy loss is aggregated per cell. Hits with < 45 MeV are invalidated.
<!-- SOURCE: python/detectors/SBTDetector.py:27-58 -->
- **Efficiency**: Hardcoded default of 0.99.
<!-- SOURCE: python/shipVeto.py:14 -->

### Upstream Background Tagger (UBT)
<!-- CLAIM: CONFIRMED -->
Positioned upstream to tag background from the muon shield. It is implemented as a simplified vacuum box scoring plane (4.4m x 6.4m x 16cm).
<!-- SOURCE: UpstreamTagger/UpstreamTagger.h:30-42 -->
- **Threshold**: Only particles with momentum > 0.1 GeV/c are counted.
<!-- SOURCE: python/shipVeto.py:67-81 -->
- **Efficiency**: Hardcoded default of 0.90.
<!-- SOURCE: python/shipVeto.py:15 -->

## Analysis Selection Logic

### Track Multiplicity Veto
<!-- CLAIM: CONFIRMED -->
Events with more than 2 converged tracks (where tracks must have nDOF >= 25) are rejected.
<!-- SOURCE: python/shipVeto.py:83-103 -->

### Fiducial Volume Check
<!-- CLAIM: CONFIRMED -->
Reconstructed vertices are verified to be inside the `DecayVacuum` blocks. The check performs a 36-step radial scan using the `TGeoManager` navigator to determine the minimum distance to the vessel wall.
<!-- SOURCE: python/shipVeto.py:111-161 -->

## Reconstruction Integration
<!-- CLAIM: CONFIRMED -->
`shipDigiReco.py` links fitted tracks to veto hits by extrapolating the track state using Genfit to the `vetoHit` position.
<!-- SOURCE: python/shipDigiReco.py:421-444 -->

## Technical Caveats
<!-- CLAIM: PROVISIONAL -->
Muon process deactivation (e.g., `muIoni`, `muBrems`) is handled via `gMC->SetProcess` in `SetCuts.C` and Geant4 macros, not a single unified routine.
<!-- SOURCE: gconfig/SetCuts.C:29-40 -->
<!-- SOURCE: gconfig/g4Config.C:62-74 -->
