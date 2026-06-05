# DIS Occurred Concept

## Overview
In FairShip simulation, a "DIS occurred" event is not marked by a single boolean flag in the output tree. Instead, the presence of Deep Inelastic Scattering (DIS) must be inferred from the particle stack and specific metadata branches.

This folder explores the technical indicators used to confirm and normalize DIS events.

## Key Indicators
1.  **Branch Presence**: The existence of the `DIS` tree in the input files (e.g., `muonDis.root`).
2.  **Particle Count**: Non-zero entries in the `DISParticles` stack.
3.  **Cross-Section Value**: A valid, non-zero cross-section weight stored in `MCTrack[1]` or the `CrossSection` branch.
4.  **Track Ancestry**: Tracks in the simulation with `MotherId == 0` (the primary muon) that originate from the `MuDISGenerator` class.

## Semantic Mapping (Code Links)
- **Generator Core**: `shipgen/MuDISGenerator.cxx` (Method: `ReadEvent`)
- **Producer Script**: `muonDIS/makeMuonDIS.py` (Method: `makeMuonDIS`)
- **Sim Post-processor**: `macro/run_simScript.py` (Block: `if options.mudis:`)

## Notes in this Folder
- [[InMuon_schema]]: Detailed element-by-element mapping of the `InMuon` TVectorD.
- [[DISParticles_stack]]: Structure and content of the DIS particle array.
- [[MCTrack_alignment]]: How simulation tracks map to DIS products and where `MCTrack[1]` is overloaded.
- [[cross_section_weighting]]: Explanation of cross-sections, weights ($W \times \frac{1}{nDIS}$), and normalization.
- [[data_structures_and_memory]]: In-depth analysis of ROOT and FairRoot variable dependencies and memory footprints.
- [[documentation_synthesis]]: Comprehensive synthesis of all internal wiki and report findings.
- [[advanced_pipeline_details]]: Technical dive into vertex placement, rejection sampling, and momentum rotation.
- [[tagger_integration]]: How SBT and UBT hits are selected, extracted, and merged back into the simulation.
- [[pythia6_configuration]]: Specific Pythia6 settings for Fixed Target DIS generation.

## Workflow Context
The "DIS occurred" state is typically determined during the **Generation** stage (`makeMuonDIS.py`) and propagated through **Transport** (`MuDISGenerator.cxx`) to the final simulation output (`cbmsim`).
