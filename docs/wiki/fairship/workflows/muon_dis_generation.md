# Muon DIS Generation (LEGACY / MERGED)

> [!IMPORTANT]
> This page has been merged into the canonical [FairShip Muon DIS Operational Pipeline](../../workflows/fairship_muon_dis_operational_pipeline.md). 
> This file is preserved for historical source anchors but should not be expanded.

## Overview
<!-- CLAIM: CONFIRMED -->
The Muon DIS workflow in FairShip handles the generation and simulation of Deep Inelastic Scattering (DIS) interactions from muons after the muon shield. It follows a staged pipeline: background selection -> DIS event generation -> transport simulation -> metadata persistence.
<!-- SOURCE: muonDIS/make_nTuple_SBT.py:1-80 -->
<!-- SOURCE: muonDIS/makeMuonDIS.py:141-220 -->
<!-- SOURCE: macro/run_simScript.py:491-570 -->

## Generation Pipeline

### 1. Muon Background Selection (`make_nTuple_SBT.py`)
<!-- CLAIM: CONFIRMED -->
Muons hitting the Surrounding Background Tagger (SBT) with momentum > 3 GeV are collected into a dedicated ntuple.
<!-- SOURCE: muonDIS/make_nTuple_SBT.py:191-248 -->
- **Input**: `ship.conical.MuonBack-TGeant4.root`
- **Output**: `muonsProduction_wsoft_SBT.root` (Tree: `MuonAndSoftInteractions`)

### 2. DIS Event Production (`makeMuonDIS.py`)
<!-- CLAIM: CONFIRMED -->
DIS interactions are generated for the selected muons using Pythia6 (FIXT mode, `gamma/mu+` or `gamma/mu-` on `p+` or `n0` targets).
<!-- SOURCE: muonDIS/makeMuonDIS.py:167-245 -->
- **Output Branches**:
  - `InMuon`: TVectorD containing incoming muon state and cross-section (index 10).
  - `DISParticles`: TClonesArray of generated DIS particles.
  - `SoftParticles`: TClonesArray of associated soft interaction products.
<!-- SOURCE: muonDIS/makeMuonDIS.py:152-166 -->

### 3. Simulation Transport (`run_simScript.py --MuDIS`)
<!-- CLAIM: CONFIRMED -->
The `MuDISGenerator` reads the DIS events, determines vertex placement using the material budget (via `shipgen::MeanMaterialBudget`), and injects particles into the simulation.
<!-- SOURCE: shipgen/MuDISGenerator.cxx:35-56 -->
<!-- SOURCE: shipgen/MuDISGenerator.cxx:115-150 -->
- **Default Z-Window**: From `Chamber1` (front of UVT) up to `TrackStation1`.
<!-- SOURCE: macro/run_simScript.py:541-555 -->

## Physics Metadata
<!-- CLAIM: CONFIRMED -->
The DIS cross-section is extracted from the `InMuon` branch and persisted as a float branch `CrossSection` in the output `cbmsim` tree.
<!-- SOURCE: macro/run_simScript.py:841-866 -->

## Technical Caveats
<!-- CLAIM: UNRESOLVED -->
Spatial labels "front", "side", and "cavern" are not explicitly defined in the core simulation code and likely refer to region cuts in external scripts.
NOT FOUND IN INDEX
Next action: Trace regions in initial MuonBack production.
