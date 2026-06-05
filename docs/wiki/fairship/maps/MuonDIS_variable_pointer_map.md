# MuonDIS Variable Pointer Map

## Purpose
This map tracks critical variables, ROOT branches, and configuration parameters used in the Muon DIS background generation and simulation pipeline. It serves as a producer-to-consumer registry to help agents and researchers navigate the code flow.

## Scope
Focuses on `muonDIS` generation, `shipgen` simulation, and `shipDigiReco` reconstruction layers.

## Variable/Branch Map

| Item | Type | Producer | Storage | Consumer | Evidence | State |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `nDIS` | Variable | `makeMuonDIS.py` | `muonDis.root` | `MuDISGenerator.cxx` | `muonDIS/makeMuonDIS.py:212-231` | CONFIRMED |
| `nmuons` | Variable | `makeMuonDIS.py` | `muonDis.root` | Analysis Macros | `muonDIS/makeMuonDIS.py:212-231` | CONFIRMED |
| `CrossSection` | Branch | `run_simScript.py` | `ship.conestoga.root` | Analysis Macros | `macro/run_simScript.py:841-902` | CONFIRMED |
| `mu_start` | Config | `run_simScript.py` | Memory | `MuDISGenerator` | `macro/run_simScript.py:491-570` | CONFIRMED |
| `InMuon` | Branch | `makeMuonDIS.py` | `muonDis.root` | `MuDISGenerator.cxx` | `muonDIS/makeMuonDIS.py:212-231` | CONFIRMED |
| `DISParticles` | Branch | `makeMuonDIS.py` | `muonDis.root` | `MuDISGenerator.cxx` | `shipgen/MuDISGenerator.cxx:1-80` | CONFIRMED |
| `SoftParticles` | Branch | `makeMuonDIS.py` | `muonDis.root` | `MuDISGenerator.cxx` | `muonDIS/makeMuonDIS.py:141-220` | CONFIRMED |
| `vetoHit` | Branch | `shipDigiReco.py` | `ship.conestoga_digi.root` | `linkVetoOnTracks` | `python/shipDigiReco.py:421-473` | CONFIRMED |
| `SBT Threshold` | Threshold | `shipVeto.py` | Config | `fiducialCheck` | `python/shipVeto.py:1-171` | CONFIRMED |
| `UBT Threshold` | Threshold | `shipDigiReco.py` | Config | `reconstruct` | `python/shipDigiReco.py:1-420` | CONFIRMED |

## Evidence Semantics
- **code-local**: Verified directly in the FairShip master index.
- **project-local-docs**: Derived from audited wiki notes or reports.

## Unresolved Gaps
- Exact `MSEL` value mapping for `muonShieldOptimization` vs `muonDIS` remains mixed.
- Secondary consumer of `SoftParticles` in analysis macros is not yet indexed.

## Query Aliases
- `nDIS` -> `nmuons`, `weight`, `normalization`
- `CrossSection` -> `InMuon[0][10]`, `promotion`
- `mu_start` -> `z-window`, `Chamber1`
- `vetoHit` -> `SBT`, `extrapolation`
