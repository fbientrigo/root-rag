# FairShip Muon-Background Pipeline Autonomy Report

## Executive summary
- `root-rag` checkout at `C:\Users\Asus\Documents\FisicoFabi\root-rag`.
- FairShip sibling repo at `C:\Users\Asus\Documents\FisicoFabi\FairShip`. Remote `https://github.com/ShipSoft/FairShip.git`, branch `master`, commit `98de16a5b264d51c36e1a3638466d1dbb7667678`.
- Post-MuonShield insertion: `macro/run_simScript.py --MuonBack` use `ROOT.MuonBackGenerator()`. Input trees with `MCTrack` + `vetoPoint`/`PlaneHAPoint`.
- Muon DIS entry: `macro/run_simScript.py --MuDIS` use `ROOT.MuDISGenerator()` on `DIS` tree files. Example: `-f $EOSSHIP/.../muonDis_1.root`.
- Reconstruction: `macro/ShipReco.py` drive `python/shipDigiReco.py` (digitization + tracking + vertexing). Output `*_rec.root`.
- Veto/selection: `python/shipVeto.py` decisions (SBT/UBT/track multiplicity) in `macro/ShipAna.py`. Candidate preselection in `python/experimental/analysis_toolkit.py`.

## Scope
- Goal: mapping FairShip muon-background workflow (post-MuonShield muons, muon DIS generation, propagation/reconstruction, veto/selection, minimal LXPLUS path).
- Local-only evidence. Graph/index artifacts are aids, not physics truth.

## What was found locally

### Repository identity and state
- `root-rag` root confirmed by git.
- Docs/scripts mention FairShip indexing/extraction (`README.md`, `docs/HOW_TO_GUIDE.md`, `docs/QUICK_START.md`, `docs/fairship_extraction.md`, `scripts/index_fairship.py`, `scripts/extract_fairship_root_usage.py`).

### FairShip location and version
- Sibling `..\FairShip` exists.
- URL: ShipSoft upstream.
- Branch: `master`.
- Commit: `98de16a5b264d51c36e1a3638466d1dbb7667678`.

### Existing root-rag FairShip artifacts
- Index in `data/indexes_fairship/`: `fairship__master__98de16a5b264__20260331T185059271533+0000Z`.
- Manifest (`data/indexes_fairship/.../index_manifest.json`):
  - `resolved_commit`: `98de16a5b264d51c36e1a3638466d1dbb7667678`
  - `chunk_count`: `386`
  - `file_count`: `163`
  - `fts_db_path`: `fts.sqlite`

## Repo-supported FairShip setup path

### In `root-rag` docs/scripts
- `docs/HOW_TO_GUIDE.md`:
  - `git clone https://github.com/ShipSoft/FairShip ../FairShip`
  - `python scripts/index_fairship.py --fairship-path ../FairShip`
- `docs/fairship_extraction.md` / `docs/QUICK_START.md`:
  - `python scripts/extract_fairship_root_usage.py --fairship-path ../FairShip`

### In FairShip docs (LXPLUS recommendation)
- `FairShip/README.md`:
  - Clone FairShip.
  - `source /cvmfs/ship.cern.ch/$SHIP_RELEASE/setUp.sh`
  - `aliBuild build FairShip --always-prefer-system --config-dir $SHIPDIST --defaults release`
  - `alienv enter FairShip/latest`

## root-rag support status for FairShip
- Build support: confirmed (`scripts/index_fairship.py --help`).
- Existing index data: confirmed in `data/indexes_fairship/`.
- Query support: partially blocked.
  - `.venv\Scripts\root-rag.exe` exists.
  - CLI resolve FairShip index but `search`/`grep` fail: `FTS5 database not found: fts.sqlite` / `Index database not found for v6-36-08`.
- SQLite evidence confirms FairShip FTS DB searchable:
  - Python query on `data/indexes_fairship/.../fts.sqlite` returns MuDIS hits (e.g. `shipgen/MuDISGenerator.cxx`).

## Evidence-backed pipeline map

### 1) Post-MuonShield muon input
- `macro/run_simScript.py` add `--MuonBack`, instantiate `ROOT.MuonBackGenerator()`.
- Input file required for `--MuonBack`.
- `shipgen/MuonBackGenerator.cxx` read:
  - tree `pythia8-Geant4` (legacy), or
  - tree `cbmsim` with `MCTrack` + `PlaneHAPoint` or `vetoPoint`.

### 2) Muon DIS generation
- `macro/run_simScript.py` expose `--MuDIS`, create `ROOT.MuDISGenerator()`.
- Require input file, e.g. `$EOSSHIP/.../muonDIS/muonDis_1.root`.
- `shipgen/MuDISGenerator.cxx` expect tree `DIS` with:
  - `InMuon`, `DISParticles`, `SoftParticles`.
- `MuDISGenerator::ReadEvent`:
  - Trajectory segment (`startZ`, `endZ` via `SetPositions`).
  - Sample geometry/material via `gGeoManager->FindNode(...)`.
  - Use `shipgen::MeanMaterialBudget(...)`.
  - Pick interaction point by density.
  - Inject muon + DIS particles + soft tracks (if `softz <= zmu`).

### 3) Pre-DIS muon-to-DIS file production
- `muonDIS/make_nTuple_SBT.py`: collect muons hitting SBT from MuonBack (`ship.conical.MuonBack-TGeant4.root`). Write `MuonAndSoftInteractions` with `imuondata`, `tracks`, `muon_vetoPoints`, `muon_UpstreamTaggerPoints`.
- `muonDIS/make_nTuple_Tr.py`: same for Tracking Station 1. Categorize `Tr`, `Tr_SBT`, `Tr_noSBT`.
- `muonDIS/makeMuonDIS.py`: read `MuonAndSoftInteractions`, run Pythia6 DIS, write `muonDis.root` (tree `DIS`) with `InMuon`, `DISParticles`, `SoftParticles`, `muon_vetoPoints`, `muon_UpstreamTaggerPoints`.

### 4) Transport and reconstruction
- Simulation: `macro/run_simScript.py`.
- Reconstruction: `macro/ShipReco.py -f sim_*.root -g geo_*.root`.
- `ShipReco.py` calls `shipDigiReco.ShipDigiReco(...)`. Loop events: `digitize()`, `reconstruct()`. Output `ship_reco_sim` in `*_rec.root`.
- `python/shipDigiReco.py` include SBT/UBT digitizers, track finding/fitting, vertexing.

### 5) Veto / candidate selection
- `python/shipVeto.py`:
  - `SBT_decision` (efficiency on `Digi_SBTHits`).
  - `UBT_decision` (hits in `UpstreamTaggerPoint`).
  - `Track_decision` (multiplicity threshold, veto if `nMultCon > 2`).
- `macro/ShipAna.py` apply via `vetoDets`.
- `python/experimental/analysis_toolkit.py::preselection_cut`: fiducial checks, distance to wall, IP, DOCA, daughter nDOF, chi2/ndf, momentum.

## Post-MuonShield injection options

### Highest-confidence (fact)
- `run_simScript.py --MuonBack -f <muonback_input>` with `MuonBackGenerator`, `cbmsim` branches `MCTrack` + `vetoPoint`/`PlaneHAPoint`.

### MuonDIS chain (fact)
- `muonDIS/make_nTuple_SBT.py` -> `muonDIS/makeMuonDIS.py` -> `run_simScript.py --MuDIS -f muonDis.root`.

### Direct ntuple mode (fact)
- `run_simScript.py --Ntuple -f <file>` activates `ROOT.NtupleGenerator()`.

## muon DIS generation path
- Modern path: `FairShip/muonDIS/*`. Consolidated there per CHANGELOG.
- Legacy: `muonShieldOptimization/makeMuonDIS.py`.
- CHANGELOG mention MuonDIS corrections + `add_muonresponse.py`.

## Reconstruction / veto / selection path
- Reconstruction: `run_simScript.py` -> `ShipReco.py` -> `*_rec.root`.
- Veto/tagging: `shipVeto.py` used by `ShipAna.py`.
- Candidate selection: `analysis_toolkit.py preselection_cut`.

## Minimal LXPLUS runbook
- See `artifacts/fairship_pipeline_autonomy/runbook/minimal_lxplus_runbook.md`.

## Version / geometry sensitivities
- Commit: `master@98de16a5b264...`.
- Geometry in MuDIS: `run_simScript.py` sets z-range. `MuDISGenerator` sample material via `gGeoManager`.
- Veto: SBT/UBT efficiencies hardcoded (`0.99`, `0.9`).

## Facts vs inferences vs gaps

### Facts
- Repos, paths, git commit.
- `--MuonBack`, `--MuDIS`, `ShipReco` signatures.
- Branches/trees in `MuonBackGenerator`, `MuDISGenerator`.
- `muonDIS/` scripts and I/O.
- Veto functions.
- Index manifests and FTS records.

### Inferences
- MuonDIS campaign order (MuonBack -> ntuple -> `makeMuonDIS.py` -> `run_simScript --MuDIS` -> `add_muonresponse.py` -> reco -> analysis).
- "Dangerous candidate" = fail veto / preselection. No explicit entity.

### Unresolved gaps
- No "front/side/cavern" labels found.
- `root-rag` CLI query error for FairShip index.
- Full physics-validation chain not established.
