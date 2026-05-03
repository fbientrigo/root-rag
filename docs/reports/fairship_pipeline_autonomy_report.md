# FairShip Muon-Background Pipeline Autonomy Report

## Executive summary
- `root-rag` checkout confirmed locally at `C:\Users\Asus\Documents\FisicoFabi\root-rag` (`git rev-parse --show-toplevel` returned this path).
- FairShip already present as sibling repo at `C:\Users\Asus\Documents\FisicoFabi\FairShip`, remote `https://github.com/ShipSoft/FairShip.git`, branch `master`, commit `98de16a5b264d51c36e1a3638466d1dbb7667678`.
- Highest-confidence post-MuonShield insertion point: `macro/run_simScript.py --MuonBack` using `ROOT.MuonBackGenerator()` over input trees/ntuples carrying `MCTrack` + `vetoPoint`/`PlaneHAPoint`.
- Highest-confidence muon DIS entry point: `macro/run_simScript.py --MuDIS` using `ROOT.MuDISGenerator()` on `DIS` tree files (example in script: `-f $EOSSHIP/.../muonDis_1.root`).
- Reconstruction step: `macro/ShipReco.py` driving `python/shipDigiReco.py` (digitization + tracking + vertexing), writing `*_rec.root`.
- Veto/selection step: `python/shipVeto.py` decisions (SBT/UBT/track multiplicity) consumed in `macro/ShipAna.py`; additional explicit candidate preselection exists in `python/experimental/analysis_toolkit.py`.

## Scope
- Goal: evidence-grounded mapping of FairShip muon-background workflow for post-MuonShield muons, muon DIS generation, propagation/reconstruction, veto/selection, and minimal LXPLUS run path.
- Constraint observed: local-only evidence. Graph/index artifacts treated as aids, not physics ground truth.

## What was found locally

### Repository identity and state
- `root-rag` repo root confirmed by git command result.
- Local tree contains docs/scripts explicitly mentioning FairShip indexing/extraction (`README.md`, `docs/HOW_TO_GUIDE.md`, `docs/QUICK_START.md`, `docs/fairship_extraction.md`, `scripts/index_fairship.py`, `scripts/extract_fairship_root_usage.py`).

### FairShip location and version
- Sibling directory `..\FairShip` exists.
- `git -C ..\FairShip remote -v` -> ShipSoft upstream URL.
- `git -C ..\FairShip rev-parse --abbrev-ref HEAD` -> `master`.
- `git -C ..\FairShip rev-parse HEAD` -> `98de16a5b264d51c36e1a3638466d1dbb7667678`.

### Existing root-rag FairShip artifacts
- Existing index directories in `data/indexes_fairship/` include index id `fairship__master__98de16a5b264__20260331T185059271533+0000Z`.
- Index manifest (`data/indexes_fairship/.../index_manifest.json`) reports:
  - `resolved_commit`: `98de16a5b264d51c36e1a3638466d1dbb7667678`
  - `chunk_count`: `386`
  - `file_count`: `163`
  - `fts_db_path`: `fts.sqlite`

## Repo-supported FairShip setup path

### In `root-rag` docs/scripts
- `docs/HOW_TO_GUIDE.md` gives:
  - `git clone https://github.com/ShipSoft/FairShip ../FairShip`
  - `python scripts/index_fairship.py --fairship-path ../FairShip`
- `docs/fairship_extraction.md` / `docs/QUICK_START.md` give:
  - `python scripts/extract_fairship_root_usage.py --fairship-path ../FairShip`

### In FairShip docs (LXPLUS recommendation)
- `FairShip/README.md` recommends on LXPLUS:
  - clone FairShip
  - `source /cvmfs/ship.cern.ch/$SHIP_RELEASE/setUp.sh`
  - `aliBuild build FairShip --always-prefer-system --config-dir $SHIPDIST --defaults release`
  - `alienv enter FairShip/latest`

## root-rag support status for FairShip
- Build support: **confirmed** (`scripts/index_fairship.py --help` documents FairShip indexing options).
- Existing FairShip index data: **confirmed** in `data/indexes_fairship/`.
- Query support through shipped `root-rag` CLI: **partially blocked in this environment**.
  - `.venv\Scripts\root-rag.exe` exists.
  - CLI resolves FairShip index id (`root_ref master`) but `search`/`grep` fail with `FTS5 database not found: fts.sqlite` / `Index database not found for v6-36-08`.
- Direct SQLite evidence confirms FairShip FTS DB is present and searchable:
  - Python sqlite query on `data/indexes_fairship/.../fts.sqlite` returns MuDIS hits (e.g. `shipgen/MuDISGenerator.cxx` chunks).

## Evidence-backed pipeline map

### 1) Post-MuonShield muon input
- `macro/run_simScript.py` adds `--MuonBack` option and instantiates `ROOT.MuonBackGenerator()` when enabled.
- Same script enforces input file for `--MuonBack` (`input file required if simEngine = Ntuple or MuonBack`).
- `shipgen/MuonBackGenerator.cxx` reads either:
  - tree `pythia8-Geant4` (legacy format), or
  - tree `cbmsim` with `MCTrack` and either `PlaneHAPoint` (STL format) or `vetoPoint` (TClonesArray format).
- This is strongest observed ingestion path for “muons after Muon Shield” into transport.

### 2) Muon DIS generation
- `macro/run_simScript.py` exposes `--MuDIS` and creates `ROOT.MuDISGenerator()`.
- For `--MuDIS`, script requires input file and gives example path under `$EOSSHIP/.../muonDIS/muonDis_1.root`.
- `shipgen/MuDISGenerator.cxx` expects tree `DIS` with branches:
  - `InMuon`, `DISParticles`, `SoftParticles`.
- Vertex/material logic in `MuDISGenerator::ReadEvent`:
  - computes trajectory segment (`startZ`,`endZ` set by `run_simScript.py` via `SetPositions`),
  - samples geometry/material via `gGeoManager->FindNode(...)`,
  - uses `shipgen::MeanMaterialBudget(...)`,
  - stochastically picks interaction point by local density,
  - injects incoming muon + DIS particles + soft tracks (soft tracks only if `softz <= zmu`).

### 3) Pre-DIS muon-to-DIS file production in current FairShip
- `muonDIS/make_nTuple_SBT.py`: collects muons hitting SBT from MuonBack samples (`ship.conical.MuonBack-TGeant4.root`), writes `MuonAndSoftInteractions` with:
  - `imuondata`,
  - `tracks` (soft tracks),
  - `muon_vetoPoints`,
  - `muon_UpstreamTaggerPoints`.
- `muonDIS/make_nTuple_Tr.py`: analogous workflow for Tracking Station 1 and keeps SBT/no-SBT categorization (`events_ = {"Tr","Tr_SBT","Tr_noSBT"}`).
- `muonDIS/makeMuonDIS.py`: reads `MuonAndSoftInteractions`, runs Pythia6 DIS per muon, writes `muonDis.root` with tree `DIS` and branches:
  - `InMuon`, `DISParticles`, `SoftParticles`,
  - `muon_vetoPoints`, `muon_UpstreamTaggerPoints`.

### 4) Transport and reconstruction to detector-level outputs
- Transport/simulation driver: `macro/run_simScript.py`.
- Reconstruction driver: `macro/ShipReco.py -f sim_*.root -g geo_*.root`.
- `ShipReco.py` calls `shipDigiReco.ShipDigiReco(...)` and loops events with:
  - `digitize()`,
  - `reconstruct()`,
  - output `ship_reco_sim` tree in `*_rec.root`.
- `python/shipDigiReco.py` includes detector digitizers and reco branches, notably:
  - SBT digitizer when `vetoPoint` branch exists,
  - UBT digitizer when `UpstreamTaggerPoint` exists,
  - track finding/fitting and vertexing.

### 5) Veto / candidate selection logic
- Veto decisions implemented in `python/shipVeto.py`:
  - `SBT_decision` (efficiency model on `Digi_SBTHits`),
  - `UBT_decision` (hits in `UpstreamTaggerPoint`),
  - `Track_decision` (converged track multiplicity threshold, veto when `nMultCon > 2`).
- `macro/ShipAna.py` applies these via:
  - `vetoDets["SBT"] = veto.SBT_decision()`
  - `vetoDets["UBT"] = veto.UBT_decision()`
  - `vetoDets["TRA"] = veto.Track_decision()`
- Additional candidate preselection cuts (analysis toolkit) in `python/experimental/analysis_toolkit.py::preselection_cut` include:
  - fiducial checks,
  - distance to inner wall / vessel entrance,
  - IP, DOCA,
  - daughter nDOF and chi2/ndf,
  - daughter momentum.

## Post-MuonShield injection options

### Highest-confidence option (fact)
- `run_simScript.py --MuonBack -f <muonback_input>` with `MuonBackGenerator` and `cbmsim` branches `MCTrack` + `vetoPoint`/`PlaneHAPoint`.

### MuonDIS chain option (fact)
- `muonDIS/make_nTuple_SBT.py` (or `make_nTuple_Tr.py`) -> `muonDIS/makeMuonDIS.py` -> `run_simScript.py --MuDIS -f muonDis.root`.

### Direct ntuple mode (fact)
- `run_simScript.py --Ntuple -f <file>` activates `ROOT.NtupleGenerator()` path.

## muon DIS generation path
- Modern path is in `FairShip/muonDIS/*` (supported by CHANGELOG note that scripts were consolidated there).
- Legacy path still exists in `muonShieldOptimization/makeMuonDIS.py`.
- CHANGELOG explicitly states modern MuonDIS corrections and `add_muonresponse.py` usage.

## Reconstruction / veto / selection path
- Reconstruction: `run_simScript.py` output (`sim_*.root`, `geo_*.root`) -> `ShipReco.py` -> `sim_*_rec.root`.
- Veto/tagging: `shipVeto.py` SBT/UBT/track decisions used by `ShipAna.py`.
- Candidate-level selection: `analysis_toolkit.py preselection_cut` (experimental toolkit).

## Minimal LXPLUS runbook
- See `artifacts/fairship_pipeline_autonomy/runbook/minimal_lxplus_runbook.md`.
- Commands are marked as either directly observed or conservative inference.

## Version / geometry sensitivities
- FairShip branch/commit sensitivity: local evidence tied to `master@98de16a5b264...`.
- Geometry sensitivity in MuDIS path:
  - `run_simScript.py` sets MuDIS active z-range from geometry (`ship_geo...`).
  - `MuDISGenerator` interaction sampling depends on geometry/material map via `gGeoManager` and `MeanMaterialBudget`.
- Veto behavior sensitivity:
  - SBT/UBT efficiencies are hardcoded defaults in `shipVeto.py` (`0.99`, `0.9`).

## Facts vs inferences vs gaps

### Facts (direct evidence)
- Repository identities, local paths, FairShip git commit.
- Presence and signatures of `--MuonBack`, `--MuDIS`, `ShipReco`.
- Branch names and tree branches used in `MuonBackGenerator` and `MuDISGenerator`.
- DIS generation scripts in `muonDIS/` and their I/O branches.
- Veto decision functions and where called.
- Existing FairShip index manifests and SQLite FTS records.

### Inferences (marked)
- Minimal operational order of full MuonDIS campaign (MuonBack sample -> ntuple extraction -> `makeMuonDIS.py` -> `run_simScript --MuDIS` -> optional `add_muonresponse.py` -> reco -> analysis) is inferred by joining script responsibilities + CHANGELOG notes.
- “Dangerous candidate” operationalized as events failing SBT/UBT/track veto and/or failing candidate preselection cuts; no single code entity named exactly “dangerous candidate”.

### Unresolved gaps
- No explicit “front/side/cavern” classifier found in scanned FairShip files with those labels.
- `root-rag` CLI query path for FairShip index currently errors despite existing index DB (needs tooling check).
- Full physics-validation chain (cross-section/systematics/geometry configuration control for thesis-grade claim) not established by this scan alone.
