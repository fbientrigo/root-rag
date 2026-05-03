# FairShip Pipeline Gap Register

## Missing repos / components
- None for core scan: local FairShip repo exists at `C:\Users\Asus\Documents\FisicoFabi\FairShip` on `master`.

## Missing env vars / runtime prerequisites
- Required for actual LXPLUS execution but not validated in this Windows session:
  - `FAIRSHIP`
  - `SHIPDIST`
  - `SHIP_RELEASE`
  - CVMFS mount at `/cvmfs/ship.cern.ch`

## Missing command confirmations
- Could not execute full FairShip runtime commands (`run_simScript.py`, `ShipReco.py`, `ShipAna.py`) in this local environment because FairShip runtime stack (alienv/CVMFS/FairRoot libs) not activated here.

## Ambiguous / partially resolved entry points
- No explicit label-level implementation found for “front / side / cavern muons” in scanned files.
- Legacy and modern MuonDIS script stacks both exist:
  - modern: `muonDIS/*`
  - legacy: `muonShieldOptimization/makeMuonDIS.py`
  - consolidation claim appears in `CHANGELOG.md`, but no single canonical workflow doc in `muonDIS/README.md` (none found).

## Tooling gaps (root-rag integration)
- FairShip index artifacts exist and SQLite FTS is queryable directly.
- `.venv\Scripts\root-rag.exe` fails to query FairShip index with CLI (`FTS5 database not found: fts.sqlite` / `Index database not found for v6-36-08`) when targeting `data/indexes_fairship`.
- Practical implication: index build artifacts are present, but turnkey FairShip search path through current CLI is not fully functional in this session.

## Physics-validation gaps
- No end-to-end numerical validation run performed for:
  - muon DIS cross-section consistency,
  - geometry-dependent interaction distributions,
  - veto performance metrics,
  - signal/background selection efficiency.
- Thesis-grade claims need controlled production/reco runs on pinned FairShip + shipdist release and documented geometry YAML/config.
