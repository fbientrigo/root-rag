# FairShip Pipeline Gap Register

## Missing components
- local FairShip repo: `C:\Users\Asus\Documents\FisicoFabi\FairShip` on `master`.

## Missing env / runtime
- Required for LXPLUS (not validated Windows): `FAIRSHIP`, `SHIPDIST`, `SHIP_RELEASE`, CVMFS `/cvmfs/ship.cern.ch`.

## Missing command confirm
- No `run_simScript.py`, `ShipReco.py`, `ShipAna.py` execution. Runtime stack not active.

## Partially resolved
- No "front / side / cavern" label implementation found in `fairship__master__98de16a5b264`.
- **Blocked promotions**:
  - `CLM-008` (SBT/UBT z-filter) blocked until `add_muonresponse.py` logic verified on LXPLUS.
  - LXPLUS end-to-end recipe (MuonBack -> MuDIS) blocked until manual validation.
- MuonDIS stacks: modern `muonDIS/*`, legacy `muonShieldOptimization/makeMuonDIS.py`. No canonical README.

## Tool gaps
- Index exists, SQLite queryable.
- CLI fails (`FTS5 database not found`). Target `data/indexes_fairship` broken in current session.

## Physics gaps
- No end-to-end validation: xsec consistency, geometry interactions, veto metrics, selection efficiency.
- Need pinned FairShip + shipdist + documented geometry YAML for thesis.
