# Open Question: PYTHIA6 to PYTHIA8 Muon DIS Mapping

**Status**: OPEN
**Category**: Simulation Configuration
**Context**: `makeMuonDIS.py` currently uses Pythia6 via the legacy FairShip interface. Migration to Pythia8 is proposed.

## Description
What are the equivalent settings in PYTHIA8 for the FairShip Muon DIS configuration?

## Lead Mapping (from [[EXTERNAL_EVIDENCE_REGISTRY#EXT-005]])
- **Process**: `MSUB(10)=1` (P6) -> `WeakBosonExchange:ff2ff(t:gmZ) = on` (P8)
- **Recoil**: `MSTP(32)` (P6) -> `SpaceShower:dipoleRecoil = on` (P8)
- **Cuts**: `CKIN(21)` (P6) -> `PhaseSpace:Q2Min` (P8)

## Verification Needed
1. Does FairShip's `MuDISGenerator` support reading Pythia8-style trees?
2. Are the DIS cross-sections consistent between the two versions for fixed-target muons?
3. What is the impact of switching to the `dipoleRecoil` scheme on the hadronic system kinematics?

## Search Targets
- `makeMuonDIS.py` Pythia6 configuration block.
- Pythia8 online manual for DIS (Lepton-Nucleon).
