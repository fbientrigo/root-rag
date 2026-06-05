# LXPLUS Preflight Checklist

**STATUS: PROVISIONAL**
*This document lists prerequisites for runtime validation. It does NOT claim that execution has occurred or will succeed.*

## 1. Environment Prerequisites
- [ ] **FairShip Commit/Index**: Verify local `fairship` version matches the target LXPLUS environment.
- [ ] **Setup Command**: `source /cvmfs/ship.cern.ch/SHiP-202X/setup.sh` (or equivalent).
- [ ] **AliBuild State**: Confirm `alienv load FairShip` completes without errors.

## 2. Input Data Verification
- [ ] **MuDIS ROOT File**: Ensure `muonDis.root` exists and contains the `DIS` tree.
- [ ] **Tree Schema**: Check for `InMuon`, `DISParticles`, and `SoftParticles` branches.
- [ ] **Branch Integrity**: Use `DIS->Print()` to verify `TClonesArray` types.

## 3. Configuration Checks
- [ ] **MuDISGenerator Compatibility**: Check `MuDISGenerator.cxx` for any hardcoded paths that may fail on LXPLUS.
- [ ] **Pythia6 Parameters**: Ensure `makeMuonDIS.py` settings are compatible with the installed ROOT version.
- [ ] **Geometry Files**: Confirm all `.root` or `.py` geometry files are accessible in the `$FAIRSHIP` path.

## 4. Expected Output Artifacts
- [ ] `ship.conestoga.root` (Simulation output).
- [ ] `ship.conestoga_digi.root` (Digitization output).
- [ ] `ship.conestoga_reco.root` (Reconstruction output).

## 5. Log Files & Monitoring
- [ ] **Standard Out**: Monitor for "Geant4 exception" or "TGeoManager overlap" warnings.
- [ ] **Memory Logs**: Keep track of RSS for long-running simulation loops.

## 6. Failure Modes to Watch
- **Library Mismatch**: `TClass` version mismatch between local index and LXPLUS.
- **Missing Symbols**: `undefined symbol` errors during `MuDISGenerator` initialization.
- **CVMFS Latency**: Slow execution or timeouts during initial setup.
