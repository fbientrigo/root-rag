# FairShip Muon DIS Workflow

## Overview
The Muon DIS workflow in FairShip handles the generation and simulation of Muon Deep Inelastic Scattering (DIS) backgrounds, typically originating from muons after the muon shield.

## Workflow Stages

### 1. Ntuple Generation (`make_nTuple_SBT.py`)
- **Stage**: Pre-selection and ntuple creation.
- **Responsibility**: Collects muons hitting the Surrounding Background Tagger (SBT).
- **Inputs**: Muon background files (e.g., `ship.conical.MuonBack-TGeant4.root`).
- **Outputs**: `muonsProduction_wsoft_SBT.root` with tree `MuonAndSoftInteractions`.
- **Evidence**: `muonDIS/make_nTuple_SBT.py:1-80`, `141-220`.

### 2. DIS Generation (`makeMuonDIS.py`)
- **Stage**: DIS event generation using Pythia6.
- **Responsibility**: Generates DIS interactions for input muons.
- **Inputs**: Muon ntuple.
- **Outputs**: `muonDis.root` with tree `DIS`.
- **Critical Branches**:
  - `InMuon`: Incoming muon state (TVectorD).
  - `DISParticles`: Generated DIS particles.
  - `SoftParticles`: Associated soft interaction products.
- **Evidence**: `muonDIS/makeMuonDIS.py:141-220`, `221-250`.

### 3. Simulation Transport (`run_simScript.py --MuDIS`)
- **Stage**: Transport and detector simulation.
- **Responsibility**: Instantiates `MuDISGenerator`, places vertices based on material budget, and propagates tracks.
- **Inputs**: `muonDis.root`.
- **Critical Configuration**:
  - `SetPositions(mu_start, mu_end)`: Defines the z-window for DIS generation.
  - Default window: `Chamber1.z - Tub1length - 10cm` to `TrackStation1.z`.
- **Evidence**: `macro/run_simScript.py:491-570`, `shipgen/MuDISGenerator.cxx:1-80`.

### 4. Cross Section Persistence
- **Stage**: Post-simulation metadata.
- **Responsibility**: Copies DIS cross-section from input to output `cbmsim` tree.
- **Logic**: `cross_section[0] = muondis_event.InMuon[0][10]`.
- **Evidence**: `macro/run_simScript.py:841-902`, `shipgen/MuDISGenerator.cxx:81-140`.

## Veto and Selection Logic

### Veto Efficiencies
- **SBT Efficiency**: 0.99
- **UBT Efficiency**: 0.9
- **Evidence**: `python/shipVeto.py:1-80`.

### Selection Cuts
- **Impact Parameter (IP)**: Default pre-selection cut at 250 cm.
- **Evidence**: `python/experimental/analysis_toolkit.py:196-210`, `examples/analysis_example.py:71-140`.

## Technical Caveats
- **Process Controls**: Muon processes (BREM, MUNU, LOSS) are controlled via `gMC->SetProcess` in `gconfig/SetCuts.C`.
- **Inactivation**: While `run_prod.py` mentions switching off processes like `muIoni`, no single routine named `InactivateMuonProcesses` was found in the indexed code.
- **Classification**: "Front / side / cavern" classification is not explicitly labeled in the core simulation code, likely defined by the input muon source or z-window selection in generation scripts.

## Technical Guide for LXPLUS (Post-Mortem Hardened)

To ensure reliable execution on LXPLUS, follow these validated operational guidelines derived from internal research audits and code verification.

### 1. Environment & Preflight
- **Environment Initialization**: Always use `alienv` to load a consistent FairShip build.
  ```bash
  alienv enter FairShip/latest
  ```
- **Required Variables**: Ensure `FAIRSHIP` and `SHIPSOFT` are exported. If `FAIRSHIP` is unset, sim-scripts will fail to locate macros and configuration files.
- **Preflight Check**: Verify script availability before batch submission:
  ```bash
  test -f "$FAIRSHIP/macro/run_simScript.py" && echo "[OK] SimScript found"
  ```

### 2. Common Failure Modes & Resource Management
- **Missing Input Abort**: `run_simScript.py --MuDIS` requires an explicit `-f` input file. If missing, the job will abort with an error.
- **Memory Overflows**: DIS simulations with high multiplicity (e.g., `nDIS > 1000`) are memory-intensive. 
  - **HTCondor Advice**: Request at least **4GB** of memory (`request_memory = 4000`) and use the `+JobFlavour = "longlunch"` (2 hours) for standard event sizes.
- **CVMFS Latency**: Jobs may fail if CVMFS is not yet mounted. Consider adding a short sleep or retry logic in your wrapper scripts.

### 3. Critical Post-Processing Steps
- **Cross-Section Promotion**: The cross-section is injected into the output `cbmsim` tree ONLY if the `--MuDIS` flag is used. Validation check:
  ```bash
  root -l -e 'cbmsim->Print()' | grep CrossSection
  ```
- **Tagger Hit Merging**: Hits produced by the muon *before* the DIS interaction vertex are NOT in the standard simulation output. Use `add_muonresponse.py` to restore upstream veto/SBT hits:
  ```bash
  python $FAIRSHIP/muonDIS/add_muonresponse.py -f output_sim.root -m input_muonDis.root
  ```

### 4. Oracle Validation (Success Criteria)
A successful simulation run must satisfy:
1. `dis_tree_exists == true` (verified in the intermediate `muonDis.root`).
2. `CrossSection` branch presence in the final simulation output.
3. Non-zero `n_DISParticles` count (verifying physical interaction generation).

