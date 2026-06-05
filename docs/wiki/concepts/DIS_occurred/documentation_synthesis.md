# In-Depth Documentation Synthesis: Muon DIS Pipeline

This document synthesizes all internal documentation and research reports found within the `root-rag` workspace regarding Muon Deep Inelastic Scattering (DIS).

## 1. Variable Exploitation (Internal Truths)

### `InMuon` Schema (Size 14)
The workspace documentation ([fairship/branches/InMuon.md]) corrects previous misconceptions about the schema size. It is a **14-element** `TVectorD`.

| Index | Name | Code Source | Semantic Role |
|---|---|---|---|
| `[10]` | `xsec` | `PARI(1)` (Pythia) | Physics normalization probability (mb). |
| `[12]` | `nDIS` | `args.nDIS` | Multiplicity factor for rare event sampling. |
| `[13]` | `nmuons`| `imuondata[9]` | Muon count in the seed MuBack event. |

**Critical Connection**: `run_simScript.py:858-861` explicitly promotes `InMuon[0][10]` to a top-level `CrossSection` branch in the output file, simplifying analysis.

### `nDIS` Normalization ([concepts/normalization/nDIS.md])
- **Weight Calculation**: Every simulated event weight is a composite: $W_{final} = W_{orig} \cdot \frac{1}{nDIS}$.
- **Target Splitting**: The pipeline generates a 50/50 split between `p+` (proton) and `n0` (neutron) targets based on `nDIS // 2`.
- **Generator Implementation**: `MuDISGenerator.cxx:94` extracts `1/(*mu)[12]` to calculate the final `DIS_multiplicity`.

## 2. Geometry & Operational Constraints

### DY Semantics ([reports/fairship_muon_dis_DY_semantics.md])
- **Parameter**: `-Y` or `dy` in `run_simScript.py`.
- **Default**: `6.0` (meters).
- **Function**: Defines the vacuum-tank height and geometry setup.
- **Workflow Link**: Flows through `geometry_config.py` and is converted using `* u.m`.

### Z-Window and Vertex Placement
- **Logic**: The `MuDISGenerator` picks an interaction vertex (`zmu`) along the muon trajectory between `startZ` and `endZ`.
- **Weighting**: The probability of picking a point is proportional to the local material density relative to the maximum density along the path (`prob2int = mat->GetDensity() / mparam[7]`).

## 3. Observable Schema ([reports/fairship_muon_dis_oracle_observable_schema.md])

The internal "Oracle" schema defines what can be safely extracted from simulation output:
- **`dis_tree_exists`**: Confirmed by `file.Get("DIS")`.
- **`n_DISParticles`**: Event-level length of the `DIS.DISParticles` array.
- **`n_SBT_hits`**: Count of hits in `DIS.muon_vetoPoints`.
- **`n_UBT_hits`**: Count of hits in `DIS.muon_UpstreamTaggerPoints`.

## 4. Unresolved Issues (Internal Gaps)
Based on `docs/wiki/open_questions.md`:
- **Truth Label**: There is no dedicated boolean for "DIS occurred" in the final tree. It must be inferred from `DISParticles` or `CrossSection`.
- **Classification**: "Front", "Side", and "Cavern" regions are not explicitly labeled in the core C++ code and likely represent analysis-level cuts on the vertex position.
- **DY Safe Range**: Documentation provides examples but no "safe" range for thesis-level simulations has been officially validated within the index.

## 5. Dependency Reference
- **Simulation Engine**: FairShip (Commit `98de16a5b264`).
- **Physics Engine**: Pythia6 (Integrated via `r.TPythia6()`).
- **I/O Engine**: ROOT (Utilizing `TClonesArray` and `TVectorD`).
- **Orchestration**: `aliBuild` / `LXPLUS` environment.
