# InMuon Branch Schema

## Definition
The `InMuon` branch is a `TClonesArray` of `TVectorD` (size 14) that stores the state of the incoming muon and metadata for the DIS interaction. It is the primary communication channel between the **Generation** (`makeMuonDIS.py`) and **Transport** (`MuDISGenerator.cxx`) stages.

## Data Layout (TVectorD(14))

| Index | Name | Unit | Description |
|---|---|---|---|
| 0 | `pid` | - | PDG ID of the incoming muon (+13 or -13). |
| 1 | `px` | GeV/c | Momentum X-component. |
| 2 | `py` | GeV/c | Momentum Y-component. |
| 3 | `pz` | GeV/c | Momentum Z-component. |
| 4 | `E` | GeV | Total Energy. |
| 5 | `x` | m | Interaction vertex X position (converted to cm in Generator). |
| 6 | `y` | m | Interaction vertex Y position (converted to cm in Generator). |
| 7 | `z` | m | Interaction vertex Z position (converted to cm in Generator). |
| 8 | `w` | - | Weight of the original background muon (normalised to spill). |
| 9 | `isProton`| bool| 1 for p+ (hydrogen/proton), 0 for n0 (neutron) target. |
| 10| `xsec` | mb | DIS total cross-section ($\sigma_{DIS}$) from Pythia `PARI(1)`. |
| 11| `time` | ns | Global time of the muon hit. |
| 12| `nDIS` | - | Multiplicity factor (inverse weight $1/nDIS$ applied in sim). |
| 13| `nmuons` | - | Number of muons in the original background event. |

## Implementation Details

### Creation (`makeMuonDIS.py`)
```python
mu = array("d", [pid, px, py, pz, E, x, y, z, w, isProton, xsec, time_muon, args.nDIS, nmuons])
muPart = r.TVectorD(14, mu)
iMuon[0] = muPart
```

### Consumption (`MuDISGenerator.cxx`)
```cpp
TVectorD* mu = dynamic_cast<TVectorD*>(iMuon->AddrAt(0));
Double_t cross_sec = (*mu)[10];
Double_t DIS_multiplicity = 1 / (*mu)[12];
```

## Related Logic
- **Time Shift**: The generator calculates `t_DIS = (t_muon + t_rmu) / 1e9` using `(*mu)[11]` and the distance to the picked interaction vertex.
- **Coordinate Conversion**: The generator multiplies indices 5, 6, and 7 by `100.0` to convert from meters (ntuple scale) to centimeters (FairRoot scale).

## Evidence Anchors
- `muonDIS/makeMuonDIS.py:211-230`: Schema definition in Python.
- `shipgen/MuDISGenerator.cxx:85-100`: Schema reading in C++.
