# Advanced Pipeline Details: Vertex Placement & Rotation

This document explores the deep technical implementation of vertex placement logic and coordinate transformations in the Muon DIS pipeline.

## 1. Vertex Placement Logic (`MuDISGenerator.cxx`)

The `MuDISGenerator` uses a rejection sampling method to pick a physical interaction point along the muon's trajectory, weighted by the material budget.

### The Algorithm
1.  **Define Bounds**: A trajectory is defined between `startZ` and `endZ`.
2.  **Material Scan**: `shipgen::MeanMaterialBudget(start, end, mparam)` is called to calculate the maximum density (`mparam[7]`) along the path.
3.  **Rejection Sampling**:
    ```cpp
    while (prob2int < gRandom->Uniform(0., 1.)) {
      zmu = gRandom->Uniform(start[2], end[2]);
      // ... calculate xmu, ymu at zmu ...
      TGeoNode* node = gGeoManager->FindNode(xmu, ymu, zmu);
      mat = node->GetVolume()->GetMaterial();
      prob2int = mat->GetDensity() / mparam[7]; // Relative probability
    }
    ```
### Implications
- **Bias**: Interactions are naturally pushed towards high-density materials (e.g., tungsten cores, magnet yokes, shielding).
- **Vacuum Interaction**: If the sampled point is in a vacuum (`density ~ 0`), `prob2int` will be near zero, making the point highly likely to be rejected.

## 2. Momentum Rotation (`makeMuonDIS.py`)

Pythia6 generates DIS events in a local frame where the collision occurs along a fixed axis. These must be rotated into the global SHiP reference frame.

### The `rotate` Function
```python
def rotate(px, py, pz, theta, phi):
    momentum = r.TVector3(px, py, pz)
    rotation = r.TRotation()
    rotation.RotateY(theta) 
    rotation.RotateZ(phi)
    rotated_momentum = rotation * momentum
    return rotated_momentum.X(), rotated_momentum.Y(), rotated_momentum.Z()
```
- **Theta ($\theta$)**: Calculated as `ACos(pz / p)` from the incoming muon.
- **Phi ($\phi$)**: Calculated as `ATan2(py, px)` from the incoming muon.
- **Transformation**: This aligns the Pythia "Z-axis" with the actual momentum vector of the background muon hitting the tagger.

## 3. Propagation & Metadata Overloading

The pipeline uses `MCTrack[1]` as a metadata anchor.

- **Cross-Section Propagation**: `MuDISGenerator` explicitly passes `cross_sec` as the `weight` parameter for the first daughter particle (`index == 0`) in the `AddTrack` call.
- **Soft Track Filtering**: 
  ```cpp
  if ((*SoftPart)[7] > zmu) { continue; }
  ```
  This ensures causality: soft interactions associated with the muon that occur *after* the DIS vertex are discarded, as the muon is assumed to have interacted at `zmu`.

## 4. Operational Metrics (Internal Logic)

| Step | Metric | Semantic Meaning |
|---|---|---|
| **Sampling** | `prob2int` | Material-weighted probability density. |
| **Causality** | `zmu` check | Filter for `SoftParticles`. |
| **Rotation** | `TRotation` | Mapping Pythia local frame to SHiP global frame. |
| **Weighting** | `mparam[0] * mparam[4]` | Material thickness weight ($g/cm^2$) applied to tracks. |

## Source Anchors
- `shipgen/MuDISGenerator.cxx:117-150`: Rejection sampling loop.
- `muonDIS/makeMuonDIS.py:49-61`: `rotate` function implementation.
- `shipgen/MuDISGenerator.cxx:193-195`: Metadata overloading for `MCTrack[1]`.
- `shipgen/MuDISGenerator.cxx:214-216`: Soft particle causality filter.
