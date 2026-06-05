---
type: node
domain: fairship
topic: muon_dis
status: CONFIRMED
claim_ids: [CLM-005]
tags: [MeanMaterialBudget, vertex, MuDIS, simulation]
---

# MeanMaterialBudget Usage

## Role
`MeanMaterialBudget` is a utility function used by `MuDISGenerator` to probabilistically determine the longitudinal position ($z$) of a Muon DIS interaction vertex based on the material density along the muon's trajectory.

## Implementation Details
1. **Initial Calculation**: `shipgen::MeanMaterialBudget(start, end, mparam)` is called to determine the total material budget and the maximum density (`mparam[7]`) along the trajectory between `mu_start` and `mu_end`.
2. **Probabilistic Sampling**: The generator uses a "rejection sampling" style loop:
    - A candidate $z$ is picked uniformly between `start` and `end`.
    - The local material density $\rho(z)$ is retrieved via `gGeoManager->FindNode(x, y, z)->GetVolume()->GetMaterial()->GetDensity()`.
    - The interaction probability is calculated as: $P_{int} = \rho(z) / \rho_{max}$.
    - If a random number $[0, 1)$ is less than $P_{int}$, the vertex is accepted.
3. **Outcome**: This ensures that DIS interactions are more likely to occur in dense materials (like the vessel walls or detector components) rather than in the vacuum.

## Source Anchors
- `shipgen/MuDISGenerator.cxx:117`: Call to `MeanMaterialBudget`.
- `shipgen/MuDISGenerator.cxx:129-142`: Loop and density-based rejection sampling logic.

## Links
- [[fairship/generators/MuDISGenerator]]
- [[fairship/geometry/TGeoManager]]
