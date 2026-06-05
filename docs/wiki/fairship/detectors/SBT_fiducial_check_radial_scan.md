---
type: node
domain: fairship
topic: veto
status: CONFIRMED
claim_ids: [CLM-007]
tags: [SBT, fiducial, radial-scan, TGeoManager]
---

# SBT Fiducial Check Radial Scan

## Role
The `fiducialCheck` method in `shipVeto.py` determines if a reconstructed vertex is inside the vacuum decay volume by performing a geometric scan of the surrounding vessel walls.

## Implementation (36-Step Scan)
1. **Initial Check**: The method first verifies if the vertex is inside one of the known `DecayVacuum_block` volumes.
2. **Radial Scan**: If inside, it performs a 36-step circular scan:
    - `nSteps = 36` ($\Delta \phi = 10^\circ$).
    - For each angle $\phi$, it sets a direction vector $(\sin\phi, \cos\phi, 0)$.
    - It uses the `TGeoManager` navigator (`FindNextBoundaryAndStep()`) to find the distance to the next boundary (the vessel wall).
3. **Longitudinal Check**: It also checks the distance to the upstream Straw Veto (`/Veto_5`) and the downstream Tracking Station (`/Tr1_1`).
4. **Output**: The method returns the minimum distance to any boundary (`distmin`).

## Source Anchors
- `python/shipVeto.py:111-161`: Full implementation of the `fiducialCheck` method.

## Links
- [[fairship/detectors/SBT]]
- [[fairship/geometry/TGeoManager]]
