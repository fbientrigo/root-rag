---
type: node
domain: fairship
topic: geometry
status: CONFIRMED
claim_ids: [CLM-003]
tags: [Chamber1, geometry, z-window, MuDIS]
---

# Chamber1 Alignment

## Role
`Chamber1` (the first tracking chamber) serves as a key geometric anchor for defining the longitudinal (Z) window for Muon Deep Inelastic Scattering (DIS) interaction generation in FairShip.

## Confirmed Facts
- **Z-Window Boundary**: The start of the MuDIS interaction window (`mu_start`) is defined relative to `Chamber1`.
- **Calculation**:
    - `mu_start = ship_geo.Chamber1.z - ship_geo.chambers.Tub1length - 10.0 * u.cm`
    - `mu_end = ship_geo.TrackStation1.z`
- **Context**: This window typically encompasses the region in front of the Upstream Veto Tagger (UVT) and extends to the first tracking station.

## Source Anchors
- `macro/run_simScript.py:549`: Calculation of `mu_start` and `mu_end` using `ship_geo.Chamber1.z`.

## Links
- [[fairship/generators/MuDISGenerator]]
- [[fairship/geometry/geometry_config]]
- [[fairship/detectors/TrackStation1]]
