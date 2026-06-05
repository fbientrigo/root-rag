---
type: node
domain: fairship
topic: muon_dis
status: CONFIRMED
claim_ids: [CLM-002]
tags: [SoftParticles, schema, MuDIS, TVectorD]
---

# SoftParticles

## Role
The `SoftParticles` branch contains associated soft interaction products (pions, etc.) generated alongside the DIS event. These tracks are often low-momentum and are used to model the full hadronic final state.

## Confirmed Facts
- **Tree Name**: `DIS`
- **Data Type**: `TClonesArray` of `TVectorD`.
- **Index Mapping**:
    - `[0]`: **PDG Code** (did)
    - `[1-3]`: **Momentum** (PX, PY, PZ) in GeV/c.
    - `[4]`: **Energy** (GeV)
    - `[5-7]`: **Start Position** (X, Y, Z) in meters.
    - `[8]`: **Start Time** (ns).
- **Filtering**: `MuDISGenerator::ReadEvent` skips soft particles that appear after the DIS interaction point (`(*SoftPart)[7] > zmu`).

## Source Anchors
- `muonDIS/makeMuonDIS.py:290-292`: TVectorD initialization (9 elements).
- `shipgen/MuDISGenerator.cxx:212-222`: Filtering and injection logic.

## Links
- [[fairship/trees/DIS_tree]]
- [[fairship/generators/MuDISGenerator]]
- [[fairship/branches/DISParticles]]
