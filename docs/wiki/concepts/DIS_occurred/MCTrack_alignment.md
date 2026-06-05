# MCTrack Alignment for Muon DIS

## Track Hierarchy and Code Calls
When running with `--MuDIS`, the `MCTrack` tree is populated by the `FairPrimaryGenerator* cpg` object in `MuDISGenerator::ReadEvent`.

### 1. The Incoming Muon (`MCTrack[0]`)
- **Code Call**: `cpg->AddTrack(static_cast<int>((*mu)[0]), ..., t_DIS, w * DIS_multiplicity)`
- **MotherId**: -1 (Primary).
- **Weight**: $W_{muon} \cdot \frac{1}{nDIS}$ (calculated as `w * DIS_multiplicity`).
- **Time**: `t_DIS` (Calculated using the distance from the original hit to the interaction vertex).

### 2. The DIS "Anchor" Particle (`MCTrack[1]`)
- **Code Call**: `cpg->AddTrack(static_cast<int>((*Part)[0]), ..., t_DIS, cross_sec)`
- **MotherId**: 0 (Muon).
- **Weight**: `cross_sec` (extracted from `InMuon[10]`).
- **Note**: This is the first entry (`index == 0`) in the `dPart` loop. Its weight stores the DIS cross-section.

### 3. Other DIS Particles (`MCTrack[2...N]`)
- **Code Call**: `cpg->AddTrack(..., t_DIS, w)`
- **MotherId**: 0 (Muon).
- **Weight**: `w` (Calculated as `mparam[0] * mparam[4]` - Average density $\times$ track length).

### 4. Soft Interaction Particles
- **Code Call**: `cpg->AddTrack(..., t_soft, w)`
- **Condition**: Only if the soft interaction point `(*SoftPart)[7]` is upstream of the DIS vertex `zmu`.
- **Note**: These tracks keep the original muon weight `w`.

## Verification Query (Python/ROOT)
```python
# Check the weight of the second track (index 1)
xsec = sTree.MCTrack[1].GetWeight() 
if xsec > 0:
    print(f"DIS Event occurred with cross-section: {xsec} mb")
```

## Caveats
- If multiple generators are used, the index `1` might not be the DIS anchor. Always check the generator type or the track's mother/process information.
- In modern FairShip versions, a dedicated `CrossSection` branch is often added to the `cbmsim` tree to avoid overloading `MCTrack` weights.

## Evidence Anchors
- `shipgen/MuDISGenerator.cxx:170-180`: Adding the muon track.
- `shipgen/MuDISGenerator.cxx:193-195`: Overloading `MCTrack[1]` weight with `cross_sec`.
