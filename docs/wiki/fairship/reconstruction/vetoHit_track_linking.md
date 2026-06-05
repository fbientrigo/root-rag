---
type: node
domain: fairship
topic: reconstruction
status: CONFIRMED
claim_ids: [CLM-008]
tags: [vetoHit, extrapolation, Genfit, track-linking]
---

# VetoHit Track Linking

## Role
During reconstruction, fitted tracks are linked to activity in the Surrounding Background Tagger (SBT) to determine if a candidate signal event is accompanied by coincident veto hits.

## Process
1. **Extrapolation**: For each "good" track, the `shipDigiReco.py` script uses the Genfit track representation (`rep`) to extrapolate the track state to the position of each `vetoHit`.
2. **Distance Calculation**: The distance between the extrapolated track position and the actual `vetoHit` position is calculated:
   `dist = (rep.getPos(state) - vetoHitPos).Mag()`
3. **Best Match**: The `vetoHit` with the minimum distance (`distMin`) is identified as the best match.
4. **Association**: The result is saved as a `vetoHitOnTrack` object, containing the hit ID and the distance.

## Source Anchors
- `python/shipDigiReco.py:421-437`: Implementation of `findVetoHitOnTrack`.
- `python/shipDigiReco.py:439-444`: Implementation of `linkVetoOnTracks`.

## Links
- [[fairship/detectors/SBT]]
- [[fairship/reconstruction/Genfit]]
