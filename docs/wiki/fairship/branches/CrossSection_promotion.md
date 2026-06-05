---
type: node
domain: fairship
topic: muon_dis
status: CONFIRMED
claim_ids: [CLM-006]
tags: [CrossSection, branch, promotion, run_simScript]
---

# CrossSection Branch Promotion

## Role
The DIS interaction cross-section, initially stored in the `InMuon` branch of the input `DIS` tree, is "promoted" to a top-level branch in the simulation output tree to simplify downstream analysis.

## Process
1. `run_simScript.py` checks for the `--MuDIS` option.
2. It opens the output `cbmsim` tree and the input `DIS` tree.
3. It creates a new float branch named `CrossSection` in the output tree.
4. During the event loop, it extracts the value at `InMuon[0][10]` (the 11th element of the `TVectorD`).
5. This value is assigned to the `CrossSection` branch for each event.

## Source Anchors
- `macro/run_simScript.py:856`: Creation of the `CrossSection` branch.
- `macro/run_simScript.py:859-860`: Value extraction and assignment loop.

## Links
- [[fairship/branches/InMuon]]
- [[fairship/scripts/run_simScript]]
