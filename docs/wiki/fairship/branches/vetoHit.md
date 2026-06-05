---
type: branch-note
subsystem: MuonDIS
claim_state: CONFIRMED code-local
evidence_type: code-local
runtime_validated: false
physics_validated: false
fairship_commit: 98de16a5b264
aliases:
  - SBT
  - extrapolation
---

# Branch: vetoHit

## What it is
`vetoHit` is a branch in the digitization and reconstruction output trees representing active signals in the Surrounding Background Tagger (SBT) or other veto systems.

## Where it appears
- **Producer**: `python/shipDigiReco.py`
- **Storage**: `ship.conestoga_digi.root` / `ship.conestoga_reco.root`
- **Consumer**: High-level analysis macros for track rejection.

## Producer / storage / consumer
During reconstruction, the `linkVetoOnTracks` method extrapolates fitted tracks to the veto volumes to check for coincident hits.

## Minimal code evidence
    # python/shipDigiReco.py:440
    def linkVetoOnTracks(sTree):
        for n in range(sTree.cbmsim.GetEntries()):
            # ...
            findVetoHitOnTrack(sTree, n)

## Interpretation
The `vetoHit` branch is the primary interface between low-level detector signals and high-level background rejection algorithms.

## Agent guidance
Verify if `vetoHit` is present in the `reco` tree. If missing, track-veto linking was likely skipped during the `DigiReco` run.

## Thesis use
Critical for the "Background Rejection" and "Veto Efficiency" chapters.

## Open questions
Does `vetoHit` include timing information for coincidence windows?
