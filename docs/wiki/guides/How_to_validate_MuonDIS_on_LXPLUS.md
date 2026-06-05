# How to validate MuonDIS on LXPLUS

## Goal
Prepare and execute runtime validation without promoting unsupported claims.

## Preflight path
1. Read [[fairship/runtime/LXPLUS_preflight]].
2. Confirm DY semantics in [[fairship/config/DY_Yheight]].
3. Use print-only smoke planning via [[fairship/runtime/MuonBack_smoke]].

## Runtime evidence handling
- Save command logs and probe outputs.
- Update runtime notes first (`fairship/runtime/*`), then maps/registries.
- Do not claim LXPLUS success unless artifacts are explicit.

## Oracle guardrails
- Use [[fairship/oracle/Oracle_schema]] as candidate schema only.
- Keep wall/fiducial/geant4 channels separated:
  - [[fairship/geometry/analysis_fiducial_fail]]
  - [[fairship/geometry/wall_like_fail]]
  - [[fairship/geometry/geant4_volume_wall_hit]]

## Risk linkage
If runtime results affect thesis risk posture, add links and notes under `docs/wiki/thesis_risks/`.
