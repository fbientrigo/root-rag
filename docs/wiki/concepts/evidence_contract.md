# Evidence Contract

Claim status model for this wiki:

- `CONFIRMED`
  - Fully backed by required source anchors.
  - For function/class/macro behavior: include path and line range evidence.
- `PROVISIONAL`
  - Partial evidence exists or analysis is incomplete.
  - Must include TODO or source context for follow-up.
- `UNRESOLVED`
  - Evidence missing or conflicting.
  - Must include explicit `Next action:`.
  - Use `NOT FOUND IN INDEX` when retrieval evidence is absent.
- `SUPERSEDED`
  - Historical claim replaced by newer evidence.
  - Must include `Superseded by:`.

Non-negotiable grounding:
- Do not infer FairShip behavior from memory.
- Do not mark a claim as `CONFIRMED` without source anchors.
- Do not claim call-order or data-flow without sufficient evidence records.
