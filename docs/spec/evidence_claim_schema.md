# Evidence Claim Schema (YAML)

Purpose: version-controlled, file-based claim records for workflow tracing. No database required.

## Status Values
- `CONFIRMED`: claim fully backed by cited root-rag evidence.
- `PROVISIONAL`: partial/indirect evidence exists, still incomplete.
- `UNRESOLVED`: evidence missing or conflicting.
- `SUPERSEDED`: replaced by newer claim record.

## Confidence Values
- `HIGH`
- `MEDIUM`
- `LOW`

## Required Fields
- `claim_id` (string): stable unique id, recommended pattern `domain.topic.short_name.vN`.
- `claim_text` (string): precise statement under evaluation.
- `status` (enum): `CONFIRMED | PROVISIONAL | UNRESOLVED | SUPERSEDED`.
- `confidence` (enum): `HIGH | MEDIUM | LOW`.
- `sources` (list): one or more source objects (empty list allowed only for `UNRESOLVED`).
- `supersedes` (string or null): prior claim id replaced by this claim.
- `added` (date string, `YYYY-MM-DD`): initial creation date.
- `reviewed` (date string, `YYYY-MM-DD` or null): most recent review date.
- `notes` (string): rationale, gaps, follow-up, or review context.

## Source Object Fields
Each item in `sources` must include:
- `file` (string): path returned by root-rag evidence output.
- `start_line` (integer >= 1)
- `end_line` (integer >= start_line)
- `root_rag_query` (string): exact query used.
- `root_rag_score` (number): score reported by root-rag for that hit.

## Operational Rules
- Claims about behavior must be grounded in `sources` from root-rag outputs.
- `CONFIRMED` should include enough sources to support claim type (e.g., call-order/data-flow typically needs multiple records).
- If evidence not found, use `status: UNRESOLVED`, `sources: []`, and note `NOT FOUND IN INDEX` in `notes`.
- Keep one YAML file per claim set or workflow area for simple git diff/review.

## Minimal YAML Template
```yaml
claim_id: fairship.muondis.example.v1
claim_text: "<statement>"
status: PROVISIONAL
confidence: LOW
sources: []
supersedes: null
added: 2026-04-27
reviewed: null
notes: "NOT FOUND IN INDEX"
```
