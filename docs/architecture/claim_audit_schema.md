# Claim Audit JSON Schema (REF-002)

## Overview
The Claim Audit JSON Schema is a hardening step for the root-rag harness. It ensures that all claims extracted from the FairShip codebase or project documentation are grounded in specific, auditable evidence.

## Why this schema exists
Previously, audits suffered from:
- **Generic `CONFIRMED` states**: Obscuring whether evidence was code-local or from a paper.
- **Missing line ranges**: Making it difficult to verify claims against the indexed source.
- **Profile mixing**: Confusing project documentation with primary source code.
- **Hidden uncertainty**: Claims appearing more certain than they were due to lack of explicit "remaining uncertainty" fields.

## Schema Specification
The schema is located at `schemas/claim_audit.schema.json`.

### Required Fields for each Claim
- `claim_id`: Unique identifier (e.g., `CLM-SBT-001`).
- `claim_text`: The literal text of the claim.
- `claim_state`: Must be one of:
    - `CONFIRMED code-local`
    - `CONFIRMED external-doc`
    - `CONFIRMED project-local-docs`
    - `PROVISIONAL`
    - `UNRESOLVED`
    - `CONTRADICTED`
- `evidence_type`: `code-local`, `external-doc`, `project-local-docs`, `mixed`, `unsupported`.
- `profile`: The retrieval profile used (e.g., `fairship`, `project_docs`).
- `source`: File path or document identifier.
- `line_range_or_section`: Explicit line numbers or section identifiers.
- `runtime_validated`: Boolean (Must be `false` unless LXPLUS execution is confirmed).
- `physics_validated`: Boolean (Must be `false` unless SME review is documented).
- `thesis_safe_sentence`: A polished sentence for the final thesis.
- `remaining_uncertainty`: Explicit description of what is still unknown.

## Usage in Vertical Slices
When performing a new audit (e.g., for digitization or reconstruction), the agent must:
1. Generate a `*.claims.json` file.
2. Ensure it validates against `schemas/claim_audit.schema.json`.
3. Use the `scripts/validate_claim_audit.py` tool to confirm compliance.

## Examples

### Valid Claim
```json
{
  "claim_id": "CLM-SBT-001",
  "claim_text": "SBT 45 MeV Threshold",
  "claim_state": "CONFIRMED code-local",
  "evidence_type": "code-local",
  "profile": "fairship",
  "source": "python/detectors/SBTDetector.py",
  "line_range_or_section": "53",
  "runtime_validated": false,
  "physics_validated": false,
  "thesis_safe_sentence": "The Scintillating Beauty Tagger (SBT) implements a 45 MeV energy deposition threshold in the simulation layer.",
  "remaining_uncertainty": "Impact of secondary particles on threshold efficiency."
}
```

### Invalid Claim (Generic CONFIRMED)
```json
{
  "claim_id": "CLM-BAD-001",
  "claim_text": "Generic confirmation",
  "claim_state": "CONFIRMED", 
  "evidence_type": "code-local",
  ...
}
```
*Result: Validation Fail (Value not in enum).*
