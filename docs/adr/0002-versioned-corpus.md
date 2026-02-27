# ADR 0002: Versioned corpus as a hard requirement

## Status
Accepted

## Context
ROOT changes over time.
A symbol, comment, implementation detail, or file path may differ between versions, branches, or commits.
The project must support:
- pinning retrieval to a specific ROOT revision
- later comparing behavior across versions
- reproducible debugging and evaluation

Without strict versioning, answers become ambiguous and evidence may be accidentally mixed across revisions.

## Decision
Treat the corpus as versioned and immutable per index build.

Rules:
- every index is tied to a user-supplied root_ref and a resolved immutable commit
- every chunk carries root_ref and resolved_commit
- retrieval requests must target one version unless multi-version mode is explicitly requested
- manifests are persisted for each corpus acquisition and index build
- caches must preserve version boundaries

## Consequences

### Positive
- reproducible answers
- easier debugging
- correct future support for version diff features
- prevents accidental cross-version contamination

### Negative
- more metadata to carry through the pipeline
- more storage because multiple indexes may coexist
- slightly more complex CLI and API contracts

### Operational
- version metadata must appear in answer payloads
- tests must verify that indexes do not mix revisions
- logs must include root_ref and resolved_commit for every retrieval

## Alternatives discarded

### Use only the latest checked-out branch
Rejected because it destroys reproducibility.

### Resolve only tags, not commits
Rejected because branches move and tags may not cover all use cases.

### Infer version from source comments or docs
Rejected because that is brittle and can be wrong.

### Store version only at the index level, not per chunk
Rejected because chunk-level provenance is necessary for integrity and future merges.
