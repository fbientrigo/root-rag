# Proposal: Contextual Retrieval

Status: scaffold only  
Feature flag: `retrieval_sota.contextual_retrieval_enabled` (default `false`)

Prerequisite:
- Frozen or at least coverage-ready Muon DIS V0 baseline.

Scope:
- Compare baseline retrieval against context-window augmentation strategy.

Must report:
- Baseline V0 metrics snapshot.
- Exact retrieval method delta.
- Exact qrel slice evaluated.
- Failure modes and regressions.
- Rollback path to baseline.

Non-goals:
- No qrel auto-promotion.
- No wiki/workflow confirmation changes.
- No change to default harness gates.
