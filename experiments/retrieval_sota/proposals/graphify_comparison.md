# Proposal: Graphify Comparator

Status: scaffold only  
Feature flag: `retrieval_sota.graphify_comparison_enabled` (default `false`)

Role:
- Comparator/exploration artifact only.
- Not source of truth for FairShip behavior.
- Not replacement for evidence-grounded qrel workflow.

Prerequisite:
- V0 baseline metrics available.

Comparison plan:
- Run baseline retrieval and Graphify-assisted retrieval on same qrel slice.
- Compare hit quality, coverage deltas, and failure clusters.
- Record where Graphify helps and where it introduces noise.

Must report:
- Baseline V0 metrics.
- Changed retrieval method definition.
- qrel slice evaluated.
- Failure modes.
- Rollback path.

Hard boundary:
- No direct promotion to qrels/wiki/workflow claims from Graphify output.
