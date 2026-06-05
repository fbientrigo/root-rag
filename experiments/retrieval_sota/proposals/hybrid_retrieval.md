# Proposal: Hybrid Retrieval

Status: scaffold only  
Feature flag: `retrieval_sota.hybrid_retrieval_enabled` (default `false`)

Prerequisite:
- Muon DIS V0 baseline exists and baseline metrics are recorded.

Scope:
- Evaluate lexical + alternate retriever blend against baseline.

Must report:
- Baseline V0 metrics.
- Hybrid method composition and ranking merge logic.
- qrel slice used.
- Failure modes (recall/precision drift, noise amplification, latency impact).
- Rollback path.

Non-goals:
- No source-of-truth changes before V0.
- No wiki claim promotion.
- No workflow graph confirmation changes.
