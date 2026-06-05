# Retrieval SOTA Proposals (Scaffold Only)

This folder contains proposal scaffolds only.

Hard constraints:
- All feature flags default to `false`.
- V0 baseline must exist before any proposal execution.
- No proposal can modify default EMV harness behavior.
- No proposal can modify qrels, review decisions, wiki claim status, workflow graph confirmation, or V0 freeze policy.
- Graphify is comparator/exploration artifact only, never source of truth.

Required report fields for any future experiment:
- baseline V0 metrics
- changed retrieval method
- qrel slice evaluated
- failure modes
- rollback path

See:
- `feature_flags.yaml`
- `proposals/contextual_retrieval.md`
- `proposals/hybrid_retrieval.md`
- `proposals/graphify_comparison.md`
