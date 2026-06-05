# ROOT-RAG Workflow Wiki

This wiki is a local, evidence-constrained knowledge layer for FairShip Muon DIS workflow tracing.

What this wiki proves:
- Claims marked `CONFIRMED` with valid source anchors.
- Explicitly bounded statements tied to root-rag evidence artifacts.

What this wiki does not prove:
- Any FairShip behavior without source anchors.
- Any inferred call graph or data flow edge without cited evidence.

Navigation:
- [Claim Format](CLAIM_FORMAT.md)
- [Evidence Contract](concepts/evidence_contract.md)
- [Muon DIS Workflow Scaffold](workflows/muon_dis_pipeline.md)
- [MuonDIS Query Pack Benchmark](benchmarks/muondis_query_pack.md)
- [DIS Occurred Concept](concepts/DIS_occurred/INDEX.md)
- [Weekly Notes Policy](weekly/README.md)
- [Open Questions](open_questions.md)

Validation commands:
```bash
python scripts/lint_wiki_claims.py docs/wiki
python scripts/validate_workflow_graph.py workflow_graphs/muon_dis_workflow.json
```

Promotion rule:
- A claim becomes `CONFIRMED` only after adding one or more valid `SOURCE` anchors from root-rag evidence output.
- Claims lacking required evidence must remain `PROVISIONAL` or `UNRESOLVED`.
