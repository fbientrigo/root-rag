# FairShip Muon DIS Thesis Plan

## Executive Verdict
The root-rag project is currently in a high-fidelity documentation and retrieval state. While it successfully maps the FairShip Muon DIS codebase with code-local evidence, it lacks runtime and physics validation. It is suitable for architectural drafting and code-level research but not for production physics results.

## Current validated state after MuonDIS retrieval benchmark (2026-05-13)
- **fairship profile**: ACTIVE (master branch, commit 98de16a5b264). High-fidelity code retrieval.
- **project_docs profile**: ACTIVE (288 files, 485 chunks). Indexes local Wiki, reports, and architecture.
- **MuonDIS Smoke Benchmark**: 12/12 PASS (100% retrieval success).
- **Agent Failure Benchmark**: HARDENED 10/10 PASS (Run 002 - Auditable with line ranges).
- **Vertical Slice (Normalization)**: COMPLETED. End-to-end trace of weighting metadata.
- **Variable Pointer Map**: COMPLETED. Provides direct mapping between code symbols and Wiki rationales.
- **Detector Rationales**: SBT 45 MeV threshold CONFIRMED via external-doc (arXiv:2112.01487).
- **Schema Registry**: InMuon (14 fields), DISParticles, and CrossSection alignment verified.
- **Remaining Limitations**: FTS5 implicit AND decay in code profiles; bracket `[]` syntax errors in raw queries.
- **Next Recommended Development Block**: Digitization & Reconstruction Trace (Mapping hits to tracks).

## Why root-rag Matters for the Thesis
Muon Deep Inelastic Scattering (DIS) is a critical background for the SHiP experiment. The legacy nature of the FairShip framework (hybrid Python/C++/ROOT) creates a high barrier to entry. root-rag enables:
1. **Semantic Navigation**: Finding "how" and "where" parameters are defined across disparate macros and C++ classes.
2. **Evidence-Grounded Drafting**: Ensuring thesis descriptions of the simulation pipeline match the actual implementation.
3. **Reproducibility**: Query packs allow future students to trace the same evidence paths.

## What root-rag Should Become
A "Verification Hub" where code retrieval meets runtime validation. It should evolve from a documentation assistant to a test-and-verify agent that can run minimal FairShip environments to confirm hypotheses.

## Current Constraints
- **Environment**: No access to LXPLUS or local ALIBUILD/CVMFS runtime.
- **Verification**: Limited to static analysis of indices.
- **Scope**: Focused on `muonDIS` and `shipgen` modules.

## Minimal Architecture
- **Tier 1**: ROOT and FairShip Source Indices.
- **Tier 2**: `evidence/` records (Grep/Ask outputs).
- **Tier 3**: `docs/wiki/` (Human-readable nodes).
- **Tier 4**: `AGENTS.md` (Operational guardrails).

## First Artifacts to Build
1. **Vertical Slice (Normalization)**: [[docs/wiki/workflows/muon_dis_vertical_slice_normalization.md]]
2. **Schema Registry**: Finalized definitions of `InMuon`, `DISParticles`, and `CrossSection`.
3. **Normalization Guide**: [[docs/wiki/fairship/normalization/MuDIS_normalization_factors.md]]

## Evaluation and Benchmark Strategy
- **Retrieval Accuracy**: Can the agent find the exact line defining a geometry parameter?
- **Evidence Ratio**: Number of `CONFIRMED` claims vs. `PROVISIONAL` ones.
- **Hallucination Rate**: Monitored via `boulder.json` success criteria.

## Risks and Mitigations
- **Risk**: Hallucinated ROOT APIs.
  - **Mitigation**: Strict `citation_required` policy in `AGENTS.md`.
- **Risk**: Over-reliance on inferred rationales (e.g., SBT threshold).
  - **Mitigation**: Demote unverified rationales to `PROVISIONAL`.

## Do-Not-Do-Yet
- Do not attempt to run `run_simScript.py` without a validated LXPLUS preflight.
- Do not claim physics correctness of the 45 MeV threshold without a cited SHiP internal note.

## Next 2-Week Roadmap
- **Week 1**: Complete the Schema Registry for all Muon DIS branches.
- **Week 2**: Draft the "Normalization and Weighting" chapter of the thesis using only `CONFIRMED` evidence.

## Success Criteria
- [ ] `AGENTS.md` fully adopted by all sub-agents.
- [ ] Zero unverified claims in `docs/wiki/fairship`.
- [ ] Query packs created for all 10 audited nodes.
