# AGENTS.md

## Project Mission
root-rag is a thesis-support and agent-acceleration layer for FairShip Muon DIS development. It provides an evidence-grounded retrieval system for CERN ROOT and FairShip codebases, ensuring that agents and researchers navigate complex frameworks without hallucinations.

## Current Operating Mode
- **Status:** HARDENED Retrieval & Documentation.
- **Activity:** Auditing wiki nodes, aligning thesis plans, and enforcing evidence rules with mandatory line-range verification.
- **Prohibition:** No modifications to source code or retrieval architecture. No claims of runtime/physics validation.

## Claim States
- **CONFIRMED code-local:** Explicitly verified in the local index with file/line citations.
- **CONFIRMED external-doc:** Verified via official documentation (e.g., ROOT website).
- **PROVISIONAL:** Partially supported; lacks primary source or runtime confirmation.
- **UNRESOLVED:** Missing evidence.
- **CONTRADICTED:** Refuted by evidence.

## Evidence Rules
- **Citation Required**: All claims must include `file path` and `line range`.
- **Primary Source Verification**: Code-local PASS requires `root-rag show` verification.
- **No Generic CONFIRMED**: Use specific states (`code-local`, `project-local-docs`, `external-doc`).
- **Profile Separation**: Explicitly label source profile; Wiki snippets are not code-local evidence.
- **Source Priority**: Code-local retrieval > External documentation > Model memory (strictly for syntax only).
- **Reasoning is not Evidence**: LLM "logical inferences" about physics or geometry are PROVISIONAL until confirmed by code or docs.

## Query Guidance & Alias Policy
- **High-Signal Symbols**: Prefer 1–3 high-signal aliases (e.g., `nDIS`, `InMuon`) for FairShip FTS5/BM25 queries.
- **Avoid NLQ in Code**: Do not use long natural-language queries for `fairship` code-local retrieval; FTS5 implicit AND often causes 0-result failures.
- **Syntax Safety**: Avoid unescaped array-style syntax (e.g., `InMuon[0][10]`) in raw FTS5 queries to prevent syntax errors.
- **Label Evidence Source**: Always explicitly label which profile (`fairship` vs `project_docs`) provided the evidence.

## FairShip/ROOT Guardrails
- **No guessing**: If an API signature is not in the index, mark as UNRESOLVED.
- **Context Preservation**: Always distinguish between `muonDIS` (generation) and `run_simScript.py` (simulation).
- **TGeoManager Safety**: Geometry scans (like SBT fiducial checks) must be traced to specific `TGeoManager` calls.
- **MIP Interpretation**: Do not assume physics rationales for thresholds unless explicitly stated in code comments or linked papers.

## Preferred Agent Workflow
1. **Research**: Query the index for the specific symbol or logic.
2. **Audit**: Compare findings against existing wiki notes for contradictions.
3. **Document**: Update wiki/reports with high-signal, evidence-linked content.
4. **Validate**: Run JSON linting on any machine-readable artifacts.

## Do-Not-Do-Yet
- **NO LXPLUS Claims**: Do not claim code execution on LXPLUS.
- **NO Physics Claims**: Do not claim "validated" physics results.
- **NO Production Ready**: The project is in a research/documentation phase.

## When Codex Rescue is Needed
- When the index is missing core FairShip dependencies.
- When `root-rag show` output is ambiguous or lacks line numbers for critical claims.

## Output Expectations
- Surgical updates to documentation.
- Valid JSON for all schema/audit reports.
- Clear "Verdict" and "Remaining Uncertainty" in all responses.
