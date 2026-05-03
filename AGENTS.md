# AGENTS.md

## Mission
- root-rag evidence is only allowed source for claims about FairShip/ROOT code behavior.
- No claim may rely on model memory, intuition, or external summaries.

## Citation Contract
- Every claim about function/class/macro must cite: `file path` + `line range` from root-rag output.
- Every call-order or data-flow claim must cite at least two independent evidence records.
- If required evidence is missing, write exactly: `NOT FOUND IN INDEX`.
- Unverified claims must be written to unresolved log artifacts, never to wiki pages.

## Output Rules (Codex/LLM)
- Always separate claims into three sections: `CONFIRMED`, `PROVISIONAL`, `UNRESOLVED`.
- `CONFIRMED`: fully backed by required citations.
- `PROVISIONAL`: partial evidence present; missing pieces explicitly listed.
- `UNRESOLVED`: missing/conflicting evidence; include `NOT FOUND IN INDEX` where applicable.
- Do not infer FairShip behavior from model memory.
- Do not use web knowledge to fill FairShip code gaps.

## Repository Artifact Conventions
- `query_packs/`: versioned query definitions for workflow tracing tasks.
- `evidence/`: retrieval outputs and normalized evidence records used as citations.
- `reports/`: generated analysis outputs that reference evidence records.
- `benchmarks/`: golden queries, scoring configs, and evaluation results.
- `docs/wiki/`: curated pages allowed to contain only `CONFIRMED` claims.

## Enforcement
- Any claim without required evidence is invalid.
- Invalid or unverified content must be redirected to unresolved logs, not promoted to wiki artifacts.

## Codex EMV Harness
- For autonomous evidence-grounded Codex work, start with `agents/codex_emv/README.md`.
