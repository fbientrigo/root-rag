# root-rag System Layers

This document formalizes the mapping of the root-rag repository into five distinct architectural layers. This mapping ensures clear separation of concerns and guides minimal refactor efforts.

## 1. Harness
**Responsibility:** Orchestration of data acquisition, indexing, benchmarking, and validation. It is the "glue" that runs the engine on specific corpora.

- **Current Files:**
  - `scripts/*.py` (e.g., `index_fairship.py`, `run_query_pack.py`)
  - `benchmark/*.yaml` (task definitions)
  - `query_packs/*.yaml` (query sets)
  - `scripts/audit_benchmark_failures.py`
- **Allowed Dependencies:** Engine, Database/State, UX (for reporting).
- **Forbidden Responsibilities:** Direct implementation of retrieval algorithms; storage of primary indexes.
- **Technical Debt:** Many scripts are "one-off" or have duplicated logic for profile handling. Lack of a formal Profile Registry makes it hard to add new data sources consistently.

## 2. Engine
**Responsibility:** Core RAG logic, including parsing, chunking, indexing, and retrieval algorithms (Lexical, Semantic, Forest).

- **Current Files:**
  - `src/root_rag/core/`
  - `src/root_rag/corpus/`
  - `src/root_rag/parser/`
  - `src/root_rag/index/`
  - `src/root_rag/retrieval/`
- **Allowed Dependencies:** Internal core modules only.
- **Forbidden Responsibilities:** Configuration of specific corpora (FairShip, ROOT); CLI argument parsing; writing final UX reports.
- **Technical Debt:** Retrieval Forest logic is slightly coupled with the FairShip profile. Chunk schema is well-defined but its enforcement is scattered.

## 3. UX
**Responsibility:** User interface (CLI) and reporting formats.

- **Current Files:**
  - `src/root_rag/cli.py`
  - `reports/*.md` (templates/logic for human-readable output)
  - `docs/HOW_TO_GUIDE.md` (user documentation)
- **Allowed Dependencies:** Engine, Harness (for status), Database/State.
- **Forbidden Responsibilities:** Implementing retrieval logic; managing corpus manifests.
- **Technical Debt:** The CLI handles profile selection with hardcoded strings in some places. Error messages for "Index not found" could be more descriptive.

## 4. Database/State
**Responsibility:** Persistence of indexes, manifests, chunks, and global project state.

- **Current Files:**
  - `data/` (indexed artifacts)
  - `boulder.json` (project state)
  - `artifacts/` (intermediate run data)
  - `IndexManifest` objects (JSON)
- **Allowed Dependencies:** None (it is a data layer).
- **Forbidden Responsibilities:** Any active logic or computation.
- **Technical Debt:** `boulder.json` is becoming a "kitchen sink" for project status. `IndexManifest` needs a stricter JSON schema.

## 5. LLM Wiki
**Responsibility:** Storage and verification of evidence-grounded knowledge, claim audits, and thesis-aligned documentation.

- **Current Files:**
  - `docs/wiki/`
  - `reports/*audit*`
  - `AGENTS.md` (mission and guardrails)
- **Allowed Dependencies:** Database/State (for evidence linkage).
- **Forbidden Responsibilities:** Claims of physics validation; claims of runtime execution.
- **Technical Debt:** Audit reports are currently manual Markdown tables. Transitioning to machine-readable JSON formats for claims would allow for automated verification (e.g., checking line-range validity).

---
**Verdict:** The architecture is fundamentally sound but exhibits "Harness Bloat" and "Wiki Fragmentation". Refactors should focus on formalizing the Harness-Engine contract (Profile Registry) and the Database-Wiki contract (JSON Claim Audits).
