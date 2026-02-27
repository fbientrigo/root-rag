# ROOT-RAG — Evidence-Grounded Retrieval for CERN ROOT

ROOT-RAG is a versioned retrieval system for the CERN ROOT codebase and documentation.
It indexes C/C++ source, headers, and Doxygen comments and answers questions by returning
**verbatim evidence with file paths and line ranges**. No hallucinated APIs, no inferred
signatures, no silent version mixing.

The project is designed as a deterministic backbone that can later power a Custom GPT
via API Actions, while preserving strict grounding and citations.

---

## Goals

- Retrieve ROOT symbols, implementations, and docs by query.
- Return **exact snippets with `file_path:start_line-end_line`**.
- Support **multiple ROOT versions** via tags/branches/commits.
- Provide **CLI and API** interfaces.
- Prevent hallucinations through an evidence-first contract.

---

## Non-Goals (MVP)

- Full semantic parsing of C++ (tree-sitter later).
- Web scraping of all ROOT sites.
- GPU-heavy local embedding models.
- UI beyond CLI/API.
- Cross-version diff reasoning (future branch).

---

## Architecture Overview

Layers:

1. **Corpus Manager**
   - Fetch ROOT repo at tag/branch/commit
   - Produce manifest with resolved commit

2. **Indexer**
   - Extract C/C++ files + Doxygen comments
   - Chunk by symbol / block
   - Store line ranges and metadata

3. **Retrieval**
   - BM25 / SQLite FTS5 lexical search
   - Optional embeddings (OpenAI)
   - Hybrid ranking

4. **Interfaces**
   - CLI
   - FastAPI service
   - Future: GPT Actions

See: `docs/architecture.md`

---

## Quick Start (MVP BM25)

```bash
git clone https://github.com/<you>/root-rag
cd root-rag

# install deps
pip install -e .

# index ROOT version
root-rag index --tag v6-32-00

# ask a question
root-rag ask "Where is TTree::Draw defined?"
````

Output:

```
Evidence:
tree/tree/inc/TTree.h:210-245
tree/tree/src/TTree.cxx:1234-1291
```

---

## Retrieval Modes

| Mode            | Description                 |
| --------------- | --------------------------- |
| BM25            | lexical symbol-first search |
| Hybrid          | BM25 + embeddings           |
| Embeddings-only | future                      |

Configured via `configs/retrieval/*.yaml`.

---

## Versioned Corpus

Each index is bound to:

* `root_ref` (tag/branch/commit requested)
* `resolved_commit`
* `build_timestamp`
* `index_schema_version`

This prevents version drift and enables reproducibility.

---

## Citation Contract

Every answer must:

* include at least one evidence chunk
* include file path and line range
* never infer missing API elements
* state uncertainty if evidence incomplete

See: `docs/adr/0003-citation-contract.md`

---

## CLI

```
root-rag index --tag v6-32-00
root-rag ask "query"
root-rag grep "symbol"
root-rag versions
```

Full spec: `docs/spec/cli_contract.md`

---

## Project Structure

```
docs/
  architecture.md
  adr/
  spec/
src/root_rag/
scripts/
tests/
configs/
data/
```

---

## Roadmap

Branch-based evolution:

* `mvp/bm25-lines`
* `feat/hybrid-openai-embeddings`
* `feat/fastapi-service`
* `feat/gpt-actions`
* `feat/version-diff`

See: `docs/GROUND_TRUTH.md`

---

## Design Principles

* Evidence > generation
* Version > convenience
* Determinism > heuristics
* CLI first
* Contracts frozen via ADRs

---

## License

MIT — see `LICENSE`

---

## Status

Early MVP under active development.
Interfaces and schema may evolve via ADR process..
