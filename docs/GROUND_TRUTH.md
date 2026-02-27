# GROUND_TRUTH

## Purpose
This document is the source of truth for the ROOT RAG MVP and its planned evolution.
Any code generation, refactor, or planning task must preserve this architecture unless an ADR updates it.
When in doubt, prefer the rules in this file over improvisation.

## Project objective
Build a version-aware RAG system for CERN ROOT that can:
- search code and documentation for a specific ROOT version, tag, branch, or commit
- return evidence anchored to file paths and line ranges
- answer symbol queries and concept queries without inventing APIs
- support a local CLI first, then a local API, then a Custom GPT / Project GPT integration through Actions

## Base plan summary
The project is built in layers:

1. Corpus Manager
   - fetch a specific ROOT revision from the official repository
   - resolve branch/tag to an immutable commit
   - persist a manifest for reproducibility

2. Parser and Chunker
   - walk C/C++ source and header files
   - extract semantically meaningful chunks
   - keep line ranges stable and attach symbol metadata when available
   - capture Doxygen comments when present

3. Indexer
   - build a lexical index first
   - optionally build a vector index for semantic retrieval
   - store enough metadata to recover exact evidence later

4. Retriever
   - use lexical retrieval as the backbone
   - optionally combine with embeddings for concept search
   - rerank using symbol exactness, docstring presence, and source type

5. Response Layer
   - return file path, start line, end line, symbol, snippet, version, commit
   - do not produce technical claims without evidence
   - fail explicitly when evidence is insufficient

6. Interfaces
   - CLI is the first-class interface
   - FastAPI service comes next
   - GPT Actions integration is layered on top of the API, never used as the system of record

## Non-negotiables
1. Versioned corpus is mandatory.
   - Every index must be tied to root_ref and resolved_commit.

2. Citations are mandatory.
   - Every technical answer must be backed by evidence with file path and line range.

3. No evidence, no claim.
   - If the system cannot find enough support, it must say so.

4. CLI before UI.
   - The first useful product is an auditable local CLI.

5. Lexical retrieval before semantic retrieval.
   - BM25 / FTS is the backbone for code and symbol search.
   - Embeddings are an optional enhancement, not the primary truth source.

6. Architecture changes require ADRs.
   - Do not silently rewrite contracts, schemas, or module responsibilities.

7. Index schema stability matters.
   - Chunks must preserve source provenance and line invariants.

8. Branch and version handling must be explicit.
   - The tool must never mix evidence across ROOT versions in one answer unless explicitly requested.

9. Logging is part of the product.
   - Retrieval decisions, selected corpus version, and failure modes must be inspectable.

10. Generated summaries must be grounded.
   - The LLM, when used, is only a formatter / summarizer over retrieved evidence.

## Repository modules (authoritative)
- src/root_rag/corpus
- src/root_rag/parser
- src/root_rag/index
- src/root_rag/retrieval
- src/root_rag/llm
- src/root_rag/api
- src/root_rag/cli

## Roadmap by branches

### Branch: mvp/bm25-lines
Goal:
- fetch one ROOT revision
- parse code files
- chunk code with line ranges
- build SQLite FTS5 lexical index
- provide CLI search and ask commands
- return evidence with path:start-end

Deliverables:
- fetch_root command or script
- manifest for selected ROOT revision
- chunk store
- FTS index
- CLI contract implemented
- smoke tests and canonical question subset

Out of scope:
- embeddings
- FastAPI
- GPT Actions
- web UI
- cross-version comparison
- auth
- distributed indexing
- fully accurate C++ AST parsing

### Branch: feat/hybrid-openai-embeddings
Goal:
- add semantic retrieval using remote embeddings
- merge lexical and semantic results
- rerank results with simple score fusion

Deliverables:
- embedding builder
- vector index backend
- retrieval mode selection
- eval updates showing recall improvement on concept queries

Out of scope:
- replacing lexical retrieval as the primary mode
- local GPU embedding pipeline
- advanced learned rerankers
- tool-calling orchestration
- full document QA over arbitrary internet sources

### Branch: feat/fastapi-service
Goal:
- expose the system through a local API

Deliverables:
- /health
- /versions
- /search
- /ask
- typed request / response models
- structured error codes

Out of scope:
- authentication / OAuth
- multi-user tenancy
- cloud deployment
- async distributed jobs
- web frontend

### Branch: feat/gpt-actions
Goal:
- connect the API to a Custom GPT / Project GPT using Actions

Deliverables:
- OpenAPI schema
- action-safe response payloads
- prompt / instruction set for grounded answering

Out of scope:
- replacing the backend retrieval engine
- uploading the whole corpus to GPT knowledge files as the primary architecture
- relaxing citation rules for convenience
- autonomous architecture changes made by the model

### Branch: feat/version-diff
Goal:
- compare evidence and symbols across two or more ROOT revisions

Deliverables:
- version selection contract
- diff-oriented retrieval
- response format that distinguishes version A vs version B evidence

Out of scope:
- semantic migration assistant
- automatic patch generation for user code
- broad release-note summarization across the entire project without grounding

## Allowed implementation freedom
The implementation may evolve in:
- parser internals
- chunk heuristics
- storage layout
- logging format
- reranking math
- test organization

Provided that it does not violate:
- the non-negotiables
- module contracts
- schema invariants
- ADR decisions

## Change control
Any proposal that changes:
- retrieval backbone
- index schema invariants
- versioning behavior
- citation behavior
- interface contracts

must update the relevant ADR and spec documents in the same change set.
