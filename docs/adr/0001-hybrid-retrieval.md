# ADR 0001: Hybrid retrieval with lexical backbone

## Status
Accepted

## Context
The project must answer questions about CERN ROOT source code and documentation.
Queries will include:
- exact symbols such as `TTree::Draw`
- file and class names
- conceptual questions such as lazy evaluation or branch handling

Code retrieval has different needs from prose retrieval.
For C/C++ codebases, exact symbol matching and file-path relevance are critical.
Pure embedding retrieval is weak when a user asks for a precise method, macro, or header, especially in a large scientific codebase with repeated terminology.

The project also targets a notebook-class machine, so the MVP must stay lightweight and auditable.

## Decision
Use hybrid retrieval, but make lexical retrieval the backbone.

Rules:
- lexical retrieval is mandatory in all production retrieval paths
- semantic retrieval is optional and additive
- score fusion and reranking may promote semantic matches, but may not suppress exact symbol matches without explicit reason
- exact symbol and file-path signals receive strong positive weight in reranking

Initial lexical backend:
- SQLite FTS5

Initial semantic backend:
- remote embeddings via API and a local vector store, only after the lexical MVP is stable

## Consequences

### Positive
- strong performance on exact symbol queries
- simple MVP implementation
- low resource usage
- easier debugging because lexical hits are human-readable
- grounded behavior for code questions

### Negative
- concept-only questions may underperform before semantic retrieval is added
- retrieval logic is slightly more complex than a single-backend system
- score fusion requires tuning

### Operational
- every retrieval response should include diagnostics for which mode was used
- evaluation must include both symbol queries and concept queries

## Alternatives discarded

### Pure lexical retrieval only
Rejected because concept questions and paraphrases would suffer.

### Pure semantic retrieval only
Rejected because precise code lookup would become unreliable and harder to debug.

### LLM-only answering over raw files
Rejected because it invites hallucination, weak provenance, and high latency.

### Full AST / compiler-index-only retrieval
Rejected for MVP because integration cost is too high relative to immediate value.
This may be revisited later for a more advanced parser or symbol graph.
