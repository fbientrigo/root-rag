# architecture

## System overview
The system is a version-aware, evidence-first RAG pipeline for CERN ROOT.

### Textual architecture diagram

```text
                      +----------------------+
                      |   User / Developer   |
                      +----------+-----------+
                                 |
                                 v
                      +----------------------+
                      |   CLI or API Layer   |
                      | (search, ask, grep)  |
                      +----------+-----------+
                                 |
                                 v
                      +----------------------+
                      |   Response Layer     |
                      | evidence selection   |
                      | grounded summary     |
                      +----------+-----------+
                                 |
                                 v
                      +----------------------+
                      |  Retrieval Layer      |
                      | lexical / semantic    |
                      | fusion / rerank       |
                      +----+-------------+----+
                           |             |
                           v             v
               +----------------+   +------------------+
               | Lexical Index  |   | Vector Index     |
               | SQLite FTS5    |   | embeddings store |
               +--------+-------+   +---------+--------+
                        |                     |
                        +----------+----------+
                                   |
                                   v
                      +----------------------+
                      | Chunk Store          |
                      | content + metadata   |
                      +----------+-----------+
                                 |
                                 v
                      +----------------------+
                      | Parser / Chunker     |
                      | symbols, comments,   |
                      | semantic code blocks |
                      +----------+-----------+
                                 |
                                 v
                      +----------------------+
                      | Corpus Manager       |
                      | fetch ROOT revision  |
                      | resolve commit       |
                      | write manifest       |
                      +----------------------+
```

## Data flow

### 1. Corpus acquisition
Input:
- user-selected root_ref (tag, branch, or commit)
- project configuration

Output:
- checked out ROOT source tree
- manifest with root_ref, resolved_commit, timestamp, and source location

### 2. Parsing and chunking
Input:
- source tree from Corpus Manager
- file include / exclude rules

Output:
- normalized chunk records with file path, line ranges, content, symbol metadata, and provenance

### 3. Index building
Input:
- chunk records

Output:
- lexical index
- optional vector index
- persisted chunk metadata store

### 4. Retrieval
Input:
- query string
- retrieval mode
- target ROOT version or index id

Output:
- ranked evidence candidates with scores and metadata

### 5. Grounded answering
Input:
- ranked evidence candidates
- user question

Output:
- answer object containing summary, evidence list, warnings, and version metadata

## Contracts between modules

### Corpus Manager
Responsibilities:
- resolve branch, tag, or commit into an immutable revision
- fetch and cache the selected ROOT corpus
- generate a manifest

Inputs:
- root_ref
- repository URL
- local cache path
- optional checkout policy

Outputs:
- local corpus path
- manifest object:
  - root_ref
  - resolved_commit
  - fetched_at
  - repository_url

Must not:
- parse source files
- build retrieval indexes
- guess versions from file contents

### Parser / Chunker
Responsibilities:
- walk selected files
- create semantically meaningful chunks
- preserve line provenance
- associate symbols and comments when available

Inputs:
- corpus path
- manifest
- parser config
- include / exclude patterns

Outputs:
- iterable or persisted chunk objects following docs/spec/index_schema.md

Must not:
- perform retrieval
- summarize answers
- drop provenance fields

### Lexical Index
Responsibilities:
- support exact and fuzzy text search over chunks
- expose ranked lexical hits

Inputs:
- chunk objects
- lexical indexing config

Outputs:
- searchable index
- lexical search results with score and chunk_id

Must not:
- mutate chunks after indexing
- hide version metadata

### Vector Index
Responsibilities:
- store embedding vectors for chunks
- support semantic nearest-neighbor retrieval

Inputs:
- chunk objects
- embeddings provider output
- vector backend config

Outputs:
- semantic search results with score and chunk_id

Must not:
- replace lexical retrieval as the sole source of truth
- drop chunk_id linkage

### Retrieval Layer
Responsibilities:
- classify query shape
- run lexical and/or semantic retrieval
- merge and rerank candidates
- attach final evidence bundle

Inputs:
- user query
- index handles
- retrieval config
- optional version selector

Outputs:
- ordered list of evidence candidates
- retrieval diagnostics:
  - mode used
  - scores
  - warnings

Must not:
- fabricate unsupported evidence
- return results from mixed versions unless explicitly requested

### Response Layer
Responsibilities:
- convert evidence into final response payload
- ensure every technical claim has evidence
- produce fail-closed warnings when support is weak

Inputs:
- user question
- evidence bundle
- optional LLM adapter

Outputs:
- answer object:
  - answer
  - evidence[]
  - warnings[]
  - root_ref
  - resolved_commit

Must not:
- answer without evidence
- hide uncertainty
- change citations

### CLI Layer
Responsibilities:
- expose local commands for indexing and retrieval
- print human-readable results and exit codes

Inputs:
- command-line args
- config path
- environment variables

Outputs:
- terminal output
- process exit code

Must not:
- silently swallow retrieval errors
- silently choose a different ROOT version than requested

### API Layer
Responsibilities:
- expose HTTP endpoints for search and ask
- serialize request / response contracts
- return structured errors

Inputs:
- HTTP request payloads
- validated models
- query params

Outputs:
- JSON responses conforming to contract

Must not:
- change core retrieval semantics
- return undocumented fields as stable contract

## LLM placement
The LLM is optional and is downstream of retrieval.
It can:
- summarize evidence
- phrase answers
- compare snippets

It cannot:
- invent file paths
- invent line ranges
- substitute missing evidence
- rewrite architecture rules

## Source of truth precedence
1. GROUND_TRUTH.md
2. ADRs
3. specs
4. implementation
5. generated summaries
