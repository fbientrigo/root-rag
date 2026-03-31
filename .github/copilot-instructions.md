# Copilot Instructions for ROOT-RAG

## Project Overview

ROOT-RAG is a **version-aware, evidence-grounded retrieval system** for the CERN ROOT codebase. The core principle: **no hallucinations**—every technical claim must be backed by exact file paths and line ranges from a specific ROOT version.

**Current Focus**: ROOT 6.36.08 (anchored to [FairShip](https://github.com/ShipSoft/FairShip) master) with seed corpus covering heavily-used classes (TTree, TFile, TGeoManager, TVector3, etc.).

## Build, Test, and Lint

### Installation
```bash
pip install -e .              # Install in development mode
pip install -e ".[dev]"       # Install with dev dependencies (pytest, pytest-cov)
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_golden_queries.py

# Run specific test function
pytest tests/test_golden_queries.py::test_golden_query_ttree_fill

# Run with coverage
pytest --cov=src/root_rag --cov-report=html

# Verbose output
pytest -v

# Show print statements
pytest -s
```

**Note**: Tests use local artifacts directory (`artifacts/pytest_runtime/`, `artifacts/pytest_cache/`) instead of `/tmp`.

### CLI Commands
```bash
# Index ROOT 6.36.08 (uses seed corpus by default)
root-rag index --root-ref v6-36-08

# Search for evidence
root-rag ask "Where is TTree::Fill defined?"
root-rag grep "TGeoManager"

# List indexed versions
root-rag versions

# Fetch corpus without indexing
root-rag fetch --root-ref v6-36-08
```

### Scripts
```bash
# Extract ROOT usage from FairShip codebase
python scripts/extract_fairship_root_usage.py --fairship-path ../FairShip

# Create mock FairShip for testing
python scripts/create_mock_fairship.py

# Run retrieval benchmark
python scripts/run_retrieval_benchmark.py
```

## Architecture Overview

The system is organized in layers, each with **strict contracts** (see `docs/architecture.md`):

```
CLI/API Layer
    ↓
Response Layer (grounded summaries, evidence selection)
    ↓
Retrieval Layer (BM25 lexical + optional embeddings)
    ↓
Index Layer (SQLite FTS5 + metadata store)
    ↓
Parser/Chunker (C++ code → semantically meaningful chunks)
    ↓
Corpus Manager (fetch ROOT revision, resolve to commit SHA)
```

### Key Modules

- **`corpus/`**: Fetch and manifest ROOT revisions
  - `fetcher.py`: Git operations, version resolution
  - `manifest.py`: Corpus metadata with `root_ref`, `resolved_commit`, `fetched_at`

- **`parser/`**: Extract chunks from C++ code
  - `chunks.py`: Main chunking logic (sliding windows, line ranges)
  - `files.py`: File discovery and filtering
  - `seed_filter.py`: Seed corpus class/file filtering

- **`index/`**: Build and locate search indices
  - `builder.py`: Orchestrates chunking → JSONL → FTS5
  - `fts.py`: SQLite FTS5 index construction
  - `locator.py`: Resolve index by `root_ref` (picks latest) or explicit `index_id`
  - `schemas.py`: `Chunk` and `IndexManifest` Pydantic models

- **`retrieval/`**: Search and ranking
  - `lexical.py`: BM25-style lexical search via FTS5
  - `pipeline.py`: Query → ranked evidence candidates
  - `transformers.py`: Query preprocessing

- **`evaluation/`**: Retrieval quality metrics
  - `metrics.py`: Precision, recall, MRR calculations

- **`cli.py`**: Command-line interface (Click-based)

## Critical Conventions

### 1. Version Integrity (Non-Negotiable)
Every index, chunk, and response is tied to:
- `root_ref`: User-requested reference (tag/branch/commit)
- `resolved_commit`: Immutable commit SHA
- `created_at`: Index build timestamp

**Never** mix evidence from different ROOT versions in a single response unless explicitly requested.

### 2. Citation Contract (ADR 0003)
Every technical claim **must** include:
- `file_path`: Repo-relative POSIX path (e.g., `tree/tree/inc/TTree.h`)
- `start_line`, `end_line`: 1-indexed, inclusive line ranges
- `root_ref` and `resolved_commit`: Version metadata

If evidence is insufficient → return warnings, not fabricated answers.

### 3. Chunk Schema Invariants
From `src/root_rag/index/schemas.py`:
- Line ranges are **1-indexed and inclusive**
- `file_path` must be repo-relative with `/` separators (no `\`)
- `content` matches exact lines from `[start_line, end_line]`
- `chunk_id` is deterministic and stable
- `doc_origin` must be one of: `source_header`, `source_impl`, `doxygen_comment`, `reference_doc`, `tutorial_doc`

### 4. Index ID Naming Convention
Index directories follow the pattern:
```
{root_ref}__{commit[:12]}__{timestamp}
```
Example: `v6-36-08__9005eb7d69f1__20260328_143022`

When resolving by `root_ref`, `resolve_index()` picks the **latest** by `manifest.created_at`.

### 5. Fail-Closed Answering
When evidence is weak or missing:
- Return explicit warnings in the response
- State uncertainty clearly
- **Do not** invent file paths, line numbers, or API signatures

### 6. Lexical-First Retrieval
BM25/FTS5 is the **backbone** for code and symbol search. Embeddings are an optional enhancement, not a replacement.

## Documentation Hierarchy (Source of Truth)

When making changes, respect this precedence order:
1. **`docs/GROUND_TRUTH.md`**: Non-negotiables, project objectives
2. **`docs/adr/`**: Architecture Decision Records (ADR 0001, 0002, 0003)
3. **`docs/spec/`**: Contracts for CLI, index schema, etc.
4. **`docs/architecture.md`**: Module responsibilities and data flow
5. **Implementation**: Code follows docs, not the other way around

## Testing Practices

### Golden Query Coverage
`tests/test_golden_queries.py` contains canonical questions that **must** pass:
- `test_golden_query_ttree_fill`: Find `TTree::Fill` definition
- `test_golden_query_tgeo_classes`: Find geometry classes
- `test_golden_query_doxygen_extraction`: Extract Doxygen comments

When changing retrieval logic, verify golden queries still pass.

### Test Fixtures
From `tests/conftest.py`:
- `tmp_path`: Repo-local temp directory (`artifacts/pytest_runtime/`)
- `git_repo_fixture`: Minimal git repo with tags/branches
- `cpp_repo_fixture`: C++ repo with TTree header/impl and Doxygen comments

### Benchmark Metadata
`tests/test_benchmark_metadata.py` validates:
- All `configs/benchmark_qrels.jsonl` entries reference valid chunks
- All `configs/benchmark_queries.json` queries are well-formed

## Common Workflows

### Adding a New Retrieval Mode
1. Update `src/root_rag/retrieval/backends.py` with new backend class
2. Implement `RetrievalBackend` interface from `interfaces.py`
3. Add tests in `tests/test_retrieval_backend_contract.py`
4. Update `docs/adr/` if changing architecture decisions
5. Run golden query tests to ensure no regressions

### Expanding the Seed Corpus
1. Edit `configs/seed_corpus_root_636.yaml`
2. Add classes/files under appropriate tier (`tier1`, `tier2`, etc.)
3. Rebuild index: `root-rag index --root-ref v6-36-08`
4. Verify new evidence appears: `root-rag grep "<new-class>"`

### Adding a CLI Command
1. Add `@main.command()` in `src/root_rag/cli.py`
2. Update `docs/spec/cli_contract.md` with new command contract
3. Add smoke test in `tests/test_cli_smoke.py`
4. Update `README.md` CLI examples section

## Debugging Tips

### Index Resolution Issues
If `root-rag ask` fails with "Index not found":
```bash
# Check available indices
root-rag versions

# Manually inspect index manifests
ls -l data/indexes/*/index_manifest.json
cat data/indexes/v6-36-08__<commit>__<timestamp>/index_manifest.json
```

### Retrieval Quality Issues
Check retrieval diagnostics:
```python
# In retrieval/lexical.py, logging is verbose
# Look for:
logger.info(f"FTS5 query: {fts_query}")
logger.info(f"Retrieved {len(results)} candidates")
```

### Chunk Provenance Verification
```bash
# Find chunks for a specific file
sqlite3 data/indexes/<index-id>/index.db \
  "SELECT file_path, start_line, end_line FROM chunks WHERE file_path LIKE '%TTree.h%' LIMIT 5;"
```

## Seed Corpus Details

The MVP uses a **conservative subset** of ROOT classes from `configs/seed_corpus_root_636.yaml`:

**Tier 1** (heavily used by FairShip):
- I/O: `TTree`, `TFile`, `TBranch`
- Histograms: `TH1`, `TH1F`, `TH2`
- Physics: `TVector3`, `TLorentzVector`
- Geometry: `TGeoManager`, `TGeoVolume`, `TGeoNode`

Full ROOT 6.36 has ~2000+ headers; seed corpus covers **90%+ of FairShip usage patterns**.

## Out of Scope for MVP

- Full semantic parsing of C++ (tree-sitter planned for later)
- Multi-version diff reasoning
- GPU-heavy local embedding models
- Web UI (CLI and API only)
- Cross-version evidence mixing

## References

- **Quick Start**: `docs/QUICK_START.md`
- **Architecture**: `docs/architecture.md`
- **Definition of Done**: `docs/definition_of_done.md` (PR checklist)
- **Citation Contract**: `docs/adr/0003-citation-contract.md`
- **Support Matrix**: `configs/support_matrix.yaml` (ROOT version constraints)
