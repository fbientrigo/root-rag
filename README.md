# ROOT-RAG — Evidence-Grounded Retrieval for CERN ROOT & FairShip

ROOT-RAG is a **version-aware, zero-hallucination retrieval system** for CERN ROOT and FairShip codebases. Ask questions about ROOT APIs or FairShip code and get **exact answers with file paths and line ranges** - no hallucinated APIs, no inferred signatures, no version mixing.

**🎯 Perfect for:**
- Learning ROOT APIs with real examples
- Understanding FairShip detector implementations
- Finding usage patterns across ROOT + FairShip
- Exploring SOFIE operators for ML inference

**Current Status:** Production-ready with 4 operational indices (ROOT Tier 1, FairShip, SOFIE, Legacy)

[![Benchmark Mode Alignment](https://github.com/fbientrigo/root-rag/actions/workflows/benchmark_mode_alignment.yml/badge.svg)](https://github.com/fbientrigo/root-rag/actions/workflows/benchmark_mode_alignment.yml)

---

## Quick Start

```bash
# Install
git clone https://github.com/fbientrigo/root-rag
cd root-rag
pip install -e .

# Query existing indices (instant)
root-rag ask "Where is TTree::Fill defined?"
root-rag ask "How does FairShip use TGeoManager?"
root-rag grep "ROperator_Conv SOFIE"

# Build new index (one-time, ~45 seconds)
root-rag index --root-ref v6-36-08 --seed-corpus configs/tier1_corpus_root_636.yaml

# List available indices
root-rag versions
```

## Opt-in S1 Semantic Retrieval

`S1-v1` is additive and off by default. Lexical retrieval remains the default backbone.

```bash
# Install optional local semantic dependencies
pip install -e .[s1]

# Build semantic artifacts from an existing lexical index
root-rag build-semantic-index --root-ref v6-36-08

# Opt in at query time
root-rag search "detector geometry assembly" --root-ref v6-36-08 --retrieval-backend hybrid
root-rag search "TTree::Draw" --root-ref v6-36-08 --retrieval-backend hybrid
```

---

## What's Indexed

| Index | Files | Chunks | Content |
|-------|-------|--------|---------|
| **ROOT Tier 1** | 53 | 1,106 | 35 most-used ROOT classes (TTree, TGeoManager, TVector3, etc.) |
| **FairShip** | 163 | 386 | FairShip master branch C++ code (detectors, generators, framework) |
| **SOFIE** | 40 | 140 | ROOT ML inference operators (Conv, Pool, Relu, RModel, etc.) |
| **Total** | **275** | **2,251** | **16 MB storage, <200ms queries** |

---

## Key Features

### ✅ Zero Hallucinations
Every answer is backed by **actual code** from indexed files. No invented APIs, no imagined signatures. If ROOT-RAG doesn't find evidence, it says so.

### ✅ Version Integrity  
Every response tagged with ROOT version + commit SHA. No mixing ROOT 6.32 APIs with 6.36 APIs. Reproducible results.

### ✅ Cross-Codebase Search
Query ROOT + FairShip simultaneously. Find API definitions and real-world usage examples in one search.

### ✅ FairShip-Optimized
Corpus prioritizes ROOT classes used by FairShip (TTree, TGeoManager, TVector3, etc.) plus FairShip's own codebase.

### ✅ SOFIE Ready
Includes ROOT's ML inference framework (39 operators) for future FairShip ML integration.

### ✅ Fast & Efficient
- **Query Speed:** <200ms for cross-index search
- **Storage:** 16 MB for all 4 indices
- **Indexing:** 10-45 seconds per corpus (one-time)

---

## Current Support Policy

**Anchor Project**: [ShipSoft/FairShip](https://github.com/ShipSoft/FairShip) (master branch)  
**ROOT Version**: 6.36.08 (commit: 9005eb7d69f1)  
**Python**: 3.10+ recommended  
**SOFIE**: Available but not yet used by FairShip (ready for future adoption)

---

## Use Cases & How-Tos

### 🔍 Use Case 1: Learn ROOT APIs

**Scenario:** You need to use TTree for data I/O but don't know the exact API.

```bash
# Find method signatures
$ root-rag ask "TTree::Fill"
→ tree/tree/inc/TTree.h:234-289 - Shows: virtual Int_t Fill()

# Find usage examples from FairShip
$ root-rag ask "TTree Fill Branch"
→ Returns: ROOT API definition + FairShip real-world usage

# Understand parameters
$ root-rag ask "TTree Branch"
→ tree/tree/inc/TTree.h:150-180 - Shows branching API
```

**Pro Tip:** Use keywords, not full sentences. "TTree Fill" works better than "How do I fill a TTree?"

👉 **[See Query Syntax Guide](docs/QUERY_SYNTAX_GUIDE.md)** for complete guide on writing effective queries.

---

### 🛠️ Use Case 2: Debug FairShip Detector Code

**Scenario:** Your detector isn't saving hits correctly.

```bash
# Find detector implementation patterns
$ root-rag ask "DetectorHit ProcessHits"
→ FairShip detector files showing ProcessHits implementations

# Find ROOT documentation for TBranch
$ root-rag ask "TBranch SetAddress"
→ ROOT API with proper usage patterns

# Cross-reference: How do other detectors do it?
$ root-rag ask "ecal ProcessHits save hits"
→ Shows ecal detector implementation as reference
```

---

### 🏗️ Use Case 3: Geometry Construction

**Scenario:** Build detector geometry using TGeoManager.

```bash
# Find ROOT geometry API
$ root-rag ask "TGeoManager MakeBox Material"
→ geom/geom/inc/TGeoManager.h - Shows shape creation API

# Find FairShip geometry examples
$ root-rag ask "TGeoManager AddNode FairShip"
→ FairShip geometry construction code

# Understand complete workflow
$ root-rag ask "TGeoVolume TGeoMedium muonShield"
→ Complete geometry workflow from API to implementation
```

---

### 🤖 Use Case 4: Explore SOFIE for ML

**Scenario:** Investigate using SOFIE for ML-based particle identification.

```bash
# Find available operators
$ root-rag ask "SOFIE ROperator Conv"
→ tmva/sofie/inc/TMVA/ROperator_Conv.hxx - CNN operator

# Understand RModel workflow
$ root-rag ask "RModel Generate ONNX"
→ RModel.hxx showing ONNX to C++ code generation

# Find operator list
$ root-rag grep "ROperator" --index sofie
→ Lists all 39 available SOFIE operators
```

**Note:** FairShip doesn't use SOFIE yet (as of 2026-04-01), but ROOT-RAG is ready for future adoption.

---

### 🔗 Use Case 5: Cross-Codebase Queries

**Scenario:** Understand how FairShip uses ROOT physics classes.

```bash
# Find TVector3 usage
$ root-rag ask "TVector3 momentum FairShip"
→ Returns: ROOT TVector3 API + FairShip physics calculations

# Find TLorentzVector patterns
$ root-rag ask "TLorentzVector energy mass"
→ ROOT API definitions + FairShip usage examples

# Understand detector hit geometry
$ root-rag ask "TGeoNode GetMedium detector"
→ Complete workflow: ROOT API → FairShip detector code
```

---

## CLI Commands

### `root-rag ask`

Ask natural questions, get evidence with citations.

```bash
# Basic query
root-rag ask "Where is TTree::Fill defined?"

# Cross-index search (searches ROOT + FairShip + SOFIE)
root-rag ask "TGeoManager MakeBox AddNode"

# Limit results
root-rag ask "TVector3 momentum" --top-k 5

# JSON output
root-rag ask "DetectorHit" --json > results.json
```

### `root-rag grep`

Fast keyword search (like grep but on indexed code).

```bash
# Search all indices
root-rag grep "ROperator_Conv"

# Search specific index
root-rag grep "TTree::Fill" --index tier1

# Limit results
root-rag grep "TGeoManager" --top-k 10
```

### `root-rag index`

Build new search indices.

```bash
# Index ROOT 6.36.08 with Tier 1 corpus
root-rag index --root-ref v6-36-08 \
  --seed-corpus configs/tier1_corpus_root_636.yaml \
  --output-dir data/indexes_tier1

# Index FairShip (requires local clone)
python scripts/index_fairship.py --fairship-path ../FairShip

# Index SOFIE
root-rag index --root-ref v6-36-08 \
  --seed-corpus configs/sofie_corpus_root_636.yaml \
  --output-dir data/indexes_sofie
```

### `root-rag versions`

List all available indices and their metadata.

```bash
root-rag versions

# Output:
# ROOT Tier 1 (v6-36-08): 1,106 chunks, 53 files
# FairShip (master): 386 chunks, 163 files  
# SOFIE (v6-36-08): 140 chunks, 40 files
```

---

## Architecture & Design

ROOT-RAG follows an evidence-first, deterministic design:

1. **Corpus Manager** - Fetch ROOT/FairShip at specific commits
2. **Indexer** - Chunk code into 80-line windows, build FTS5 search
3. **Retrieval** - BM25 lexical search with cross-index support
4. **CLI** - `index`, `ask`, `grep`, `versions` commands

**Key Principles:**
- Evidence > generation (no hallucinations)
- Version integrity (no version mixing)
- Determinism > heuristics (reproducible)
- Fail-closed (state uncertainty when evidence weak)

See [`docs/architecture.md`](docs/architecture.md) and [`docs/GROUND_TRUTH.md`](docs/GROUND_TRUTH.md) for details.

---

## Project Structure

```
root-rag/
├── src/root_rag/          # Core Python package
│   ├── cli.py            # CLI commands
│   ├── corpus/           # Git fetching, version resolution
│   ├── parser/           # File discovery, chunking
│   ├── index/            # FTS5 builder, manifest
│   └── retrieval/        # Search backends, cross-index
├── tests/                 # 125 tests (122 passing)
├── configs/              # Corpus definitions, golden queries
├── scripts/              # Indexing, extraction tools
├── docs/                 # Architecture, ADRs, guides
└── data/                 # Indices, corpora cache
    ├── indexes_tier1/    # ROOT Tier 1 index
    ├── indexes_fairship/ # FairShip index
    └── indexes_sofie/    # SOFIE index
```

---

## Current Status

**Version:** 0.2.0 (Post-SOFIE Fix)  
**Health:** Production-Ready ✅

### Completed Milestones

✅ **Test Infrastructure** (2026-03-31)
- Fixed failing tests, added SOFIE coverage
- 125 tests total (122 passing, 0 failures)

✅ **Tier 1 Corpus** (2026-03-31)
- Expanded to 35 ROOT classes (100% FairShip coverage)
- 1,106 chunks from 53 files

✅ **FairShip Indexing** (2026-03-31)
- Indexed FairShip master branch
- 386 chunks from 163 files

✅ **Cross-Index Search** (2026-03-31)
- Query ROOT + FairShip + SOFIE simultaneously
- 7 comprehensive tests

✅ **FairShip Golden Queries** (2026-03-31)
- 7 benchmark queries with validation
- Ensures retrieval quality

✅ **SOFIE Indexing** (2026-04-01)
- Fixed P5 (.hxx extension support)
- 140 chunks from 40 files (7x improvement)

### Quality Metrics

- **Tests:** 125 total (122 passing, 3 intentional skips, 0 failures)
- **Coverage:** 83% (production-ready for MVP)
- **Storage:** 16 MB (highly efficient)
- **Query Speed:** <200ms (cross-index)
- **Indexing Speed:** 10-45 seconds per corpus

### Capabilities

✅ Index ROOT 6.36.08 (Tier 1 corpus)  
✅ Index FairShip codebase  
✅ Index SOFIE operators (39 operators)  
✅ Lexical BM25 search (SQLite FTS5)  
✅ Cross-index search (multi-source)  
✅ Version-tagged evidence (no mixing)  
✅ Golden query benchmarks  
✅ CLI commands (index, ask, grep, versions)  

### Limitations (Intentional for MVP)

- **Single ROOT version:** 6.36.08 only (no multi-version yet)
- **No provider-backed embeddings yet:** lexical BM25 plus local `S0` semantic-hash baseline only
- **Curated corpus:** 35/2000+ ROOT classes (focused on FairShip usage)
- **CLI only:** No web UI or REST API yet

### Future Work (Low Priority)

⏳ Hybrid retrieval (BM25 + embeddings)  
⏳ Multi-version support (ROOT 6.32, 6.34, 6.36)  
⏳ FastAPI REST service  
⏳ SOFIE golden queries (when FairShip adopts SOFIE)  

---

## Documentation

- [**GROUND_TRUTH.md**](docs/GROUND_TRUTH.md) - Project principles and non-negotiables
- [**QUICK_QUERY_REFERENCE.md**](docs/QUICK_QUERY_REFERENCE.md) - TL;DR for writing queries
- [**QUERY_SYNTAX_GUIDE.md**](docs/QUERY_SYNTAX_GUIDE.md) - Complete guide to query syntax, stop words, and best practices
- [**architecture.md**](docs/architecture.md) - System architecture and data flow
- [**QUICK_START.md**](docs/QUICK_START.md) - Detailed setup and usage guide
- [**retrieval_quality_checks.md**](docs/retrieval_quality_checks.md) - Official B0/B1 benchmark and audit quality gates
- [**benchmark_mode_alignment.md**](docs/benchmark_mode_alignment.md) - End-to-end reproducible mode comparison workflow
- [**ADRs**](docs/adr/) - Architecture Decision Records (ADR 0001-0003)
- [**CLI Contract**](docs/spec/cli_contract.md) - Command specifications
- [**Citation Contract**](docs/adr/0003-citation-contract.md) - Evidence requirements

---

## Contributing

### Test Suite

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_golden_queries.py

# Run with coverage
pytest --cov=src/root_rag --cov-report=html

# Verbose output
pytest -v
```

### Code Style

- Follow existing patterns
- Add tests for new features
- Update documentation
- Run `pytest` before committing

### Benchmark Mode Alignment (B0/B1)

```bash
# Local deterministic mode alignment run
python scripts/run_benchmark_mode_tracks.py
```

Manual GitHub Actions run:
1. Open the repository Actions tab.
2. Select workflow `benchmark-mode-alignment`.
3. Click `Run workflow` (`workflow_dispatch`).
4. Download uploaded artifact `benchmark-mode-alignment` from the run summary.

### Adding a Corpus

1. Create corpus config in `configs/` (YAML)
2. Add golden queries for validation
3. Build index with `root-rag index`
4. Add tests in `tests/test_*_corpus.py`
5. Update README with new capabilities

---

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Query (single-index) | <100ms | Lexical BM25 search |
| Query (cross-index) | <200ms | 3 indices merged |
| Index build (Tier 1) | ~45s | 53 files, 1,106 chunks |
| Index build (FairShip) | ~30s | 163 files, 386 chunks |
| Index build (SOFIE) | ~10s | 40 files, 140 chunks |

**Storage:**
- Tier 1: 5.15 MB
- FairShip: 2.76 MB
- SOFIE: 0.74 MB
- **Total: ~16 MB** (very efficient)

---
