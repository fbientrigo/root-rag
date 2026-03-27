# ROOT-RAG — Evidence-Grounded Retrieval for CERN ROOT

ROOT-RAG is a versioned retrieval system for the CERN ROOT codebase and documentation.
It indexes C/C++ source, headers, and Doxygen comments and answers questions by returning
**verbatim evidence with file paths and line ranges**. No hallucinated APIs, no inferred
signatures, no silent version mixing.

**Current Focus**: ROOT 6.36.08 (anchored to FairShip master) with seed corpus for auditable MVP.

The project is designed as a deterministic backbone that can later power a Custom GPT
via API Actions, while preserving strict grounding and citations.

---

## Current Support Policy

**Anchor Project**: [ShipSoft/FairShip](https://github.com/ShipSoft/FairShip) (master branch)  
**ROOT Version**: 6.36.08 (hard-pinned from [shipdist/root.sh](https://github.com/ShipSoft/shipdist/blob/main/root.sh))  
**C++ Standard**: C++20  
**CVMFS Releases**: 26.02, 26.03 (tested by FairShip CI)  
**SOFIE/ONNX**: Available in ROOT 6.36.08 (experimental), **not currently used by FairShip**

root-rag targets the ROOT version ecosystem required by FairShip master to ensure:
- Accurate API retrieval (no version mixing)
- FairShip-relevant corpus prioritization
- Future-proofing for SOFIE adoption

**Seed Corpus (MVP)**: Conservative subset of ROOT classes heavily used by FairShip:
- I/O: TTree, TFile, TBranch
- Histogramming: TH1, TH1F, TH2
- Physics Vectors: TVector3, TLorentzVector
- Geometry: TGeoManager, TGeoVolume, TGeoNode

See [`configs/support_matrix.yaml`](configs/support_matrix.yaml) for machine-readable constraints,
[`configs/seed_corpus_root_636.yaml`](configs/seed_corpus_root_636.yaml) for corpus scope,
and [`reports/fairship_root_sofie_audit.md`](reports/fairship_root_sofie_audit.md) for full audit.

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

## Quick Start (Retrieval MVP)

```bash
# Clone repository
git clone https://github.com/fbientrigo/root-rag
cd root-rag

# Install dependencies
pip install -e .

# Index ROOT 6.36.08 (uses seed corpus by default)
root-rag index

# Ask questions
root-rag ask "Where is TTree::Fill defined?"
root-rag grep "TGeoManager"
root-rag versions
```

## Extract ROOT Usage from FairShip (T1 Tool)

To derive Tier 1/Tier 2 corpus from evidence:

```bash
# Extract ROOT usage from local FairShip clone
python scripts/extract_fairship_root_usage.py --fairship-path ../FairShip

# Outputs:
#   - artifacts/fairship_root_usage_inventory.json (machine-readable)
#   - reports/fairship_root_usage_inventory.md (human-readable)

# See docs/QUICK_START.md for details
```

Expected output:

```
Evidence (ROOT v6-36-08, commit 1a2b3c4d5e6f):

[1] tree/tree/inc/TTree.h:234-289
    Symbol: TTree::Fill
[2] tree/tree/src/TTree.cxx:567-612
```

**Note**: First `index` command will clone ROOT 6.36.08 (~1GB) and build the search index (~5-10min).
Subsequent queries are instant.

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
root-rag index --root-ref v6-32-00
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

**Retrieval MVP**: ✅ Complete  
**T1 Extraction Tool**: ✅ Complete

**Completed**:
- ✅ FairShip ROOT/SOFIE/ONNX audit ([report](reports/fairship_root_sofie_audit.md))
- ✅ Support matrix config ([support_matrix.yaml](configs/support_matrix.yaml))
- ✅ Seed corpus definition ([seed_corpus_root_636.yaml](configs/seed_corpus_root_636.yaml))
- ✅ GitHub issue backlog (6 issues in `.github/ISSUE_TEMPLATE/`)
- ✅ Working CLI commands: `index`, `ask`, `grep`, `versions`
- ✅ SQLite FTS5 lexical retrieval
- ✅ Evidence-based output with version tagging
- ✅ Golden query test suite
- ✅ **T1: FairShip ROOT usage extraction** ([summary](reports/T1_implementation_summary.md), [guide](docs/QUICK_START.md))

**Current Capabilities**:
- Index ROOT 6.36.08 with FairShip-focused seed corpus
- Retrieve evidence with file:line citations
- Version-tagged responses (no version mixing)
- Zero-hallucination contract (evidence-only output)

**Limitations (Intentional for MVP)**:
- Seed corpus only (11 classes, ~25 files)
  - Full ROOT 6.36 has ~2000+ headers
  - Seed covers 90%+ of FairShip usage patterns
- Lexical search only (no embeddings yet)
- ROOT 6.36.08 only (no multi-version support yet)
- CLI only (no API server yet)

**Next Steps** (see issue templates):
- Expand to Tier 1 corpus (Issue #2: full FairShip API usage)
- Add SOFIE experimental docs (Issue #3)
- Version monitoring automation (Issue #1)
- Golden question expansion (Issue #6)

Interfaces and schema are stabilizing. Breaking changes will be announced via ADRs.
