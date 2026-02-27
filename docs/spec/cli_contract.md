# cli_contract

## Purpose
This document defines the supported command-line interface for the ROOT RAG MVP and its near-term extensions.

## General rules
- Commands must print actionable error messages.
- Commands must return documented exit codes.
- Commands must not silently switch ROOT versions.
- Commands must not produce unsupported grounded answers without warnings.
- Human-readable output is required by default.
- A machine-readable JSON mode may be added, but must be explicit.

## Global options
These options may be supported across commands:
- `--config <path>`: path to config file
- `--root-ref <ref>`: target ROOT tag, branch, or commit
- `--index-id <id>`: explicit prebuilt index identifier
- `--verbose`: enable detailed logs
- `--json`: emit JSON output when supported

## Supported commands

### `root-rag fetch`
Fetch a ROOT corpus revision and write a manifest.

Example:
```bash
root-rag fetch --root-ref v6-32-00
root-rag fetch --root-ref master
root-rag fetch --root-ref 0123456789abcdef
```

Expected behavior:
- resolves the requested reference
- checks out or updates the local corpus cache
- writes manifest metadata

Primary outputs:
- corpus path
- root_ref
- resolved_commit
- manifest location

### `root-rag index`
Build an index for a selected ROOT corpus.

Example:
```bash
root-rag index --root-ref v6-32-00
root-rag index --root-ref v6-32-00 --config configs/retrieval/bm25_only.yaml
```

Expected behavior:
- loads the target corpus
- parses files into chunks
- builds lexical index
- optionally builds vector index based on configuration
- persists build metadata

Primary outputs:
- index id
- schema version
- chunk count
- retrieval mode(s) built

### `root-rag search`
Run a retrieval query and return ranked evidence, without forcing a natural-language answer.

Example:
```bash
root-rag search "TTree::Draw" --root-ref v6-32-00
root-rag search "lazy evaluation in RDataFrame" --root-ref v6-32-00
```

Expected behavior:
- performs retrieval against the selected version
- returns ranked evidence candidates
- includes file path, line range, symbol if known, score, and source type

### `root-rag ask`
Run retrieval and produce a grounded answer.

Example:
```bash
root-rag ask "Where is TTree::Draw declared?" --root-ref v6-32-00
root-rag ask "How does RDataFrame use lazy evaluation?" --root-ref v6-32-00
```

Expected behavior:
- retrieves evidence
- produces an answer only from evidence
- includes warnings when evidence is partial
- refuses unsupported claims when evidence is insufficient

### `root-rag grep`
Run an exact or near-exact text search over indexed chunks or raw files.

Example:
```bash
root-rag grep "TBranchElement" --root-ref v6-32-00
root-rag grep "#include \"TTree.h\"" --root-ref v6-32-00
```

Expected behavior:
- fast literal or regex-like lookup
- useful for developer debugging
- returns matching locations

### `root-rag versions`
List available corpora and indexes.

Example:
```bash
root-rag versions
```

Expected behavior:
- list known root_ref values
- list resolved commits
- list available index ids and build timestamps

## Suggested output structure for `search`
Human-readable:
```text
[1] tree/tree/inc/TTree.h:210-245  symbol=TTree::Draw  score=18.21
    virtual Long64_t Draw(...)

[2] tree/tree/src/TTree.cxx:1234-1291  symbol=TTree::Draw  score=17.02
    Long64_t TTree::Draw(...)
```

## Suggested output structure for `ask`
Human-readable:
```text
Answer:
TTree::Draw is declared in the TTree header and implemented in the corresponding source file for this ROOT revision.

Evidence:
- tree/tree/inc/TTree.h:210-245  symbol=TTree::Draw
- tree/tree/src/TTree.cxx:1234-1291  symbol=TTree::Draw

Warnings:
- Signature details may vary across ROOT versions; answer is scoped to v6-32-00 at commit <sha>.
```

## Exit codes
| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Generic runtime failure |
| `2` | Invalid command-line usage or argument validation error |
| `3` | Requested ROOT revision not found or could not be resolved |
| `4` | Index not found or not built for the requested version |
| `5` | Retrieval completed but no evidence matched the query |
| `6` | Answer generation refused due to insufficient evidence |
| `7` | Configuration error |
| `8` | External dependency failure, e.g. embeddings provider unavailable |

## Contract notes
- `search` is for evidence retrieval; it should not hallucinate summaries.
- `ask` may use an LLM, but only downstream of retrieval.
- `ask` must still return evidence and warnings.
- `grep` is a debugging and exact-match convenience tool, not a replacement for indexed retrieval.

## Forward compatibility
Future commands may be added, but existing commands and exit codes must not change meaning without updating this document and the relevant ADR/spec.
