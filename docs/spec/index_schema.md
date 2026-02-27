# index_schema

## Purpose
This document defines the canonical chunk schema for indexed ROOT evidence.
All parsers, index builders, retrievers, and response layers must preserve these fields or derive them losslessly.

## Canonical chunk schema

### Required fields
| Field | Type | Description |
|---|---|---|
| `chunk_id` | string | Stable identifier for the chunk. Must be unique within an index build. |
| `root_ref` | string | User-requested ROOT reference used to build the corpus, such as a tag, branch, or commit. |
| `resolved_commit` | string | Immutable commit SHA resolved from `root_ref`. |
| `file_path` | string | Repository-relative path to the source file. |
| `language` | string | Language identifier, e.g. `cpp`, `c`, `text`, `markdown`. |
| `start_line` | integer | Inclusive start line in the original source file. |
| `end_line` | integer | Inclusive end line in the original source file. |
| `content` | string | Raw text content of the chunk. |
| `doc_origin` | string | Origin category, e.g. `source_header`, `source_impl`, `doxygen_comment`, `reference_doc`, `tutorial_doc`. |
| `index_schema_version` | string | Version of this schema used when the chunk was produced. |

### Strongly recommended fields
| Field | Type | Description |
|---|---|---|
| `symbol_path` | string or null | Canonical symbol path if known, e.g. `TTree::Draw`. |
| `symbol_kind` | string or null | Symbol kind, e.g. `class`, `method`, `function`, `namespace`, `macro`. |
| `keywords` | array[string] | Extracted keywords or aliases used to improve retrieval. |
| `has_doxygen` | boolean | Whether this chunk is associated with a Doxygen comment or block. |
| `imports` | array[string] | Nearby includes or referenced symbols if available. |
| `parser_name` | string | Name of parser strategy used, e.g. `ctags`, `regex`, `tree_sitter`. |
| `parser_version` | string | Parser version or build identifier. |
| `source_hash` | string | Hash of the exact chunk content for integrity checks. |
| `build_id` | string | Identifier for the index build that produced this chunk. |

## Example chunk
```json
{
  "chunk_id": "a8f8b9...",
  "root_ref": "v6-32-00",
  "resolved_commit": "0123456789abcdef0123456789abcdef01234567",
  "file_path": "tree/tree/inc/TTree.h",
  "language": "cpp",
  "start_line": 210,
  "end_line": 245,
  "content": "virtual Long64_t Draw(...);",
  "doc_origin": "source_header",
  "index_schema_version": "1.0.0",
  "symbol_path": "TTree::Draw",
  "symbol_kind": "method",
  "keywords": ["TTree", "Draw"],
  "has_doxygen": true,
  "imports": ["TString", "Option_t"],
  "parser_name": "ctags",
  "parser_version": "1",
  "source_hash": "....",
  "build_id": "root-v6-32-00-2026-02-27T23-00-00Z"
}
```

## Invariants
1. `start_line` and `end_line` refer to the original file on disk in the checked-out ROOT revision.
2. `start_line` >= 1.
3. `end_line` >= `start_line`.
4. `file_path` is repository-relative and must not be absolute.
5. `resolved_commit` must be a real immutable commit SHA for the selected corpus.
6. `root_ref` must preserve the user-facing reference that produced the corpus, even if it resolves to the same commit as another ref.
7. `content` must correspond exactly to the text spanning the chunk's provenance, subject only to normalization explicitly documented by the parser.
8. `chunk_id` must be stable for a given build strategy and input corpus, or the build must document why stability cannot be guaranteed.
9. `index_schema_version` must change when a backward-incompatible schema change is introduced.
10. `doc_origin` must be one of a documented finite set known to the parser and retriever.
11. A chunk may not mix text from multiple files.
12. A chunk may not mix text from multiple ROOT revisions.
13. If `symbol_path` is present, it must refer to the best-known symbol represented by the chunk and not to an unrelated neighboring symbol.

## Allowed normalization
The parser may:
- normalize line endings
- strip a trailing final newline for storage
- store UTF-8 normalized text

The parser may not:
- renumber lines
- merge non-contiguous file regions into one chunk
- rewrite identifiers
- remove provenance fields

## Compatibility rules
- Any module consuming chunks must ignore unknown future fields safely.
- Required fields are contractually stable unless changed through an ADR and schema version update.
