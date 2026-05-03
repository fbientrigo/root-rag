# Query Packs

Lightweight YAML convention for workflow tracing query packs.

## Top-Level Fields
- `pack_id`: unique pack identifier string.
- `rq`: research question or workflow objective string.
- `created`: ISO date (`YYYY-MM-DD`).
- `tags`: list of short labels for filtering/grouping.
- `queries`: list of query records.

## Per-Query Fields
- `id`: stable query identifier.
- `natural_language`: short human-readable query text.
- `bm25_tokens`: keyword token list for lexical retrieval.
- `expected_files`: optional file hints from known evidence; keep empty if unknown.
- `tier`: priority level (example: `mvp`, `extended`).
- `golden`: boolean flag for benchmark/golden-set usage.

## Authoring Rules
- Keep YAML compatible with Python `yaml.safe_load`.
- `bm25_tokens` should be keyword-based tokens, not full natural-language sentences.
- Do not invent file paths. Use `expected_files: []` until evidence confirms paths.
- Keep records deterministic and versionable.

## Notes
- This convention defines data shape only.
- Query execution logic is intentionally out of scope here.
