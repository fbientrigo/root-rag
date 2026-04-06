# S1-v1 implementation note

Date: 2026-04-04

Scope:
- Implement an opt-in `S1-v1` semantic milestone without changing default lexical behavior.
- Keep `B0` and `B1` benchmark contracts unchanged.
- Reuse current chunk/corpus artifacts and retrieval backend contract.

Conservative decisions:
- `deep-research-report.md` is used as guidance, but current repo contracts remain authoritative.
- Existing lexical search and benchmark tracks stay untouched; `S1` is additive.
- Reuse existing `chunks.jsonl` / benchmark corpus row patterns instead of redesigning chunking.
- Build semantic artifacts beside existing index artifacts, not inside the lexical FTS path.
- Use deterministic local embeddings only, exact FAISS only, normalized vectors only.
- No reranker, no ANN, no vector DB service, no remote APIs.
- Symbol-like queries keep lexical dominance via lexical pinning and weighted rank fusion.

Planned work:
1. Add a minimal local embedding abstraction and deterministic embedding corpus text builder.
2. Add a semantic artifact build path: embeddings manifest + vectors + FAISS index.
3. Add an opt-in semantic backend and an opt-in hybrid backend with deterministic fusion.
4. Add `S1` benchmark support and comparison artifact generation.
5. Add focused tests and minimal docs updates.

Assumptions to document:
- A local embedding model dependency is acceptable as an optional install/runtime requirement for S1 only.
- Tests should avoid downloading a real model; they will use stub embedders or synthetic vectors.
- If local model dependencies are unavailable, lexical/B0/B1 behavior must still work unchanged.
