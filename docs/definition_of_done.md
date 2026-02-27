# definition_of_done

## Purpose
This checklist applies to every pull request.
A PR is not done when the code merely runs.
It is done when contracts, tests, provenance, and documentation remain trustworthy.

## Required checklist for every PR

### 1. Build and tests
- [ ] Unit tests pass locally or in CI
- [ ] New code paths have tests, or the PR explains why tests are not applicable
- [ ] Canonical question coverage was checked if retrieval behavior changed
- [ ] No existing command contract was broken without an explicit spec update

### 2. Grounding and hallucination control
- [ ] No new path allows technical answers without evidence
- [ ] Citations still include file path and line range
- [ ] Version metadata remains attached to evidence and answers
- [ ] Failure modes return warnings or refusal rather than fabricated certainty

### 3. Logging and observability
- [ ] Relevant logs were added or preserved for new retrieval / indexing behavior
- [ ] Logs include enough context to debug version, index id, and retrieval mode
- [ ] Errors are actionable and do not fail silently

### 4. Contracts and schema
- [ ] Changes to CLI behavior update `docs/spec/cli_contract.md`
- [ ] Changes to chunk structure update `docs/spec/index_schema.md`
- [ ] Changes to architecture or non-negotiables update `docs/GROUND_TRUTH.md`
- [ ] Major decision changes include a new or updated ADR

### 5. Versioning integrity
- [ ] The PR does not mix multiple ROOT versions unintentionally
- [ ] New indexes or manifests preserve `root_ref` and `resolved_commit`
- [ ] Tests or manual checks confirm the requested revision is respected

### 6. Documentation
- [ ] README or architecture docs were updated if usage changed
- [ ] New commands or flags are documented
- [ ] New limitations or out-of-scope behavior are documented honestly

### 7. Scope discipline
- [ ] The PR stays within the target branch scope
- [ ] Out-of-scope work was not slipped in without review
- [ ] Temporary hacks are labeled clearly, tracked, or rejected

## Additional checklist by branch

### For `mvp/bm25-lines`
- [ ] FTS search works on a pinned ROOT version
- [ ] Search results expose path:start-end
- [ ] Ask mode refuses unsupported claims

### For `feat/hybrid-openai-embeddings`
- [ ] Lexical retrieval remains available and tested
- [ ] Semantic retrieval is additive, not a replacement
- [ ] Embedding-provider failures degrade gracefully

### For `feat/fastapi-service`
- [ ] API responses match documented models
- [ ] HTTP error codes are stable and documented
- [ ] Health endpoint and version endpoint are covered

### For `feat/gpt-actions`
- [ ] Action payloads preserve evidence and warnings
- [ ] GPT-facing prompts do not override grounding rules
- [ ] The backend remains the source of truth

### For `feat/version-diff`
- [ ] Version comparisons are explicit and labeled
- [ ] Evidence from each version is clearly separated
- [ ] Cross-version tests exist

## PR review prompts
Reviewers should ask:
- Does this preserve grounded answering?
- Can I trace a claim back to a file and lines?
- Could this accidentally mix versions?
- Did the contract change without docs?
- Is the architecture becoming simpler or just weirder?

## Merge rule
A PR should not be merged if any unchecked item affects:
- evidence integrity
- version integrity
- user-facing contracts
- architecture decisions
