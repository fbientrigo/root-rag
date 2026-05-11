# Wiki Claim Format

All claims in `docs/wiki/` use HTML comment markers.

Required status marker:
```markdown
<!-- CLAIM: CONFIRMED -->
Claim text here.
<!-- SOURCE: path/to/file.py:10-25 -->
```

Allowed statuses:
- `CONFIRMED`
- `PROVISIONAL`
- `UNRESOLVED`
- `SUPERSEDED`

Rules:
- `CONFIRMED`: must include at least one `SOURCE`.
- `CONFIRMED` workflow-relevance claims must include qrel/review evidence reference.
- `CONFIRMED` claims must have no unresolved contradiction against available evidence.
- `PROVISIONAL`: must include at least one `SOURCE` or `TODO`, and must not use confirmed language.
- `UNRESOLVED`: must include `Next action:`.
- `SUPERSEDED`: must include `Superseded by:`.

Source format:
- `path/to/file.ext:start-end`
- Example: `macro/run_simScript.py:120-182`

Notes:
- Do not infer FairShip internals from memory.
- If evidence is missing, keep claim `UNRESOLVED` and include `NOT FOUND IN INDEX` in text.
- Do not promote current wiki claims to `CONFIRMED` until qrels are manually approved/confirmed.
