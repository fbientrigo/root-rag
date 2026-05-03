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
- `PROVISIONAL`: must include at least one `SOURCE` or `TODO`.
- `UNRESOLVED`: must include `Next action:`.
- `SUPERSEDED`: must include `Superseded by:`.

Source format:
- `path/to/file.ext:start-end`
- Example: `macro/run_simScript.py:120-182`

Notes:
- Do not infer FairShip internals from memory.
- If evidence is missing, keep claim `UNRESOLVED` and include `NOT FOUND IN INDEX` in text.
