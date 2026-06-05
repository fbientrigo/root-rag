# How to extend the FairShip LLMWiki

## Add a new atomic note
1. Place note under correct domain folder in `docs/wiki/fairship/` or `docs/wiki/thesis_risks/`.
2. Use sections: `Status`, `Summary`, `Claims`, `Evidence anchors`, `Connections`, `What this does NOT prove`.
3. Keep note focused on one stable concept.

## Add evidence anchors
- Cite `file:start-end` references.
- For missing evidence, write `NOT FOUND IN INDEX`.
- Keep micro-snippets short and optional.

## Update indexes and maps
- Add compact entry to [[indexes/NODE_REGISTRY]].
- Add/adjust edge entries in [[indexes/EDGE_REGISTRY]].
- If relationship-level change exists, update relevant map edge row.

## Avoid overclaiming
- Never convert code links into runtime truth automatically.
- Keep runtime-dependent edges as `RUNTIME_UNVALIDATED` until artifact-backed.
- Link new risk-relevant findings to thesis risk scaffold notes.
