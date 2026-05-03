# Heartbeat Protocol

## Definition

- A heartbeat is one autonomous Codex work cycle.

## Start Requirements

Every heartbeat must start by reading:

1. `AGENTS.md`
2. `agents/codex_emv/README.md`
3. `agents/codex_emv/heartbeat/current.md`
4. `agents/codex_emv/heartbeat/next_prompt.md` if present and non-empty

## Closure Semantics

- If the previous verdict was `ACCEPT`, `next_prompt.md` may be empty.
- If the previous verdict was `ACCEPT WITH NOTES`, `next_prompt.md` must contain a concrete continuation prompt.
- If the previous verdict was `BLOCKED`, `next_prompt.md` must contain a concrete unblock prompt.
- Codex must not claim closure while `next_prompt.md` contains unresolved operational work.

## File Roles

- `current.md`: current focus, accepted slice, and immediate operational need.
- `next_prompt.md`: operational prompt for the following heartbeat.
- `state.json`: minimal machine-readable heartbeat state.
- `history/`: versioned snapshots of heartbeat transitions.
