# 04 Optional Tooling

Optional tools can help explore code shape, but root-rag evidence remains the source of truth for FairShip claims. Do not auto-install tools without explicit approval. Detect availability first and record tool versions.

## Detection Commands

```powershell
rg --version
```

```powershell
git --version
```

```powershell
python -c "import sys; print(sys.version)"
```

```powershell
python -c "import graphify; print(getattr(graphify, '__version__', 'unknown'))"
```

```powershell
ast-grep --version
```

```powershell
pyan3 --version
```

## Guardrails

- Graphify or graph tools are optional exploration aids.
- `ast-grep`, `ripgrep`, and `pyan3` are optional exploration aids.
- Never let optional tooling override root-rag evidence artifacts.
- Never promote optional-tool output to wiki `CONFIRMED` claims.
- If a tool is missing, document it as unavailable and continue with root-rag evidence.
- Do not add install commands here unless the repository adopts that tooling explicitly.
