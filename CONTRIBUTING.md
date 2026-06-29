# Contributing to root-rag

Thanks for helping improve root-rag. This guide covers the local development
workflow and the quality gates that CI enforces.

## Development setup

```bash
git clone https://github.com/fbientrigo/root-rag
cd root-rag
make install        # pip install -e ".[dev,s1]"
```

`[dev]` brings in pytest and ruff; `[s1]` adds the optional semantic-retrieval
stack (numpy, faiss-cpu, sentence-transformers).

## Code quality

The project standardizes on [Ruff](https://docs.astral.sh/ruff/) for both linting
and formatting. Configuration lives in `pyproject.toml` under `[tool.ruff]`.

```bash
make lint           # ruff check src scripts tests
make format         # ruff format + ruff check --fix
make format-check   # verify formatting without writing
make test           # pytest -q
make check          # lint + format-check + test (mirrors CI)
```

Optionally install the git pre-commit hooks so lint/format run automatically:

```bash
pip install pre-commit
pre-commit install
```

## Tests

- Run the suite with `make test` (or `pytest -q`).
- Integration tests that need pre-built indices (`data/indexes_fairship`, etc.)
  or benchmark artifacts **self-skip** on a bare checkout — see
  `tests/conftest.py`. Build the relevant index or run the benchmark tracks to
  exercise them locally.

## CI

Two workflows run on every pull request:

- **ci** (`.github/workflows/ci.yml`): `ruff check`, `ruff format --check`, and
  `pytest`.
- **benchmark-mode-alignment**: the B0/B1/S0 retrieval benchmark tracks (runs
  only when retrieval/evaluation/benchmark inputs change).

Please make sure `make check` passes before opening a PR.
