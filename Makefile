.PHONY: help install lint format format-check test check

help:
	@echo "Targets:"
	@echo "  install       Install the package with dev + semantic extras (editable)"
	@echo "  lint          Run ruff lint checks"
	@echo "  format        Auto-format the code with ruff"
	@echo "  format-check  Verify formatting without writing changes"
	@echo "  test          Run the test suite"
	@echo "  check         Run lint, format-check, and tests (CI parity)"

install:
	python -m pip install -e ".[dev,s1]"

lint:
	ruff check src scripts tests

format:
	ruff format src scripts tests
	ruff check src scripts tests --fix

format-check:
	ruff format --check src scripts tests

test:
	pytest -q

check: lint format-check test
