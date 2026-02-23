# Contributing

## Setup

```bash
pip install -e ".[dev]"
```

## Code style

- Use [ruff](https://docs.astral.sh/ruff/) for linting: `ruff check src tests`
- Config: `pyproject.toml` → `[tool.ruff]`

## Tests

```bash
pytest tests/ -v
```

Tests require no GPU. Some tests skip if optional dependencies (e.g. `torch`) are missing.

## Submitting changes

1. Keep the scope of a change focused.
2. Ensure tests pass and ruff is clean.
3. Update the README or CHANGELOG if behavior or usage changes.
