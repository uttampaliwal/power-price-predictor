# Contributing

## Workflow

1. Fork / branch from `main` (`feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `paper/<topic>`).
2. Keep changes small and scoped; one theme per commit (conventional style).
3. Validate before pushing:

```bash
pip install -e ".[dev]"
ruff format --check src/ tests/ && ruff check src/ tests/
python -m pytest tests/ -v
```

4. Open a PR that states: what changed, why, whether any metric affects committed CSVs, and which validation you ran.

## Standards

- **Determinism:** any new randomness gets `random_state=42` (matching `src/models/*` and `src/probabilistic.py`).
- **No silently-regenerating artifacts:** if a script rewrites files under `data/`, `models/`, `predictions/`, `results/`, or `powerbi_data/`, call it out in the PR explicitly.
- **Numbers come from files:** commits that change metric claims in README/blog/paper must include the regenerated CSV (or skip the claim change).
- No secrets in code; scrapers stay keyless. Never commit a real `.env`.
- Docstrings on new public functions; docstring claims must match behavior.

## Code style

ruff (format + lint) with the repo's `pyproject.toml` settings, line length 120. If CI fails `ruff format --check`, run `ruff format src/ tests/` locally.

## PR checklist

- [ ] tests pass locally (`python -m pytest tests/ -v`)
- [ ] ruff lint + format checks pass
- [ ] no secrets / no new keyless-dependency breakage
- [ ] metric-affecting changes regenerated their CSV snapshots
- [ ] docs updated if behavior/data meaning changed
- [ ] no incidental artifact churn in the diff
