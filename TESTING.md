# TESTING.md

## Matrix

- **Unit tests:** pytest — `tests/` (3 files, 25 tests: `test_config.py`, `test_evaluate.py`, `test_probabilistic.py`).
- **Format/lint gate:** CI enforces `ruff format --check src/ tests/` + `ruff check src/ tests/` (stricter than laziness: run `ruff format src/ tests/` before committing).
- **CI matrix:** Python 3.10 / 3.11 / 3.12 (`.github/workflows/ci.yml` `test` job).

## Commands

```bash
pip install -e ".[dev]"        # dev extra = pytest + pytest-cov (pyproject.toml)
python -m pytest tests/ -v --tb=short         # matching CI
ruff format --check src/ tests/ && ruff check src/ tests/   # matching CI lint job
```

## Conventions & current focus

- Tests target **pure decision logic** — config invariants (`src/config.py`: target column, feature list, holdout date), metric definitions edge-cases (`src/evaluate.py`), and coverage sanity of `src/probabilistic.py` (SplitConformal / AdaptiveConformal / CQR / QuantileForecaster behavior across seeds).
- Seeds are set (`np.random.seed(42)`) in tests; keep them deterministic.
- Integration coverage today deliberately excludes the training scripts' artifact-writing behavior (expensive, disk-heavy) and the Streamlit app. If you touch:
  - `src/probabilistic.py` → run `tests/test_probabilistic.py` + `src/run_conformal.py` (smoke, small horizon) and compare against `results/conformal/conformal_comparison.csv`.
  - `src/models/*_model.py` → rerun one family end-to-end and confirm `predictions/<fam>/` + metrics CSVs stay compatible with the committed snapshots' schema.
  - `src/fetch_data.py` / `src/fetch_weather.py` → test against `data/sample_dam.csv` locally; never commit a data refresh silently.
  - `app.py` → run Streamlit manually; the Data Management page shells out to pipeline scripts, so break it → break daily ops.

## Coverage expectations

- Target: keep the committed `conformal_comparison.csv` reachable from code (change → regenerate → commit both, or don't change either).
- New modules under `src/` should come with at least one behavior test and one edge-case test.
- No hard coverage threshold yet (see ROADMAP.md).
