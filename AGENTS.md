# AGENTS.md

## Mission

This is a **reproducible forecasting research project** (Indian Energy Exchange Day-Ahead Market price forecasting with principled uncertainty via conformal prediction), structured for publication-grade credibility — not a throwaway script collection.

Golden rule: **every claimed number must regenerate from code you can point to** (train split: raw 2020–2024; pure holdout: 2025 onward, frozen in `src/config.py::HOLDOUT_START_DATE`). Never quote metrics from memory in docs/README/paper — copy them from committed CSVs (`predictions/benchmark_metrics.csv`, `results/conformal/conformal_comparison.csv`) or re-run the script and paste fresh output.

## Non-negotiable rules

1. No secrets, API keys, or credentials in code (none exist today — keep it that way). The IEX scraper and Open-Meteo fetcher are keyless; do not "helpfully" add a keyed path without a `src/config.py` + `.env.example` entry.
2. **Fixed seeds everywhere:** `random_state=42` is the convention in every learner (`src/models/*_model.py`) and bootstrap (`src/probabilistic.py`). New random sources get seeds too.
3. Do not mutate committed result artifacts (models/, predictions/, results/, data/) silently. Outputs are regenerable — say so, and list the exact command in the commit message.
4. Data refresh changes metrics: `src/fetch_data.py` appends into current-year raw CSVs. A refreshed dataset is a **different experiment** — never mix a data refresh with a metrics claim without regenerating everything downstream and updating the CSV snapshots.
5. Keep **exploratory research out of production code paths**: `src/api_v2.py` (serving) must not import experiment-only state; `app.py` (Streamlit) shells out to pipeline scripts — don't move pipeline logic into it.
6. Do not add new frameworks (Streamlit/FastAPI/sklearn/XGBoost/LightGBM/torch are set). Do not remove `mlflow`, even though `mlruns/` is not committed — experiment tracking is planned.
7. Keep changes minimal, atomic, and reversible; one theme per commit; conventional-style messages (`feat:`, `fix:`, `docs:`, `refactor:`, `data:`, `paper:`).
8. Tests accompany behavior or metric-affecting changes (`tests/` — pytest; the evaluation of conformal modules must keep passing: `tests/test_probabilistic.py`).

## Repo facts agents must know

- Two result narratives exist and must not be conflated: `src/trading_simulation.py` (results/conformal/trading_simulation.csv — Interval-Arbitrage strategy, Sharpe 19.28) vs `src/backtest.py` (predictions/backtest/ — Block-Shift strategy, Sharpe 56.67). README/paper quote the former. Check with the maintainer before touching the latter, adding a third framework, or "reconciling" them.
- `models_no_weather/` + `src/models/*_no_weather_model.py` are a deliberate **ablation axis** (remove weather features), not accidental duplication. A parameterized merge is planned (see ROADMAP.md) — ask before doing it.
- `src/models/lstm_model.py`, `prophet_model.py`, `tft_model.py` have **no trained artifacts on disk**. LSTM is in `MODEL_LIST` but excluded from app training (`app.py`); prophet/tft are dead experimental scaffolding. Do not advertise them as results.
- `src/api.py` (v1) is legacy; only `src/api_v2.py` ships (Dockerfile, README). Don't build on v1.
- `paths` are centralized in `src/config.py`. No env-var layer exists by design; if a knob must be configurable, extend `src/config.py` and document it, or introduce `os.environ` deliberately with an `.env.example` entry.
- Data policy: raw (data/raw/**) and processed (data/processed/**.parquet) datasets ARE committed (unlike the original claim in data/README.md, which has been corrected). Treat the committed raw CSVs as the canonical snapshot for the current paper numbers; heavy generated outputs stay committed to keep results checkable at clone time — do not "clean them up", that's a maintainer decision (see ROADMAP.md).
- Paper builds: LaTeX under `paper/`; CI does not build the paper. Do not commit PDFs outside `paper/` (root `main.pdf` duplicates were removed for a reason).

## Code standards

- Python >= 3.9, line length 120 (`ruff`); format with `ruff format`, lint with `ruff check` (CI enforces `ruff format --check`).
- Type hints appreciated but not enforced; docstrings on public functions — especially the probabilistic module where method semantics (SCP/ACP/CQR/EnbPI) must stay textbook-correct.
- Tests import via `sys.path.insert` into `src/` today; when adding tests, follow the existing pattern in `tests/` (pytest, deterministic seeds).

## Validation before completing anything

```bash
ruff format --check src/ tests/ && ruff check src/ tests/
python -m pytest tests/ -v
```

If you touched a model/conformal module: re-run the relevant script in report mode and confirm the committed CSVs still match (or regenerate them and flag the diff loudly).

## Escalation

Stop and get an explicit decision before: changing the holdout split date, modifying data/README.md claims, editing `src/backtest.py`'s strategy, deleting any committed artifact under data/, models/, predictions/, results/, or publishing/reworking numbers in README/blog/paper.
