# Roadmap

## Phase 1 — Foundation (current)

- [x] CI lint + tests on Python 3.10/3.11/3.12
- [x] Full baseline docs (AGENTS/ARCHITECTURE/CONTRIBUTING/TESTING/SECURITY/DEPLOYMENT/DEVELOPMENT/ROADMAP + LICENSE, .env.example, .editorconfig, .pre-commit-config.yaml, Dependabot)
- [x] Repo-level cleanup of dead/duplicated artifacts (root main.pdf, paper/main.spl, exploded submission docx, orphaned images/, broken .agents symlink)
- [x] Research write-ups consolidated under blog/

## Phase 2 — Reproducibility hardening

- [ ] Pin a lockfile (pip-compile/uv) and stop ignoring `*.lock` in `.gitignore`
- [ ] Dataset versioning: stamp raw CSV slices with checksums + record in `data/README.md` on every refresh
- [ ] Optional experiment registry check-in: lightweight manifest (JSON) per training run instead of MLflow server
- [ ] CI job that regenerates one small slice end-to-end to keep committed CSV schemas honest
- [ ] Adopt pip-audit / dependency audit in CI (`security` stage)

## Phase 3 — Science quality

- [ ] Fix or remove the random-forest artifact gap (retrain RF or drop it from comparisons)
- [ ] Resolve the `src/trading_simulation.py` vs `src/backtest.py` strategy/P&L divergence into one documented framework
- [ ] Reconcile `benchmark_metrics.csv` (LGBM 0.872 best) vs README crown (XGBoost 0.871) — pick the true champion basis
- [ ] Add ablation/contact studies for conformal methods on spike regimes (build from `conformal_results.json` regime split)
- [ ] Paper: regenerate `paper/main.pdf` from TeX source in-repo (drop stale builds), fix `ARXIV_SUBMISSION.md` phantom file references

## Phase 4 — Engineering maturity

- [ ] Merge `*_no_weather` trainer twins into one parameterized script (config-driven output dirs)
- [ ] Move LSTM/Prophet/TFT scaffolding to `src/experimental/` (or remove)
- [ ] Retire `src/api.py` (v1) — port anything still used into v2 and delete
- [ ] Split `app.py` into an `app/` package (pages/, components/, orchestration/)
- [ ] Add `functions`-level (cron) data-refresh workflow + docs refresh cadence
- [ ] Right-size committed outputs: move `predictions/*test_predictions.csv`, `results/conformal/*_intervals.csv`, `data/processed/*.parquet` to LFS/DVC or release assets; keep metric CSVs + JSON summaries
- [ ] Docker: compose the Streamlit dashboard as a second service (dev-only profile)
