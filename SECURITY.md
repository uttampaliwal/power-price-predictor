# Security

## Reporting

Email the maintainer directly (see the contact in `pyproject.toml` author info / GitHub profile @uttampaliwal) or open a private advisory via GitHub Security Advisories. Commands below are a maintainer-response outline; users should report rather than patch loudly.

## Threat model summary

Trusted environment assumed — this is a **single-operator research app**. Real attack surfaces:

| Surface | Risk | Current state |
|---|---|---|
| Keyed secrets | Leaked API keys | **None exist** — scraper (IEX) and weather (Open-Meteo) are keyless. Keep it that way unless keys are necessary; then load from env, never code. |
| Serving APIs | Open serving surface | `src/api_v2.py` (the deployed FastAPI) runs with CORS `allow_origins=["*"]`. Acceptable for local/single-host research; production deployments must restrict origins and add auth first. |
| Scraper | Remote-content parsing | `src/fetch_data.py` parses scraped HTML via `pd.read_html` with a spoofed browser UA. Handle upstream layout changes defensively; never eval scraped content. |
| Streamlit app | Command injection into UI | `app.py` Data Management shell-outs (`subprocess.Popen` to pipeline scripts with arg lists). Only safe because inputs are the repo's own script paths — do not make them user-configurable. |
| Subprocesses | Privilege/egress | Pipeline scripts write only inside the repo tree (paths from `src/config.py`). Don't widen. |
| Artifacts | Pickle loading | Loading trained artifacts (`joblib.load` of committed pickles) is intrinsic to this repo; only commit/load artifacts you trained locally. Note `*.pkl` is gitignored by default — the tracked ones predate the rule. |

## Rules for contributors

- Never commit credentials/tokens/keys — includes `.env` and any real email/personal data in paper submission materials.
- Keep `random_state`, file paths, and shell-outs inside `src/config.py` / repo tree.
- Dependency hygiene: CI runs the test job; there is no automated `pip-audit` gate yet (roadmap). Keep dependencies from `pyproject.toml`/`requirements.txt` synchronized.
- Stay current with the repo's disclosure norms: if you find a vulnerability, describe a minimal PoC, don't release exploit code publicly.

## Review checklist (use in PRs)

- [ ] no new key/token literal
- [ ] no new user-controlled path or shell input
- [ ] no new wildcard CORS without justification
- [ ] artifacts regenerated locally only
- [ ] `.env` not committed; `.gitignore` untouched (or changes documented)
