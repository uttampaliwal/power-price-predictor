"""Parity guards for the merged no-weather trainer twins (2026-11 refactor).

The former copy-twins (`src/models/<name>_no_weather_model.py`) were merged
into their base modules behind a `--no-weather` flag. These tests keep the
public contract the twins used to provide: output dirs, feature lists and
the `run(use_weather=...)` entry point, so api.py / benchmark.py /
backtest.py keep consuming both variants unchanged.
"""

import importlib.util
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, "models", f"{name}_model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# base module name -> no-weather artifact filename the old twin saved
_TWINS = {
    "lightgbm": "lightgbm_no_weather.txt",
    "xgboost": "xgboost_no_weather.json",
    "random_forest": "random_forest_no_weather.pkl",
    "ridge": "ridge_no_weather_pipeline.pkl",
}

WEATHER_FEATURES = {"delhi_apparent_temp", "mumbai_apparent_temp"}


def test_no_weather_output_dirs_match_old_twin_layout():
    for name, artifact in _TWINS.items():
        mod = _load(name)
        assert mod.MODELS_DIR.endswith(os.path.join("models", name))
        assert mod.MODELS_NO_WEATHER_DIR.endswith(os.path.join("models_no_weather", name))
        assert mod.PREDS_DIR.endswith(os.path.join("predictions", name))
        assert mod.PREDS_NO_WEATHER_DIR.endswith(os.path.join("predictions", f"{name}_no_weather"))
        # ridge appends the suffix before `_pipeline.pkl` (not after `.pkl`)
        expected = f"{name}_no_weather_pipeline.pkl" if name == "ridge" else artifact
        assert artifact == expected


def test_feature_split_is_exactly_the_weather_columns():
    for name, _artifact in _TWINS.items():
        mod = _load(name)
        assert set(mod.FEATURE_COLS) - set(mod.FEATURE_COLS_NO_WEATHER) == WEATHER_FEATURES
        assert len(mod.FEATURE_COLS) == len(mod.FEATURE_COLS_NO_WEATHER) + 2


def test_all_merged_modules_expose_parameterized_entrypoint():
    for name, _artifact in _TWINS.items():
        mod = _load(name)
        params = inspect.signature(mod.run).parameters
        assert "use_weather" in params, f"{name}_model.run() lost the ablation switch"
        assert params["use_weather"].default is True
