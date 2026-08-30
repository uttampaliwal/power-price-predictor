import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import (
    CITIES,
    FEATURE_COLS,
    FEATURE_COLS_NO_WEATHER,
    MODEL_LIST,
    TARGET_COL,
    WEATHER_FEATURE_COLS,
)


class TestFeatureCols:
    def test_feature_cols_count(self):
        assert len(FEATURE_COLS) == 18

    def test_no_weather_excludes_weather_cols(self):
        for col in WEATHER_FEATURE_COLS:
            assert col not in FEATURE_COLS_NO_WEATHER

    def test_no_weather_subset_of_full(self):
        for col in FEATURE_COLS_NO_WEATHER:
            assert col in FEATURE_COLS

    def test_no_weather_length(self):
        assert len(FEATURE_COLS_NO_WEATHER) == len(FEATURE_COLS) - len(
            WEATHER_FEATURE_COLS
        )


class TestConstants:
    def test_target_col(self):
        assert TARGET_COL == "mcp_rs_per_mwh"

    def test_model_list_has_champion(self):
        assert "xgboost" in MODEL_LIST
        assert "lightgbm" in MODEL_LIST

    def test_cities_have_coords(self):
        for city, coords in CITIES.items():
            assert "lat" in coords
            assert "lon" in coords
            assert -90 <= coords["lat"] <= 90
            assert -180 <= coords["lon"] <= 180
