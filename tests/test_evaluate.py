import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from evaluate import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_all_metrics,
    price_direction,
    evaluate_by_segment,
)


class TestRegressionMetrics:
    def test_perfect_predictions(self):
        y = np.array([100, 200, 300, 400, 500], dtype=float)
        m = compute_regression_metrics(y, y)
        assert m["RMSE"] == 0.0
        assert m["MAE"] == 0.0
        assert m["R2"] == 1.0
        assert m["WAPE"] == 0.0

    def test_known_values(self):
        y_true = np.array([100, 200, 300], dtype=float)
        y_pred = np.array([110, 190, 310], dtype=float)
        m = compute_regression_metrics(y_true, y_pred)
        assert m["RMSE"] > 0
        assert m["MAE"] > 0
        assert 0 < m["R2"] < 1
        assert m["WAPE"] > 0

    def test_handles_nan(self):
        y_true = np.array([100, 200, np.nan, 400], dtype=float)
        y_pred = np.array([110, 190, 300, 410], dtype=float)
        m = compute_regression_metrics(y_true, y_pred)
        assert not np.isnan(m["RMSE"])

    def test_empty_arrays(self):
        m = compute_regression_metrics(np.array([]), np.array([]))
        assert all(np.isnan(v) for v in m.values())


class TestClassificationMetrics:
    def test_price_direction(self):
        prices = np.array([100, 110, 105, 120, 90], dtype=float)
        d = price_direction(prices, threshold_pct=1.0)
        assert d[0] == 0
        assert d[1] == 1
        assert d[2] == 0
        assert d[3] == 1
        assert d[4] == 0

    def test_direction_single_class(self):
        y = np.array([100, 100, 100, 100], dtype=float)
        m = compute_classification_metrics(y, y)
        assert all(np.isnan(v) for v in m.values())


class TestAllMetrics:
    def test_returns_all_keys(self):
        y = np.array([100, 200, 300, 400, 500], dtype=float)
        m = compute_all_metrics(y, y)
        expected = {"RMSE", "MAE", "MAPE", "R2", "WAPE", "AUC_ROC", "F1"}
        assert set(m.keys()) == expected


class TestEvaluateBySegment:
    def test_by_season(self):
        df = pd.DataFrame({
            "mcp_rs_per_mwh": [100, 200, 300, 400],
            "season": ["summer", "summer", "winter", "winter"],
        })
        y_pred = np.array([110, 210, 290, 390], dtype=float)
        result = evaluate_by_segment(df, y_pred, "season")
        assert len(result) == 3
        assert "season=summer" in result["segment"].values
        assert "season=winter" in result["segment"].values
        assert "OVERALL" in result["segment"].values
