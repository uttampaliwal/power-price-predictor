import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from probabilistic import (
    QuantileForecaster,
    ConformalForecaster,
    compute_interval_metrics,
)


class TestQuantileForecaster:
    def test_fit_predict(self):
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5)
        y = X @ np.array([1, 2, 0, -1, 0.5]) + np.random.randn(n) * 0.5

        qf = QuantileForecaster(quantiles=[0.10, 0.50, 0.90])
        qf.fit(X, y)
        result = qf.predict(X)

        assert "P10" in result.columns
        assert "P50" in result.columns
        assert "P90" in result.columns
        assert len(result) == n
        # P10 should be <= P50 <= P90 on average
        assert result["P10"].mean() <= result["P50"].mean()
        assert result["P50"].mean() <= result["P90"].mean()


class TestConformalForecaster:
    def test_fit_predict(self):
        from sklearn.linear_model import LinearRegression

        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5)
        y = X @ np.array([1, 2, 0, -1, 0.5]) + np.random.randn(n) * 0.5

        model = LinearRegression()
        model.fit(X[:400], y[:400])

        cf = ConformalForecaster(alpha=0.20)
        cf.fit(model, X[400:450], y[400:450])
        result = cf.predict(model, X[450:])

        assert "P10" in result.columns
        assert "P50" in result.columns
        assert "P90" in result.columns
        assert len(result) == 50

    def test_fit_required(self):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        cf = ConformalForecaster()
        try:
            cf.predict(model, np.zeros((5, 3)))
            assert False, "Should raise ValueError"
        except ValueError:
            pass


class TestIntervalMetrics:
    def test_perfect_intervals(self):
        y_true = np.array([100, 200, 300, 400, 500], dtype=float)
        intervals = pd.DataFrame({
            "P10": y_true - 10,
            "P50": y_true,
            "P90": y_true + 10,
        })
        m = compute_interval_metrics(y_true, intervals)
        assert m["PICP"] == 100.0
        assert m["P50_RMSE"] == 0.0

    def test_wide_intervals(self):
        y_true = np.array([100, 200, 300], dtype=float)
        intervals = pd.DataFrame({
            "P10": y_true - 1000,
            "P50": y_true,
            "P90": y_true + 1000,
        })
        m = compute_interval_metrics(y_true, intervals)
        assert m["PICP"] == 100.0
        assert m["PINAW"] > 100  # very wide intervals
