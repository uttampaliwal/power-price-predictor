import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from probabilistic import (
    AdaptiveConformal,
    ConformalizedQuantileRegression,
    QuantileForecaster,
    SplitConformal,
    compute_interval_metrics,
    evaluate_conformal,
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


class TestSplitConformal:
    def test_fit_predict(self):
        from sklearn.linear_model import LinearRegression

        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5)
        y = X @ np.array([1, 2, 0, -1, 0.5]) + np.random.randn(n) * 0.5

        model = LinearRegression()
        model.fit(X[:400], y[:400])

        scp = SplitConformal(alpha=0.10)
        scp.fit(model, X[400:450], y[400:450])
        result = scp.predict(model, X[450:])

        assert "y_pred" in result.columns
        assert "lower" in result.columns
        assert "upper" in result.columns
        assert len(result) == 50

    def test_coverage(self):
        from sklearn.linear_model import LinearRegression

        np.random.seed(42)
        n = 1000
        X = np.random.randn(n, 5)
        y = X @ np.array([1, 2, 0, -1, 0.5]) + np.random.randn(n) * 0.5

        model = LinearRegression()
        model.fit(X[:500], y[:500])

        scp = SplitConformal(alpha=0.10)
        scp.fit(model, X[500:700], y[500:700])
        result = scp.predict(model, X[700:])

        covered = (y[700:] >= result["lower"].values) & (
            y[700:] <= result["upper"].values
        )
        coverage = np.mean(covered)
        # Should be close to 90%
        assert 0.85 < coverage < 0.95

    def test_fit_required(self):
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        scp = SplitConformal()
        try:
            scp.predict(model, np.zeros((5, 3)))
            assert False, "Should raise ValueError"
        except ValueError:
            pass


class TestAdaptiveConformal:
    def test_fit_predict(self):
        from sklearn.linear_model import LinearRegression

        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5)
        y = X @ np.array([1, 2, 0, -1, 0.5]) + np.random.randn(n) * 0.5

        model = LinearRegression()
        model.fit(X[:400], y[:400])

        acp = AdaptiveConformal(alpha=0.10, window=200)
        acp.fit(model, X[400:450], y[400:450])
        result = acp.predict(model, X[450:])

        assert "y_pred" in result.columns
        assert "lower" in result.columns
        assert "upper" in result.columns
        assert len(result) == 50


class TestCQR:
    def test_fit_predict(self):
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5)
        y = X @ np.array([1, 2, 0, -1, 0.5]) + np.random.randn(n) * 0.5

        cqr = ConformalizedQuantileRegression(alpha=0.10)
        cqr.fit_quantiles(X[:400], y[:400], X[400:450], y[400:450])
        result = cqr.predict(X[450:])

        assert "y_pred" in result.columns
        assert "lower" in result.columns
        assert "upper" in result.columns
        assert len(result) == 50


class TestEvaluateConformal:
    def test_basic(self):
        np.random.seed(42)
        n = 100
        y_true = np.random.randn(n) * 100 + 500
        intervals = pd.DataFrame(
            {
                "y_pred": y_true + np.random.randn(n) * 5,
                "lower": y_true - 100,
                "upper": y_true + 100,
            }
        )
        result = evaluate_conformal(y_true, intervals, alpha=0.10)

        assert "PICP" in result
        assert "PINAW" in result
        assert "Winkler" in result
        assert "ACI" in result
        assert result["n_test"] == n

    def test_with_regimes(self):
        np.random.seed(42)
        n = 100
        y_true = np.random.randn(n) * 100 + 500
        intervals = pd.DataFrame(
            {
                "y_pred": y_true + np.random.randn(n) * 5,
                "lower": y_true - 100,
                "upper": y_true + 100,
            }
        )
        regimes = np.where(y_true > 500, "spike", "normal")
        result = evaluate_conformal(
            y_true, intervals, alpha=0.10, regime_labels=regimes
        )

        assert "regime_normal_PICP" in result
        assert "regime_spike_PICP" in result


class TestIntervalMetrics:
    def test_perfect_intervals(self):
        y_true = np.array([100, 200, 300, 400, 500], dtype=float)
        intervals = pd.DataFrame(
            {
                "P10": y_true - 10,
                "P50": y_true,
                "P90": y_true + 10,
            }
        )
        m = compute_interval_metrics(y_true, intervals)
        assert m["PICP"] == 100.0
        assert m["P50_RMSE"] == 0.0

    def test_wide_intervals(self):
        y_true = np.array([100, 200, 300], dtype=float)
        intervals = pd.DataFrame(
            {
                "P10": y_true - 1000,
                "P50": y_true,
                "P90": y_true + 1000,
            }
        )
        m = compute_interval_metrics(y_true, intervals)
        assert m["PICP"] == 100.0
        assert m["PINAW"] > 100  # very wide intervals
