"""
probabilistic.py — Probabilistic Forecasting with Confidence Bands

Provides prediction intervals (P10/P50/P90) using:
1. Quantile Regression (XGBoost/LightGBM with quantile objective)
2. Conformal Prediction (distribution-free, model-agnostic)

Usage:
    from probabilistic import QuantileForecaster, ConformalForecaster

    qf = QuantileForecaster()
    intervals = qf.predict(X_train, y_train, X_test)

    cf = ConformalForecaster()
    intervals = cf.fit_predict(model, X_train, y_train, X_test)
"""

import numpy as np
import pandas as pd
from typing import Optional


# Quantiles for prediction intervals
QUANTILES = [0.10, 0.50, 0.90]


class QuantileForecaster:
    """
    Trains separate XGBoost models for each quantile.
    Produces P10/P50/P90 prediction intervals.
    """

    def __init__(self, quantiles=None, **xgb_kwargs):
        self.quantiles = quantiles or QUANTILES
        self.models = {}
        self.xgb_kwargs = {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "n_jobs": -1,
            "random_state": 42,
            **xgb_kwargs,
        }

    def fit(self, X: np.ndarray, y: np.ndarray, eval_pct: float = 0.15):
        """Fit quantile models. Uses last eval_pct of training data for early stopping."""
        import xgboost as xgb

        split_idx = int(len(X) * (1 - eval_pct))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        for q in self.quantiles:
            model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                **self.xgb_kwargs,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self.models[q] = model
        return self

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """Predict quantile intervals."""
        preds = {}
        for q, model in self.models.items():
            preds[f"P{int(q*100):02d}"] = model.predict(X)
        return pd.DataFrame(preds)


class ConformalForecaster:
    """
    Distribution-free prediction intervals via split conformal prediction.

    Uses any sklearn-compatible model + calibration data to produce
    guaranteed coverage intervals.
    """

    def __init__(self, alpha: float = 0.20):
        """
        Args:
            alpha: miscoverage rate (0.20 → 80% coverage, i.e., P10-P90)
        """
        self.alpha = alpha
        self.cal_scores = None
        self.quantile_levels = QUANTILES

    def fit(self, model, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Compute nonconformity scores on calibration set.

        Args:
            model: fitted sklearn-compatible model
            X_cal: calibration features
            y_cal: calibration targets
        """
        y_cal_pred = model.predict(X_cal)
        self.cal_scores = np.abs(y_cal - y_cal_pred)
        return self

    def predict(self, model, X: np.ndarray) -> pd.DataFrame:
        """
        Predict with conformal intervals.

        Uses standard split conformal prediction:
        - margin = quantile of calibration scores at (1 - alpha) level
        - P10 = y_pred - margin, P90 = y_pred + margin

        Args:
            model: fitted sklearn-compatible model
            X: test features

        Returns:
            DataFrame with P10, P50, P90 columns
        """
        if self.cal_scores is None:
            raise ValueError("Must call fit() first")

        y_pred = model.predict(X)

        # Standard conformal: margin = (1-alpha) quantile of calibration scores
        margin = np.quantile(self.cal_scores, 1 - self.alpha)

        results = {
            "P10": y_pred - margin,
            "P50": y_pred,
            "P90": y_pred + margin,
        }

        return pd.DataFrame(results)


def compute_interval_metrics(y_true: np.ndarray, intervals: pd.DataFrame) -> dict:
    """
    Compute evaluation metrics for prediction intervals.

    Returns:
        - P50 RMSE, MAE (point forecast quality)
        - PICP: Prediction Interval Coverage Probability (actual % within interval)
        - PINAW: Prediction Interval Normalized Average Width (narrower is better)
    """
    y_true = np.asarray(y_true, dtype=float)
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]

    p10 = intervals["P10"].values[mask]
    p50 = intervals["P50"].values[mask]
    p90 = intervals["P90"].values[mask]

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse = np.sqrt(mean_squared_error(y_true, p50))
    mae = mean_absolute_error(y_true, p50)

    # Coverage: what % of actuals fall within P10-P90?
    within = (y_true >= p10) & (y_true <= p90)
    picp = np.mean(within) * 100

    # Average width of interval (normalized by mean of actuals)
    width = p90 - p10
    pinaw = np.mean(width) / np.mean(np.abs(y_true)) * 100

    return {
        "P50_RMSE": round(rmse, 4),
        "P50_MAE": round(mae, 4),
        "PICP": round(picp, 2),
        "PINAW": round(pinaw, 2),
        "Interval": "P10-P90 (80%)",
    }
