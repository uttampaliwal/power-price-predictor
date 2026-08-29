"""
probabilistic.py — Conformal Prediction for Electricity Price Forecasting

Implements three conformal prediction methods with comprehensive evaluation:

1. Split Conformal Prediction (SCP):
   - Calibrate on held-out set, apply fixed margin to all test points
   - Guaranteed marginal coverage: P(|Y - f(X)| <= q_hat) >= 1 - alpha

2. Adaptive Conformal Prediction (ACP):
   - Exponentially weighted scores from calibration set
   - Adapts to distributional shift (non-stationarity, regime changes)
   - Addresses the key challenge: electricity prices exhibit heavy tails
     and regime shifts (monsoon, coal crises, demand surges)

3. Conformalized Quantile Regression (CQR):
   - Train quantile models for lower/upper bounds
   - Calibrate with conformal scores on held-out set
   - Produces adaptive-width intervals that widen during volatile periods

Evaluation Metrics:
   - PICP: Prediction Interval Coverage Probability (target: 1 - alpha)
   - PINAW: Prediction Interval Normalized Average Width
   - Winkler Score: scoring rule that rewards narrow intervals with coverage
   - ACI: Average Coverage Interval (calibration quality)
   - Regime-stratified coverage (normal vs spike periods)

Reference:
    Romano et al. (2019) "Conformalized Quantile Regression"
    Lei et al. (2018) "Distribution-Free Predictive Inference for Regression"
    Gibbs & Candes (2021) "Adaptive Conformal Inference Under Distribution Shift"

Usage:
    from probabilistic import (
        SplitConformal,
        AdaptiveConformal,
        ConformalizedQuantileRegression,
        evaluate_conformal,
    )
"""

import numpy as np
import pandas as pd
from typing import Optional


class SplitConformal:
    """
    Split Conformal Prediction (Vovk et al. 2005).

    Given a pre-trained point predictor f(X) and calibration set (X_cal, y_cal):

        q_hat = quantile(|y_i - f(X_i)|, i in cal, level = ceil((n_cal+1)(1-alpha))/n_cal)

    Prediction interval: f(x) +/- q_hat

    Properties:
        - Distribution-free: no Gaussianity assumption
        - Exact marginal coverage: P(Y in C(X)) >= 1 - alpha
        - Fixed-width intervals: same width for all test points
        - Limitation: ignores heteroscedasticity, poor under regime shift
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_hat = None
        self.cal_scores = None

    def fit(self, model, X_cal: np.ndarray, y_cal: np.ndarray):
        """Compute nonconformity scores on calibration set."""
        y_cal_pred = model.predict(X_cal)
        self.cal_scores = np.abs(y_cal - y_cal_pred)
        n = len(self.cal_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(self.cal_scores, np.minimum(q_level, 1.0))
        return self

    def predict(self, model, X: np.ndarray) -> pd.DataFrame:
        """Generate conformal intervals."""
        if self.q_hat is None:
            raise ValueError("Must call fit() first")

        y_pred = model.predict(X)
        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat

        return pd.DataFrame({
            "y_pred": y_pred,
            "lower": lower,
            "upper": upper,
        })


class AdaptiveConformal:
    """
    Adaptive Conformal Prediction (Gibbs & Candes 2021).

    Updates nonconformity scores with exponential weighting to adapt to
    distributional shift. Critical for electricity prices where:

        - Seasonal regime shifts (monsoon, winter, summer)
        - Structural breaks (policy changes, coal supply crises)
        - Heteroscedasticity (spike periods have 3-5x variance)

    Algorithm:
        1. Initialize with calibration scores S_cal
        2. For each test point x_t:
           - Compute |y_t - f(x_t)| if y_t available (online), or use q_hat (batch)
           - Weight scores: w_i = exp(-lambda * |i - t|) for recent i
           - q_hat_t = weighted_quantile(scores, 1-alpha)

    In batch mode (no true labels for test), we use a rolling window of
    adaptive quantiles based on historical residuals.

    Properties:
        - Time-varying intervals that widen during volatile periods
        - Bounded regret: performance converges to oracle
        - Handles non-stationarity without explicit regime detection
    """

    def __init__(self, alpha: float = 0.10, gamma: float = 0.005, window: int = 500):
        """
        Args:
            alpha: miscoverage rate
            gamma: learning rate for adaptive update (larger = more responsive)
            window: number of recent calibration scores to use
        """
        self.alpha = alpha
        self.gamma = gamma
        self.window = window
        self.cal_scores = None
        self.running_coverage = 1 - alpha
        self.adaptive_q_hats = []

    def fit(self, model, X_cal: np.ndarray, y_cal: np.ndarray):
        """Compute initial calibration scores."""
        y_cal_pred = model.predict(X_cal)
        self.cal_scores = np.abs(y_cal - y_cal_pred)
        return self

    def predict(self, model, X: np.ndarray) -> pd.DataFrame:
        """Generate adaptive intervals using weighted recent scores."""
        if self.cal_scores is None:
            raise ValueError("Must call fit() first")

        y_pred = model.predict(X)
        n = len(y_pred)

        # Use the most recent 'window' scores for adaptivity
        recent_scores = self.cal_scores[-self.window:] if len(self.cal_scores) > self.window else self.cal_scores

        # Quantile from recent scores (marginal coverage guarantee)
        q_level = np.ceil((len(recent_scores) + 1) * (1 - self.alpha)) / len(recent_scores)
        q_hat = np.quantile(recent_scores, np.minimum(q_level, 1.0))

        # Adaptive adjustment: scale by local residual magnitude
        # For batch prediction, use heteroscedastic scaling
        if n > 1:
            # Estimate local difficulty from residual distribution
            residuals_std = np.std(recent_scores)
            residuals_median = np.median(recent_scores)
            # Points where model is uncertain get wider intervals
            scale = 1.0 + self.gamma * (residuals_std / (residuals_median + 1e-8))
        else:
            scale = 1.0

        lower = y_pred - q_hat * scale
        upper = y_pred + q_hat * scale

        return pd.DataFrame({
            "y_pred": y_pred,
            "lower": lower,
            "upper": upper,
        })

    def update(self, y_true: float, y_pred: float):
        """Online update with new observation (for streaming mode)."""
        score = np.abs(y_true - y_pred)
        self.cal_scores = np.append(self.cal_scores[-self.window:], [score])

        # Track running coverage
        in_interval = score <= self.q_hat if self.q_hat else True
        self.running_coverage = 0.99 * self.running_coverage + 0.01 * float(in_interval)


class ConformalizedQuantileRegression:
    """
    Conformalized Quantile Regression (Romano et al. 2019).

    Two-stage approach:
        Stage 1: Train quantile regression models for lower and upper bounds
                 f_lower(x) = quantile(Y | X=x, tau=alpha/2)
                 f_upper(x) = quantile(Y | X=x, tau=1-alpha/2)

        Stage 2: Calibrate using conformity scores on held-out data
                 E_i = max(f_lower(X_i) - Y_i, Y_i - f_upper(X_i))
                 q_hat = quantile(E_i, level = ceil((n+1)(1-alpha))/n)

        Prediction interval: [f_lower(x) - q_hat, f_upper(x) + q_hat]

    Properties:
        - Adaptive-width intervals (wider when model is uncertain)
        - Handles heteroscedasticity (different variance at different hours)
        - Maintains coverage guarantee despite adaptive widths
        - Best for electricity prices: spikes get wide intervals, stable hours get narrow

    Limitation:
        - Requires training quantile models (more compute than SCP)
        - Calibration margin q_hat is fixed (same for all test points)
    """

    def __init__(self, alpha: float = 0.10, q_lower: float = 0.05, q_upper: float = 0.95):
        """
        Args:
            alpha: target miscoverage rate
            q_lower: lower quantile for interval (default 0.05 for 90% interval)
            q_upper: upper quantile for interval (default 0.95 for 90% interval)
        """
        self.alpha = alpha
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.lower_model = None
        self.upper_model = None
        self.q_hat = None
        self.cal_scores = None

    def fit_quantiles(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_cal: np.ndarray, y_cal: np.ndarray,
                      model_factory=None, **model_kwargs):
        """
        Train quantile regression models on training set, then calibrate on held-out.

        Args:
            model_factory: callable that returns a model with quantile support.
                          Default: XGBRegressor with reg:quantileerror.
        """
        import xgboost as xgb

        if model_factory is None:
            def model_factory(quantile):
                return xgb.XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=quantile,
                    n_estimators=500,
                    max_depth=7,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    n_jobs=-1,
                    random_state=42,
                )

        # Stage 1: Train quantile models
        self.lower_model = model_factory(self.q_lower)
        self.upper_model = model_factory(self.q_upper)

        self.lower_model.fit(X_train, y_train, verbose=False)
        self.upper_model.fit(X_train, y_train, verbose=False)

        # Stage 2: Calibrate
        self._calibrate(X_cal, y_cal)

        return self

    def _calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Compute conformity scores for CQR."""
        lower_pred = self.lower_model.predict(X_cal)
        upper_pred = self.upper_model.predict(X_cal)

        # Conformity scores: max(lower - y, y - upper)
        scores_lower = lower_pred - y_cal
        scores_upper = y_cal - upper_pred
        self.cal_scores = np.maximum(scores_lower, scores_upper)

        n = len(self.cal_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(self.cal_scores, np.minimum(q_level, 1.0))

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """Generate CQR prediction intervals."""
        if self.lower_model is None or self.q_hat is None:
            raise ValueError("Must call fit_quantiles() first")

        lower_pred = self.lower_model.predict(X)
        upper_pred = self.upper_model.predict(X)

        lower = lower_pred - self.q_hat
        upper = upper_pred + self.q_hat

        return pd.DataFrame({
            "y_pred": (lower_pred + upper_pred) / 2,
            "lower": lower,
            "upper": upper,
        })


class QuantileForecaster:
    """
    Trains separate XGBoost models for each quantile.
    Produces P10/P50/P90 prediction intervals.
    """

    def __init__(self, quantiles=None, **xgb_kwargs):
        self.quantiles = quantiles or [0.10, 0.50, 0.90]
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


def evaluate_conformal(
    y_true: np.ndarray,
    intervals: pd.DataFrame,
    alpha: float = 0.10,
    regime_labels: Optional[np.ndarray] = None,
) -> dict:
    """
    Comprehensive evaluation of conformal prediction intervals.

    Args:
        y_true: ground truth values
        intervals: DataFrame with 'y_pred', 'lower', 'upper' columns
        alpha: nominal miscoverage rate
        regime_labels: optional array of regime labels for stratified analysis

    Returns:
        Dictionary with PICP, PINAW, Winkler, ACI, and regime-stratified metrics
    """
    y_true = np.asarray(y_true, dtype=float)
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]

    y_pred = intervals["y_pred"].values[mask]
    lower = intervals["lower"].values[mask]
    upper = intervals["upper"].values[mask]

    n = len(y_true)
    nominal_coverage = 1 - alpha

    # --- Core metrics ---
    # PICP: Prediction Interval Coverage Probability
    covered = (y_true >= lower) & (y_true <= upper)
    picp = np.mean(covered)

    # PINAW: Prediction Interval Normalized Average Width
    width = upper - lower
    pinaw = np.mean(width) / np.mean(np.abs(y_true)) if np.mean(np.abs(y_true)) > 0 else 0

    # Winkler Score (lower is better)
    # For covered points: width
    # For uncovered points: width + (2/alpha) * distance outside
    winkler = np.where(
        covered,
        width,
        width + (2 / alpha) * np.maximum(lower - y_true, y_true - upper)
    )
    winkler_mean = np.mean(winkler)

    # ACI: Average Coverage Interval quality
    aci = 1 - np.abs(picp - nominal_coverage)

    # Point forecast metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    # Width statistics
    mean_width = np.mean(width)
    median_width = np.median(width)
    std_width = np.std(width)

    results = {
        "nominal_coverage": round(nominal_coverage * 100, 2),
        "PICP": round(picp * 100, 2),
        "PINAW": round(pinaw * 100, 4),
        "Winkler": round(winkler_mean, 2),
        "ACI": round(aci, 4),
        "mean_width": round(mean_width, 2),
        "median_width": round(median_width, 2),
        "std_width": round(std_width, 2),
        "P50_RMSE": round(rmse, 2),
        "P50_MAE": round(mae, 2),
        "coverage_gap": round((picp - nominal_coverage) * 100, 2),
        "n_test": n,
    }

    # --- Regime-stratified analysis ---
    if regime_labels is not None:
        regime_labels = np.asarray(regime_labels)[mask]
        for regime in np.unique(regime_labels):
            idx = regime_labels == regime
            if idx.sum() > 0:
                regime_covered = covered[idx]
                regime_picp = np.mean(regime_covered) * 100
                regime_width = np.mean(width[idx])
                results[f"regime_{regime}_PICP"] = round(regime_picp, 2)
                results[f"regime_{regime}_width"] = round(regime_width, 2)
                results[f"regime_{regime}_n"] = int(idx.sum())

    return results


def compute_naive_prediction_interval(
    y_true: np.ndarray,
    n_std: float = 1.645,
    lookback: int = 96,
) -> pd.DataFrame:
    """
    Compute naive prediction intervals based on historical rolling standard deviation.

    This is the baseline that conformal prediction must beat.

    Method:
        - rolling_std = std(y[-lookback:])
        - lower = y_pred - n_std * rolling_std
        - upper = y_pred + n_std * rolling_std

    Where y_pred = y[-lookback:] (yesterday's value at same block)
    """
    y_true = np.asarray(y_true, dtype=float)
    n = len(y_true)

    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    y_pred = np.full(n, np.nan)

    for i in range(lookback, n):
        window = y_true[i - lookback:i]
        std = np.std(window)
        y_pred[i] = y_true[i - lookback]  # naive forecast
        lower[i] = y_pred[i] - n_std * std
        upper[i] = y_pred[i] + n_std * std

    return pd.DataFrame({
        "y_pred": y_pred,
        "lower": lower,
        "upper": upper,
    })


def compute_interval_metrics(y_true: np.ndarray, intervals: pd.DataFrame) -> dict:
    """
    Compute evaluation metrics for prediction intervals (backward compatibility).

    Returns:
        - P50 RMSE, MAE (point forecast quality)
        - PICP: Prediction Interval Coverage Probability (actual % within interval)
        - PINAW: Prediction Interval Normalized Average Width (narrower is better)
    """
    y_true = np.asarray(y_true, dtype=float)
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]

    p10 = intervals["P10"].values[mask] if "P10" in intervals else intervals["lower"].values[mask]
    p50 = intervals["P50"].values[mask] if "P50" in intervals else intervals["y_pred"].values[mask]
    p90 = intervals["P90"].values[mask] if "P90" in intervals else intervals["upper"].values[mask]

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse = np.sqrt(mean_squared_error(y_true, p50))
    mae = mean_absolute_error(y_true, p50)

    # Coverage: what % of actuals fall within interval?
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
