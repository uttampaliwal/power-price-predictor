"""
run_conformal.py — Run conformal prediction evaluation on all benchmark models.

Applies Split Conformal, Adaptive Conformal, CQR, EnbPI, and SPCI to each model.
Generates coverage analysis, comparison tables, and plot data.

Usage:
    python src/run_conformal.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from config import FEATURE_COLS, TARGET_COL, DATA_PROCESSED_DIR, PREDS_DIR
from probabilistic import (
    SplitConformal,
    AdaptiveConformal,
    ConformalizedQuantileRegression,
    EnsembleBatchConformal,
    SequentialPredictiveConformal,
    evaluate_conformal,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "conformal")


def load_data():
    """Load training and holdout data, split into train/cal/test."""
    train = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "training_features.parquet"))
    holdout = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "holdout_features.parquet"))

    # Drop rows with NaN
    train = train.dropna(subset=FEATURE_COLS + [TARGET_COL])
    holdout = holdout.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X_train_full = train[FEATURE_COLS].values
    y_train_full = train[TARGET_COL].values
    X_test = holdout[FEATURE_COLS].values
    y_test = holdout[TARGET_COL].values

    # Split training into train + calibration (last 20% for calibration)
    cal_size = int(len(X_train_full) * 0.20)
    X_train = X_train_full[:-cal_size]
    y_train = y_train_full[:-cal_size]
    X_cal = X_train_full[-cal_size:]
    y_cal = y_train_full[-cal_size:]

    return X_train, y_train, X_cal, y_cal, X_test, y_test, holdout


def train_xgboost(X_train, y_train):
    """Train XGBoost model."""
    split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lightgbm(X_train, y_train):
    """Train LightGBM model."""
    split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    return model


def train_ridge(X_train, y_train):
    """Train Ridge model."""
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model


def compute_spike_regime(y_true, threshold_percentile=85):
    """Label time points as 'spike' or 'normal' based on price level."""
    threshold = np.percentile(y_true, threshold_percentile)
    return np.where(y_true >= threshold, "spike", "normal")


def run_conformal_for_model(model_name, model, X_train, y_train, X_cal, y_cal, X_test, y_test, holdout):
    """Run all five CP methods on a single model."""
    print(f"\n{'='*60}")
    print(f"  Conformal Prediction: {model_name.upper()}")
    print(f"{'='*60}")

    results = {}
    regimes = compute_spike_regime(y_test)

    # Method 1: Split Conformal
    print(f"\n  [1/5] Split Conformal Prediction")
    scp = SplitConformal(alpha=0.10)
    scp.fit(model, X_cal, y_cal)
    intervals_scp = scp.predict(model, X_test)
    metrics_scp = evaluate_conformal(y_test, intervals_scp, alpha=0.10, regime_labels=regimes)
    results["split_conformal"] = metrics_scp
    print(f"         PICP: {metrics_scp['PICP']:.2f}%  (target: 90.0%)")
    print(f"         PINAW: {metrics_scp['PINAW']:.4f}")
    print(f"         Winkler: {metrics_scp['Winkler']:.2f}")
    print(f"         Coverage gap: {metrics_scp['coverage_gap']:+.2f}%")

    # Method 2: Adaptive Conformal
    print(f"\n  [2/5] Adaptive Conformal Prediction")
    acp = AdaptiveConformal(alpha=0.10, gamma=0.005, window=500)
    acp.fit(model, X_cal, y_cal)
    intervals_acp = acp.predict(model, X_test)
    metrics_acp = evaluate_conformal(y_test, intervals_acp, alpha=0.10, regime_labels=regimes)
    results["adaptive_conformal"] = metrics_acp
    print(f"         PICP: {metrics_acp['PICP']:.2f}%  (target: 90.0%)")
    print(f"         PINAW: {metrics_acp['PINAW']:.4f}")
    print(f"         Winkler: {metrics_acp['Winkler']:.2f}")
    print(f"         Coverage gap: {metrics_acp['coverage_gap']:+.2f}%")

    # Method 3: CQR
    print(f"\n  [3/5] Conformalized Quantile Regression")
    cqr = ConformalizedQuantileRegression(alpha=0.10, q_lower=0.05, q_upper=0.95)
    cqr.fit_quantiles(X_train, y_train, X_cal, y_cal)
    intervals_cqr = cqr.predict(X_test)
    metrics_cqr = evaluate_conformal(y_test, intervals_cqr, alpha=0.10, regime_labels=regimes)
    results["cqr"] = metrics_cqr
    print(f"         PICP: {metrics_cqr['PICP']:.2f}%  (target: 90.0%)")
    print(f"         PINAW: {metrics_cqr['PINAW']:.4f}")
    print(f"         Winkler: {metrics_cqr['Winkler']:.2f}")
    print(f"         Coverage gap: {metrics_cqr['coverage_gap']:+.2f}%")

    # Method 4: EnbPI
    print(f"\n  [4/5] Ensemble Batch Prediction Intervals (EnbPI)")
    from sklearn.linear_model import Ridge as RidgeFactory

    def ridge_factory():
        return Ridge(alpha=1.0)

    enbpi = EnsembleBatchConformal(alpha=0.10, n_bootstraps=10, sample_ratio=0.8)
    enbpi.fit(ridge_factory, X_train, y_train)
    intervals_enbpi = enbpi.predict(X_test)
    metrics_enbpi = evaluate_conformal(y_test, intervals_enbpi, alpha=0.10, regime_labels=regimes)
    results["enbpi"] = metrics_enbpi
    print(f"         PICP: {metrics_enbpi['PICP']:.2f}%  (target: 90.0%)")
    print(f"         PINAW: {metrics_enbpi['PINAW']:.4f}")
    print(f"         Winkler: {metrics_enbpi['Winkler']:.2f}")
    print(f"         Coverage gap: {metrics_enbpi['coverage_gap']:+.2f}%")

    # Method 5: SPCI
    print(f"\n  [5/5] Sequential Predictive Conformal Inference (SPCI)")
    spci = SequentialPredictiveConformal(alpha=0.10, decay=0.99, min_scores=100)
    spci.fit(model, X_cal, y_cal)
    intervals_spci = spci.predict(model, X_test)
    metrics_spci = evaluate_conformal(y_test, intervals_spci, alpha=0.10, regime_labels=regimes)
    results["spci"] = metrics_spci
    print(f"         PICP: {metrics_spci['PICP']:.2f}%  (target: 90.0%)")
    print(f"         PINAW: {metrics_spci['PINAW']:.4f}")
    print(f"         Winkler: {metrics_spci['Winkler']:.2f}")
    print(f"         Coverage gap: {metrics_spci['coverage_gap']:+.2f}%")

    # Point forecast metrics
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results["point_forecast_RMSE"] = round(rmse, 2)
    print(f"\n  Point forecast RMSE: {rmse:.2f} Rs/MWh")

    return results, intervals_scp, intervals_acp, intervals_cqr


def generate_comparison_table(all_results):
    """Generate a comparison table across all models and CP methods."""
    rows = []
    for model_name, model_results in all_results.items():
        for method in ["split_conformal", "adaptive_conformal", "cqr", "enbpi", "spci"]:
            if method in model_results:
                m = model_results[method]
                rows.append({
                    "Model": model_name,
                    "Method": method.replace("_", " ").title(),
                    "PICP (%)": m["PICP"],
                    "Target (%)": m["nominal_coverage"],
                    "Gap (%)": m["coverage_gap"],
                    "PINAW": m["PINAW"],
                    "Winkler": m["Winkler"],
                    "ACI": m["ACI"],
                    "Mean Width": m["mean_width"],
                    "RMSE": model_results["point_forecast_RMSE"],
                })

    df = pd.DataFrame(rows)
    return df


def generate_regime_table(all_results):
    """Generate regime-stratified comparison."""
    rows = []
    for model_name, model_results in all_results.items():
        for method in ["split_conformal", "adaptive_conformal", "cqr", "enbpi", "spci"]:
            if method in model_results:
                m = model_results[method]
                rows.append({
                    "Model": model_name,
                    "Method": method.replace("_", " ").title(),
                    "Normal PICP (%)": m.get("regime_normal_PICP", np.nan),
                    "Normal Width": m.get("regime_normal_width", np.nan),
                    "Spike PICP (%)": m.get("regime_spike_PICP", np.nan),
                    "Spike Width": m.get("regime_spike_width", np.nan),
                    "Normal N": m.get("regime_normal_n", 0),
                    "Spike N": m.get("regime_spike_n", 0),
                })

    df = pd.DataFrame(rows)
    return df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Conformal Prediction Evaluation Pipeline")
    print("  Target: Indian DAM at 15-min resolution")
    print("  Methods: SCP, ACP, CQR")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    X_train, y_train, X_cal, y_cal, X_test, y_test, holdout = load_data()
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Calibration: {X_cal.shape[0]} samples")
    print(f"  Test (holdout): {X_test.shape[0]} samples")

    # Train models
    print("\nTraining models...")
    models = {
        "XGBoost": train_xgboost(X_train, y_train),
        "LightGBM": train_lightgbm(X_train, y_train),
        "Ridge": train_ridge(X_train, y_train),
    }

    # Run conformal prediction for each model
    all_results = {}
    all_intervals = {}
    for model_name, model in models.items():
        results, intervals_scp, intervals_acp, intervals_cqr = run_conformal_for_model(
            model_name, model, X_train, y_train, X_cal, y_cal, X_test, y_test, holdout
        )
        all_results[model_name] = results
        all_intervals[model_name] = {
            "scp": intervals_scp,
            "acp": intervals_acp,
            "cqr": intervals_cqr,
        }

    # Generate comparison tables
    print("\n\n" + "=" * 60)
    print("  COMPARISON TABLE")
    print("=" * 60)
    comp_table = generate_comparison_table(all_results)
    print(comp_table.to_string(index=False))

    print("\n\n" + "=" * 60)
    print("  REGIME-STRATIFIED ANALYSIS")
    print("=" * 60)
    regime_table = generate_regime_table(all_results)
    print(regime_table.to_string(index=False))

    # Save results
    comp_table.to_csv(os.path.join(RESULTS_DIR, "conformal_comparison.csv"), index=False)
    regime_table.to_csv(os.path.join(RESULTS_DIR, "conformal_regime.csv"), index=False)

    # Save full results as JSON
    with open(os.path.join(RESULTS_DIR, "conformal_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save intervals for plot generation
    for model_name, intervals in all_intervals.items():
        for method, df in intervals.items():
            df.to_csv(os.path.join(RESULTS_DIR, f"{model_name.lower()}_{method}_intervals.csv"), index=False)

    print(f"\n\nResults saved to {RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
