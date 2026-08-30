"""
tracking_cp.py — MLflow Experiment Tracking for Conformal Prediction

Logs conformal prediction experiments to MLflow with:
- Model hyperparameters
- Coverage metrics (PICP, PINAW, Winkler)
- Regime-stratified results
- Artifacts (intervals, comparison tables)

Usage:
    python src/tracking_cp.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import json

import mlflow
import mlflow.sklearn
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "conformal")
MLRUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "mlruns")


def log_experiment():
    """Log all conformal prediction experiments to MLflow."""
    db_path = os.path.join(os.path.dirname(__file__), "..", "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(db_path)}")
    mlflow.set_experiment("conformal_prediction_iex_dam")

    # Load results
    with open(os.path.join(RESULTS_DIR, "conformal_results.json")) as f:
        results = json.load(f)

    # Load comparison table
    comp_table = pd.read_csv(os.path.join(RESULTS_DIR, "conformal_comparison.csv"))

    # Load regime table
    regime_table = pd.read_csv(os.path.join(RESULTS_DIR, "conformal_regime.csv"))

    for model_name, model_results in results.items():
        for method in ["split_conformal", "adaptive_conformal", "cqr", "enbpi", "spci"]:
            if method not in model_results:
                continue

            m = model_results[method]

            with mlflow.start_run(run_name=f"{model_name}_{method}"):
                # Log parameters
                mlflow.log_param("base_model", model_name)
                mlflow.log_param("cp_method", method)
                mlflow.log_param("alpha", 0.10)
                mlflow.log_param("target_coverage", 90.0)
                mlflow.log_param("dataset", "IEX DAM 15-min")
                mlflow.log_param("train_period", "2020-01 to 2024-12")
                mlflow.log_param("holdout_period", "2025-01 to 2026-08")
                mlflow.log_param("n_test", m.get("n_test", 0))

                # Log metrics
                mlflow.log_metric("picp", m["PICP"])
                mlflow.log_metric("coverage_gap", m["coverage_gap"])
                mlflow.log_metric("pinaw", m["PINAW"])
                mlflow.log_metric("winkler", m["Winkler"])
                mlflow.log_metric("aci", m["ACI"])
                mlflow.log_metric("mean_width", m["mean_width"])
                mlflow.log_metric("median_width", m["median_width"])
                mlflow.log_metric("std_width", m["std_width"])
                mlflow.log_metric("p50_rmse", m["P50_RMSE"])
                mlflow.log_metric("p50_mae", m["P50_MAE"])
                mlflow.log_metric(
                    "point_forecast_rmse", model_results["point_forecast_RMSE"]
                )

                # Log regime metrics
                for regime in ["normal", "spike"]:
                    picp_key = f"regime_{regime}_PICP"
                    width_key = f"regime_{regime}_width"
                    n_key = f"regime_{regime}_n"
                    if picp_key in m:
                        mlflow.log_metric(f"{regime}_picp", m[picp_key])
                        mlflow.log_metric(f"{regime}_width", m[width_key])
                        mlflow.log_metric(f"{regime}_n", m[n_key])

                # Log artifacts
                mlflow.log_artifact(
                    os.path.join(RESULTS_DIR, "conformal_comparison.csv")
                )
                mlflow.log_artifact(os.path.join(RESULTS_DIR, "conformal_regime.csv"))

                if os.path.exists(os.path.join(RESULTS_DIR, "hourly_coverage.csv")):
                    mlflow.log_artifact(
                        os.path.join(RESULTS_DIR, "hourly_coverage.csv")
                    )

                if os.path.exists(os.path.join(RESULTS_DIR, "trading_simulation.csv")):
                    mlflow.log_artifact(
                        os.path.join(RESULTS_DIR, "trading_simulation.csv")
                    )

                print(f"  Logged: {model_name}/{method}")


if __name__ == "__main__":
    print("=" * 60)
    print("  MLflow Experiment Tracking")
    print("=" * 60)

    log_experiment()

    print(f"\n  Results logged to {MLRUNS_DIR}")
    print("  Run 'mlflow ui' to view experiments")
    print("Done.")
