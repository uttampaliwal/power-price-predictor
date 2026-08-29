"""
tracking.py — MLflow Experiment Tracking

Logs model training runs, metrics, parameters, and artifacts to MLflow.
Integrates with the existing training scripts.

Usage:
    python src/tracking.py --log-all         # Log all trained models
    python src/tracking.py --log-metrics     # Log metrics from existing models
    python src/tracking.py --show-experiments  # Show all tracked experiments

MLflow UI:
    mlflow ui --port 5000
    # Then open http://localhost:5000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from config import (
    MODELS_DIR, MODELS_NO_WEATHER_DIR, PREDS_DIR, DATA_PROCESSED_DIR,
    FEATURE_COLS, FEATURE_COLS_NO_WEATHER, TARGET_COL,
)
from evaluate import compute_all_metrics


EXPERIMENT_NAME = "IEX DAM Price Forecasting"


def setup_experiment():
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_model_run(model_name: str, use_weather: bool = True):
    """Log a trained model's metrics and parameters to MLflow."""
    setup_experiment()
    weather_str = "with_weather" if use_weather else "no_weather"

    base_dir = MODELS_DIR if use_weather else MODELS_NO_WEATHER_DIR
    pred_dir = os.path.join(PREDS_DIR, f"{model_name}{'_no_weather' if not use_weather else ''}")

    # Load metrics
    metrics_path = os.path.join(base_dir, model_name, "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"  [SKIP] No metrics for {model_name} ({weather_str})")
        return

    metrics_df = pd.read_csv(metrics_path)
    metrics = metrics_df.to_dict(orient="records")[0]

    # Load predictions for direction metrics
    pred_path = os.path.join(pred_dir, "test_predictions.csv")
    direction_metrics = {}
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        if "mcp_rs_per_mwh" in pred_df.columns and "predicted_mcp" in pred_df.columns:
            dir_metrics = compute_all_metrics(
                pred_df["mcp_rs_per_mwh"].values,
                pred_df["predicted_mcp"].values,
            )
            direction_metrics = dir_metrics

    # Hyperparameters (common defaults)
    params = {
        "model_name": model_name,
        "use_weather": use_weather,
        "training_period": "2020-01-01 to 2024-12-31",
        "holdout_period": "2025-01-01 to 2026-08-28",
        "n_features": len(FEATURE_COLS if use_weather else FEATURE_COLS_NO_WEATHER),
    }

    # Model-specific params
    if model_name == "xgboost":
        params.update({"max_depth": 7, "learning_rate": 0.05, "n_estimators": 1000, "early_stopping_rounds": 50})
    elif model_name == "lightgbm":
        params.update({"num_leaves": 63, "max_depth": 7, "learning_rate": 0.05, "min_child_samples": 20})
    elif model_name == "random_forest":
        params.update({"n_estimators": 200, "max_depth": None})
    elif model_name == "ridge":
        params.update({"alpha": 1.0})

    with mlflow.start_run(run_name=f"{model_name}_{weather_str}"):
        mlflow.log_params(params)

        # Log regression metrics
        for key in ["RMSE", "MAE", "MAPE", "R2", "WAPE"]:
            if key in metrics:
                mlflow.log_metric(key, metrics[key])

        # Log direction metrics
        for key in ["AUC_ROC", "F1", "Dir_Accuracy"]:
            if key in direction_metrics and direction_metrics[key] is not None:
                mlflow.log_metric(key, direction_metrics[key])

        # Log by-season metrics
        season_path = os.path.join(base_dir, model_name, "metrics_by_season.csv")
        if os.path.exists(season_path):
            season_df = pd.read_csv(season_path)
            for _, row in season_df.iterrows():
                season = row.get("segment", "unknown").replace("season=", "")
                for col in ["RMSE", "MAE", "R2", "WAPE"]:
                    if col in row:
                        mlflow.log_metric(f"season_{season}_{col}", row[col])

        # Log feature importances
        fi_path = os.path.join(base_dir, model_name, "feature_importance.csv")
        if os.path.exists(fi_path):
            mlflow.log_artifact(fi_path)

        print(f"  [OK] Logged {model_name} ({weather_str}) → MLflow")


def log_backtest_results(model_name: str = "xgboost"):
    """Log backtest results to MLflow."""
    setup_experiment()
    bt_dir = os.path.join(PREDS_DIR, "backtest")
    summary_path = os.path.join(bt_dir, "backtest_summary.csv")

    if not os.path.exists(summary_path):
        print("  [SKIP] No backtest results found. Run backtest.py first.")
        return

    summary = pd.read_csv(summary_path)

    with mlflow.start_run(run_name=f"backtest_{model_name}"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("backtest_type", "holdout_2025_2026")

        for _, row in summary.iterrows():
            strategy = row["strategy"].lower().replace(" ", "_").replace("-", "_")
            for col in summary.columns:
                if col != "strategy":
                    mlflow.log_metric(f"{strategy}_{col}", row[col])

        # Log backtest artifacts
        for f in os.listdir(bt_dir):
            if f.endswith(".csv") or f.endswith(".png"):
                mlflow.log_artifact(os.path.join(bt_dir, f))

        print(f"  [OK] Logged backtest results → MLflow")


def show_experiments():
    """Display all tracked experiments."""
    setup_experiment()
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print("No experiments found. Run --log-all first.")
        return

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    print(f"\n{'='*80}")
    print(f"  Experiment: {EXPERIMENT_NAME} (ID: {experiment.experiment_id})")
    print(f"  Total runs: {len(runs)}")
    print(f"{'='*80}\n")

    for run in runs:
        m = run.data.metrics
        p = run.data.params
        print(f"  Run: {run.info.run_name}")
        print(f"    Model: {p.get('model_name', '?')} | Weather: {p.get('use_weather', '?')}")
        print(f"    R²={m.get('R2', '?'):.4f}  RMSE={m.get('RMSE', '?'):.1f}  "
              f"WAPE={m.get('WAPE', '?'):.1f}%  AUC={m.get('AUC_ROC', '?')}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--log-all", action="store_true", help="Log all trained models")
    group.add_argument("--log-backtest", action="store_true", help="Log backtest results")
    group.add_argument("--show", action="store_true", help="Show tracked experiments")
    args = parser.parse_args()

    if args.log_all:
        for model in ["xgboost", "lightgbm", "random_forest", "ridge"]:
            for wx in [True, False]:
                log_model_run(model, wx)
        log_backtest_results()

    elif args.log_backtest:
        log_backtest_results()

    elif args.show:
        show_experiments()
