"""
benchmark.py — Model Comparison and Drift Analysis

Loads the holdout test predictions from ALL trained models and produces:
  1. predictions/benchmark_metrics.csv     — Model comparison table (all metrics)
  2. predictions/drift_plot.png            — Monthly RMSE trend per model
  3. predictions/model_comparison.png      — Bar chart of all metrics

Usage:
    python src/benchmark.py
"""

import os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import compute_all_metrics
from config import DATA_PROCESSED_DIR, PREDS_DIR, MODELS_DIR, MODEL_LIST
import sys
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = DATA_PROCESSED_DIR
MODELS = MODEL_LIST


def load_predictions(model_name: str) -> pd.DataFrame | None:
    pred_path = os.path.join(PREDS_DIR, model_name, "test_predictions.csv")
    if not os.path.exists(pred_path):
        print(f"  [SKIP] No predictions found for '{model_name}': {pred_path}")
        return None
    df = pd.read_csv(pred_path, parse_dates=["date"])
    return df


def load_holdout_meta() -> pd.DataFrame:
    """Load holdout features for metadata (date, block, season, hour_bucket)."""
    path = os.path.join(DATA_DIR, "holdout_features.parquet")
    return pd.read_parquet(path, columns=["date", "block", "time_block",
                                          "mcp_rs_per_mwh", "season", "hour_bucket", "month", "year"])


def drift_plot(drift_df: pd.DataFrame, out_path: str, period_str: str):
    """Monthly RMSE trend per model."""
    fig, ax = plt.subplots(figsize=(14, 6))
    palette = sns.color_palette("tab10", n_colors=len(drift_df["model"].unique()))

    for i, (model, grp) in enumerate(drift_df.groupby("model")):
        ax.plot(grp["year_month"], grp["RMSE"], marker="o", label=model, color=palette[i])

    ax.set_title(f"Monthly RMSE Drift — Holdout Period {period_str}", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("RMSE (Rs/MWh)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [OK] Drift plot → {out_path}")


def comparison_plot(summary: pd.DataFrame, out_path: str, period_str: str):
    """Bar chart comparing metrics across models."""
    metrics = ["RMSE", "MAE", "MAPE", "R2", "AUC_ROC", "F1"]
    available = [m for m in metrics if m in summary.columns]
    n = len(available)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    palette = sns.color_palette("viridis", n_colors=len(summary))

    for ax, metric in zip(axes, available):
        vals = summary.set_index("model")[metric].dropna()
        bars = ax.bar(vals.index, vals.values, color=palette[:len(vals)])
        ax.set_title(metric, fontsize=12)
        ax.set_xticklabels(vals.index, rotation=45, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Model Comparison — Holdout Test {period_str}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [OK] Comparison chart → {out_path}")


def run():
    os.makedirs(PREDS_DIR, exist_ok=True)
    holdout_meta = load_holdout_meta()

    summary_rows = []
    drift_rows   = []

    for model_name in MODELS:
        pred_df = load_predictions(model_name)
        if pred_df is None:
            continue

        # Must have "predicted_mcp" and "mcp_rs_per_mwh" (true) columns
        if "predicted_mcp" not in pred_df.columns:
            print(f"  [WARN] {model_name}: 'predicted_mcp' column missing, skipping.")
            continue
        if "mcp_rs_per_mwh" not in pred_df.columns:
            # Try merging from holdout_meta by date+block
            if "date" in pred_df.columns and "block" in pred_df.columns:
                pred_df = pred_df.merge(
                    holdout_meta[["date", "block", "mcp_rs_per_mwh", "season", "hour_bucket", "month", "year"]],
                    on=["date", "block"], how="inner",
                )
            else:
                print(f"  [WARN] {model_name}: cannot get true values, skipping.")
                continue

        y_true = pred_df["mcp_rs_per_mwh"].values
        y_pred = pred_df["predicted_mcp"].values
        metrics = compute_all_metrics(y_true, y_pred)
        summary_rows.append({"model": model_name, **metrics})
        print(f"  {model_name}: RMSE={metrics.get('RMSE', 'N/A')}  R2={metrics.get('R2', 'N/A')}")

        # Monthly drift: compute RMSE per year_month
        if "date" in pred_df.columns:
            pred_df["year_month"] = pd.to_datetime(pred_df["date"]).dt.to_period("M").astype(str)
            for ym, grp in pred_df.groupby("year_month"):
                m = compute_all_metrics(grp["mcp_rs_per_mwh"].values, grp["predicted_mcp"].values)
                drift_rows.append({"model": model_name, "year_month": ym, **m})

    if not summary_rows:
        print("\n[ERROR] No model predictions found. Run individual model scripts first.")
        return

    summary = pd.DataFrame(summary_rows)
    drift_df = pd.DataFrame(drift_rows)

    period_str = f"({holdout_meta['date'].min().strftime('%b %Y')} - {holdout_meta['date'].max().strftime('%b %Y')})"

    print("\n" + "="*70)
    print(f"  MODEL BENCHMARK SUMMARY (Holdout {period_str})")
    print("="*70)
    print(summary.to_string(index=False))

    summary.to_csv(os.path.join(PREDS_DIR, "benchmark_metrics.csv"), index=False)
    if not drift_df.empty:
        drift_df.to_csv(os.path.join(PREDS_DIR, "drift_by_month.csv"), index=False)

    if not drift_df.empty:
        drift_plot(drift_df, os.path.join(PREDS_DIR, "drift_plot.png"), period_str)
    comparison_plot(summary, os.path.join(PREDS_DIR, "model_comparison.png"), period_str)

    print(f"\n[OK] All results saved to {PREDS_DIR}/")


if __name__ == "__main__":
    run()
