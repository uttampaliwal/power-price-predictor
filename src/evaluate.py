"""
evaluate.py — Metrics Suite for Power Price Prediction

Provides both regression and classification (price direction) metrics.
All models import from here to ensure consistent evaluation.

Functions:
    compute_regression_metrics(y_true, y_pred)  → dict
    compute_classification_metrics(y_true, y_pred) → dict
    compute_all_metrics(y_true, y_pred) → dict
    evaluate_by_segment(df, y_pred_col) → DataFrame
    print_metrics_table(metrics_dict)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    roc_auc_score, f1_score,
)
import sys
sys.stdout.reconfigure(encoding="utf-8")


# Regression Metrics

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, MAPE, R², WAPE for regression targets.

    Returns: dict with metric names as keys.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {m: np.nan for m in ["RMSE", "MAE", "MAPE", "R2", "WAPE"]}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    # MAPE — guard against near-zero true values
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

    return {
        "RMSE":  round(rmse, 4),
        "MAE":   round(mae,  4),
        "MAPE":  round(mape, 4),
        "R2":    round(r2,   4),
        "WAPE":  round(wape, 4),
    }



# Classification Metrics (price direction: up vs down vs flat)

def price_direction(series: np.ndarray, threshold_pct: float = 1.0) -> np.ndarray:
    """
    Convert price series to binary direction labels relative to previous day same block.
    1 = price went UP by ≥ threshold_pct %, 0 = DOWN or flat.
    """
    series = np.asarray(series, dtype=float)
    direction = np.zeros(len(series), dtype=int)
    direction[1:] = ((series[1:] - series[:-1]) / np.abs(series[:-1] + 1e-9) * 100 >= threshold_pct).astype(int)
    return direction


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    
    """
    Compute AUC-ROC and F1 for price direction
    (up vs not-up, ≥1% change threshold).

    Parameters:
        y_true: actual MCP prices (time-ordered)
        y_pred: predicted MCP prices (time-ordered)

    Returns: dict with classification metric names as keys.
    """
    
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    true_dir = price_direction(y_true)
    pred_dir = price_direction(y_pred)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    true_dir, pred_dir = true_dir[mask], pred_dir[mask]

    if len(np.unique(true_dir)) < 2:
        return {m: np.nan for m in ["AUC_ROC", "F1"]}

    try:
        auc = roc_auc_score(true_dir, pred_dir)
    except Exception:
        auc = np.nan

    return {
        "AUC_ROC": round(auc, 4),
        "F1":      round(f1_score(true_dir, pred_dir, zero_division=0), 4),
    }


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return merged dict of regression + classification metrics."""
    reg = compute_regression_metrics(y_true, y_pred)
    clf = compute_classification_metrics(y_true, y_pred)
    return {**reg, **clf}


# Segment-level Evaluation

def evaluate_by_segment(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    segment_col: str = "season",
) -> pd.DataFrame:
    """
    Compute RMSE + MAPE broken down by a segment column (e.g., season, hour_bucket).

    Args:
        df:          the test/holdout DataFrame with mcp_rs_per_mwh and segment_col
        y_pred:      predicted MCP values (same order as df)
        segment_col: column to group by

    Returns: DataFrame with one row per segment value + overall row.
    """
    df = df.copy()
    df["_pred"] = y_pred
    rows = []
    for seg_val, grp in df.groupby(segment_col):
        m = compute_regression_metrics(grp["mcp_rs_per_mwh"].values, grp["_pred"].values)
        rows.append({"segment": f"{segment_col}={seg_val}", **m})
    # Overall
    overall = compute_regression_metrics(df["mcp_rs_per_mwh"].values, y_pred)
    rows.append({"segment": "OVERALL", **overall})
    return pd.DataFrame(rows)


def print_metrics_table(model_name: str, metrics: dict):
    print(f"\n{'='*50}")
    print(f"  Model: {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")
    print(f"{'='*50}\n")
