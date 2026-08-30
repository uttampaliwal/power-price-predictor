"""
hourly_coverage.py — Hourly/Conditional Coverage Analysis

Evaluates conformal prediction coverage at each hour of the day,
revealing whether coverage is uniform or concentrated in specific periods.

Usage:
    python src/hourly_coverage.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from probabilistic import evaluate_conformal

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "conformal")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_data():
    """Load test data with hour labels."""
    test = pd.read_parquet(os.path.join(DATA_DIR, "holdout_features.parquet"))
    test = test.dropna(subset=["mcp_rs_per_mwh"])
    return test


def load_intervals(model_name="lightgbm", method="scp"):
    """Load conformal prediction intervals."""
    filename = f"{model_name}_{method}_intervals.csv"
    return pd.read_csv(os.path.join(RESULTS_DIR, filename))


def compute_hourly_coverage(test, intervals, alpha=0.10):
    """Compute coverage metrics for each hour of the day."""
    # Ensure same length
    min_len = min(len(test), len(intervals))
    test = test.iloc[:min_len].reset_index(drop=True)
    intervals = intervals.iloc[:min_len].reset_index(drop=True)

    # Add hour column
    test["hour"] = test["block"] // 4  # 4 blocks per hour

    results = []
    for hour in range(24):
        mask = test["hour"] == hour
        if mask.sum() < 10:
            continue

        y_true = test.loc[mask, "mcp_rs_per_mwh"].values
        hour_intervals = intervals.loc[mask, ["y_pred", "lower", "upper"]]

        metrics = evaluate_conformal(y_true, hour_intervals, alpha=alpha)
        metrics["hour"] = hour
        metrics["n_samples"] = int(mask.sum())
        metrics["mean_price"] = round(float(y_true.mean()), 2)
        metrics["std_price"] = round(float(y_true.std()), 2)

        results.append(metrics)

    return pd.DataFrame(results)


def compute_monthly_coverage(test, intervals, alpha=0.10):
    """Compute coverage metrics for each month."""
    min_len = min(len(test), len(intervals))
    test = test.iloc[:min_len].reset_index(drop=True)
    intervals = intervals.iloc[:min_len].reset_index(drop=True)

    # Extract month from date
    if "date" in test.columns:
        test["month"] = pd.to_datetime(test["date"]).dt.month
    else:
        return pd.DataFrame()

    results = []
    for month in range(1, 13):
        mask = test["month"] == month
        if mask.sum() < 10:
            continue

        y_true = test.loc[mask, "mcp_rs_per_mwh"].values
        month_intervals = intervals.loc[mask, ["y_pred", "lower", "upper"]]

        metrics = evaluate_conformal(y_true, month_intervals, alpha=alpha)
        metrics["month"] = month
        metrics["n_samples"] = int(mask.sum())
        metrics["mean_price"] = round(float(y_true.mean()), 2)

        results.append(metrics)

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("  Hourly/Monthly Coverage Analysis")
    print("=" * 60)

    test = load_data()
    intervals = load_intervals("lightgbm", "scp")

    # Hourly coverage
    print("\n  Computing hourly coverage...")
    hourly = compute_hourly_coverage(test, intervals, alpha=0.10)
    print("\n  Hourly Coverage (LightGBM + SCP):")
    print(
        hourly[
            ["hour", "PICP", "coverage_gap", "mean_width", "mean_price", "n_samples"]
        ].to_string(index=False)
    )

    # Monthly coverage
    print("\n  Computing monthly coverage...")
    monthly = compute_monthly_coverage(test, intervals, alpha=0.10)
    if len(monthly) > 0:
        print("\n  Monthly Coverage (LightGBM + SCP):")
        print(
            monthly[
                [
                    "month",
                    "PICP",
                    "coverage_gap",
                    "mean_width",
                    "mean_price",
                    "n_samples",
                ]
            ].to_string(index=False)
        )

    # Save results
    hourly.to_csv(os.path.join(RESULTS_DIR, "hourly_coverage.csv"), index=False)
    if len(monthly) > 0:
        monthly.to_csv(os.path.join(RESULTS_DIR, "monthly_coverage.csv"), index=False)

    # Generate plot data
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hourly coverage plot
    ax = axes[0]
    hours = hourly["hour"].values
    picps = hourly["PICP"].values
    ax.bar(
        hours,
        picps,
        color=[
            "#4CAF50" if 88 <= p <= 92 else "#FF9800" if p > 92 else "#F44336"
            for p in picps
        ],
    )
    ax.axhline(y=90, color="red", linestyle="--", linewidth=2, label="Target 90%")
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("PICP (%)", fontsize=12)
    ax.set_title("Coverage by Hour", fontsize=14, fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Monthly coverage plot
    if len(monthly) > 0:
        ax = axes[1]
        months = monthly["month"].values
        picps_m = monthly["PICP"].values
        ax.bar(
            months,
            picps_m,
            color=[
                "#4CAF50" if 88 <= p <= 92 else "#FF9800" if p > 92 else "#F44336"
                for p in picps_m
            ],
        )
        ax.axhline(y=90, color="red", linestyle="--", linewidth=2, label="Target 90%")
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("PICP (%)", fontsize=12)
        ax.set_title("Coverage by Month", fontsize=14, fontweight="bold")
        ax.set_xticks(range(1, 13))
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULTS_DIR, "..", "..", "paper", "cp_hourly_monthly_coverage.png"
        ),
        dpi=150,
        bbox_inches="tight",
    )
    print("\n  Plot saved to paper/cp_hourly_monthly_coverage.png")

    # Summary statistics
    print(f"\n  Overall PICP: {hourly['PICP'].mean():.2f}%")
    print(
        f"  Hourly PICP range: {hourly['PICP'].min():.2f}% - {hourly['PICP'].max():.2f}%"
    )
    print(f"  Standard deviation: {hourly['PICP'].std():.2f}%")

    if len(monthly) > 0:
        print(
            f"  Monthly PICP range: {monthly['PICP'].min():.2f}% - {monthly['PICP'].max():.2f}%"
        )

    print(f"\n  Results saved to {RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
