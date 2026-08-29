"""
backtest.py — Backtesting and P&L Simulation Framework

Simulates trading strategies using model predictions against actual IEX DAM prices.
Measures: cumulative P&L, Sharpe ratio, max drawdown, win rate, profit factor.

Strategies:
  1. Naive buy-low-sell-high: buy when predicted price < actual, sell when > actual
  2. Block-shift arbitrage: shift demand to predicted-cheap blocks
  3. Directional: go long when model predicts price will rise vs yesterday

Usage:
    python src/backtest.py
    python src/backtest.py --model xgboost --start 2025-01-01 --end 2025-06-30
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import BLOCKS_PER_DAY
from config import PREDS_DIR, DATA_RAW_DIR, MODELS_DIR, MODELS_NO_WEATHER_DIR, FEATURE_COLS, FEATURE_COLS_NO_WEATHER, TARGET_COL


def load_predictions(model_name: str, use_weather: bool = True) -> pd.DataFrame:
    """Load holdout test predictions."""
    suffix = "" if use_weather else "_no_weather"
    path = os.path.join(PREDS_DIR, f"{model_name}{suffix}", "test_predictions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_actual_prices(split: str = "holdout") -> pd.DataFrame:
    """Load actual DAM prices."""
    raw_dir = os.path.join(DATA_RAW_DIR, split)
    csvs = sorted([f for f in os.listdir(raw_dir) if f.startswith("dam_") and f.endswith(".csv")])
    dfs = [pd.read_csv(os.path.join(raw_dir, f), parse_dates=["date"]) for f in csvs]
    df = pd.concat(dfs, ignore_index=True)
    # Add block index from time_block
    if "block" not in df.columns:
        df["block"] = df.groupby("date").cumcount()
    return df.sort_values(["date", "block"]).reset_index(drop=True)


# ─── Strategy Implementations ───────────────────────────────────────────
def strategy_arbitrage(preds: pd.DataFrame, actuals: pd.DataFrame, capacity_mw: float = 100.0) -> pd.DataFrame:
    """
    Block-shift arbitrage strategy.
    Buy electricity at predicted-cheap blocks, use at predicted-expensive blocks.
    P&L = (actual_expensive - actual_cheap) * capacity
    """
    merged = preds.merge(actuals[["date", "block", "mcp_rs_per_mwh"]],
                         on=["date", "block"], suffixes=("_pred", "_actual"))

    daily_pnl = []
    for date, grp in merged.groupby("date"):
        grp = grp.sort_values("block")
        pred_vals = grp["predicted_mcp"].values
        actual_vals = grp["mcp_rs_per_mwh_actual"].values

        # Predict cheapest 12 blocks to buy, most expensive 12 to use
        buy_blocks = np.argsort(pred_vals)[:12]
        use_blocks = np.argsort(pred_vals)[-12:]

        buy_cost = np.mean(actual_vals[buy_blocks]) * capacity_mw
        use_value = np.mean(actual_vals[use_blocks]) * capacity_mw
        pnl = use_value - buy_cost

        daily_pnl.append({
            "date": date,
            "pnl_rs": pnl,
            "buy_price_actual": np.mean(actual_vals[buy_blocks]),
            "use_price_actual": np.mean(actual_vals[use_blocks]),
            "predicted_buy_price": np.mean(pred_vals[buy_blocks]),
            "predicted_use_price": np.mean(pred_vals[use_blocks]),
        })

    return pd.DataFrame(daily_pnl)


def strategy_directional(preds: pd.DataFrame, actuals: pd.DataFrame,
                         position_size: float = 100.0) -> pd.DataFrame:
    """
    Directional trading: go long when model predicts price will rise vs yesterday.
    P&L = position_size * (actual_price_tomorrow - actual_price_today) * signal
    """
    merged = preds.merge(actuals[["date", "block", "mcp_rs_per_mwh"]],
                         on=["date", "block"], suffixes=("_pred", "_actual"))
    merged = merged.sort_values(["date", "block"])

    # Compute yesterday's actual price per block
    merged["yesterday_actual"] = merged.groupby("block")["mcp_rs_per_mwh_actual"].shift(1)
    merged = merged.dropna(subset=["yesterday_actual"])

    # Signal: predict rise vs yesterday
    merged["signal"] = (merged["predicted_mcp"] > merged["yesterday_actual"] * 1.01).astype(int)

    # P&L: if signal=1 (long), profit = actual - yesterday_actual; if signal=0, flat
    merged["pnl"] = merged["signal"] * (merged["mcp_rs_per_mwh_actual"] - merged["yesterday_actual"]) * position_size

    daily = merged.groupby("date").agg(
        pnl_rs=("pnl", "sum"),
        n_trades=("signal", "sum"),
        total_blocks=("signal", "count"),
    ).reset_index()
    daily["win_rate"] = daily["n_trades"] / daily["total_blocks"]
    return daily


def strategy_peak_shaving(preds: pd.DataFrame, actuals: pd.DataFrame,
                           shift_mw: float = 50.0) -> pd.DataFrame:
    """
    Peak-shaving: shift load from predicted-peak to predicted-valley blocks.
    Savings = (peak_actual - valley_actual) * shift_mw
    """
    merged = preds.merge(actuals[["date", "block", "mcp_rs_per_mwh"]],
                         on=["date", "block"], suffixes=("_pred", "_actual"))

    daily = []
    for date, grp in merged.groupby("date"):
        grp = grp.sort_values("block")
        pred_vals = grp["predicted_mcp"].values
        actual_vals = grp["mcp_rs_per_mwh_actual"].values

        peak_blocks = np.argsort(pred_vals)[-6:]   # top 6 predicted peak blocks
        valley_blocks = np.argsort(pred_vals)[:6]   # bottom 6 predicted valley blocks

        peak_actual = np.mean(actual_vals[peak_blocks])
        valley_actual = np.mean(actual_vals[valley_blocks])
        savings = (peak_actual - valley_actual) * shift_mw

        daily.append({
            "date": date,
            "savings_rs": savings,
            "peak_price": peak_actual,
            "valley_price": valley_actual,
        })

    return pd.DataFrame(daily)


# ─── Metrics ────────────────────────────────────────────────────────────
def compute_trading_metrics(daily_pnl: pd.Series) -> dict:
    """Compute standard trading performance metrics."""
    pnl = daily_pnl.values
    total_pnl = pnl.sum()
    mean_daily = pnl.mean()
    std_daily = pnl.std()

    sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max()

    win_days = (pnl > 0).sum()
    total_days = len(pnl)
    win_rate = win_days / total_days * 100

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_pnl_rs": round(total_pnl, 2),
        "mean_daily_pnl_rs": round(mean_daily, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_rs": round(max_drawdown, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "trading_days": total_days,
    }


# ─── Visualization ──────────────────────────────────────────────────────
def plot_backtest(results: dict, out_path: str):
    """Plot backtest results: cumulative P&L + drawdown for each strategy."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    palette = sns.color_palette("Set2", n_colors=len(results))

    for i, (name, data) in enumerate(results.items()):
        pnl = data["daily_pnl"]
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative

        axes[0].plot(pnl.index, cumulative, label=name, color=palette[i], linewidth=1.5)
        axes[1].fill_between(pnl.index, -drawdown, alpha=0.3, color=palette[i], label=name)

    axes[0].set_ylabel("Cumulative P&L (Rs)")
    axes[0].set_title("Backtest: Cumulative P&L by Strategy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Drawdown (Rs)")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Drawdown")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Backtest plot → {out_path}")


# ─── Main ───────────────────────────────────────────────────────────────
def run_backtest(model_name: str = "xgboost", use_weather: bool = True,
                 start_date: str = None, end_date: str = None):
    print(f"\n{'='*60}")
    print(f"  BACKTEST: {model_name} ({'with weather' if use_weather else 'no weather'})")
    print(f"{'='*60}\n")

    preds = load_predictions(model_name, use_weather)
    actuals = load_actual_prices("holdout")

    if start_date:
        preds = preds[preds["date"] >= start_date]
    if end_date:
        preds = preds[preds["date"] <= end_date]

    print(f"  Predictions: {len(preds)} rows, {preds['date'].min().date()} → {preds['date'].max().date()}")

    # Run strategies
    arb_daily = strategy_arbitrage(preds, actuals)
    dir_daily = strategy_directional(preds, actuals)
    peak_daily = strategy_peak_shaving(preds, actuals)

    # Compute metrics
    arb_metrics = compute_trading_metrics(arb_daily["pnl_rs"])
    dir_metrics = compute_trading_metrics(dir_daily["pnl_rs"])
    peak_metrics = compute_trading_metrics(peak_daily["savings_rs"])

    results = {
        "Block-Shift Arbitrage": {"daily_pnl": arb_daily.set_index("date")["pnl_rs"], "metrics": arb_metrics},
        "Directional Trading": {"daily_pnl": dir_daily.set_index("date")["pnl_rs"], "metrics": dir_metrics},
        "Peak Shaving": {"daily_pnl": peak_daily.set_index("date")["savings_rs"], "metrics": peak_metrics},
    }

    # Print summary
    print(f"\n{'─'*60}")
    print(f"  {'Strategy':<25} {'Total P&L':>12} {'Sharpe':>8} {'Max DD':>10} {'Win%':>6}")
    print(f"{'─'*60}")
    for name, data in results.items():
        m = data["metrics"]
        print(f"  {name:<25} {m['total_pnl_rs']:>12,.0f} {m['sharpe_ratio']:>8.3f} {m['max_drawdown_rs']:>10,.0f} {m['win_rate_pct']:>5.1f}%")
    print(f"{'─'*60}")

    # Save
    out_dir = os.path.join(PREDS_DIR, "backtest")
    os.makedirs(out_dir, exist_ok=True)

    for name, data in results.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        data["daily_pnl"].to_csv(os.path.join(out_dir, f"{safe_name}_daily.csv"))

    summary = pd.DataFrame([
        {"strategy": name, **data["metrics"]}
        for name, data in results.items()
    ])
    summary.to_csv(os.path.join(out_dir, "backtest_summary.csv"), index=False)

    plot_backtest(results, os.path.join(out_dir, "backtest_plot.png"))

    print(f"\n[OK] Results saved to {out_dir}/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    run_backtest(args.model, not args.no_weather, args.start, args.end)
