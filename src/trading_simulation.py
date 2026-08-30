"""
trading_simulation.py — Economic Value of Conformal Prediction Intervals

Simulates trading strategies that use prediction intervals for:
1. Risk-aware position sizing (wider intervals → smaller positions)
2. Interval-based arbitrage (buy when price below lower bound, sell when above upper)
3. Hedging value (how much does coverage quality affect P&L?)

Usage:
    python src/trading_simulation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "conformal")


def load_intervals_and_data():
    """Load conformal prediction intervals and test data."""
    test = pd.read_parquet(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "processed",
            "holdout_features.parquet",
        )
    )
    test = test.dropna(subset=["mcp_rs_per_mwh"])

    # Load LightGBM SCP intervals (best performer)
    intervals = pd.read_csv(os.path.join(RESULTS_DIR, "lightgbm_scp_intervals.csv"))

    return test, intervals


def strategy_interval_arbitrage(test, intervals, capacity_mw=100):
    """
    Interval-based arbitrage strategy.

    Logic:
    - If predicted price + upper_bound_margin < actual → buy opportunity
    - If predicted price - lower_bound_margin > actual → sell opportunity
    - Position size proportional to interval width (wider = less confident = smaller position)

    This simulates a trader who uses prediction intervals to identify
    mispriced blocks and size positions accordingly.
    """
    y_true = test["mcp_rs_per_mwh"].values
    y_pred = intervals["y_pred"].values
    lower = intervals["lower"].values
    upper = intervals["upper"].values

    n = len(y_true)
    interval_width = upper - lower

    # Position sizing: inversely proportional to interval width
    # Wider intervals → less confident → smaller position
    median_width = np.median(interval_width)
    position_scale = np.clip(median_width / interval_width, 0.2, 2.0)

    # Strategy: buy blocks where predicted price is below lower bound
    # (model says price will be higher than current)
    daily_pnl = []

    for day_start in range(0, n, 96):
        day_end = min(day_start + 96, n)
        day_true = y_true[day_start:day_end]
        day_pred = y_pred[day_start:day_end]
        day_lower = lower[day_start:day_end]
        day_upper = upper[day_start:day_end]
        day_scale = position_scale[day_start:day_end]

        # Identify buy opportunities: price below lower bound
        buy_mask = day_true < day_lower
        # Identify sell opportunities: price above upper bound
        sell_mask = day_true > day_upper

        # P&L from buy opportunities (buy at actual, sell at predicted)
        buy_pnl = (
            np.sum(
                (day_pred[buy_mask] - day_true[buy_mask])
                * capacity_mw
                * day_scale[buy_mask]
            )
            if buy_mask.any()
            else 0
        )

        # P&L from sell opportunities (sell at actual, buy at predicted)
        sell_pnl = (
            np.sum(
                (day_true[sell_mask] - day_pred[sell_mask])
                * capacity_mw
                * day_scale[sell_mask]
            )
            if sell_mask.any()
            else 0
        )

        daily_pnl.append(buy_pnl + sell_pnl)

    return np.array(daily_pnl)


def strategy_confidence_weighted(test, intervals, capacity_mw=100):
    """
    Confidence-weighted trading strategy.

    Uses interval width as a confidence measure:
    - Narrow intervals → high confidence → large position
    - Wide intervals → low confidence → small position

    Trades in the direction of the predicted price change.
    """
    y_true = test["mcp_rs_per_mwh"].values
    y_pred = intervals["y_pred"].values
    lower = intervals["lower"].values
    upper = intervals["upper"].values

    n = len(y_true)
    interval_width = upper - lower

    # Confidence: inversely proportional to width
    median_width = np.median(interval_width)
    confidence = np.clip(median_width / interval_width, 0.1, 2.0)

    daily_pnl = []

    for day_start in range(0, n, 96):
        day_end = min(day_start + 96, n)
        day_true = y_true[day_start:day_end]
        day_pred = y_pred[day_start:day_end]
        day_conf = confidence[day_start:day_end]

        # Use naive forecast (yesterday's price) as reference
        if day_start >= 96:
            prev_true = y_true[day_start - 96 : day_start]
            # Direction: predicted vs yesterday
            direction = np.sign(day_pred - prev_true)
            # Actual price change
            actual_change = day_true - prev_true
            # P&L: position * actual change
            pnl = np.sum(direction * actual_change * capacity_mw * day_conf)
        else:
            pnl = 0

        daily_pnl.append(pnl)

    return np.array(daily_pnl)


def strategy_hedging_value(test, intervals, capacity_mw=100):
    """
    Hedging value analysis.

    Compares:
    1. Unhedged: full exposure to price volatility
    2. Hedged with SCP intervals: reduce position when intervals are wide
    3. Perfect hedge: know the true price in advance (oracle)

    Shows the economic value of having prediction intervals.
    """
    y_true = test["mcp_rs_per_mwh"].values
    y_pred = intervals["y_pred"].values
    lower = intervals["lower"].values
    upper = intervals["upper"].values

    n = len(y_true)
    interval_width = upper - lower

    daily_pnl_unhedged = []
    daily_pnl_hedged = []

    for day_start in range(0, n, 96):
        day_end = min(day_start + 96, n)
        day_true = y_true[day_start:day_end]
        day_pred = y_pred[day_start:day_end]
        day_width = interval_width[day_start:day_end]

        # Unhedged: full exposure
        unhedged_pnl = -np.sum(np.abs(day_true - day_pred) * capacity_mw)

        # Hedged: reduce position when intervals are wide
        median_width = np.median(interval_width)
        hedge_ratio = np.clip(day_width / median_width, 0.5, 1.5)
        hedged_pnl = -np.sum(np.abs(day_true - day_pred) * capacity_mw / hedge_ratio)

        daily_pnl_unhedged.append(unhedged_pnl)
        daily_pnl_hedged.append(hedged_pnl)

    return np.array(daily_pnl_unhedged), np.array(daily_pnl_hedged)


def compute_trading_metrics(daily_pnl, capacity_mw=100, name="Strategy"):
    """Compute trading performance metrics."""
    total_pnl = np.sum(daily_pnl)
    n_days = len(daily_pnl)

    if n_days == 0:
        return {}

    # Sharpe ratio (annualized, assuming 365 trading days)
    if np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(365)
    else:
        sharpe = 0

    # Win rate
    win_rate = np.mean(daily_pnl > 0) * 100

    # Max drawdown
    cumulative = np.cumsum(daily_pnl)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # Profit factor
    gross_profit = np.sum(daily_pnl[daily_pnl > 0])
    gross_loss = np.abs(np.sum(daily_pnl[daily_pnl < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "strategy": name,
        "total_pnl_rs": round(total_pnl, 2),
        "total_pnl_millions": round(total_pnl / 1e6, 2),
        "sharpe_ratio": round(sharpe, 2),
        "win_rate_pct": round(win_rate, 2),
        "max_drawdown_rs": round(max_drawdown, 2),
        "profit_factor": round(profit_factor, 2)
        if profit_factor != float("inf")
        else "inf",
        "n_trading_days": n_days,
    }


def main():
    print("=" * 60)
    print("  Trading Simulation: Economic Value of CP Intervals")
    print("=" * 60)

    # Load data
    test, intervals = load_intervals_and_data()
    print(f"\n  Loaded {len(test)} test observations")
    print(f"  Loaded {len(intervals)} prediction intervals")

    # Ensure same length
    min_len = min(len(test), len(intervals))
    test = test.iloc[:min_len].reset_index(drop=True)
    intervals = intervals.iloc[:min_len].reset_index(drop=True)

    # Strategy 1: Interval Arbitrage
    print("\n  [1/3] Interval-Based Arbitrage Strategy")
    pnl_arb = strategy_interval_arbitrage(test, intervals, capacity_mw=100)
    metrics_arb = compute_trading_metrics(pnl_arb, name="Interval Arbitrage")
    print(f"         Total P&L: Rs{metrics_arb['total_pnl_millions']}M")
    print(f"         Sharpe Ratio: {metrics_arb['sharpe_ratio']}")
    print(f"         Win Rate: {metrics_arb['win_rate_pct']}%")

    # Strategy 2: Confidence-Weighted Trading
    print("\n  [2/3] Confidence-Weighted Trading Strategy")
    pnl_cw = strategy_confidence_weighted(test, intervals, capacity_mw=100)
    metrics_cw = compute_trading_metrics(pnl_cw, name="Confidence-Weighted")
    print(f"         Total P&L: Rs{metrics_cw['total_pnl_millions']}M")
    print(f"         Sharpe Ratio: {metrics_cw['sharpe_ratio']}")
    print(f"         Win Rate: {metrics_cw['win_rate_pct']}%")

    # Strategy 3: Hedging Value
    print("\n  [3/3] Hedging Value Analysis")
    pnl_unhedged, pnl_hedged = strategy_hedging_value(test, intervals, capacity_mw=100)
    metrics_unhedged = compute_trading_metrics(pnl_unhedged, name="Unhedged")
    metrics_hedged = compute_trading_metrics(pnl_hedged, name="Hedged with SCP")
    print(f"         Unhedged P&L: Rs{metrics_unhedged['total_pnl_millions']}M")
    print(f"         Hedged P&L: Rs{metrics_hedged['total_pnl_millions']}M")
    print(
        f"         Hedging Improvement: {metrics_hedged['sharpe_ratio'] - metrics_unhedged['sharpe_ratio']:.2f} Sharpe"
    )

    # Save results
    all_metrics = [metrics_arb, metrics_cw, metrics_unhedged, metrics_hedged]
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(os.path.join(RESULTS_DIR, "trading_simulation.csv"), index=False)

    # Save daily P&L for plotting
    pd.DataFrame(
        {
            "date": test["date"].values[: len(pnl_arb)],
            "interval_arbitrage": pnl_arb,
            "confidence_weighted": pnl_cw,
            "unhedged": pnl_unhedged,
            "hedged": pnl_hedged,
        }
    ).to_csv(os.path.join(RESULTS_DIR, "daily_pnl.csv"), index=False)

    print(f"\n  Results saved to {RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
