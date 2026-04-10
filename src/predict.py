"""
predict.py — Next-Day Power Price Prediction

Loads a trained model and generates predictions for all 96 time blocks
of a given date. Outputs a CSV and a matplotlib chart.

Usage:
    python src/predict.py --model xgboost --date 2026-03-22
    python src/predict.py --model lstm    --date 2026-03-22
    python src/predict.py --model lightgbm                    # defaults to tomorrow
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import requests
import sys
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PREDS_DIR  = os.path.join(os.path.dirname(__file__), "..", "predictions")

FEATURE_COLS = [
    "block", "hour", "day_of_week", "day_of_year", "month", "year",
    "is_weekend", "is_holiday", "season", "hour_bucket",
    "mcp_lag_1d", "mcp_lag_7d",
    "mcp_rolling_7d_mean", "mcp_rolling_7d_std",
    "mcp_rolling_30d_mean", "mcp_rolling_30d_std",
    "delhi_apparent_temp", "mumbai_apparent_temp",
]
BLOCKS_PER_DAY = 96


def load_tabular_model(model_name: str):
    if model_name == "xgboost":
        import xgboost as xgb
        m = xgb.XGBRegressor()
        m.load_model(os.path.join(MODELS_DIR, "xgboost", "xgboost.json"))
        return m, "tabular"
    elif model_name == "lightgbm":
        import lightgbm as lgb
        m = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lightgbm", "lightgbm.txt"))
        return m, "lightgbm_booster"
    elif model_name == "ridge":
        pipe = joblib.load(os.path.join(MODELS_DIR, "ridge", "ridge_pipeline.pkl"))
        return pipe, "tabular"
    elif model_name == "random_forest":
        m = joblib.load(os.path.join(MODELS_DIR, "random_forest", "random_forest.pkl"))
        return m, "tabular"
    elif model_name == "naive":
        return None, "naive"
    elif model_name == "lstm":
        return None, "lstm"
    else:
        raise ValueError(f"Unknown model: {model_name}")


def fetch_daily_weather(target_date: date, lat: float, lon: float):
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    if target_date < date.today() - timedelta(days=5):
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "apparent_temperature",
        "timezone": "Asia/Kolkata",
        "start_date": str(target_date),
        "end_date": str(target_date)
    }
    try:
        http = urllib3.PoolManager(cert_reqs='CERT_NONE')
        response = http.request('GET', url, fields=params, timeout=10.0)
        if response.status == 200:
            temps = response.json()["hourly"]["apparent_temperature"]
            return [t if t is not None else np.nan for t in temps]
    except Exception as e:
        print(f"Error fetching weather: {e}")
    return [np.nan] * 24

def get_features_for_date(target_date: date, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature row(s) for the target date using recent history.
    Returns a DataFrame with 96 rows (one per block).
    """
    import holidays
    india_hol = holidays.India()

    print("  Fetching target date weather forecast...")
    delhi_weather = fetch_daily_weather(target_date, 28.6139, 77.2090)
    mumbai_weather = fetch_daily_weather(target_date, 19.0760, 72.8777)

    rows = []
    for block in range(BLOCKS_PER_DAY):
        hour = block // 4
        # Get lag values from history
        lag1_date = target_date - timedelta(days=1)
        lag7_date = target_date - timedelta(days=7)

        def get_block_mcp(hist, d, b):
            row = hist[(hist["date"].dt.date == d) & (hist["block"] == b)]
            return float(row["mcp_rs_per_mwh"].values[0]) if len(row) else np.nan

        mcp_lag_1d = get_block_mcp(history_df, lag1_date, block)
        mcp_lag_7d = get_block_mcp(history_df, lag7_date, block)

        # Rolling stats: mean/std of last 7 days same block
        mask_7d = (
            (history_df["block"] == block) &
            (history_df["date"].dt.date >= target_date - timedelta(days=7)) &
            (history_df["date"].dt.date <  target_date)
        )
        recent_7 = history_df[mask_7d]["mcp_rs_per_mwh"]
        rolling_7d_mean = recent_7.mean() if len(recent_7) > 0 else np.nan
        rolling_7d_std  = recent_7.std()  if len(recent_7) > 1 else 0.0

        mask_30d = (
            (history_df["block"] == block) &
            (history_df["date"].dt.date >= target_date - timedelta(days=30)) &
            (history_df["date"].dt.date <  target_date)
        )
        recent_30 = history_df[mask_30d]["mcp_rs_per_mwh"]
        rolling_30d_mean = recent_30.mean() if len(recent_30) > 0 else np.nan
        rolling_30d_std  = recent_30.std()  if len(recent_30) > 1 else 0.0

        target_dt = pd.Timestamp(target_date)
        season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 0}

        def hour_bucket(h):
            if h < 6:  return 0
            if h < 10: return 1
            if h < 14: return 2
            if h < 18: return 3
            if h < 22: return 4
            return 0

        rows.append({
            "block":              block,
            "hour":               hour,
            "day_of_week":        target_dt.dayofweek,
            "day_of_year":        target_dt.dayofyear,
            "month":              target_dt.month,
            "year":               target_dt.year,
            "is_weekend":         int(target_dt.dayofweek >= 5),
            "is_holiday":         int(target_date in india_hol),
            "season":             season_map[target_dt.month],
            "hour_bucket":        hour_bucket(hour),
            "mcp_lag_1d":         mcp_lag_1d,
            "mcp_lag_7d":         mcp_lag_7d,
            "mcp_rolling_7d_mean":  rolling_7d_mean,
            "mcp_rolling_7d_std":   rolling_7d_std,
            "mcp_rolling_30d_mean": rolling_30d_mean,
            "mcp_rolling_30d_std":  rolling_30d_std,
            "delhi_apparent_temp": delhi_weather[hour] if hour < len(delhi_weather) else np.nan,
            "mumbai_apparent_temp": mumbai_weather[hour] if hour < len(mumbai_weather) else np.nan,
        })

    return pd.DataFrame(rows)


def make_plot(blocks: np.ndarray, prices: np.ndarray, model_name: str, target_date: date, out_path: str):
    hours = blocks / 4
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hours, prices, color="#2563EB", linewidth=1.8, label=f"{model_name} prediction")
    ax.fill_between(hours, prices * 0.9, prices * 1.1, alpha=0.12, color="#2563EB")
    ax.set_title(f"Predicted DAM MCP — {target_date}  [{model_name}]", fontsize=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MCP (Rs/MWh)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [OK] Chart saved → {out_path}")


def run(model_name: str, target_date: date, save_files: bool = True) -> pd.DataFrame:
    out_dir = os.path.join(PREDS_DIR, model_name, "daily")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Predicting with {model_name} for {target_date} ===\n")

    # Load recent history from training + holdout processed data
    train_path   = os.path.join(DATA_DIR, "training_features.parquet")
    holdout_path = os.path.join(DATA_DIR, "holdout_features.parquet")
    dfs = []
    for p in [train_path, holdout_path]:
        if os.path.exists(p):
            dfs.append(pd.read_parquet(p, columns=["date", "block", "time_block", "mcp_rs_per_mwh"]))
    if not dfs:
        raise FileNotFoundError("No processed data found. Run preprocess.py first.")
    history = pd.concat(dfs).sort_values("date")
    history["date"] = pd.to_datetime(history["date"])

    model, mode = load_tabular_model(model_name)

    if mode == "naive":
        # Use 7-day lag directly from history
        lag7_date = target_date - timedelta(days=7)
        lag_rows  = history[history["date"].dt.date == lag7_date].sort_values("block")
        if lag_rows.empty:
            raise ValueError(f"No history found for lag date {lag7_date}")
        y_pred = lag_rows["mcp_rs_per_mwh"].values
        blocks = lag_rows["block"].values
        time_blocks = lag_rows["time_block"].values if "time_block" in lag_rows else [f"block_{b}" for b in blocks]

    elif mode == "lstm":
        raise NotImplementedError("LSTM predict not yet implemented in predict.py. "
                                  "Use lstm_model.py to evaluate on holdout instead.")
    else:
        feat_df = get_features_for_date(target_date, history)
        X  = feat_df[FEATURE_COLS].values.astype(float)

        if mode == "lightgbm_booster":
            y_pred = model.predict(X)
        else:
            y_pred = model.predict(X)

        blocks      = feat_df["block"].values
        time_blocks = [f"{(b//4):02d}:{(b%4)*15:02d}" for b in blocks]

    # Save CSV
    csv_path = os.path.join(out_dir, f"prediction_{target_date}.csv")
    
    if mode != "naive" and mode != "lstm":
        out_df = pd.DataFrame({
            "date":          str(target_date),
            "block":         blocks,
            "time_block":    time_blocks,
            "predicted_mcp": np.round(y_pred, 2),
            "delhi_weather": np.round(feat_df["delhi_apparent_temp"].values, 2),
            "mumbai_weather": np.round(feat_df["mumbai_apparent_temp"].values, 2),
        })
    else:
        out_df = pd.DataFrame({
            "date":          str(target_date),
            "block":         blocks,
            "time_block":    time_blocks,
            "predicted_mcp": np.round(y_pred, 2),
            "delhi_weather": np.nan,
            "mumbai_weather": np.nan,
        })
        
    # Append Actual MCP if it exists for this past date
    actual_data = history[history["date"].dt.date == target_date].sort_values("block")
    if not actual_data.empty and len(actual_data) == len(out_df):
        out_df["actual_mcp"] = actual_data["mcp_rs_per_mwh"].values
    else:
        out_df["actual_mcp"] = np.nan

    if save_files:
        # Save CSV
        csv_path = os.path.join(out_dir, f"prediction_{target_date}.csv")
        out_df.to_csv(csv_path, index=False)
        print(f"  [OK] Predictions → {csv_path}")
        print(f"       Min: {y_pred.min():.1f}  Max: {y_pred.max():.1f}  Mean: {y_pred.mean():.1f} Rs/MWh")
    
        # Save Plot
        png_path = os.path.join(out_dir, f"plot_{target_date}.png")
        make_plot(blocks, y_pred, model_name, target_date, png_path)
        print(f"  [OK] Plot → {png_path}\n")

    return out_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["naive", "ridge", "random_forest", "xgboost", "lightgbm", "lstm"])
    parser.add_argument("--date", default=None,
                        help="Date to predict (YYYY-MM-DD). Defaults to tomorrow.")
    args = parser.parse_args()

    if args.date:
        target = date.fromisoformat(args.date)
    else:
        target = date.today() + timedelta(days=1)

    run(args.model, target)
