"""
preprocess.py — Feature Engineering for DAM Price Data

Loads all monthly CSVs from data/raw/<split>/, merges, cleans, and adds
time features + lag features. Saves as Parquet for fast loading.

Usage:
    python src/preprocess.py --split training
    python src/preprocess.py --split holdout
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import holidays
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")

INDIA_HOL = holidays.India()


# ────────────────────────────────────────────────────────────────────────────────
# Load & Merge
# ────────────────────────────────────────────────────────────────────────────────
def load_raw_csvs(split: str) -> pd.DataFrame:
    pattern = os.path.join(DATA_RAW_DIR, split, "dam_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found at: {pattern}")
    print(f"Loading {len(files)} CSV files from {split}...")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["date"])
            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Could not read {f}: {e}")
    merged = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows loaded: {len(merged):,}")
    return merged


# ────────────────────────────────────────────────────────────────────────────────
# Parse time block → hour + block index
# ────────────────────────────────────────────────────────────────────────────────
def parse_time_block(tb_str: str):
    """'00:00-00:15' → (0, 0)  |  '06:00-06:15' → (6, 24)"""
    try:
        start_str = tb_str.split("-")[0].strip()
        h, m = map(int, start_str.split(":"))
        block = h * 4 + m // 15  # 0–95
        return h, block
    except Exception:
        return np.nan, np.nan


# ────────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ────────────────────────────────────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"], df["block"] = zip(*df["time_block"].apply(parse_time_block))
    df["hour"]  = df["hour"].astype("Int16")
    df["block"] = df["block"].astype("Int16")

    df["day_of_week"]  = df["date"].dt.dayofweek          # 0=Mon
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["month"]        = df["date"].dt.month
    df["year"]         = df["date"].dt.year
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"]   = df["date"].apply(lambda d: int(d in INDIA_HOL))

    # Season (India): Summer=3-5, Monsoon=6-9, Post-monsoon=10-11, Winter=12-2
    season_map = {
        1: 0, 2: 0,                          # Winter
        3: 1, 4: 1, 5: 1,                    # Summer
        6: 2, 7: 2, 8: 2, 9: 2,             # Monsoon
        10: 3, 11: 3, 12: 0,                 # Post-monsoon / Winter
    }
    df["season"] = df["month"].map(season_map)

    # Hour-bucket: Off-peak / Morning-peak / Afternoon / Evening-peak / Night
    def hour_bucket(h):
        if h < 6:    return 0   # off-peak night
        if h < 10:   return 1   # morning peak
        if h < 14:   return 2   # afternoon
        if h < 18:   return 3   # afternoon-pre-evening
        if h < 22:   return 4   # evening peak
        return 0                # night
    df["hour_bucket"] = df["hour"].apply(hour_bucket)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features per block index (0–95)."""
    df = df.sort_values(["block", "date"]).copy()

    for lag_days in [1, 7]:
        col = f"mcp_lag_{lag_days}d"
        df[col] = df.groupby("block")["mcp_rs_per_mwh"].shift(lag_days)

    for window in [7, 30]:
        col = f"mcp_rolling_{window}d_mean"
        df[col] = (
            df.groupby("block")["mcp_rs_per_mwh"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
        )
        col_std = f"mcp_rolling_{window}d_std"
        df[col_std] = (
            df.groupby("block")["mcp_rs_per_mwh"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=3).std())
        )

    df = df.sort_values(["date", "block"]).reset_index(drop=True)
    return df


def merge_weather_data(df: pd.DataFrame, split: str) -> pd.DataFrame:
    weather_path = os.path.join(DATA_RAW_DIR, f"weather_{split}.csv")
    if not os.path.exists(weather_path):
        print(f"  [WARN] No weather data found at {weather_path}, proceeding without weather features.")
        return df
    
    print("  Merging weather data...")
    df_weather = pd.read_csv(weather_path, parse_dates=["datetime"])
    
    # We have 'date' (timestamp 00:00:00) and 'hour' (Int16 0-23)
    df["merge_key"] = df["date"] + pd.to_timedelta(df["hour"].astype(int), unit="h")
    
    merged = pd.merge(df, df_weather, left_on="merge_key", right_on="datetime", how="left")
    merged = merged.drop(columns=["merge_key", "datetime"])
    
    merged["delhi_apparent_temp"] = merged["delhi_apparent_temp"].ffill().bfill()
    merged["mumbai_apparent_temp"] = merged["mumbai_apparent_temp"].ffill().bfill()
    
    return merged


# ────────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ────────────────────────────────────────────────────────────────────────────────
def run(split: str):
    df = load_raw_csvs(split)
    df = df.dropna(subset=["date", "mcp_rs_per_mwh"])
    df = df.sort_values(["date", "time_block"]).reset_index(drop=True)

    print("  Adding time features...")
    df = add_time_features(df)

    df = merge_weather_data(df, split)

    print("  Adding lag & rolling features...")
    df = add_lag_features(df)

    # Drop rows with NaN lags (first few days)
    # COMMENTED OUT: We want to keep the first 7 days in the parquet file so the 
    # Dashboard 'Live Tracker' can still display them. The ML models already have 
    # their own `df.dropna(subset=FEATURE_COLS)` steps before training.
    feature_cols = [c for c in df.columns if c.startswith("mcp_lag") or c.startswith("mcp_rolling")]
    before = len(df)
    # df = df.dropna(subset=feature_cols)
    print(f"  (Kept {before} rows. Models will drop the first 7 days internally for training.)")

    out_dir = DATA_PROCESSED_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}_features.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\n[OK] Saved → {out_path}")
    print(f"     Shape: {df.shape}  |  Date range: {df['date'].min().date()} to {df['date'].max().date()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["training", "holdout"])
    args = parser.parse_args()
    run(args.split)
