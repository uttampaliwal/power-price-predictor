"""
fetch_data.py — IEX India DAM Market Snapshot Scraper
Uses urllib and pandas to scrape the 15-min DAM price data table day by day,
and save year-wise CSVs. This ensures we capture all 96 blocks correctly.

Usage:
    python src/fetch_data.py --start 2020-01-01 --end 2024-12-31 --split training
    python src/fetch_data.py --start 2025-01-01 --end 2025-12-31 --split holdout
"""

import argparse
import os
import time
import io
import urllib.request
from datetime import date, timedelta
import pandas as pd
from tqdm import tqdm
from config import DATA_RAW_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "https://www.iexindia.com/market-data/day-ahead-market/market-snapshot"

COLUMNS_MAPPING = {
    'Date': 'date',
    'Time Block': 'time_block',
    'Purchase Bid (MW)': 'purchase_bid_mw',
    'Sell Bid (MW)': 'sell_bid_mw',
    'MCV (MW)': 'mcv_mw',
    'Final Scheduled Volume (MW)': 'final_scheduled_volume_mw',
    'MCP (Rs/MWh) *': 'mcp_rs_per_mwh'
}

def date_to_ddmmyyyy(d: date) -> str:
    return d.strftime("%d-%m-%Y")

def scrape_day(target_date: date) -> pd.DataFrame:
    dt_str = date_to_ddmmyyyy(target_date)
    url = f"{BASE_URL}?interval=ONE_FOURTH_HOUR&dp=SELECT_RANGE&showGraph=false&fromDate={dt_str}&toDate={dt_str}"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8')
            dfs = pd.read_html(io.StringIO(html))
            if dfs and len(dfs[0]) > 0:
                df = dfs[0]
                
                if 'Time Block' not in df.columns:
                    continue
                
                # Exclude summary rows
                df = df[~df['Time Block'].astype(str).str.contains('Avg|Max|Min|Total', case=False, na=False)]
                
                # Drop unwanted 'Hour' column if it exists
                if 'Hour' in df.columns:
                    df = df.drop(columns=['Hour'])
                    
                # Rename columns
                df = df.rename(columns=COLUMNS_MAPPING)

                # Convert Date format
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
                
                # Convert numerics
                numeric_cols = [c for c in df.columns if c not in ('date', 'time_block')]
                for col in numeric_cols:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                return df
            return pd.DataFrame()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [ERROR] Failed to fetch {dt_str}: {e}")
            time.sleep(2)
            
    return pd.DataFrame()

def _get_last_date_in_csv(filepath: str) -> date | None:
    """Read a CSV and return the maximum date found, or None."""
    try:
        df = pd.read_csv(filepath, usecols=["date"], parse_dates=["date"])
        if df.empty:
            return None
        return df["date"].max().date()
    except Exception:
        return None


def fetch_all(start: date, end: date, split: str, overwrite: bool = False):
    out_dir = os.path.join(DATA_RAW_DIR, split)
    os.makedirs(out_dir, exist_ok=True)

    # Generate all dates
    all_dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    years = sorted(list(set(d.year for d in all_dates)))
    
    print(f"\nFetching {len(all_dates)} days of DAM data → {out_dir}\n")
    
    today = date.today()

    for year in years:
        dates_in_year = [d for d in all_dates if d.year == year]
        fname = os.path.join(out_dir, f"dam_{year}.csv")
        
        if os.path.exists(fname) and not overwrite:
            # Check if this year is fully complete (past year)
            year_is_past = (year < today.year)
            if year_is_past:
                print(f"  [SKIP] {fname} — year {year} is complete.")
                continue

            # Current/incomplete year — check last date and append new data
            last_date = _get_last_date_in_csv(fname)
            if last_date is not None:
                next_date = last_date + timedelta(days=1)
                year_end = max(d for d in dates_in_year)
                if next_date > year_end:
                    print(f"  [SKIP] {fname} — already up to {last_date}, nothing new to fetch.")
                    continue
                # Only fetch from next_date onwards
                dates_in_year = [d for d in dates_in_year if d >= next_date]
                print(f"\n  [APPEND] {fname} — has data up to {last_date}, fetching {len(dates_in_year)} new days...")
            else:
                print(f"\n  [REFETCH] {fname} exists but is empty, fetching full year {year}...")
        elif overwrite and os.path.exists(fname):
            print(f"\n  [OVERWRITE] Removing {fname} and re-fetching year {year}...")
            os.remove(fname)
        
        if not dates_in_year:
            continue

        print(f"\nFetching Year {year} ({len(dates_in_year)} days)...")
        yearly_dfs = []
        for d in tqdm(dates_in_year, desc=f"Year {year}"):
            df = scrape_day(d)
            if not df.empty:
                yearly_dfs.append(df)
            time.sleep(0.5)
            
        if yearly_dfs:
            new_df = pd.concat(yearly_dfs, ignore_index=True)
            new_df = new_df.dropna(subset=['mcp_rs_per_mwh'])

            # Append to existing file if it exists (incremental mode)
            if os.path.exists(fname) and not overwrite:
                existing_df = pd.read_csv(fname, parse_dates=["date"])
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
                final_df = final_df.drop_duplicates(subset=["date", "time_block"])
                final_df = final_df.sort_values(["date", "time_block"]).reset_index(drop=True)
            else:
                final_df = new_df

            final_df.to_csv(fname, index=False)
            print(f"  [OK] Saved {len(final_df)} rows → {fname}")
        else:
            print(f"  [WARN] No data found for year {year}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch IEX India DAM data")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--split", required=True, choices=["training", "holdout"],
        help="Which data split to save into"
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Force re-fetch even if data files already exist"
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    fetch_all(start, end, args.split, overwrite=args.overwrite)
