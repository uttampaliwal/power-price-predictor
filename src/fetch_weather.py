"""
fetch_weather.py — Open-Meteo Historical Fetcher
Automates fetching the 'Apparent Temperature' (Feels Like) for National Grid proxies.

Usage:
    python src/fetch_weather.py --start 2020-01-01 --end 2024-12-31 --split training
    python src/fetch_weather.py --start 2025-01-01 --end 2026-03-24 --split holdout
"""
import argparse
import os
import pandas as pd
import urllib3
from config import DATA_RAW_DIR, CITIES
import sys
sys.stdout.reconfigure(encoding="utf-8")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOCATIONS = CITIES  # Keep alias for backward compatibility within script

def fetch_historical_weather(city, lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "apparent_temperature",
        "timezone": "Asia/Kolkata"
    }
    
    print(f"Fetching {city} weather from {start} to {end}...")
    http = urllib3.PoolManager(cert_reqs='CERT_NONE')
    response = http.request('GET', url, fields=params, timeout=30.0)
    if response.status != 200:
        raise Exception(f"API Error {response.status}: {response.data.decode()}")
        
    data = response.json()
    df = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        f"{city}_apparent_temp": data["hourly"]["apparent_temperature"]
    })
    return df
def run(start, end, split):
    out_dir = DATA_RAW_DIR
    os.makedirs(out_dir, exist_ok=True)
    
    dfs = []
    for city, coords in LOCATIONS.items():
        df_city = fetch_historical_weather(city, coords["lat"], coords["lon"], start, end)
        dfs.append(df_city)
        
    # Merge on datetime
    final_df = dfs[0]
    for df in dfs[1:]:
        final_df = pd.merge(final_df, df, on="datetime", how="outer")
        
    out_file = os.path.join(out_dir, f"weather_{split}.csv")
    final_df.to_csv(out_file, index=False)
    print(f"  [OK] Saved {len(final_df)} rows of weather data → {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--split", required=True, choices=["training", "holdout"])
    args = parser.parse_args()
    
    run(args.start, args.end, args.split)
