"""
powerbi_live_sync.py — Real-time data sync for Power BI

Creates a simple mechanism to detect new data and update CSV files.
Run this after fetch_data or preprocess to update the powerbi_data folder.

Usage:
    python src/powerbi_live_sync.py

Or integrate with fetch_data.py to auto-run after updates.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import glob
from datetime import datetime
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, MODELS_NO_WEATHER_DIR, PREDS_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "powerbi_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METADATA_FILE = os.path.join(OUTPUT_DIR, "sync_metadata.csv")


def get_latest_file_time(directory, pattern="*.csv"):
    """Get the latest modification time of files matching pattern."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(os.path.getmtime(f) for f in files)


def load_or_create_metadata():
    """Load existing sync metadata or create new."""
    if os.path.exists(METADATA_FILE):
        return pd.read_csv(METADATA_FILE)
    return pd.DataFrame(columns=["data_source", "last_sync", "last_update"])


def save_metadata(df):
    """Save sync metadata."""
    df.to_csv(METADATA_FILE, index=False)


def check_and_update(source_name, directory, pattern, force=False):
    """Check if data has changed and update if needed."""
    metadata = load_or_create_metadata()
    
    # Get current latest file time
    current_time = get_latest_file_time(directory, pattern)
    if not current_time:
        return False, "No files found"
    
    # Check last sync time for this source
    last_sync_row = metadata[metadata["data_source"] == source_name]
    if last_sync_row.empty:
        last_sync = 0
    else:
        last_sync = last_sync_row["last_sync"].iloc[0]
    
    # Update if changed or forced
    if current_time > last_sync or force:
        # Update metadata
        metadata = metadata[metadata["data_source"] != source_name]
        new_row = pd.DataFrame([{
            "data_source": source_name,
            "last_sync": current_time,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        metadata = pd.concat([metadata, new_row], ignore_index=True)
        save_metadata(metadata)
        return True, f"Updated (files changed)"
    
    return False, "No changes detected"


def quick_export():
    """Fast export of key files that change frequently."""
    print("🔄 Quick sync for Power BI...")
    
    # Check raw data
    changed1, _ = check_and_update("raw_dam", DATA_RAW_DIR, "dam_*.csv", force=True)
    
    # Check processed data
    changed2, _ = check_and_update("processed", DATA_PROCESSED_DIR, "*.parquet", force=True)
    
    # Check predictions
    changed3, _ = check_and_update("predictions", PREDS_DIR, "test_predictions.csv", force=True)
    
    # If anything changed, do full export
    if changed1 or changed2 or changed3:
        print("   → Changes detected, running full export...")
        os.system(f'"{sys.executable}" "{os.path.join(os.path.dirname(__file__), "powerbi_exporter.py")}"')
    else:
        print("   ✓ No new data to sync")
    
    # Always update metadata file timestamp
    metadata = load_or_create_metadata()
    metadata["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata.to_csv(os.path.join(OUTPUT_DIR, "sync_status.csv"), index=False)
    
    print("   ✅ Sync complete")


def auto_sync_trigger():
    """Call this from other scripts to trigger auto-sync."""
    # This can be integrated into fetch_data.py or preprocess.py
    quick_export()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force full export")
    args = parser.parse_args()
    
    if args.force:
        # Force full export
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "powerbi_exporter.py")])
    else:
        quick_export()