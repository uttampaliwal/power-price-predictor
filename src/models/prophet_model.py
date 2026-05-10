"""
models/prophet_model.py — Prophet Time-Series Model

Note: This model requires the `prophet` package.
      Uncomment and run when integrating.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from evaluate import compute_all_metrics, evaluate_by_segment, print_metrics_table

try:
    from prophet import Prophet
except ImportError:
    print("Prophet is not installed. Please run: pip install prophet")
    sys.exit(0)

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "prophet")
PREDS_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "predictions", "prophet")

TARGET = "mcp_rs_per_mwh"

def load_data():
    train = pd.read_parquet(os.path.join(DATA_DIR, "training_features.parquet"))
    test  = pd.read_parquet(os.path.join(DATA_DIR, "holdout_features.parquet"))
    return train, test

def run():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREDS_DIR,  exist_ok=True)

    print("=== Prophet Model ===\n")
    train_df, test_df = load_data()

    # Prophet expects 'ds' (datetime) and 'y' (target)
    # We will combine 'date' and 'time_block' to create a datetime column if needed,
    # or just use the datetime index. Assuming 'date' and 'time_block' are available.
    
    def prepare_prophet_df(df):
        df_p = df.copy()
        # Combine date and time to create a continuous sequence
        if 'timestamp' not in df_p.columns:
            df_p['timestamp'] = pd.to_datetime(df_p['date'].astype(str) + ' ' + df_p['time_block'].astype(str))
        
        df_prophet = pd.DataFrame({
            'ds': df_p['timestamp'],
            'y': df_p[TARGET]
        })
        return df_prophet, df_p

    train_p, train_clean = prepare_prophet_df(train_df)
    test_p, test_clean = prepare_prophet_df(test_df)

    # Note: Training a single Prophet model for 15-minute interval data can be slow.
    print(f"  Train: {len(train_p):,}  Test: {len(test_p):,}")
    print("  Training Prophet model...")
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    # Add external regressors if needed, here we just use univariate for baseline
    model.fit(train_p)

    print("  Predicting on holdout...")
    future = pd.DataFrame({'ds': test_p['ds']})
    forecast = model.predict(future)
    y_pred = forecast['yhat'].values
    y_test = test_p['y'].values

    metrics = compute_all_metrics(y_test, y_pred)
    print_metrics_table("Prophet (Holdout Test)", metrics)

    by_season = evaluate_by_segment(test_clean, y_pred, "season")
    by_hour   = evaluate_by_segment(test_clean, y_pred, "hour_bucket")
    print("\nBy Season:\n", by_season.to_string(index=False))
    print("\nBy Hour Bucket:\n", by_hour.to_string(index=False))

    # Save
    import json
    from prophet.serialize import model_to_json
    with open(os.path.join(MODELS_DIR, "prophet.json"), "w") as fout:
        json.dump(model_to_json(model), fout)
        
    pd.DataFrame([{"model": "prophet", **metrics}]).to_csv(
        os.path.join(MODELS_DIR, "metrics.csv"), index=False)
    by_season.to_csv(os.path.join(MODELS_DIR, "metrics_by_season.csv"), index=False)
    by_hour.to_csv(os.path.join(MODELS_DIR, "metrics_by_hour.csv"), index=False)

    preds_df = test_clean[["date", "time_block", "block", TARGET]].copy()
    preds_df["predicted_mcp"] = y_pred
    preds_df.to_csv(os.path.join(PREDS_DIR, "test_predictions.csv"), index=False)

    print(f"\n[OK] Model saved to {MODELS_DIR}")

if __name__ == "__main__":
    run()
