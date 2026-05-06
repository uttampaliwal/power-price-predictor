"""
powerbi_exporter.py — Export data for Power BI Dashboard

Generates CSV files from all available data sources for Power BI import.
Run this after training models, running benchmarks, or fetching new data.

Usage:
    python src/powerbi_exporter.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import glob
from datetime import date, timedelta
from config import (
    DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_DIR, MODELS_NO_WEATHER_DIR, 
    PREDS_DIR, HOLDOUT_START_DATE, CITIES
)
import sys
sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "powerbi_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_model_metrics():
    """Export all model metrics (with and without weather)."""
    print("📊 Exporting model metrics...")
    
    rows = []
    model_dirs = {
        "Yes": MODELS_DIR,
        "No": MODELS_NO_WEATHER_DIR
    }
    
    for weather_flag, base_dir in model_dirs.items():
        if not os.path.exists(base_dir):
            continue
        
        for model_name in os.listdir(base_dir):
            metrics_path = os.path.join(base_dir, model_name, "metrics.csv")
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                df["weather"] = weather_flag
                df["model_id"] = f"{model_name}_{weather_flag}"
                df["Model Category"] = "With Weather" if weather_flag == "Yes" else "Without Weather"
                # Rename for readability in Power BI
                df = df.rename(columns={
                    "R2": "R2 Score",
                    "WAPE": "WAPE %",
                    "RMSE": "RMSE (Lower is Better)",
                    "MAE": "MAE (Lower is Better)",
                    "MAPE": "MAPE %",
                })
                rows.append(df)
    
    if rows:
        df = pd.concat(rows, ignore_index=True)
        # Reorder columns for better readability
        cols = ["model", "weather", "Model Category", "model_id", "R2 Score", "WAPE %", "RMSE (Lower is Better)", 
                "MAE (Lower is Better)", "MAPE %", "AUC_ROC", "F1"]
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]
        df.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)
        print(f"   ✅ model_metrics.csv ({len(df)} records)")
    else:
        print("   ⚠️ No model metrics found")


def export_predictions():
    """Export all model predictions (actual vs predicted)."""
    print("📈 Exporting predictions...")
    
    all_preds = []
    
    # With weather predictions
    for model_name in ["xgboost", "lightgbm", "random_forest", "ridge"]:
        pred_path = os.path.join(PREDS_DIR, model_name, "test_predictions.csv")
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path, parse_dates=["date"])
            df["model"] = model_name
            df["weather"] = "Yes"
            df["model_id"] = f"{model_name}_Yes"
            df["Model Category"] = "With Weather"
            df["Prediction Error"] = df["predicted_mcp"] - df["mcp_rs_per_mwh"]
            df["Abs Error %"] = abs(df["Prediction Error"]) / df["mcp_rs_per_mwh"] * 100
            all_preds.append(df)
    
    # Without weather predictions
    for model_name in ["xgboost", "lightgbm", "random_forest", "ridge"]:
        pred_path = os.path.join(PREDS_DIR, f"{model_name}_no_weather", "test_predictions.csv")
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path, parse_dates=["date"])
            df["model"] = model_name
            df["weather"] = "No"
            df["model_id"] = f"{model_name}_No"
            df["Model Category"] = "Without Weather"
            df["Prediction Error"] = df["predicted_mcp"] - df["mcp_rs_per_mwh"]
            df["Abs Error %"] = abs(df["Prediction Error"]) / df["mcp_rs_per_mwh"] * 100
            all_preds.append(df)
    
    if all_preds:
        df = pd.concat(all_preds, ignore_index=True)
        # Rename for clarity
        df = df.rename(columns={
            "mcp_rs_per_mwh": "Actual MCP",
            "predicted_mcp": "Predicted MCP",
        })
        df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
        print(f"   ✅ predictions.csv ({len(df)} records)")
    else:
        print("   ⚠️ No predictions found")


def export_daily_prices():
    """Export historical MCP prices from holdout data."""
    print("💰 Exporting daily prices...")
    
    parquet_path = os.path.join(DATA_PROCESSED_DIR, "holdout_features.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path, columns=[
            "date", "block", "time_block", "hour", "mcp_rs_per_mwh"
        ])
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(os.path.join(OUTPUT_DIR, "daily_prices.csv"), index=False)
        print(f"   ✅ daily_prices.csv ({len(df)} records)")
    else:
        print("   ⚠️ Holdout parquet not found")


def export_weather_data():
    """Export weather data for visualization."""
    print("🌡️ Exporting weather data...")
    
    weather_files = glob.glob(os.path.join(DATA_RAW_DIR, "weather_*.csv"))
    
    for wf in weather_files:
        split = os.path.basename(wf).replace("weather_", "").replace(".csv", "")
        df = pd.read_csv(wf, parse_dates=["datetime"])
        df["split"] = split
        df.to_csv(os.path.join(OUTPUT_DIR, f"weather_{split}.csv"), index=False)
        print(f"   ✅ weather_{split}.csv ({len(df)} records)")


def export_feature_importance():
    """Export feature importance for all models."""
    print("🧬 Exporting feature importance...")
    
    all_fi = []
    model_dirs = {
        "Yes": MODELS_DIR,
        "No": MODELS_NO_WEATHER_DIR
    }
    
    for weather_flag, base_dir in model_dirs.items():
        if not os.path.exists(base_dir):
            continue
        
        for model_name in os.listdir(base_dir):
            fi_path = os.path.join(base_dir, model_name, "feature_importance.csv")
            if os.path.exists(fi_path):
                df = pd.read_csv(fi_path)
                base_model = model_name.replace("_no_weather", "")
                df["model"] = base_model
                df["weather"] = weather_flag
                model_id = f"{base_model}_{weather_flag}"
                df["model_id"] = model_id
                all_fi.append(df)
    
    if all_fi:
        df = pd.concat(all_fi, ignore_index=True)
        df["model_id"] = df["model"] + "_" + df["weather"]
        df["Model Category"] = df["weather"].apply(lambda x: "With Weather" if x == "Yes" else "Without Weather")
        df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
        print(f"   ✅ feature_importance.csv ({len(df)} records)")
    else:
        print("   ⚠️ No feature importance found")


def export_benchmark_metrics():
    """Export benchmark results."""
    print("🏁 Exporting benchmark metrics...")
    
    benchmark_path = os.path.join(PREDS_DIR, "benchmark_metrics.csv")
    if os.path.exists(benchmark_path):
        df = pd.read_csv(benchmark_path)
        df["model_id"] = df.apply(lambda x: f"{x['model'].replace('_no_weather', '')}_{x['weather']}", axis=1)
        df["Model Category"] = df["weather"].apply(lambda x: "With Weather" if x == "Yes" else "Without Weather")
        # Rename for readability
        df = df.rename(columns={
            "R2": "R2 Score",
            "WAPE": "WAPE %",
            "RMSE": "RMSE (Lower is Better)",
        })
        df.to_csv(os.path.join(OUTPUT_DIR, "benchmark_metrics.csv"), index=False)
        print(f"   ✅ benchmark_metrics.csv ({len(df)} records)")
    else:
        print("   ⚠️ Benchmark not run yet")


def export_monthly_drift():
    """Export monthly drift data."""
    print("📉 Exporting monthly drift...")
    
    drift_path = os.path.join(PREDS_DIR, "drift_by_month.csv")
    if os.path.exists(drift_path):
        df = pd.read_csv(drift_path)
        df["model_id"] = df.apply(lambda x: f"{x['model'].replace('_no_weather', '')}_{x['weather']}", axis=1)
        df["Model Category"] = df["weather"].apply(lambda x: "With Weather" if x == "Yes" else "Without Weather")
        df = df.rename(columns={"RMSE": "RMSE (Lower is Better)", "R2": "R2 Score"})
        df.to_csv(os.path.join(OUTPUT_DIR, "monthly_drift.csv"), index=False)
        print(f"   ✅ monthly_drift.csv ({len(df)} records)")
    else:
        print("   ⚠️ Drift data not available")


def export_weather_impact_comparison():
    """Create specific weather impact comparison table."""
    print("🔍 Exporting weather impact comparison...")
    
    rows = []
    for model in ["xgboost", "lightgbm", "random_forest", "ridge"]:
        # With weather
        with_path = os.path.join(MODELS_DIR, model, "metrics.csv")
        without_path = os.path.join(MODELS_NO_WEATHER_DIR, model, "metrics.csv")
        
        with_weather = pd.read_csv(with_path) if os.path.exists(with_path) else None
        without_weather = pd.read_csv(without_path) if os.path.exists(without_path) else None
        
        if with_weather is not None and without_weather is not None:
            w_r2 = with_weather["R2"].iloc[0]
            wo_r2 = without_weather["R2"].iloc[0]
            w_wape = with_weather["WAPE"].iloc[0]
            wo_wape = without_weather["WAPE"].iloc[0]
            
            rows.append({
                "model": model,
                "r2_with_weather": w_r2,
                "r2_without_weather": wo_r2,
                "r2_difference": w_r2 - wo_r2,
                "r2_improvement_pct": ((w_r2 - wo_r2) / wo_r2) * 100 if wo_r2 != 0 else 0,
                "wape_with_weather": w_wape,
                "wape_without_weather": wo_wape,
                "wape_difference": w_wape - wo_wape,
                "wape_improvement_pct": ((wo_wape - w_wape) / wo_wape) * 100 if wo_wape != 0 else 0,
            })
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUTPUT_DIR, "weather_impact_comparison.csv"), index=False)
        print(f"   ✅ weather_impact_comparison.csv ({len(df)} records)")
    else:
        print("   ⚠️ Need both weather and no-weather models trained")


def export_forecast_forecasts():
    """Export recent forecast data."""
    print("🔮 Exporting forecast data...")
    
    forecast_files = glob.glob(os.path.join(PREDS_DIR, "*", "daily", "*.csv"))
    
    for ff in forecast_files:
        df = pd.read_csv(ff, parse_dates=["date"])
        model_name = os.path.basename(os.path.dirname(os.path.dirname(ff)))
        date_str = os.path.basename(ff).replace("prediction_", "").replace(".csv", "")
        df["model"] = model_name
        df["forecast_date"] = date_str
        df.to_csv(os.path.join(OUTPUT_DIR, f"forecast_{date_str}_{model_name}.csv"), index=False)
    
    print(f"   ✅ Exported {len(forecast_files)} forecast files")


def run():
    print("=" * 60)
    print("  POWER BI DATA EXPORTER")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}")
    print()
    
    export_model_metrics()
    export_predictions()
    export_daily_prices()
    export_weather_data()
    export_feature_importance()
    export_benchmark_metrics()
    export_monthly_drift()
    export_weather_impact_comparison()
    export_forecast_forecasts()
    
    print()
    print("=" * 60)
    print("  ✅ Export complete!")
    print("=" * 60)
    print(f"\n  Files saved to: {OUTPUT_DIR}")
    print("\n  To use in Power BI:")
    print("  1. Open Power BI Desktop (free)")
    print("  2. Get Data → Folder → Select powerbi_data folder")
    print("  3. Combine all CSVs or import individually")
    print("  4. Build your dashboard!")
    

if __name__ == "__main__":
    run()