"""
visualize_pipeline.py — Comprehensive EDA & Model Benchmark Report
Generates an interactive HTML report using Plotly summarizing raw vs processed data,
overall metrics, and confusion matrices for the trained models.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import evaluate
import sys
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PREDS_DIR = os.path.join(os.path.dirname(__file__), "..", "predictions")
MODELS = ["naive", "ridge", "random_forest", "xgboost", "lightgbm"]

def create_report():
    print("Loading data...")
    # Load 1 year of training data for visualization (2024)
    raw_file = os.path.join(DATA_DIR, "raw", "training", "dam_2024.csv")
    if os.path.exists(raw_file):
        df_raw = pd.read_csv(raw_file)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        # Sample 1 week of raw data for clarity
        sample_raw = df_raw[(df_raw['date'] >= '2024-01-01') & (df_raw['date'] <= '2024-01-07')].copy()
        sample_raw['datetime'] = pd.to_datetime(sample_raw['date'].astype(str)) + pd.to_timedelta((sample_raw['time_block'].str.split('-').str[0] + ':00').str.strip(), errors='coerce')
    else:
        sample_raw = pd.DataFrame()

    print("Loading predictions and calculating metrics...")
    metrics_list = []
    cm_dict = {}
    
    for model in MODELS:
        pred_path = os.path.join(PREDS_DIR, model, "test_predictions.csv")
        if os.path.exists(pred_path):
            df_pred = pd.read_csv(pred_path)
            y_true = df_pred["mcp_rs_per_mwh"].values
            y_pred_val = df_pred["predicted_mcp"].values
            
            m = evaluate.compute_all_metrics(y_true, y_pred_val)
            m["Model"] = model.replace("_", " ").title()
            metrics_list.append(m)
            
            # Compute Confusion Matrix
            true_dir = evaluate.price_direction(y_true)
            pred_dir = evaluate.price_direction(y_pred_val)
            cm = confusion_matrix(true_dir, pred_dir, labels=[0, 1])
            cm_dict[model] = cm

    df_metrics = pd.DataFrame(metrics_list)
    
    # --------------------------------------------------------------------------------
    # 1. HTML Header
    html_content = [
        "<html><head><title>Power Price Pipeline Report</title>",
        "<style>body { font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }",
        "h1, h2 { color: #2c3e50; } .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; }",
        ".metric-table { width: 100%; border-collapse: collapse; text-align: left; }",
        ".metric-table th, .metric-table td { padding: 12px; border-bottom: 1px solid #ddd; }",
        ".metric-table th { background-color: #f2f2f2; }</style></head><body>",
        "<h1>⚡ Power Price Prediction - End-to-End Report</h1>"
    ]

    # --------------------------------------------------------------------------------
    # 2. Raw vs Processed Data Sample Visualization
    if not sample_raw.empty:
        html_content.append("<div class='card'><h2>1. Raw Data Insights (1 Week Sample: Jan 2024)</h2>")
        html_content.append("<p>Illustrating the true 96-block 15-minute granularity fetched by the new scraper.</p>")
        
        fig_raw = px.line(sample_raw, x="datetime", y="mcp_rs_per_mwh", title="Actual Market Clearing Price (Rs/MWh)")
        fig_raw.update_layout(xaxis_title="Date & Time", yaxis_title="MCP (Rs/MWh)", height=400)
        html_content.append(fig_raw.to_html(full_html=False, include_plotlyjs='cdn'))
        html_content.append("</div>")

    # --------------------------------------------------------------------------------
    # 3. Model Benchmark Tabular Data
    html_content.append("<div class='card'><h2>2. Model Metrics Comparison</h2>")
    if not df_metrics.empty:
        # Reorder columns
        cols = ['Model', 'RMSE', 'MAE', 'MAPE', 'WAPE', 'R2', 'AUC_ROC', 'F1', 'Precision', 'Recall', 'Accuracy']
        df_m = df_metrics[[c for c in cols if c in df_metrics.columns]]
        html_content.append(df_m.to_html(classes='metric-table', index=False, float_format="%.4f"))
    html_content.append("</div>")

    # --------------------------------------------------------------------------------
    # 4. Graphical Model Comparison (Bar Charts)
    if not df_metrics.empty:
        html_content.append("<div class='card'><h2>3. Graphical Performance Summary</h2>")
        
        fig_comp = make_subplots(rows=2, cols=2, subplot_titles=("RMSE (Lower is better)", "WAPE (Lower is better)", "R² Score (Higher is better)", "Accuracy (Higher is better)"))
        
        # Sort by RMSE
        df_rmse = df_metrics.sort_values("RMSE")
        fig_comp.add_trace(go.Bar(x=df_rmse["Model"], y=df_rmse["RMSE"], marker_color='#EF4444', showlegend=False), row=1, col=1)
        
        # Sort by WAPE
        df_wape = df_metrics.sort_values("WAPE")
        fig_comp.add_trace(go.Bar(x=df_wape["Model"], y=df_wape["WAPE"], marker_color='#F59E0B', showlegend=False), row=1, col=2)
        
        # Sort by R2
        df_r2 = df_metrics.sort_values("R2", ascending=False)
        fig_comp.add_trace(go.Bar(x=df_r2["Model"], y=df_r2["R2"], marker_color='#10B981', showlegend=False), row=2, col=1)
        
        # Sort by Accuracy
        df_acc = df_metrics.sort_values("Accuracy", ascending=False)
        fig_comp.add_trace(go.Bar(x=df_acc["Model"], y=df_acc["Accuracy"], marker_color='#3B82F6', showlegend=False), row=2, col=2)
        
        fig_comp.update_layout(height=700, template="plotly_white")
        html_content.append(fig_comp.to_html(full_html=False, include_plotlyjs=False))
        html_content.append("</div>")

    # --------------------------------------------------------------------------------
    # 5. Confusion Matrices
    if cm_dict:
        html_content.append("<div class='card'><h2>4. Price Direction Confusion Matrices</h2>")
        html_content.append("<p>Testing the models' ability to predict whether the price increases by >1% compared to the previous 15-minute block. <i>Class 0: Flat/Down, Class 1: Up.</i></p>")
        
        fig_cm = make_subplots(rows=1, cols=len(cm_dict), subplot_titles=[m.replace("_", " ").title() for m in cm_dict.keys()])
        
        for idx, (model, cm) in enumerate(cm_dict.items()):
            z_text = [[str(y) for y in x] for x in cm]
            fig_cm.add_trace(
                go.Heatmap(
                    z=cm, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], 
                    text=z_text, texttemplate="%{text}", colorscale="Blues", showscale=False
                ),
                row=1, col=idx+1
            )
        
        fig_cm.update_layout(height=400, template="plotly_white")
        html_content.append(fig_cm.to_html(full_html=False, include_plotlyjs=False))
        html_content.append("</div>")

    html_content.append("</body></html>")

    out_file = os.path.join(os.path.dirname(__file__), "..", "pipeline_report.html")
    with open(out_file, "w") as f:
        f.write("\n".join(html_content))
    print(f"Report successfully generated at: {out_file}")

if __name__ == "__main__":
    create_report()
