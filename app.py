import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import glob
import subprocess
import shutil
from datetime import date, timedelta
import joblib
import urllib3

# Ensure the src module can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import run as run_prediction
from fetch_data import fetch_all as _fetch_dam_data, _get_last_date_in_csv

# Configuration & Styling

st.set_page_config(
    page_title="IEX Power Price Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import (
    DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_DIR, MODELS_NO_WEATHER_DIR, PREDS_DIR, SRC_DIR, MODEL_LIST, HOLDOUT_START_DATE
)

# Overwrite config for specific needs in dashboard if necessary
ALL_TRAINABLE_MODELS = [m for m in MODEL_LIST if m not in ["prophet", "tft"]]
DATA_DIR = DATA_PROCESSED_DIR
RAW_DATA_DIR = DATA_RAW_DIR


def inject_premium_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Animated gradient header */
        .hero-banner {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            background-size: 200% 200%;
            animation: gradientShift 8s ease infinite;
            padding: 28px 32px;
            border-radius: 16px;
            margin-bottom: 24px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        @keyframes gradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .hero-banner h1 { color: #fff; margin: 0 0 4px 0; font-size: 1.8rem; }
        .hero-banner p  { color: rgba(255,255,255,0.65); margin: 0; font-size: 0.95rem; }

        /* Glassmorphism Metric Cards */
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(12px);
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 30px rgba(59,130,246,0.15);
        }

        /* Headers */
        h1, h2, h3, h4 {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
        }

        /* Buttons */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #3B82F6, #2563EB);
            color: white;
            border-radius: 10px;
            border: none;
            padding: 12px 28px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(59,130,246,0.3);
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #2563EB, #1D4ED8);
            box-shadow: 0 6px 25px rgba(59,130,246,0.5);
            transform: translateY(-2px);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #12141a;
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        /* Plotly chart wrapper */
        [data-testid="stPlotlyChart"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.06);
        }

        /* Dataframe styling */
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Separator */
        hr { border-color: rgba(255,255,255,0.08); }
        </style>
    """, unsafe_allow_html=True)

inject_premium_css()

# Data Loading
@st.cache_data
def load_historical_data():
    """Load both training + holdout features for browsing & actual MCP."""
    frames = []
    for fname in ["holdout_features.parquet", "training_features.parquet"]:
        p = os.path.join(DATA_DIR, fname)
        if os.path.exists(p):
            df = pd.read_parquet(p, columns=["date", "time_block", "block", "mcp_rs_per_mwh"])
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_values(["date", "block"]).drop_duplicates(subset=["date", "block"])
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_all_model_metrics():
    """Read metrics.csv from every trained model directory (both with and without weather)."""
    rows = []
    
    # Models WITH weather
    for m in MODEL_LIST:
        p = os.path.join(MODELS_DIR, m, "metrics.csv")
        if os.path.exists(p):
            row = pd.read_csv(p).iloc[0].to_dict()
            row["weather"] = "Yes"
            rows.append(row)
    
    # Models WITHOUT weather
    no_weather_models = ["xgboost", "lightgbm", "random_forest", "ridge"]
    for m in no_weather_models:
        p = os.path.join(MODELS_NO_WEATHER_DIR, m, "metrics.csv")
        if os.path.exists(p):
            row = pd.read_csv(p).iloc[0].to_dict()
            row["weather"] = "No"
            rows.append(row)
    
    return pd.DataFrame(rows) if rows else pd.DataFrame()

@st.cache_data
def load_feature_importance(model_name="xgboost", include_weather=True):
    """Load feature importance for a specific model and weather configuration."""
    if include_weather:
        base_dir = MODELS_DIR
    else:
        base_dir = MODELS_NO_WEATHER_DIR
    p = os.path.join(base_dir, model_name, "feature_importance.csv")
    if os.path.exists(p):
        return pd.read_csv(p).sort_values("importance", ascending=False)
    return pd.DataFrame()

def fetch_live_weather(lat, lon):
    """Fetch both actual and apparent (feels-like) temperature for a city."""
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        http = urllib3.PoolManager(cert_reqs='CERT_NONE')
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,apparent_temperature",
            "timezone": "Asia/Kolkata"
        }
        response = http.request('GET', url, fields=params, timeout=10.0)
        
        if response.status == 200:
            data = response.json()
            current = data.get("current", {})
            actual = current.get("temperature_2m")
            apparent = current.get("apparent_temperature")
            return actual, apparent
        else:
            print(f"Weather API error: status {response.status}")
    except Exception as e:
        print(f"Weather fetch error: {e}")
    return None, None


# Sidebar

st.sidebar.markdown("# ⚡ Prediction Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Live Tracker", "Forecast Sandbox", "Model Scorecard", "Data Management"],
    index=0,
)

# Best Model
metrics_df = load_all_model_metrics()
if not metrics_df.empty:
    champ = metrics_df.loc[metrics_df["R2"].idxmax()]
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 🏆 Best Model")
    st.sidebar.success(f"**{champ['model'].upper()}**  —  R² = {champ['R2']:.4f}")

# Live weather
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌡️ Live Weather")

delhi_actual, delhi_apparent = fetch_live_weather(28.6139, 77.2090)
mumbai_actual, mumbai_apparent = fetch_live_weather(19.0760, 72.8777)

def _weather_card(city, actual, apparent):
    if actual is not None and apparent is not None:
        delta = apparent - actual
        delta_sign = "+" if delta >= 0 else ""
        # Color: green if feels cooler, red/orange if feels hotter
        delta_color = "#EF4444" if delta > 0 else "#10B981"
        return f"""
        <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
                    border-radius:12px; padding:12px 14px; margin-bottom:8px;">
            <div style="font-size:0.75rem; color:rgba(255,255,255,0.5); margin-bottom:4px;">{city}</div>
            <div style="display:flex; align-items:baseline; gap:6px;">
                <span style="font-size:1.3rem; font-weight:700; color:#fff;">{actual}°C</span>
                <span style="font-size:0.75rem; color:rgba(255,255,255,0.4);">actual</span>
            </div>
            <div style="display:flex; align-items:baseline; gap:6px; margin-top:4px;">
                <span style="font-size:1rem; font-weight:600; color:#F59E0B;">{apparent}°C</span>
                <span style="font-size:0.7rem; color:rgba(255,255,255,0.4);">feels like</span>
                <span style="font-size:0.7rem; font-weight:600; color:{delta_color};">({delta_sign}{delta:.1f}°)</span>
            </div>
        </div>
        """
    return f"""
    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
                border-radius:12px; padding:12px 14px; margin-bottom:8px;">
        <div style="font-size:0.75rem; color:rgba(255,255,255,0.5);">{city}</div>
        <div style="font-size:1.1rem; color:rgba(255,255,255,0.3); margin-top:4px;">N/A</div>
    </div>
    """

st.sidebar.markdown(_weather_card("🏛️ Delhi", delhi_actual, delhi_apparent), unsafe_allow_html=True)
st.sidebar.markdown(_weather_card("🌊 Mumbai", mumbai_actual, mumbai_apparent), unsafe_allow_html=True)
st.sidebar.caption("Actual = air temp · Feels Like = model input")

# Data freshness
df_hist = load_historical_data()
if not df_hist.empty:
    latest = df_hist["date"].max()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"📅 Data up to: **{latest.date()}**")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · Open-Meteo · XGBoost")


# PAGE 1: LIVE TRACKER

if page == "Live Tracker":
    st.markdown("""<div class="hero-banner">
        <h1>⚡ Live Market Tracker</h1>
        <p>Browse historical Market Clearing Prices (MCP) from the IEX Day-Ahead Market.</p>
    </div>""", unsafe_allow_html=True)

    if df_hist.empty:
        st.warning("No historical data found. Run the data pipeline first.")
    else:
        available_dates = sorted(df_hist["date"].dt.date.unique(), reverse=True)
        selected_date = st.date_input("Select Date", value=available_dates[0],
                                      min_value=available_dates[-1], max_value=available_dates[0])

        df_day = df_hist[df_hist["date"].dt.date == selected_date].sort_values("block").copy()

        if df_day.empty:
            st.warning(f"No data available for {selected_date}.")
        else:
            prev_date = selected_date - timedelta(days=1)
            df_prev = df_hist[df_hist["date"].dt.date == prev_date].sort_values("block")

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            avg_price = df_day["mcp_rs_per_mwh"].mean()
            prev_avg  = df_prev["mcp_rs_per_mwh"].mean() if not df_prev.empty else avg_price
            diff_avg = avg_price - prev_avg
            c1.metric("Avg Price (RTC)", f"₹{avg_price:,.0f}", 
                      delta=f"{'+' if diff_avg > 0 else ''}{diff_avg:,.0f} ₹", 
                      delta_color="normal")

            max_price = df_day["mcp_rs_per_mwh"].max()
            prev_max  = df_prev["mcp_rs_per_mwh"].max() if not df_prev.empty else max_price
            diff_max = max_price - prev_max
            c2.metric("Peak Price", f"₹{max_price:,.0f}", 
                      delta=f"{'+' if diff_max > 0 else ''}{diff_max:,.0f} ₹", 
                      delta_color="normal")

            min_price = df_day["mcp_rs_per_mwh"].min()
            prev_min  = df_prev["mcp_rs_per_mwh"].min() if not df_prev.empty else min_price
            diff_min = min_price - prev_min
            c3.metric("Min Price", f"₹{min_price:,.0f}", 
                      delta=f"{'+' if diff_min > 0 else ''}{diff_min:,.0f} ₹", 
                      delta_color="normal")

            peak_block = df_day.loc[df_day["mcp_rs_per_mwh"].idxmax(), "time_block"]
            c4.metric("Peak Block", str(peak_block))

            # Volatility indicator
            day_std = df_day["mcp_rs_per_mwh"].std()
            week_ago = selected_date - timedelta(days=7)
            df_week = df_hist[(df_hist["date"].dt.date >= week_ago) & (df_hist["date"].dt.date <= selected_date)]
            avg_daily_std = df_week.groupby(df_week["date"].dt.date)["mcp_rs_per_mwh"].std().mean()
            if pd.notna(avg_daily_std) and avg_daily_std > 0:
                vol_ratio = day_std / avg_daily_std
                vol_label = "🔴 High" if vol_ratio > 1.3 else ("🟡 Moderate" if vol_ratio > 0.8 else "🟢 Low")
                st.info(f"**Volatility**: {vol_label}  (Today σ = ₹{day_std:,.0f}  vs  7-day avg σ = ₹{avg_daily_std:,.0f})")

            # Main price chart
            st.markdown(f"### Price Curve — {selected_date}")
            hours = df_day["block"].values / 4
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hours, y=df_day["mcp_rs_per_mwh"],
                                     mode="lines+markers", name="MCP",
                                     line=dict(color="#10B981", width=3, shape="spline"),
                                     fill="tozeroy", fillcolor="rgba(16,185,129,0.08)"))
            fig.update_layout(xaxis_title="Hour of Day", yaxis_title="MCP (₹/MWh)",
                              xaxis=dict(tickmode="linear", tick0=0, dtick=2),
                              height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, width="stretch")

            # 7-day trend mini
            st.markdown("### 7-Day Average Price Trend")
            df_trend = df_hist[(df_hist["date"].dt.date >= week_ago) & (df_hist["date"].dt.date <= selected_date)]
            if not df_trend.empty:
                daily_avg = df_trend.groupby(df_trend["date"].dt.date)["mcp_rs_per_mwh"].mean().reset_index()
                daily_avg.columns = ["Date", "Avg MCP"]
                fig_trend = px.bar(daily_avg, x="Date", y="Avg MCP",
                                   color_discrete_sequence=["#3B82F6"],
                                   text_auto=".0f")
                fig_trend.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                        yaxis_title="Avg MCP (₹/MWh)")
                st.plotly_chart(fig_trend, width="stretch")



# PAGE 2: FORECAST SANDBOX

elif page == "Forecast Sandbox":
    st.markdown("""<div class="hero-banner">
        <h1>🔮 Forecast Sandbox</h1>
        <p>Run real-time ML inference. Compare models. Inspect individual blocks.</p>
    </div>""", unsafe_allow_html=True)

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        target_date = st.date_input("Target Date", value=date.today() + timedelta(days=1))
    with col_in2:
        compare_mode = st.checkbox("Multi-model comparison", value=False)
        if compare_mode:
            selected_models = st.multiselect("Select Models", MODEL_LIST + ["naive"], default=["xgboost", "lightgbm"])
        else:
            selected_models = [st.selectbox("Predictive Model", MODEL_LIST + ["naive"])]

    if st.button("Generate Forecast", icon="⚡"):
        results = {}
        for model_name in selected_models:
            with st.spinner(f"Running {model_name}..."):
                try:
                    df = run_prediction(model_name=model_name, target_date=target_date, save_files=False)
                    results[model_name] = df
                except Exception as e:
                    st.error(f"Error with {model_name}: {e}")
        if results:
            st.session_state["forecast_results"] = results
            st.session_state["forecast_date"] = str(target_date)
            st.success(f"Forecast generated for {target_date}!")

    # --- Render results ---
    if "forecast_results" in st.session_state:
        results = st.session_state["forecast_results"]
        
        model_names = list(results.keys())
        best_model = model_names[0]
        if not metrics_df.empty:
            best_r2 = -float("inf")
            for m in model_names:
                m_row = metrics_df[metrics_df["model"] == m]
                if not m_row.empty:
                    m_r2 = m_row["R2"].iloc[0]
                    if m_r2 > best_r2:
                        best_r2 = m_r2
                        best_model = m
                        
        if len(model_names) > 1:
            st.markdown("### 🔍 Select Model for Metrics & Inspector")
            primary_model = st.radio("Selected Model:", options=model_names, index=model_names.index(best_model), horizontal=True)
            st.markdown("---")
        else:
            primary_model = model_names[0]
            
        df_pred = results[primary_model]

        has_act = "actual_mcp" in df_pred.columns and not df_pred["actual_mcp"].isna().all()
        
        # Metrics for primary model
        m1, m2, m3, m4 = st.columns(4)
        p_peak = df_pred['predicted_mcp'].max()
        p_min = df_pred['predicted_mcp'].min()
        p_mean = df_pred['predicted_mcp'].mean()
        p_peak_idx = df_pred["predicted_mcp"].idxmax()
        p_peak_block = df_pred.loc[p_peak_idx, "time_block"]

        def _render_card(col, title, val, act=None):
            sub = ""
            if act is not None:
                sub = f'<div style="font-size:0.8rem; color:rgba(255,255,255,0.6); margin-top:6px; display:inline-block; padding:3px 10px; background:rgba(255,255,255,0.08); border-radius:12px;">{act}</div>'
            col.markdown(f"""
            <div data-testid="stMetric" style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1); 
                        backdrop-filter:blur(12px); padding:20px; border-radius:14px; 
                        box-shadow:0 4px 20px rgba(0,0,0,0.2); transition: transform 0.2s ease, box-shadow 0.2s ease;">
                <div style="font-size:0.9rem; color:rgba(255,255,255,0.6); margin-bottom:8px; font-weight:500;">{title}</div>
                <div style="font-size:1.8rem; font-weight:700; color:#fff; line-height:1.2;">{val}</div>
                {sub}
            </div>
            """, unsafe_allow_html=True)

        if has_act:
            a_peak = df_pred['actual_mcp'].max()
            a_min = df_pred['actual_mcp'].min()
            a_mean = df_pred['actual_mcp'].mean()
            # Handle potential NaNs in actual_mcp
            valid_actuals = df_pred['actual_mcp'].dropna()
            if not valid_actuals.empty:
                a_peak_idx = valid_actuals.idxmax()
                a_peak_block = df_pred.loc[a_peak_idx, "time_block"]
                
                _render_card(m1, "Expected Peak", f"₹{p_peak:,.0f}", f"Actual: ₹{a_peak:,.0f}")
                _render_card(m2, "Expected Min", f"₹{p_min:,.0f}", f"Actual: ₹{a_min:,.0f}")
                _render_card(m3, "Expected Avg (RTC)", f"₹{p_mean:,.0f}", f"Actual: ₹{a_mean:,.0f}")
                _render_card(m4, "Peak Block", str(p_peak_block), str(a_peak_block))
            else:
                _render_card(m1, "Expected Peak", f"₹{p_peak:,.0f}")
                _render_card(m2, "Expected Min", f"₹{p_min:,.0f}")
                _render_card(m3, "Expected Avg (RTC)", f"₹{p_mean:,.0f}")
                _render_card(m4, "Peak Block", str(p_peak_block))
        else:
            _render_card(m1, "Expected Peak", f"₹{p_peak:,.0f}")
            _render_card(m2, "Expected Min", f"₹{p_min:,.0f}")
            _render_card(m3, "Expected Avg (RTC)", f"₹{p_mean:,.0f}")
            _render_card(m4, "Peak Block", str(p_peak_block))

        # ---- Main chart ----
        st.markdown("### 96-Block Forecast Curve")
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        colors = {"xgboost": "#3B82F6", "lightgbm": "#8B5CF6", "random_forest": "#F59E0B",
                  "ridge": "#EC4899", "naive": "#6B7280"}

        for model_name, df in results.items():
            hours = df["block"] / 4
            color = colors.get(model_name, "#3B82F6")

            # Confidence band (only for the primary model to keep it clean)
            if model_name == primary_model:
                fig.add_trace(go.Scatter(
                    x=pd.concat([pd.Series(hours), pd.Series(hours[::-1])]),
                    y=pd.concat([df["predicted_mcp"] * 1.10, (df["predicted_mcp"] * 0.90).iloc[::-1]]),
                    fill="toself", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.07)",
                    line=dict(width=0), name="±10% Band", showlegend=True, hoverinfo="skip"
                ), secondary_y=False)

            fig.add_trace(go.Scatter(
                x=hours, y=df["predicted_mcp"],
                mode="lines+markers" if len(results) == 1 else "lines",
                name=f"{model_name.replace('_',' ').title()} Prediction",
                line=dict(color=color, width=3, shape="spline"),
            ), secondary_y=False)

        # Actual MCP overlay
        if "actual_mcp" in df_pred.columns and not df_pred["actual_mcp"].isna().all():
            fig.add_trace(go.Scatter(
                x=df_pred["block"]/4, y=df_pred["actual_mcp"],
                mode="lines", name="Actual Price",
                line=dict(color="#10B981", width=3, dash="dash"),
            ), secondary_y=False)

        # Weather overlay (from primary model)
        if "delhi_weather" in df_pred.columns and not df_pred["delhi_weather"].isna().all():
            fig.add_trace(go.Scatter(
                x=df_pred["block"]/4, y=df_pred["delhi_weather"],
                mode="lines", name="Delhi Temp (°C)",
                line=dict(color="#EF4444", width=2, dash="dot"),
            ), secondary_y=True)
            
        if "mumbai_weather" in df_pred.columns and not df_pred["mumbai_weather"].isna().all():
            fig.add_trace(go.Scatter(
                x=df_pred["block"]/4, y=df_pred["mumbai_weather"],
                mode="lines", name="Mumbai Temp (°C)",
                line=dict(color="#F59E0B", width=2, dash="dot"),
            ), secondary_y=True)

        fig.update_layout(
            xaxis_title="Hour of Day",
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        fig.update_yaxes(title_text="Price (₹/MWh)", secondary_y=False)
        fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # ---- Block Inspector ----
        st.subheader("🔍 Block Inspector")
        selected_block = st.selectbox("Select Time Block", df_pred["time_block"].tolist())
        b_row = df_pred[df_pred["time_block"] == selected_block].iloc[0]

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric(f"Predicted Price", f"₹{b_row['predicted_mcp']:,.2f}")

        if pd.notna(b_row.get("actual_mcp")):
            error = b_row["predicted_mcp"] - b_row["actual_mcp"]
            bc2.metric("Actual Price", f"₹{b_row['actual_mcp']:,.2f}")
            bc3.metric("Error (Δ)", f"₹{error:,.2f}", delta=f"{error/b_row['actual_mcp']*100:.1f}%")
        else:
            bc2.metric("Actual Price", "N/A (future)")

        if pd.notna(b_row.get("delhi_weather")):
            bc3_col = bc3 if pd.isna(b_row.get("actual_mcp")) else bc4
            d_temp = b_row['delhi_weather']
            m_temp = b_row.get('mumbai_weather')
            if pd.notna(m_temp):
                bc3_col.metric("Temp (Delhi | Mumbai)", f"{d_temp}°C  |  {m_temp}°C")
            else:
                bc3_col.metric("Delhi Temp", f"{d_temp}°C")

        st.markdown("---")

        # ---- IEX Official Report ----
        st.subheader("📊 Official IEX Regulatory Summary")
        def get_iex_category(block):
            hour = block // 4
            if hour < 6 or hour == 23: return "Night (01-06, 24 Hrs)"
            if 6 <= hour < 10:  return "Morning (07-10 Hrs)"
            if 10 <= hour < 17: return "Day (11-17 Hrs)"
            if 17 <= hour < 23: return "Evening Peak (18-23 Hrs)"
            return "Unknown"

        df_pred["iex_category"] = df_pred["block"].apply(get_iex_category)
        has_actual = "actual_mcp" in df_pred.columns and not df_pred["actual_mcp"].isna().all()

        def safe_mean(subset, col):
            if subset.empty or subset[col].isna().all(): return np.nan
            return subset[col].mean()

        categories = [
            ("Average (RTC)", None),
            ("Evening Peak", "Evening Peak (18-23 Hrs)"),
            ("Non-Peak", "!Evening Peak (18-23 Hrs)"),
            ("Morning", "Morning (07-10 Hrs)"),
            ("Day", "Day (11-17 Hrs)"),
            ("Night", "Night (01-06, 24 Hrs)"),
        ]

        report_data = []
        for display_name, fcat in categories:
            if fcat is None:
                sub = df_pred
            elif fcat.startswith("!"):
                sub = df_pred[df_pred["iex_category"] != fcat[1:]]
            else:
                sub = df_pred[df_pred["iex_category"] == fcat]

            pred_val = safe_mean(sub, "predicted_mcp")
            row = {"Category": display_name, "Predicted Avg (₹)": round(pred_val, 2) if pd.notna(pred_val) else "—"}
            if has_actual:
                act_val = safe_mean(sub, "actual_mcp")
                row["Actual Avg (₹)"] = round(act_val, 2) if pd.notna(act_val) else "—"
                if pd.notna(pred_val) and pd.notna(act_val) and act_val != 0:
                    row["Error %"] = f"{(pred_val - act_val) / act_val * 100:+.1f}%"
            report_data.append(row)

        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, width="stretch", hide_index=True)

        # ---- Download Buttons ----
        col_dl1, col_dl2 = st.columns(2)
        
        display_df = df_pred.copy()
        cols = list(display_df.columns)
        if "actual_mcp" in cols and "predicted_mcp" in cols:
            cols.remove("actual_mcp")
            p_idx = cols.index("predicted_mcp")
            cols.insert(p_idx, "actual_mcp")
            display_df = display_df[cols]
            
            actual = display_df["actual_mcp"].replace(0, np.nan)
            err = (display_df["predicted_mcp"] - actual) / actual * 100
            display_df.insert(p_idx + 2, "error_pct", err)
        
        with col_dl1:
            csv_pred = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions CSV", csv_pred,
                               file_name=f"forecast_{primary_model}_{st.session_state.get('forecast_date','')}.csv",
                               mime="text/csv")
        with col_dl2:
            csv_report = report_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download IEX Summary CSV", csv_report,
                               file_name=f"iex_summary_{primary_model}_{st.session_state.get('forecast_date','')}.csv",
                               mime="text/csv")

        # ---- Exact Tabular Predictions ----
        with st.expander("📋 View Exact Tabular Predictions"):
            st.dataframe(display_df, width="stretch", hide_index=True)

    else:
        st.info("Select a date and model, then hit **Generate Forecast** to begin.")



# PAGE 3: MODEL SCORECARD

elif page == "Model Scorecard":
    st.markdown("""<div class="hero-banner">
        <h1>📈 Model Scorecard</h1>
        <p>Compare trained models on holdout data. Understand feature drivers.</p>
    </div>""", unsafe_allow_html=True)

    if metrics_df.empty:
        st.warning("No model metrics found. Train models first.")
    else:
        # --- Weather Filter ---
        st.subheader("🔍 Weather Feature Analysis")
        weather_filter = st.radio(
            "Filter by Weather Configuration:",
            ["All Models", "With Weather", "Without Weather"],
            horizontal=True
        )
        
        if weather_filter == "With Weather":
            filtered_df = metrics_df[metrics_df["weather"] == "Yes"]
        elif weather_filter == "Without Weather":
            filtered_df = metrics_df[metrics_df["weather"] == "No"]
        else:
            filtered_df = metrics_df
        
        # --- Comparison Table ---
        st.subheader("Holdout Performance")
        display_cols = ["model", "weather", "R2", "WAPE", "RMSE", "MAE", "MAPE", "AUC_ROC", "F1"]
        avail_cols = [c for c in display_cols if c in filtered_df.columns]
        styled = filtered_df[avail_cols].copy()
        styled["model"] = styled["model"].str.replace("_", " ").str.title()
        
        # Color code weather column
        def weather_color(val):
            if val == "Yes":
                return "🟢 Yes"
            elif val == "No":
                return "🔴 No"
            return val
        if "weather" in styled.columns:
            styled["weather"] = styled["weather"].apply(weather_color)
        
        st.dataframe(styled, width="stretch", hide_index=True)

        # --- Weather Impact Comparison (only when showing all models) ---
        if weather_filter == "All Models" and "weather" in metrics_df.columns:
            st.markdown("### 📊 Weather Impact Analysis")
            
            # Get base model names
            base_models = ["xgboost", "lightgbm", "random_forest", "ridge"]
            
            comparison_data = []
            for model in base_models:
                with_weather = metrics_df[(metrics_df["model"] == model) & (metrics_df["weather"] == "Yes")]
                without_weather = metrics_df[(metrics_df["model"] == f"{model}_no_weather") & (metrics_df["weather"] == "No")]
                
                if not with_weather.empty and not without_weather.empty:
                    w_r2 = with_weather["R2"].iloc[0]
                    wo_r2 = without_weather["R2"].iloc[0]
                    w_wape = with_weather["WAPE"].iloc[0]
                    wo_wape = without_weather["WAPE"].iloc[0]
                    
                    r2_diff = w_r2 - wo_r2
                    wape_diff = w_wape - wo_wape  # negative is better for WAPE
                    
                    comparison_data.append({
                        "Model": model.replace("_", " ").title(),
                        "R² (Weather)": w_r2,
                        "R² (No Weather)": wo_r2,
                        "R² Δ": r2_diff,
                        "WAPE (Weather)": w_wape,
                        "WAPE (No Weather)": wo_wape,
                        "WAPE Δ": wape_diff,
                    })
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                
                # Display comparison
                st.dataframe(comp_df, width="stretch", hide_index=True)
                
                # Visual comparison
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### R² Comparison")
                    fig_r2_comp = go.Figure()
                    models = comp_df["Model"].tolist()
                    fig_r2_comp.add_trace(go.Bar(
                        name="With Weather",
                        x=models,
                        y=comp_df["R² (Weather)"],
                        marker_color="#10B981"
                    ))
                    fig_r2_comp.add_trace(go.Bar(
                        name="Without Weather",
                        x=models,
                        y=comp_df["R² (No Weather)"],
                        marker_color="#EF4444"
                    ))
                    fig_r2_comp.update_layout(
                        barmode="group", height=350,
                        yaxis_title="R² (Higher is Better)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_r2_comp, width="stretch")
                
                with c2:
                    st.markdown("#### WAPE Comparison")
                    fig_wape_comp = go.Figure()
                    fig_wape_comp.add_trace(go.Bar(
                        name="With Weather",
                        x=models,
                        y=comp_df["WAPE (Weather)"],
                        marker_color="#10B981"
                    ))
                    fig_wape_comp.add_trace(go.Bar(
                        name="Without Weather",
                        x=models,
                        y=comp_df["WAPE (No Weather)"],
                        marker_color="#EF4444"
                    ))
                    fig_wape_comp.update_layout(
                        barmode="group", height=350,
                        yaxis_title="WAPE % (Lower is Better)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_wape_comp, width="stretch")
                
                # Summary
                st.markdown("#### 📋 Summary")
                improved = comp_df[comp_df["R² Δ"] > 0]
                degraded = comp_df[comp_df["R² Δ"] < 0]
                if not improved.empty:
                    st.success(f"✅ Weather improved R² for: {', '.join(improved['Model'].tolist())}")
                if not degraded.empty:
                    st.warning(f"⚠️ Weather reduced R² for: {', '.join(degraded['Model'].tolist())}")
                
        # --- Bar charts (filtered) ---
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### R² Score (Higher is Better)")
            fig_r2 = px.bar(filtered_df.sort_values("R2", ascending=True),
                            x="R2", y="model", orientation="h",
                            color="R2", color_continuous_scale="Blues",
                            text_auto=".4f")
            fig_r2.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                 coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig_r2, width="stretch")

        with c2:
            st.markdown("#### WAPE % (Lower is Better)")
            fig_wape = px.bar(filtered_df.sort_values("WAPE", ascending=False),
                              x="WAPE", y="model", orientation="h",
                              color="WAPE", color_continuous_scale="Reds_r",
                              text_auto=".2f")
            fig_wape.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                   coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig_wape, width="stretch")

        st.markdown("---")

        # --- Feature Importance ---
        st.subheader("🧬 Feature Importance")
        
        # Model selection based on filter
        if weather_filter == "With Weather":
            available_models = [c for c in metrics_df[metrics_df["weather"] == "Yes"]['model'].tolist() 
                               if os.path.exists(os.path.join(MODELS_DIR, c, "feature_importance.csv"))]
        elif weather_filter == "Without Weather":
            available_models = [c.replace("_no_weather", "") for c in metrics_df[metrics_df["weather"] == "No"]['model'].tolist()]
            available_models = [m for m in available_models 
                               if os.path.exists(os.path.join(MODELS_NO_WEATHER_DIR, m, "feature_importance.csv"))]
        else:
            # Show both
            available_with = [c for c in metrics_df[metrics_df["weather"] == "Yes"]['model'].tolist() 
                             if os.path.exists(os.path.join(MODELS_DIR, c, "feature_importance.csv"))]
            available_without = [c.replace("_no_weather", "") for c in metrics_df[metrics_df["weather"] == "No"]['model'].tolist()]
            available_without = [m for m in available_without 
                                if os.path.exists(os.path.join(MODELS_NO_WEATHER_DIR, m, "feature_importance.csv"))]
            available_models = available_with + available_without
        
        if not available_models:
            st.warning("No feature importance files found for trained models.")
        else:
            # Model selection with weather indicator
            model_options = []
            for m in available_models:
                weather_path = os.path.join(MODELS_DIR, m, "feature_importance.csv")
                no_weather_path = os.path.join(MODELS_NO_WEATHER_DIR, m, "feature_importance.csv")
                if os.path.exists(weather_path) and os.path.exists(no_weather_path):
                    model_options.append(f"{m} (Both)")
                elif os.path.exists(weather_path):
                    model_options.append(f"{m} (Weather)")
                elif os.path.exists(no_weather_path):
                    model_options.append(f"{m} (No Weather)")
            
            selected_model = st.selectbox("Select Model for Feature Importance", model_options)
            
            # Determine if we show with or without weather
            base_model = selected_model.split(" (")[0]
            if "(Both)" in selected_model:
                # Let user choose which version
                weather_choice = st.radio("Weather Configuration:", ["With Weather", "Without Weather"], horizontal=True)
                include_weather = (weather_choice == "With Weather")
            elif "(Weather)" in selected_model:
                include_weather = True
            else:
                include_weather = False
            
            fi = load_feature_importance(base_model, include_weather)
            if not fi.empty:
                fi_top = fi.head(12)
                fig_fi = px.bar(fi_top.iloc[::-1], x="importance", y="feature", orientation="h",
                                color="importance", color_continuous_scale="Viridis",
                                text_auto=".4f")
                fig_fi.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                                     coloraxis_showscale=False, yaxis_title="",
                                     xaxis_title="Gain / Importance")
                st.plotly_chart(fig_fi, width="stretch")
    
                # Highlight weather features if showing weather model
                if include_weather:
                    weather_feats = fi[fi["feature"].str.contains("temp", case=False, na=False)]
                    if not weather_feats.empty:
                        total_imp = fi["importance"].sum()
                        weather_pct = weather_feats["importance"].sum() / total_imp * 100
                        st.info(f"🌡️ **Weather features** contribute **{weather_pct:.1f}%** of total model importance for {base_model.replace('_', ' ').title()}.")
            else:
                st.warning(f"Feature importance file is empty for {base_model}.")



# PAGE 4: DATA MANAGEMENT

elif page == "Data Management":
    st.markdown("""<div class="hero-banner">
        <h1>🗄️ Data Management</h1>
        <p>Fetch new data, train models, run benchmarks, and manage your train/holdout split.</p>
    </div>""", unsafe_allow_html=True)

    # ── Helper: get date range from a split directory ──
    def _get_split_info(split: str):
        """Return (min_date, max_date, total_rows, file_count) for a split."""
        split_dir = os.path.join(RAW_DATA_DIR, split)
        files = sorted(glob.glob(os.path.join(split_dir, "dam_*.csv")))
        if not files:
            return None, None, 0, 0
        min_d, max_d, total = None, None, 0
        for f in files:
            try:
                df = pd.read_csv(f, usecols=["date"], parse_dates=["date"])
                if df.empty:
                    continue
                fmin, fmax = df["date"].min().date(), df["date"].max().date()
                total += len(df)
                if min_d is None or fmin < min_d:
                    min_d = fmin
                if max_d is None or fmax > max_d:
                    max_d = fmax
            except Exception:
                pass
        return min_d, max_d, total, len(files)

    # ── Helper: run a shell command and stream output ──
    def _run_pipeline_cmd(cmd: list[str], status_container):
        """Run a command, streaming output to a Streamlit status container."""
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
            cwd=os.path.dirname(__file__)
        )
        if result.stdout:
            status_container.code(result.stdout[-3000:], language="text")
        if result.returncode != 0:
            status_container.error(f"Command failed (exit code {result.returncode})")
            if result.stderr:
                status_container.code(result.stderr[-2000:], language="text")
            return False
        return True


    # Section 1: Data Status

    st.subheader("📊 Data Status")

    train_min, train_max, train_rows, train_files = _get_split_info("training")
    hold_min, hold_max, hold_rows, hold_files = _get_split_info("holdout")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Training Data")
        if train_min:
            st.metric("Date Range", f"{train_min} → {train_max}")
            st.metric("Total Rows", f"{train_rows:,}")
            st.caption(f"{train_files} CSV file(s)")
        else:
            st.warning("No training data found.")

    with c2:
        st.markdown("#### Holdout Data")
        if hold_min:
            st.metric("Date Range", f"{hold_min} → {hold_max}")
            st.metric("Total Rows", f"{hold_rows:,}")
            st.caption(f"{hold_files} CSV file(s)")
        else:
            st.warning("No holdout data found.")

    # Coverage timeline
    if train_min and hold_min:
        total_train_months = (train_max.year - train_min.year) * 12 + (train_max.month - train_min.month)
        total_hold_months = (hold_max.year - hold_min.year) * 12 + (hold_max.month - hold_min.month)
        total_months = total_train_months + total_hold_months
        train_pct = total_train_months / total_months * 100 if total_months > 0 else 0
        hold_pct = 100 - train_pct

        st.markdown(f"""
        <div style="margin: 16px 0;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span style="font-size:0.85rem; color:rgba(255,255,255,0.6);">
                    Training ({total_train_months}mo, {train_pct:.0f}%)
                </span>
                <span style="flex-grow:1;"></span>
                <span style="font-size:0.85rem; color:rgba(255,255,255,0.6);">
                    Holdout ({total_hold_months}mo, {hold_pct:.0f}%)
                </span>
            </div>
            <div style="display:flex; height:24px; border-radius:12px; overflow:hidden; border:1px solid rgba(255,255,255,0.1);">
                <div style="width:{train_pct}%; background:linear-gradient(90deg,#3B82F6,#2563EB); display:flex; align-items:center; justify-content:center;">
                    <span style="font-size:0.7rem; color:white; font-weight:600;">{train_min} → {train_max}</span>
                </div>
                <div style="width:{hold_pct}%; background:linear-gradient(90deg,#F59E0B,#D97706); display:flex; align-items:center; justify-content:center;">
                    <span style="font-size:0.7rem; color:white; font-weight:600;">{hold_min} → {hold_max}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


    # Feature Refresh (Global)

    st.markdown("---")
    st.subheader("🔄 Processing Center")
    st.caption("Regenerate all feature Parquet files from raw CSVs. Use this after fixing preprocessing logic.")

    # Weather configuration for preprocessing
    col_weather_1, col_weather_2 = st.columns([1, 2])
    with col_weather_1:
        include_weather = st.toggle("Include Weather Features", value=True, key="include_weather_toggle")
    
    if st.button("🚀 Refresh All Features", key="btn_refresh_all"):
        python = sys.executable
        weather_label = "with Weather" if include_weather else "without Weather"
        with st.status(f"Refreshing all features ({weather_label})...", expanded=True) as status:
            for split in ["training", "holdout"]:
                st.write(f"Processing **{split}** features ({weather_label})...")
                cmd = [python, os.path.join(SRC_DIR, "preprocess.py"), "--split", split]
                if not include_weather:
                    cmd.append("--no-weather")
                ok = _run_pipeline_cmd(cmd, st)
                if not ok:
                    status.update(label=f"❌ Failed during {split} preprocessing", state="error")
                    st.stop()
            status.update(label="✅ All features refreshed!", state="complete")
        st.cache_data.clear()
        st.success(f"Feature parquets updated successfully ({weather_label})!")

    st.markdown("---")


    # Section: Update Training Data

    st.subheader("📅 Update Training Data")
    st.caption("Manage historical data (typically 2020-2024). Use primarily for initial setup or full resets.")
    
    col_t1, col_t2, col_t3 = st.columns([1, 1, 1])
    with col_t1:
        t_start = st.date_input("Start Date", value=date(2020, 1, 1), key="t_start")
    with col_t2:
        t_end = st.date_input("End Date", value=date(2024, 12, 31), key="t_end")
    with col_t3:
        t_force = st.checkbox("Overwrite", value=False, key="t_force")

    if st.button("📥 Fetch Training Data", key="btn_fetch_training"):
        python = sys.executable
        with st.status("Fetching training data...", expanded=True) as status:
            cmd = [python, os.path.join(SRC_DIR, "fetch_data.py"), "--start", str(t_start), "--end", str(t_end), "--split", "training"]
            if t_force: cmd.append("--overwrite")
            if _run_pipeline_cmd(cmd, st):
                st.write("Fetching training weather...")
                cmd_w = [python, os.path.join(SRC_DIR, "fetch_weather.py"), "--start", str(t_start), "--end", str(t_end), "--split", "training"]
                if _run_pipeline_cmd(cmd_w, st):
                    status.update(label="✅ Training data updated!", state="complete")
                    st.success("Training data fetch complete!")
                    st.rerun()
                else:
                    status.update(label="⚠️ Weather fetch failed", state="error")
            else:
                status.update(label="❌ Data fetch failed", state="error")

    st.markdown("---")


    # Section 2: Update Holdout Data

    st.subheader("📥 Update Holdout Data")
    st.caption("Fetch new IEX DAM price data and weather data for the holdout period.")

    col_date, col_btn = st.columns([2, 1])
    with col_date:
        fetch_end_date = st.date_input(
            "Fetch data up to",
            value=date.today(),
            min_value=HOLDOUT_START_DATE,
            max_value=date.today(),
            key="fetch_end_date",
        )
    with col_btn:
        force_overwrite = st.checkbox("Force overwrite", value=False,
                                       help="Re-fetch all data from scratch instead of appending new days.")

    if hold_max:
        if fetch_end_date <= hold_max and not force_overwrite:
            st.info(f"Holdout data is already up to **{hold_max}**. Select a later date or enable 'Force overwrite'.")

    if st.button("🔄 Fetch Holdout Data", key="btn_fetch_holdout"):
        python = sys.executable
        with st.status("Updating holdout data...", expanded=True) as status:
            # Step 1: Fetch DAM data
            st.write("**Step 1/3:** Fetching IEX DAM price data...")
            cmd = [
                python, os.path.join(SRC_DIR, "fetch_data.py"),
                "--start", str(HOLDOUT_START_DATE),
                "--end", str(fetch_end_date),
                "--split", "holdout",
            ]
            if force_overwrite:
                cmd.append("--overwrite")
            ok = _run_pipeline_cmd(cmd, st)
            if not ok:
                status.update(label="❌ Failed at data fetch", state="error")
                st.stop()

            # Step 2: Fetch weather data
            st.write("**Step 2/3:** Fetching weather data...")
            cmd_weather = [
                python, os.path.join(SRC_DIR, "fetch_weather.py"),
                "--start", str(HOLDOUT_START_DATE),
                "--end", str(fetch_end_date),
                "--split", "holdout",
            ]
            ok = _run_pipeline_cmd(cmd_weather, st)
            if not ok:
                status.update(label="⚠️ Weather fetch failed (continuing...)", state="running")

            # Step 3: Preprocess
            st.write("**Step 3/3:** Preprocessing holdout features...")
            cmd_preprocess = [
                python, os.path.join(SRC_DIR, "preprocess.py"),
                "--split", "holdout",
            ]
            ok = _run_pipeline_cmd(cmd_preprocess, st)
            if not ok:
                status.update(label="❌ Failed at preprocessing", state="error")
                st.stop()

            status.update(label="✅ Holdout data updated!", state="complete")

        # Clear cache so new data is reflected
        st.cache_data.clear()
        st.success(f"Data pipeline complete! Holdout data should now be up to **{fetch_end_date}**.")
        st.rerun()

    st.markdown("---")


    # Section 3: Train Models

    st.markdown("---")
    st.subheader("🧠 Train Models")
    st.caption("Retrain models using the current training + holdout data.")

    # Weather configuration for model training
    col_train_weather_1, col_train_weather_2 = st.columns([1, 2])
    with col_train_weather_1:
        train_with_weather = st.toggle("Train WITH Weather Features", value=True, key="train_weather_toggle")
    
    if train_with_weather:
        model_label = "With Weather"
        available_models = [m for m in ALL_TRAINABLE_MODELS if m in ["xgboost", "lightgbm", "random_forest", "ridge"]]
    else:
        model_label = "Without Weather"
        available_models = ["xgboost", "lightgbm", "random_forest", "ridge"]
    
    selected_train_models = st.multiselect(
        f"Select models to train ({model_label})",
        available_models,
        default=[available_models[0]] if available_models else None,
        key="train_model_select",
    )

    if st.button("🚀 Train Selected Models", key="btn_train_models"):
        if not selected_train_models:
            st.warning("Please select at least one model.")
        else:
            python = sys.executable
            for model_name in selected_train_models:
                with st.status(f"Training {model_name} ({model_label})...", expanded=True) as status:
                    if train_with_weather:
                        script = os.path.join(SRC_DIR, "models", f"{model_name}_model.py")
                    else:
                        script = os.path.join(SRC_DIR, "models", f"{model_name}_no_weather_model.py")
                    
                    if not os.path.exists(script):
                        status.update(label=f"❌ Script not found: {os.path.basename(script)}", state="error")
                        continue
                    
                    ok = _run_pipeline_cmd([python, script], st)
                    if ok:
                        status.update(label=f"✅ {model_name} trained! ({model_label})", state="complete")
                    else:
                        status.update(label=f"❌ {model_name} training failed", state="error")

            st.cache_data.clear()
            st.success("Model training complete!")

    st.markdown("---")


    # Section 4: Run Benchmark

    st.subheader("📈 Run Benchmark")
    st.caption("Compare all trained models on the holdout dataset and generate drift plots.")

    if st.button("🏁 Run Benchmark", key="btn_benchmark"):
        python = sys.executable
        with st.status("Running benchmark...", expanded=True) as status:
            ok = _run_pipeline_cmd(
                [python, os.path.join(SRC_DIR, "benchmark.py")], st
            )
            if ok:
                status.update(label="✅ Benchmark complete!", state="complete")
            else:
                status.update(label="❌ Benchmark failed", state="error")

        st.cache_data.clear()
        st.success("Benchmark updated! Check the **Model Scorecard** page.")


    # Section 5: Power BI Export

    st.markdown("---")
    st.subheader("📊 Power BI Export")
    st.caption("Export data to CSV files for Power BI dashboard. Run after training or fetching new data.")

    col_pb1, col_pb2 = st.columns([2, 1])
    with col_pb1:
        st.info("💡 Export generates CSV files in `powerbi_data/` folder for Power BI import.")
    with col_pb2:
        if st.button("📥 Export to Power BI", key="btn_powerbi_export"):
            python = sys.executable
            with st.status("Exporting data for Power BI...", expanded=True) as status:
                ok = _run_pipeline_cmd(
                    [python, os.path.join(SRC_DIR, "powerbi_exporter.py")], st
                )
                if ok:
                    status.update(label="✅ Export complete!", state="complete")
                    st.success("Data exported to `powerbi_data/` folder!")
                else:
                    status.update(label="❌ Export failed", state="error")

    st.markdown("---")


    # Section 5: Graduate Data (Holdout → Training)

    st.subheader("🎓 Graduate Data")
    st.caption("Move the oldest holdout year(s) into the training set for improved model performance.")

    if hold_min and hold_max and train_max:
        hold_months = (hold_max.year - hold_min.year) * 12 + (hold_max.month - hold_min.month)
        graduate_enabled = hold_months >= 18

        if not graduate_enabled:
            st.info(
                f"📏 Holdout covers **{hold_months} months** ({hold_min} → {hold_max}). "
                f"Graduation is available once holdout exceeds **18 months** to maintain a healthy train/test split."
            )
        else:
            st.success(
                f"📏 Holdout is **{hold_months} months** — graduation is available!"
            )

            # Determine which holdout years can be graduated
            holdout_dir = os.path.join(RAW_DATA_DIR, "holdout")
            holdout_files = sorted(glob.glob(os.path.join(holdout_dir, "dam_*.csv")))
            holdout_years = []
            for f in holdout_files:
                basename = os.path.basename(f)
                year_str = basename.replace("dam_", "").replace(".csv", "")
                try:
                    holdout_years.append(int(year_str))
                except ValueError:
                    pass

            # Only allow graduating years that are fully complete (past years)
            graduable_years = [y for y in holdout_years if y < date.today().year]

            if not graduable_years:
                st.info("No fully completed holdout years available for graduation yet.")
            else:
                years_to_graduate = st.multiselect(
                    "Select holdout year(s) to move to training",
                    graduable_years,
                    key="graduate_years",
                )

                if years_to_graduate:
                    # Calculate before/after ranges
                    new_train_max = date(max(years_to_graduate), 12, 31)
                    if new_train_max > train_max:
                        proposed_train_max = new_train_max
                    else:
                        proposed_train_max = train_max

                    remaining_holdout_years = [y for y in holdout_years if y not in years_to_graduate]
                    if remaining_holdout_years:
                        proposed_hold_min = None
                        for ry in sorted(remaining_holdout_years):
                            rf = os.path.join(holdout_dir, f"dam_{ry}.csv")
                            try:
                                rdf = pd.read_csv(rf, usecols=["date"], parse_dates=["date"])
                                fmin = rdf["date"].min().date()
                                if proposed_hold_min is None or fmin < proposed_hold_min:
                                    proposed_hold_min = fmin
                            except Exception:
                                pass
                    else:
                        proposed_hold_min = None

                    # Show before/after comparison
                    st.markdown("#### Before & After Comparison")
                    ba1, ba2 = st.columns(2)
                    with ba1:
                        st.markdown("**🔵 BEFORE**")
                        st.markdown(f"- Training: `{train_min}` → `{train_max}`")
                        st.markdown(f"- Holdout: `{hold_min}` → `{hold_max}`")
                    with ba2:
                        st.markdown("**🟢 AFTER**")
                        st.markdown(f"- Training: `{train_min}` → `{proposed_train_max}`")
                        if proposed_hold_min:
                            st.markdown(f"- Holdout: `{proposed_hold_min}` → `{hold_max}`")
                        else:
                            st.markdown("- Holdout: ⚠️ **Empty** (all data moved)")

                    # Safety checks
                    overlap = proposed_train_max >= (proposed_hold_min or date.max)
                    reduces_training = proposed_train_max < train_max

                    if overlap:
                        st.error("❌ This would cause overlap between training and holdout. Aborting.")
                    elif reduces_training:
                        st.error("❌ This would reduce training data. Aborting.")
                    elif proposed_hold_min is None:
                        st.warning("⚠️ This will empty the holdout set. Only do this if you plan to fetch new holdout data immediately after.")
                    else:
                        st.info("✅ No overlap. Training data only grows.")

                    # Confirmation checkbox + button
                    confirm_graduate = st.checkbox(
                        f"I confirm I want to graduate year(s) **{years_to_graduate}** from holdout → training",
                        key="confirm_graduate",
                    )

                    if st.button("🎓 Graduate Data", key="btn_graduate", disabled=overlap or reduces_training):
                        if not confirm_graduate:
                            st.warning("Please check the confirmation box first.")
                        else:
                            training_dir = os.path.join(RAW_DATA_DIR, "training")
                            with st.status("Graduating data...", expanded=True) as status:
                                for yr in years_to_graduate:
                                    src_file = os.path.join(holdout_dir, f"dam_{yr}.csv")
                                    dst_file = os.path.join(training_dir, f"dam_{yr}.csv")
                                    if os.path.exists(src_file):
                                        shutil.move(src_file, dst_file)
                                        st.write(f"  Moved `dam_{yr}.csv` → training/")

                                # Reprocess both splits
                                python = sys.executable
                                st.write("Reprocessing training data...")
                                _run_pipeline_cmd(
                                    [python, os.path.join(SRC_DIR, "preprocess.py"), "--split", "training"], st
                                )

                                st.write("Reprocessing holdout data...")
                                _run_pipeline_cmd(
                                    [python, os.path.join(SRC_DIR, "preprocess.py"), "--split", "holdout"], st
                                )

                                # Update weather splits
                                st.write("Updating weather data for training split...")
                                new_train_end = date(max(years_to_graduate), 12, 31)
                                _run_pipeline_cmd([
                                    python, os.path.join(SRC_DIR, "fetch_weather.py"),
                                    "--start", str(train_min),
                                    "--end", str(new_train_end),
                                    "--split", "training",
                                ], st)

                                if proposed_hold_min:
                                    st.write("Updating weather data for holdout split...")
                                    _run_pipeline_cmd([
                                        python, os.path.join(SRC_DIR, "fetch_weather.py"),
                                        "--start", str(proposed_hold_min),
                                        "--end", str(hold_max),
                                        "--split", "holdout",
                                    ], st)

                                status.update(label="✅ Graduation complete!", state="complete")

                            st.cache_data.clear()
                            st.success("Data graduated! Please **retrain models** to take advantage of the expanded training set.")
                            st.rerun()
    else:
        st.warning("Not enough data to evaluate graduation eligibility.")

