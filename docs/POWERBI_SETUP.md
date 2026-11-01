# Power BI Dashboard Setup Guide

## Overview
This guide helps you build a Power BI dashboard using data exported from your power price prediction system.

## Data Export
Run the exporter to generate CSV files:
```bash
python src/powerbi_exporter.py
```

Or use the "Export to Power BI" button in the Data Management section of the Streamlit app.

## Output Files (powerbi_data/)

| File | Description | Recommended Use |
|------|-------------|-----------------|
| `model_metrics.csv` | All model performance metrics | Model comparison cards, R²/WAPE charts |
| `predictions.csv` | Actual vs predicted prices | Scatter plots, error analysis |
| `daily_prices.csv` | Historical MCP prices | Time series charts, price trends |
| `weather_holdout.csv` | Weather data | Correlation analysis with prices |
| `feature_importance.csv` | Feature importance | Bar charts, model explainability |
| `benchmark_metrics.csv` | Benchmark results | Comprehensive model comparison |
| `monthly_drift.csv` | Monthly RMSE by model | Drift analysis over time |
| `weather_impact_comparison.csv` | Weather impact | WITH vs WITHOUT comparison |

## Power BI Setup Steps

### 1. Import Data
1. Open Power BI Desktop (free version)
2. Click **Get Data** → **Folder**
3. Select the `powerbi_data` folder
4. Choose "Combine" to merge all CSVs, or import individually

### 2. Recommended Visualizations

#### A. Model Performance Overview
- **Table/Matrix**: Model | Weather | R² | WAPE | RMSE | MAPE
- **Bar Chart**: R² by model (horizontal)
- **Gauge**: Best model R²

#### B. Weather Impact Analysis
- **Stacked Bar Chart**: R² with/without weather by model
- **Line Chart**: WAPE comparison
- **Table**: Weather impact summary

#### C. Price Trends
- **Line Chart**: Daily average MCP over time
- **Area Chart**: Price distribution by hour
- **Heatmap**: Price by day-of-week and hour

#### D. Forecast vs Actual
- **Scatter Plot**: Predicted vs Actual
- **Line Chart**: Both lines overlaid
- **Histogram**: Error distribution

#### E. Feature Importance
- **Bar Chart**: Top 10 features (horizontal)
- **Treemap**: Feature importance visualization

#### F. Monthly Drift Analysis
- **Line Chart**: RMSE by month per model
- **Area Chart**: Model performance over time

### 3. Useful Calculated Columns (DAX)

```dax
// Error calculation
Error = [predicted_mcp] - [actual_mcp]

// Absolute Error
Abs Error = ABS([Error])

// Error Percentage
Error % = DIVIDE([Error], [actual_mcp]) * 100

// Day of Week Name
Day Name = FORMAT([date], "dddd")

// Month Name
Month Name = FORMAT([date], "MMMM")

// Hour Bucket
Hour Bucket = SWITCH(TRUE(),
    [hour] < 6, "Night",
    [hour] < 10, "Morning Peak",
    [hour] < 14, "Afternoon",
    [hour] < 18, "Pre-Evening",
    [hour] < 22, "Evening Peak",
    "Night"
)

// Weekend Flag
Is Weekend = IF(WEEKDAY([date]) IN {1, 7}, "Weekend", "Weekday")
```

### 4. Recommended Filters/Slicers
- Date range slider
- Model selector (multi-select)
- Weather toggle (Yes/No)
- Hour bucket
- Day of week
- Season

## Additional Analytics Ideas

### A. Seasonal Analysis
- Compare model performance across seasons (Summer, Monsoon, Winter)
- Price patterns by season

### B. Peak vs Off-Peak Analysis
- Evening peak (18-23 hrs) performance
- Night (00-06 hrs) performance

### C. Holiday Impact
- Compare holidays vs non-holidays
- Model behavior during holidays

### D. Model Confidence Intervals
- If available, visualize prediction bands

### E. Cost Savings Calculator
- Calculate potential savings using predictions vs actuals

## Auto-Refresh Setup

For automatic updates:

1. **Option A: OneDrive/SharePoint**
   - Upload CSV files to OneDrive
   - Connect Power BI to OneDrive for automatic refresh
   
2. **Option B: Power BI Dataflows**
   - Set up dataflows to pull from your data sources
   
3. **Option C: Scheduled Refresh (Pro required)**
   - Publish to Power BI Service
   - Set up scheduled refresh

For **free account**, manual export is recommended:
1. Run `python src/powerbi_exporter.py` after each update
2. Refresh data in Power BI manually

## Quick Start Checklist

- [ ] Run `python src/powerbi_exporter.py`
- [ ] Open Power BI → Get Data → Folder
- [ ] Select `powerbi_data` folder
- [ ] Create Model Performance table
- [ ] Add R² bar chart
- [ ] Add Weather Impact comparison
- [ ] Add Price Trend line chart
- [ ] Save dashboard as `.pbix`

## Support
For issues with specific visualizations or data connections, refer to Power BI documentation or check data types in the CSV files.