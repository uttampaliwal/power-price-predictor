## Data Description

Due to file size constraints, raw and processed datasets are **not included** in this repository.

### Data Source
- **Exchange**: Indian Energy Exchange (IEX)
- **Market**: Day-Ahead Market (DAM)
- **URL**: https://www.iexindia.com/market-data/day-ahead-market/market-snapshot

### Granularity
- 15-minute blocks (96 blocks per day)
- Date range: Jan 2020 – present

### Data Fields
| Field | Description |
|-------|-------------|
| `date` | Date of the time block |
| `block` | Block number (1-96) |
| `time_block` | Human-readable time (HH:MM-HH:MM) |
| `purchase_bid_mw` | Purchase bid in MW |
| `sell_bid_mw` | Sell bid in MW |
| `mcv_mw` | Market Clearing Volume in MW |
| `mcp_rs_per_mwh` | Market Clearing Price (Rs/MWh) — TARGET |

### Files (Generated Locally)

| Directory | Description | Size |
|----------|-------------|------|
| `data/raw/training/` | Monthly DAM CSV files (Jan 2020 – Dec 2024) | ~30 MB |
| `data/raw/holdout/` | Monthly DAM CSV files (Jan 2025 – present) | ~15 MB |
| `data/raw/weather_*.csv` | Delhi & Mumbai temperatures (Open-Meteo API) | ~5 MB |
| `data/processed/` | Feature-engineered Parquet files | ~50 MB |
| `models/` | Trained model artifacts (.json, .pkl) | ~460 MB |
| `predictions/` | Model predictions for all dates | ~22 MB |
| `powerbi_data/` | CSV exports for Power BI dashboards | ~50 MB |

### How to Generate Data

Run the data pipeline to recreate all files locally:

```bash
# 1. Fetch historical prices from IEX
python src/fetch_data.py --start 2020-01-01 --end 2026-04-08

# 2. Fetch weather data
python src/fetch_weather.py --start 2020-01-01 --end 2026-04-08

# 3. Generate features
python src/preprocess.py --split training
python src/preprocess.py --split holdout

# 4. Train models
python src/models/xgboost_model.py
python src/models/lightgbm_model.py

# 5. Export to Power BI
python src/powerbi_exporter.py
```

### Sample Data

A minimal sample dataset is provided in `data/sample_dam.csv` for code testing and demonstration.

### Why Data Is Not Pushed

1. **File size**: GitHub has a 100 MB hard limit per file; datasets exceed this
2. **Proprietary**: IEX data licensing restrictions
3. **Best practice**: Real ML projects never commit production data to Git

### Key Insight

Time-based features (hour, day-of-week, season) explain 99% of price variance.
Weather adds only +0.1% improvement. The model works reliably with price history alone.

---

*See [README.md](../README.md) for full project overview and [OPERATIONS_GUIDE.md](../OPERATIONS_GUIDE.md) for pipeline commands.*