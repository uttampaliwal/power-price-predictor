# Weather Impact Analysis

## Objective
Compare model performance WITH weather features vs WITHOUT weather features to evaluate whether weather data improves power price predictions.

## Methodology

### Data Configuration
- **Training Data**: Same date range for both configurations
- **Holdout Data**: Same date range for both configurations
- **Only Difference**: Weather features (delhi_apparent_temp, mumbai_apparent_temp) are either included or excluded

### Models Tested
- XGBoost
- LightGBM
- Random Forest
- Ridge Regression

### Training Configuration
- Same hyperparameters for both weather and no-weather versions
- Same validation split strategy (last 60 days of training)
- Same test set (holdout data)

## Results Summary

*To be filled after running training for both configurations*

| Model | R² (Weather) | R² (No Weather) | R² Δ | WAPE (Weather) | WAPE (No Weather) | WAPE Δ |
|-------|-------------|-----------------|------|----------------|-------------------|--------|
| XGBoost | - | - | - | - | - | - |
| LightGBM | - | - | - | - | - | - |
| Random Forest | - | - | - | - | - | - |
| Ridge | - | - | - | - | - | - |

## Analysis

### Key Questions Answered
1. Does adding weather features improve model performance?
2. Which models benefit most from weather data?
3. Is the improvement statistically significant?
4. Are there cases where weather features degrade performance?

### Interpretation Guide
- **R² Δ > 0**: Weather improves model (higher is better)
- **R² Δ < 0**: Weather degrades model
- **WAPE Δ < 0**: Weather improves model (lower is better)
- **WAPE Δ > 0**: Weather degrades model

## Files Generated

### With Weather
- `models/xgboost/metrics.csv`
- `models/lightgbm/metrics.csv`
- `models/random_forest/metrics.csv`
- `models/ridge/metrics.csv`

### Without Weather
- `models_no_weather/xgboost/metrics.csv`
- `models_no_weather/lightgbm/metrics.csv`
- `models_no_weather/random_forest/metrics.csv`
- `models_no_weather/ridge/metrics.csv`

### Data Files
- `data/processed/training_features.parquet` (with weather)
- `data/processed/training_features_no_weather.parquet` (without weather)
- `data/processed/holdout_features.parquet` (with weather)
- `data/processed/holdout_features_no_weather.parquet` (without weather)

## Recommendation

*To be filled after analysis*

Based on the results:
- If majority of models show improvement with weather → Recommend using weather features
- If majority show degradation → Consider removing weather features
- If mixed results → May need to tune which weather features to include or use model-specific recommendations

## How to Reproduce

1. Generate no-weather parquet files:
```bash
python src/preprocess.py --split training --no-weather
python src/preprocess.py --split holdout --no-weather
```

2. Train no-weather models:
```bash
python src/models/xgboost_no_weather_model.py
python src/models/lightgbm_no_weather_model.py
python src/models/random_forest_no_weather_model.py
python src/models/ridge_no_weather_model.py
```

3. View comparison in Streamlit Model Scorecard page with "All Models" filter selected.