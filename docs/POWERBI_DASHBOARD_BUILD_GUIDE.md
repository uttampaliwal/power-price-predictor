# POWER BI DASHBOARD BUILD GUIDE
# Complete step-by-step instructions for building your Power BI Dashboard

================================================================================
PART 1: DATA PREPARATION
================================================================================

Before building the dashboard, ensure you have exported the data:

1. Open your terminal/command prompt
2. Navigate to your project folder
3. Activate your virtual environment (venv\Scripts\activate)
4. Run: python src/powerbi_exporter.py
5. Verify files in powerbi_data/ folder

You should have these files:
- model_metrics.csv
- predictions.csv
- daily_prices.csv
- weather_impact_comparison.csv
- feature_importance.csv
- benchmark_metrics.csv
- monthly_drift.csv


================================================================================
PART 2: IMPORTING DATA INTO POWER BI
================================================================================

STEP 1: Open Power BI Desktop

STEP 2: Get Data - Folder
- Click "Get Data" in the toolbar
- Select "Folder" from the list
- Click "Connect"

STEP 3: Select your data folder
- Browse to: C:\Users\uttam.paliwal\Downloads\Project\powerbi_data
- Click OK

STEP 4: Choose data loading option
- You'll see a preview showing all CSV files
- Click "Combine & Load" (top button)

STEP 5: Wait for loading
- Power BI will combine all files
- This creates a single table with a "Source.Name" column showing which file each row came from

ALTERNATIVE (RECOMMENDED): Import files separately

For better analysis, import each file separately:

1. Get Data → Text/CSV
2. Import these files one by one:
   - model_metrics.csv → Name: "Model Metrics"
   - daily_prices.csv → Name: "Daily Prices"
   - weather_impact_comparison.csv → Name: "Weather Impact"
   - predictions.csv → Name: "Predictions"
   - feature_importance.csv → Name: "Feature Importance"
   - monthly_drift.csv → Name: "Monthly Drift"

3. Do this for each file by repeating: Get Data → Text/CSV → select file


================================================================================
PART 3: DATA MODEL RELATIONSHIPS
================================================================================

After importing all files, go to "Model" view (left sidebar, 3rd icon)

Create these relationships:

1. Model Metrics ↔ Weather Impact
   - Join on: model

2. Model Metrics ↔ Feature Importance  
   - Join on: model

3. Predictions ↔ Daily Prices
   - Join on: date, block

4. Model Metrics ↔ Monthly Drift
   - Join on: model

Note: Some files may have different column names - adjust accordingly


================================================================================
PART 4: BUILDING THE DASHBOARD - PAGE BY PAGE
================================================================================

==================== PAGE 1: MODEL PERFORMANCE OVERVIEW ====================

VISUALIZATION 1: Model Metrics Table
-------------------------------------
- Visual type: Table or Matrix
- Fields: model, weather, R2, WAPE, RMSE, MAE, MAPE
- Sort by: R2 (descending)
- Format: Add conditional formatting (green for high R2, red for low)

VISUALIZATION 2: R² Score Bar Chart
-------------------------------------
- Visual type: Bar Chart (horizontal)
- Y-axis: model
- X-axis: R2
- Color: Add "weather" to Legend
- Title: "R² Score by Model (Higher is Better)"

VISUALIZATION 3: WAPE Comparison
----------------------------------
- Visual type: Bar Chart (horizontal)
- Y-axis: model  
- X-axis: WAPE
- Color: Add "weather" to Legend
- Title: "WAPE % by Model (Lower is Better)"

VISUALIZATION 4: Model Performance Cards
------------------------------------------
- Visual type: Card (4 cards)
- Card 1: Best R² = MAX of R2
- Card 2: Best WAPE = MIN of WAPE
- Card 3: Total Models = COUNT of model
- Card 4: Weather Models = COUNT of model where weather = "Yes"

VISUALIZATION 5: Filter Pane
------------------------------
Add slicers (filters):
- Model (multi-select)
- Weather (Yes/No)
- Date Range (if applicable)


==================== PAGE 2: WEATHER IMPACT ANALYSIS ======================

This is the KEY comparison page!

VISUALIZATION 1: Weather Impact Summary Table
------------------------------------------------
- Visual type: Table
- Fields: model, r2_with_weather, r2_without_weather, r2_difference, 
          wape_with_weather, wape_without_weather, wape_difference
- Format: Add data bars for numeric columns
- Add conditional formatting: green for positive R2 diff, red for negative

VISUALIZATION 2: R² Comparison Clustered Bar Chart
----------------------------------------------------
- Visual type: Clustered Bar Chart
- X-axis: model
- Y-axis: R2
- Legend: weather (Yes/No)
- Title: "R²: With Weather vs Without Weather"
- This shows side-by-side comparison

VISUALIZATION 3: WAPE Comparison Clustered Bar Chart
-------------------------------------------------------
- Visual type: Clustered Bar Chart
- X-axis: model
- Y-axis: WAPE
- Legend: weather
- Title: "WAPE: With Weather vs Without Weather"

VISUALIZATION 4: R² Improvement Gauge
--------------------------------------
- Visual type: Gauge
- Value: Average of r2_difference
- Title: "Average R² Impact"
- Color: Green if positive, Red if negative

VISUALIZATION 5: Weather Impact Summary Cards
------------------------------------------------
- Card: Models Improved = COUNT where r2_difference > 0
- Card: Models Degraded = COUNT where r2_difference < 0
- Card: Avg R² Change = AVERAGE of r2_difference
- Card: Best Improvement = MAX of r2_improvement_pct

VISUALIZATION 6: Scatter - R² vs WAPE by Model
-----------------------------------------------
- Visual type: Scatter
- X-axis: r2_with_weather
- Y-axis: wape_with_weather
- Size: r2_improvement_pct
- Tooltip: model


==================== PAGE 3: PRICE TRENDS ======================

VISUALIZATION 1: Daily MCP Price Over Time (Line Chart)
----------------------------------------------------------
- Visual type: Line Chart
- X-axis: date
- Y-axis: Average of mcp_rs_per_mwh
- Title: "Daily Average MCP Price Trend"

VISUALIZATION 2: Price by Hour of Day (Line Chart)
---------------------------------------------------
- Visual type: Line Chart
- X-axis: hour
- Y-axis: Average of mcp_rs_per_mwh
- Title: "Average Price by Hour"

VISUALIZATION 3: Price Heatmap - Hour vs Day of Week
-----------------------------------------------------
- Visual type: Matrix or Table
- Rows: day_of_week
- Columns: hour
- Values: Average of mcp_rs_per_mwh
- Format: Apply conditional formatting - color scale (green=low, red=high)

VISUALIZATION 4: Monthly Price Distribution (Box Plot Alternative)
--------------------------------------------------------------------
- Visual type: Stacked Bar Chart
- X-axis: month
- Y-axis: Average of mcp_rs_per_mwh
- Stack by: year (if available)

VISUALIZATION 5: Price Statistics Cards
----------------------------------------
- Card: Min Price = MIN of mcp_rs_per_mwh
- Card: Max Price = MAX
- Card: Avg Price = AVERAGE
- Card: Price Std Dev = STDEV


==================== PAGE 4: FORECAST ACCURACY ======================

VISUALIZATION 1: Predicted vs Actual Scatter
---------------------------------------------
- Visual type: Scatter
- X-axis: actual_mcp (or mcp_rs_per_mwh)
- Y-axis: predicted_mcp
- Size: block
- Color: model
- Add diagonal reference line (average)

VISUALIZATION 2: Prediction Error Distribution (Histogram)
-------------------------------------------------------------
- Visual type: Bar Chart (histogram)
- X-axis: Error (calculated = predicted - actual)
- Y-axis: Count
- Title: "Prediction Error Distribution"

VISUALIZATION 3: Error by Hour
-------------------------------
- Visual type: Line Chart
- X-axis: hour
- Y-axis: Average of ABS(Error)
- Title: "Average Absolute Error by Hour"

VISUALIZATION 4: MAPE by Model (Bar Chart)
-------------------------------------------
- Visual type: Bar Chart
- Y-axis: model
- X-axis: Average of MAPE
- Title: "MAPE by Model (Lower is Better)"

VISUALIZATION 5: Error Over Time
---------------------------------
- Visual type: Line Chart
- X-axis: date
- Y-axis: Average of ABS(Error)
- Color: model
- Title: "Prediction Error Over Time"


==================== PAGE 5: FEATURE IMPORTANCE ====================

VISUALIZATION 1: Top Features Bar Chart (Horizontal)
-----------------------------------------------------
- Visual type: Bar Chart (horizontal)
- Y-axis: feature
- X-axis: importance (SUM or AVG)
- Filter: Top N = 15
- Color: importance (gradient)
- Title: "Top 15 Most Important Features"

VISUALIZATION 2: Feature Importance by Model
---------------------------------------------
- Visual type: Stacked Bar Chart
- X-axis: feature
- Y-axis: importance
- Legend: model
- Filter: Top 10 features

VISUALIZATION 3: Weather Feature Impact
----------------------------------------
- Visual type: Bar Chart
- Filter: feature contains "temp"
- Show importance of weather features specifically

VISUALIZATION 4: Feature Importance Table
------------------------------------------
- Visual type: Table
- Fields: feature, importance, model, weather
- Sort by: importance (descending)


==================== PAGE 6: MONTHLY DRIFT ANALYSIS ================

VISUALIZATION 1: Monthly RMSE Trend (Line Chart)
-------------------------------------------------
- Visual type: Line Chart
- X-axis: year_month
- Y-axis: Average of RMSE
- Color: model
- Title: "Monthly RMSE Drift by Model"

VISUALIZATION 2: Model Performance Over Time (Area Chart)
----------------------------------------------------------
- Visual type: Area Chart
- X-axis: year_month
- Y-axis: R2
- Color: model
- Title: "Model R² Over Time"

VISUALIZATION 3: Monthly Performance Table
-------------------------------------------
- Visual type: Table
- Fields: year_month, model, RMSE, MAE, R2
- Sort by: year_month

VISUALIZATION 4: Drift Summary Cards
-------------------------------------
- Card: Highest Drift Month = MAX of RMSE
- Card: Most Consistent Model = MIN of Average RMSE
- Card: Total Months Analyzed = COUNT of year_month


================================================================================
PART 5: FORMATTING & DESIGN TIPS
================================================================================

THEME:
- Use dark theme (if available) or customize colors
- Match your brand colors: Blue (#3B82F6), Green (#10B981), Orange (#F59E0B)

PAGE DESIGN:
- Use page background: solid color or image
- Set page size: 16:9 (recommended)
- Use consistent titles and subtitles

VISUAL FORMATTING:
- Turn off gridlines for cleaner look
- Use data labels sparingly (only key values)
- Add tooltips for detailed information

FILTERS:
- Add "Sync" slicers across pages for consistent filtering
- Place important filters at top of each page

TITLES:
- Add page titles with icons
- Add descriptions explaining each visualization


================================================================================
PART 6: AUTOMATIC REFRESH (LIMITED ON FREE ACCOUNT)
================================================================================

Since you have a free account, you cannot set up automatic cloud refresh.
However, you can make updates easier:

OPTION 1: Manual Refresh Button
---------------------------------
In Power BI:
- File → Options and Settings → Options
- Under "Preview Features", enable "Show refresh button"
- Now you can click Refresh on the toolbar after exporting new data

OPTION 2: Keyboard Shortcut
----------------------------
- After exporting new CSVs, press F5 in Power BI to refresh all data

OPTION 3: Simple Workflow (Recommended for Free Account)
----------------------------------------------------------
1. Make changes in Streamlit (fetch data, train models, etc.)
2. Click "Export to Power BI" button in Data Management
3. Open your saved .pbix file
4. Press F5 or click Refresh
5. Dashboard updates!


================================================================================
PART 7: SAVING AND SHARING
================================================================================

SAVE:
- File → Save As
- Name: "IEX Power Price Dashboard.pbix"
- Location: Your project folder

SHARE (Free Account Options):
- Export to PDF: File → Export → PowerPoint or PDF
- Share screen in meetings
- Publish to Power BI Free (limited features)


================================================================================
QUICK START CHECKLIST
================================================================================

□ Run: python src/powerbi_exporter.py
□ Open Power BI
□ Import model_metrics.csv
□ Import daily_prices.csv  
□ Import weather_impact_comparison.csv
□ Build Page 1: Model Performance (2-3 visuals)
□ Build Page 2: Weather Impact (key comparison)
□ Add at least one filter/slicer
□ Save .pbix file
□ Test: Export new data → Refresh in Power BI

================================================================================
END OF GUIDE
================================================================================