# Model Card: Demand Forecasting at Scale (Walmart)

---

## 1. Model Details

| Field | Details |
|---|---|
| **Framework Name** | Retail Demand Forecasting at Scale — Multi-Model Time Series Framework |
| **Python Version** | 3.10 |
| **Analysis Date** | March 2026 |
| **Recommended Model** | XGBoost Regressor (production deployment via `train.py`) |
| **Primary Metric** | WMAE — Weighted Mean Absolute Error (holiday weeks weighted 5×) |
| **Secondary Metrics** | MAE, RMSE, MAPE%, SMAPE% |
| **Live App** | [Streamlit App](https://demand-forecasting-at-scale-live-app.streamlit.app/) |

---

## 2. Intended Use

- **Primary Use Case:** Predict weekly sales for 99 departments across 45 Walmart stores, with special accuracy emphasis on high-volatility holiday weeks (Super Bowl, Labor Day, Thanksgiving, Christmas).
- **Target Users:** Supply Chain Analysts, Inventory Planners, Retail Data Scientists.
- **Out of Scope:** Real-time intra-day forecasting, product-level SKU forecasting, or stores outside the 45-store dataset.

---

## 3. Dataset

| Property | Value |
|---|---|
| **Source Files** | `train.csv` + `stores.csv` + `features.csv` |
| **Total Rows (merged)** | 421,570 |
| **Stores** | 45 |
| **Departments** | 99 |
| **Timeframe** | Weekly historical sales (2010–2012) |
| **Max Weekly Sales** | $693,099 (Store 14, Dept 92, Black Friday 2010) |
| **Min Weekly Sales** | Negative returns possible (returns > sales in low-traffic depts) |
| **Holiday Weeks** | Super Bowl · Labor Day · Thanksgiving · Christmas |

**Merge Logic:**
```
train.csv ──LEFT JOIN── stores.csv ──LEFT JOIN── features.csv
(on Store + Date, IsHoliday deduplicated from features)
```

**Sample data (Store 1, Dept 1, Nov–Dec 2011):**

| Date | Weekly Sales | IsHoliday | Markdown3 | CPI |
|---|---|---|---|---|
| 2011-11-11 | £18,689.54 | False | £215.07 | 217.998 |
| 2011-11-25 | £20,911.25 | **True** | £55,805.51 | 218.468 |
| 2011-12-09 | £33,305.92 | False | £105.02 | 218.962 |

> Markdown3 spikes dramatically on the Thanksgiving holiday week — confirming markdown-holiday interaction as a key signal.

---

## 4. Feature Engineering

**Original columns:** 16 → **Engineered columns:** 37 total features

### Calendar Features
| Feature | Description |
|---|---|
| `week_of_year` | ISO week number (1–52) |
| `month` | Month of year |
| `year` | Calendar year |
| `quarter` | Q1–Q4 |

### Lag Features (per Store–Department series)
| Feature | Lag |
|---|---|
| `sales_lag_1w` | 1 week prior |
| `sales_lag_2w` | 2 weeks prior |
| `sales_lag_4w` | 4 weeks prior |
| `sales_lag_8w` | 8 weeks prior |
| `sales_lag_13w` | 13 weeks (quarter) |
| `sales_lag_26w` | 26 weeks (half-year) |
| `sales_lag_52w` | 52 weeks (year-over-year) |

### Rolling Statistics (shift(1) applied to prevent data leakage)
| Feature | Window |
|---|---|
| `roll_mean_4w`, `roll_std_4w` | 4-week rolling |
| `roll_mean_8w`, `roll_std_8w` | 8-week rolling |
| `roll_mean_13w`, `roll_std_13w` | 13-week rolling |

### Markdown & Interaction Features
| Feature | Description |
|---|---|
| `total_markdown` | Sum of Markdown1–5 (NaN filled with 0) |
| `markdown_holiday_interaction` | `total_markdown × isholiday` — key signal for promotional spikes |

### Other Features
| Feature | Description |
|---|---|
| `time_idx` | Cumulative week index per store-dept series |
| `store_type_enc` | Label-encoded store type (A, B, C) |

---


<<<<<<< Updated upstream
### 5. Methodology & Pipeline Architecture
 
```
┌─────────────────────────────────────────────────────────────────────┐
│                        1. DATA LAYER                                │
│  train.csv + stores.csv + features.csv                              │
│         │                                                           │
│         ▼  load_and_merge_data()                                    │
│  Unified DataFrame (421k+ rows, hierarchically sorted)              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                     2. FEATURE ENGINEERING                          │
│  engineer_features()                                                │
│  ├── Calendar features (week, month, quarter, year)                 │
│  ├── Lag features (1w–52w per store-dept)                           │
│  ├── Rolling stats (4w/8w/13w mean & std, shift-safe)               │
│  ├── Markdown aggregation + holiday interaction term                │
│  └── Store type encoding + time index                               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                  3. EDA & STATISTICAL VALIDATION                    │
│  ├── ADF stationarity tests (per series)                            │
│  ├── ACF / PACF analysis → SARIMAX order selection                  │
│  ├── Seasonal decomposition (trend / seasonal / residual)           │
│  └── Correlation analysis (heatmap, scatter, group means)           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                     4. MODEL TRAINING                               │
│  Temporal split: train < 2012  |  test = 2012                       │
│  ├── Baseline: SARIMAX (per series, interpretability)               │
│  └── Primary:  XGBRegressor (n_estimators=500, lr=0.05, depth=6)    │
│                                                                     │
│  Loss: WMAE (holiday weeks weighted 5×)                             │
│  Secondary: MAE, RMSE                                               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                  5. EXPERIMENT TRACKING (MLflow)                    │
│  sqlite:///mlflow.db · Experiment: "Demand_Forecasting_Walmart"     │
│  Logs: params, metrics (MAE / RMSE / WMAE), model artifact          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                  6. PACKAGING & CI/CD                               │
│  ├── model.joblib → serialized artifact                             │
│  ├── Docker → containerized environment (Dockerfile)                │
│  ├── GitHub Actions → CI pipeline (unit tests via PyTest on push)   │
│  └── scripts/deploy.sh → automated deployment script                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                     7. PRODUCTION SERVING                           │
│  ├── FastAPI backend microservice (AWS EC2)                         │
│  ├── Streamlit frontend (Streamlit Cloud)                           │
│  └── MLflow artifact registry for live experiment management        │
└─────────────────────────────────────────────────────────────────────┘
```
 
---

## 6. Correlation Insights (from EDA)
=======

## 5. Correlation Insights (from EDA)
>>>>>>> Stashed changes

| Feature | Correlation with Weekly Sales | Insight |
|---|---|---|
| `size` | **+0.21** | Strongest positive driver — larger stores sell more |
| `dept` | +0.14 | Certain departments inherently adgenerate higher revenue |
| `temperature` | ~0.00 | Sales resilient to weather fluctuations |
| `fuel_price` | ~0.00 | Near-zero correlation |
| `cpi` | ~0.00 | Near-zero correlation |
| `unemployment` | ~0.00 | Near-zero correlation |

> External macro-economic factors (CPI, unemployment, fuel price) show near-zero correlation — store size and department type are far stronger predictors.

**Additional findings:**
- Store Type **A** consistently generates highest weekly sales
- Holiday weeks show meaningfully higher sales (IsHoliday = True)
- Positive correlation between fuel price and temperature observed

---

## 7. Time Series Diagnostics (Store 1, Dept 1)

| Test | Result |
|---|---|
| **ADF Statistic** | -3.3022 |
| **p-value** | 0.0148 |
| **Conclusion** | ✅ **Stationary** — no differencing required |

---

## 8. Model Comparison Results

All models evaluated on Store 16, Dept 72, 12-week test horizon:

| Model | MAE | RMSE | MAPE% | SMAPE% | Train Time |
|---|---|---|---|---|---|
| Naive (Last Value) | 1,951.47 | 3,464.90 | 8.98% | 10.09% | 0.00s |
| Seasonal Naive (52w) | 1,951.47 | 3,464.90 | 8.98% | 10.09% | 0.00s |
| SARIMA(1,1,1)(1,1,1,52) | 1,951.47 | 3,464.90 | 8.98% | 10.09% | 0.41s |
| **LSTM (PyTorch)** | **2,227.20** | **3,601.88** | **11.32%** | **11.45%** | **0.27s** |
| XGBoost (lag features) | 7,135.38 | 9,211.05 | 41.04% | 32.09% | 0.18s |

**Best model (lowest SMAPE on single series):** Naive / Seasonal Naive

> **Note:** XGBoost underperforms on a single short series due to insufficient lag feature history. It excels at scale across all 45 stores × 99 departments where rich lag features are available — which is why it is the production model in `train.py`.

### Model Selection Guide
```
Short horizon (1–4 weeks), limited data     → Seasonal Naive / SARIMA
Medium horizon (4–12 weeks), rich features  → XGBoost
Long horizon (12+ weeks), large dataset     → LSTM / N-BEATS
```

---

## 9. Production Model — XGBoost (train.py)

**Train/Test Split:** Year < 2012 = train | Year = 2012 = test

**Hyperparameters:**

| Parameter | Value |
|---|---|
| `n_estimators` | 500 |
| `learning_rate` | 0.05 |
| `max_depth` | 6 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `random_state` | 42 |

**Holiday Weighting:**
```python
weights = test_df['isholiday'].apply(lambda x: 5 if x else 1)
WMAE = Σ(weights × |y_true - y_pred|) / Σ(weights)
```
Holiday weeks contribute 5× more to the error metric — intentionally penalising missed forecasts during peak demand.

**Artifacts saved:** `model.joblib` (loaded by Streamlit app)

---

## 10. Forecast Uncertainty — Conformal Prediction Intervals

| Interval | Half-Width |
|---|---|
| **80% PI** | ±18,735 units |
| **95% PI** | ±27,326 units |

---

## 11. Inventory Cost Optimisation

Using conformal prediction intervals to drive safety stock decisions:

| Service Level | z-score | Avg Safety Stock | Avg Order Qty | Annual Holding Cost | Annual Stockout Cost | **Total Annual Cost** |
|---|---|---|---|---|---|---|
| 85% | 1.04 | 15,222 units | 40,503 units | £395,778 | £103,574 | **£499,352** |
| 90% | 1.28 | 18,735 units | 44,016 units | £487,111 | £103,574 | **£590,686** |
| 95% | 1.65 | 24,151 units | 49,431 units | £627,917 | £103,574 | **£731,491** |
| 99% | 2.33 | 34,104 units | 59,384 units | £886,695 | £103,574 | **£990,269** |

> Moving from 90% to 99% service level costs an additional **£399,583/year** — the explicit trade-off the supply chain team must decide.

**Recommended service level: 90%** — optimal balance between holding cost and stockout risk for most departments.

---

## 12. Ethical Considerations & Limitations

- **Temporal Drift:** The dataset covers 2010–2012. Retail patterns have shifted significantly — retraining on recent data is essential before production use.
- **Holiday Generalisation:** The 5× holiday weighting assumes all holiday weeks are equally important. Thanksgiving and Christmas typically outperform Super Bowl and Labor Day — per-holiday weights would improve accuracy.
- **Negative Sales:** Some department-weeks show negative `weekly_sales` (returns exceeding purchases). These are valid but can destabilise lag features — a clipping floor of 0 should be evaluated.
- **Store Closure:** The dataset doesn't explicitly flag store closures, which can produce anomalous zeros that corrupt rolling statistics.
- **External Shocks:** Events like COVID-19 or supply chain disruptions are not represented — the model should be retrained or augmented with exogenous indicators in such scenarios.

---

## 13. Infrastructure & Tools

| Category | Tool |
|---|---|
| Language | Python 3.10 |
| ML / Forecasting | XGBoost, SARIMA (statsmodels), LSTM (PyTorch) |
| Baselines | Naive, Seasonal Naive (52-week) |
| Statistical Tests | ADF (Augmented Dickey-Fuller), ACF/PACF |
| Uncertainty | Conformal Prediction Intervals |
| API Framework | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Experiment Tracking | MLflow (SQLite backend) |
| Testing | Pytest + pytest-cov |
| Coverage Reporting | Codecov |
| CI/CD | GitHub Actions |
| Containerisation | Docker |
| Cloud Infrastructure | AWS EC2 |
| Model Serialisation | Joblib (`model.joblib`) |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Tableau |

---

## 14. Final Decision Summary

```
══════════════════════════════════════════════════════════════
        DEMAND FORECASTING — EXECUTIVE SUMMARY REPORT
══════════════════════════════════════════════════════════════
Dataset:         421,570 rows | 45 stores | 99 departments
Time Period:     Weekly historical data (2010–2012)
Primary KPI:     WMAE (holiday weeks weighted 5×)
══════════════════════════════════════════════════════════════
PRODUCTION MODEL:  XGBoost Regressor
n_estimators:      500 | learning_rate: 0.05 | max_depth: 6
══════════════════════════════════════════════════════════════
KEY DESIGN DECISIONS:
1. WMAE used (not MAE) — holiday accuracy prioritised
2. Year-based temporal split — no future data leakage
3. 52-week lag feature captures year-over-year seasonality
4. Markdown-holiday interaction captures promo-driven spikes
5. Conformal prediction intervals for inventory optimisation
══════════════════════════════════════════════════════════════
PRODUCTION RECOMMENDATIONS:
• Retrain monthly as new weekly sales data arrives
• Monitor WMAE drift — alert if >10% degradation
• Use 90% service level for standard inventory ordering
• Flag Thanksgiving and Christmas weeks for manual review
• Log all predictions to MLflow for audit trail
══════════════════════════════════════════════════════════════
```
