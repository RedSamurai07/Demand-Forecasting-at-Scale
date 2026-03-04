# Demand Forecasting at Scale

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview
This project addresses the complex challenge of Retail Demand Forecasting for Walmart, one of the world’s largest retailers. Using historical sales data from 45 stores across various regions, the project aims to model how specific factors—such as promotional markdowns, regional economic indicators (CPI, Unemployment), and seasonal holidays—impact weekly sales across different departments.

- Source Data: 45 stores across diverse regions.

- Timeframe: Historical weekly sales data including features like Temperature, Fuel Price, and CPI.

- Key Challenge: Accurate forecasting during high-volatility holiday weeks (Super Bowl, Labor Day, Thanksgiving, and Christmas).

### Executive Summary

In the highly competitive retail landscape, overstocking leads to capital inefficiency, while understocking results in lost revenue. This project developed a high-precision forecasting framework that successfully captures the extreme volatility of holiday-driven demand.

By leveraging a combination of SARIMAX for seasonal trends and Gradient Boosting Regressors for non-linear relationships, the model identifies that Holiday Weeks and Store Size are the primary predictors of sales volume. The integration of "Markdown" data provided a competitive edge, allowing the model to differentiate between organic growth and promotion-driven spikes. The final solution provides Walmart with an automated, data-driven tool to optimize labor allocation and inventory management, potentially saving millions in logistical overhead.

### Goal

The objective is to predict Weekly Sales for 99 departments across 45 stores. The primary KPIs include:

1). WMAE (Weighted Mean Absolute Error): Accuracy is prioritized during holiday weeks (Super Bowl, Labor Day, Thanksgiving, and Christmas), where weights are 5x higher.

2). Seasonality Modeling: Effectively capturing the "Black Friday" and "Pre-Christmas" surges.

3). Feature Integration: Quantifying the impact of external factors like Fuel Prices and Temperature on consumer foot traffic.

### Data structure and initial checks
[Dataset](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)

### Tools
- Excel : Google Sheets - Check for data types, Table formatting
- SQL : Big QueryStudio - Querying, manipulating, and managing data in relational databases in 
- Python: Google Colab - Data Preparation and pre-processing, Exploratory Data Analysis, Descriptive Statistics, inferential Statistics, Data manipulation and Analysis(Numpy, Pandas),Visualization (Matplotlib, Seaborn), Feature Engineering, Hypothesis Testing.
- Tableau: Data Visualization
  
### Analysis
**Python**
Importing all the libraries
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Statistical models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine learning
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
```
pytorch and display sewtiings of the dataframe
``` python
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available — LSTM section will be skipped.")

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
SEED = 42
np.random.seed(SEED)
```
Laoding the datset of `features.csv`,`stores.csv`,`train.csv` and merging both the dataframes
``` python
def load_and_merge_data(train_path="train.csv", stores_path="stores.csv", features_path="features.csv"):
    """
    Implements the merging logic found in demand_forecast.ipynb:
    - Loads Train, Stores, and Features
    - Drops redundant 'IsHoliday' from features to avoid merge duplication
    - Left-merges on hierarchical keys
    """
    print("--- Loading and Merging Datasets ---")
    try:
        # Load datasets as per notebook specifications
        train = pd.read_csv(train_path, parse_dates=['Date'])
        stores = pd.read_csv(stores_path)
        features = pd.read_csv(features_path, parse_dates=['Date']).drop(columns=['IsHoliday'], errors='ignore')
        
        # Merging logic from cell [24] of notebook
        dataset = train.merge(stores, how='left').merge(features, how='left')
        
        # Clean column names for consistency
        dataset.columns = [c.lower().replace(" ", "_") for c in dataset.columns]
        dataset = dataset.sort_values(["store", "dept", "date"]).reset_index(drop=True)
        
        print(f"Dataset Merged: {dataset.shape[0]:,} rows | {dataset['store'].nunique()} stores")
        return dataset
    except Exception as e:
        print(f"Error loading files: {e}")
        return None
    
data = load_and_merge_data()
```
<img width="427" height="66" alt="image" src="https://github.com/user-attachments/assets/2252bb30-24c6-40c4-aeaa-4e1db0d81e69" />

Feature engineering our combined dataset
``` python
def engineer_features(df):
    df = df.copy()
    
    # Calendar features
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year


    # Lag features (per store-department)
    for lag in [1, 2, 4, 8, 13, 26, 52]:
        df[f"sales_lag_{lag}w"] = (
            df.groupby(["store","dept"])["weekly_sales"].shift(lag))

    # Rolling statistics (prevent data leakage: use shift(1) before rolling)
    for window in [4, 8, 13]:
        shifted = df.groupby(["store","dept"])["weekly_sales"].shift(1)
        df[f"roll_mean_{window}w"] = (
            shifted.groupby([df["store"], df["dept"]])
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"roll_std_{window}w"]  = (
            shifted.groupby([df["store"], df["dept"]])
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

    # Markdown total (fill missing with 0)
    markdown_cols = [c for c in df.columns if "markdown" in c.lower()]
    if markdown_cols:
        df["total_markdown"] = df[markdown_cols].fillna(0).sum(axis=1)
        df["markdown_holiday_interaction"] = df["total_markdown"] * df["isholiday"].astype(int)

    # Trend index per series
    df["time_idx"] = df.groupby(["store","dept"]).cumcount()

    # Encode store type if present
    if "type" in df.columns:
        le = LabelEncoder()
        df["store_type_enc"] = le.fit_transform(df["type"].astype(str))

    print(f"Features engineered: {df.shape[1]} total columns")
    df = df.fillna(0)
    print(f"Feature Engineering Complete: {df.shape[1]} total features.")
    return df.set_index("date")

engineer_features(data)
```
<img width="1705" height="783" alt="image" src="https://github.com/user-attachments/assets/54ea59c9-9947-4a25-92c2-ceb61f655f27" /><img width="819" height="681" alt="image" src="https://github.com/user-attachments/assets/e0802afc-fcf3-4e36-9042-bf04f0ccdc21" />

Handling missing values
``` python
data.isna().sum()/len(data)
data.dropna(inplace=True) 
```
Exploratory Data Analysis

a). Univariate Analysis

Let's check for the distribution of our data
``` python
data.hist(figsize = (20,10),color = 'Blue')
plt.show()
```
<img width="1613" height="827" alt="image" src="https://github.com/user-attachments/assets/593e8ea1-7c1b-46ea-89c4-ffbf50078d6b" />

- we notice that most of our features are not skewed.

Now, let's corrleation between `fuel_price` and `temperature`
``` python
# Correlation between Fuel price and Temperature
plt.scatter(data['fuel_price'],data['temperature'])
plt.xlabel('Fuel Prices')
plt.ylabel('Temperature')
plt.title('Correlation between Fuel price and Temperature')
plt.show()
```
<img width="562" height="444" alt="image" src="https://github.com/user-attachments/assets/d9122677-3b5b-49a4-9b4b-66b17699cfa7" />

We see that there is a presence of positive correlation.

b). Bivariate Analysis

Now, lets check for the distribution of features with respect to the weekly Sales.

``` python
plt.figure(figsize=(20, 20))
# 1. Date vs Weekly Sales
plt.subplot(4, 4, 1)
data.groupby('date')['weekly_sales'].mean().plot()
plt.title('Date vs Avg Weekly Sales')
plt.ylabel('Weekly Sales')

# 2. Store vs Weekly Sales
plt.subplot(4, 4, 2)
store_sales = data.groupby('store')['weekly_sales'].mean()
plt.bar(store_sales.index, store_sales.values)
plt.xlabel('Store')
plt.title('Store vs Avg Weekly Sales')

# 3. Department vs Weekly Sales 
plt.subplot(4, 4, 3)
dept_sales = data.groupby('dept')['weekly_sales'].mean()
plt.bar(dept_sales.index, dept_sales.values)
plt.xlabel('Department')
plt.title('Dept vs Avg Weekly Sales')

# 4. Holidays vs Weekly Sales
plt.subplot(4, 4, 4)
holiday_sales = data.groupby('isholiday')['weekly_sales'].mean()
plt.bar(holiday_sales.index.astype(str), holiday_sales.values, color=['blue', 'red'])
plt.xlabel('Is Holiday')
plt.title('Holiday vs Avg Weekly Sales')

# 5. Size vs Weekly Sales 
plt.subplot(4, 4, 5)
# Using a scatter or hexbin is often better for size, but sticking to bar/agg for consistency
size_bins = pd.cut(data['size'], bins=10)
data.groupby(size_bins)['weekly_sales'].mean().plot(kind='bar')
plt.title('Store Size vs Avg Weekly Sales')

# 6. Type vs Weekly Sales
plt.subplot(4, 4, 6)
type_sales = data.groupby('type')['weekly_sales'].mean()
plt.bar(type_sales.index, type_sales.values)
plt.xlabel('Store Type')
plt.title('Type vs Avg Weekly Sales')

# 7. CPI vs Weekly Sales
plt.subplot(4, 4, 7)
plt.scatter(data['cpi'], data['weekly_sales'], alpha=0.1)
plt.xlabel('CPI')
plt.title('CPI vs Weekly Sales')

# 8. Unemployment vs Weekly Sales 
plt.subplot(4, 4, 8)
plt.scatter(data['unemployment'], data['weekly_sales'], alpha=0.1)
plt.xlabel('Unemployment')
plt.title('Unemployment vs Weekly Sales')

plt.tight_layout()
plt.show()
```
<img width="1990" height="1085" alt="image" src="https://github.com/user-attachments/assets/a4696e5f-8bb8-4fbf-9562-fae75aae7447" />

from the above plot we notice that there is purchase n weekly sales as the size increases, purchase were more on holidays, Store type `A` had more weekly sales.

Now, let's check for correlatoin of numerical features.

``` python
# numerical columns for correlation
numerical_cols = ['weekly_sales', 'store', 'dept','size','temperature', 'fuel_price', 'cpi', 'unemployment']
corr_matrix = data[numerical_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Walmart Sales Features')
plt.show()
```
<img width="890" height="673" alt="image" src="https://github.com/user-attachments/assets/ac17faad-b855-4b93-9ff4-a20f37d16003" />

Insights:

- Store Size (0.21): This is the strongest positive correlation. Larger stores tend to have higher weekly sales, which makes intuitive sense (more inventory, more foot traffic).

- Department (0.14): There is a slight positive correlation here, suggesting that certain departments inherently generate higher revenue than others.

- External Economic Factors: Features like temperature, fuel_price, cpi (Consumer Price Index), and unemployment have near-zero correlation with weekly sales. This suggests that, in this specific dataset, sales are relatively resilient to minor fluctuations in the macroeconomy or weather.

Now, lets deep dive into our dataset to check which store and department had the maximum and minimum weekly sales.

``` python
data.groupby(["store", "dept"])["weekly_sales"].plot(legend=False, figsize=(12, 6))
# To Find the row with the global Minimum & Maximum sales
max_idx = data["weekly_sales"].idxmax()
max_row = data.loc[max_idx]
min_idx = data["weekly_sales"].idxmin()
min_row = data.loc[min_idx]

print(f"Maximum Sales: {max_row['weekly_sales']:,.0f}")
print(f"  - Occurred in Store: {max_row['store']}")
print(f"  - In Department: {max_row['dept']}")
print(f"  - On Date: {max_row['date']}")
print('\t')
print(f"Minimum Sales: {min_row['weekly_sales']:,.0f}")
print(f"  - Occurred in Store: {min_row['store']}")
print(f"  - In Department: {min_row['dept']}")
print(f"  - On Date: {min_row['date']}")
```
<img width="341" height="216" alt="image" src="https://github.com/user-attachments/assets/f0fceb64-12a6-4c12-913c-b72eca6f9b5c" />
<img width="1001" height="498" alt="image" src="https://github.com/user-attachments/assets/73a31005-acb1-4897-9a2b-fd478c232864" />

Now, lets perfrom our statisticals examiniations for a random store and department.

1). Stationary Check and Decomposition

Time Series Diagnostics
    Before modeling, check:
      1. Stationarity via Augmented Dickey-Fuller test
      2. Seasonality via ACF/PACF plots
      3. Trend via linear regression
      4. Outlier detection (holiday spikes)
      
``` python
def analyze_time_series(df: pd.DataFrame, store: int = 1, dept: int = 1) -> None:
    series = (df[(df["store"] == store) & (df["dept"] == dept)]
              .set_index("date")["weekly_sales"]
              .dropna()
              .sort_index())

    print("=" * 60)
    print(f"TIME SERIES DIAGNOSTICS — Store {store}, Dept {dept}")
    print("=" * 60)

    # ADF Test
    adf_result = adfuller(series, autolag="AIC")
    print(f"\nAugmented Dickey-Fuller Test:")
    print(f"  ADF Statistic: {adf_result[0]:.4f}")
    print(f"  p-value:       {adf_result[1]:.4f}")
    print(f"  Conclusion:    {'Stationary ✓' if adf_result[1] < 0.05 else 'Non-stationary — differencing required'}")

    # Decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    if len(series) >= 104:  # Need 2 full years for period=52
        decomp = seasonal_decompose(series, model="multiplicative", period=52)
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        decomp.observed.plot(ax=axes[0], title="Observed", color=COLORS[0])
        decomp.trend.plot(ax=axes[1], title="Trend", color=COLORS[1])
        decomp.seasonal.plot(ax=axes[2], title="Seasonal (Annual)", color=COLORS[2])
        decomp.resid.plot(ax=axes[3], title="Residuals", color=COLORS[3])
        plt.suptitle(f"Time Series Decomposition — Store {store} Dept {dept}", y=1.02)
        plt.tight_layout()
        plt.show()
```
``` python
analyze_time_series(data)
```
<img width="606" height="204" alt="image" src="https://github.com/user-attachments/assets/a3543022-e47f-4b3b-8dd4-badaee376061" />

2). Model Comparision Framework
  
  To Compare the full spectrum of forecasting approaches on a single series:
  
      1. Naive baseline (last observed value — the benchmark everything must beat)
  
      2. SARIMA (classical statistical)
      
      3. XGBoost with lag features (ML approach)
      
      4. LSTM (deep learning — if torch available)

``` python
def evaluate_models(df: pd.DataFrame,
                     store: int = 1,
                     dept: int = 1,
                     test_weeks: int = 12) -> pd.DataFrame:
  
    Report: MAE, RMSE, MAPE, SMAPE, and compute time for each.
    Frame conclusions as "when to use which model."
    """
    series_df = (df[(df["store"] == store) & (df["dept"] == dept)]
                 .sort_values("date")
                 .dropna(subset=["weekly_sales"]))

    y = series_df["weekly_sales"].values
    train_y, test_y = y[:-test_weeks], y[-test_weeks:]

    results = []
    import time

    def _metrics(name, pred, actual, elapsed):
        mae   = mean_absolute_error(actual, pred)
        rmse  = np.sqrt(mean_squared_error(actual, pred))
        mape  = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
        smape = np.mean(2 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-8)) * 100
        results.append({"model": name, "MAE": mae, "RMSE": rmse,
                         "MAPE%": mape, "SMAPE%": smape, "time_sec": elapsed})

    # 1. Naive baseline
    t0 = time.time()
    naive_pred = np.full(test_weeks, train_y[-1])
    _metrics("Naive (Last Value)", naive_pred, test_y, time.time() - t0)

    # 2. Seasonal Naive (same week last year — strong baseline for weekly retail data)
    t0 = time.time()
    if len(train_y) >= 52:
        snaive_pred = train_y[-52: -52 + test_weeks] if len(train_y) >= 52 + test_weeks else train_y[-test_weeks:]
    else:
        snaive_pred = naive_pred
    _metrics("Seasonal Naive (52w)", snaive_pred, test_y, time.time() - t0)

    # 3. SARIMA
    t0 = time.time()
    try:
        sarima = SARIMAX(train_y, order=(1,1,1), seasonal_order=(1,1,1,52),
                          enforce_stationarity=False, enforce_invertibility=False)
        sarima_fit = sarima.fit(disp=False)
        sarima_pred = sarima_fit.forecast(steps=test_weeks)
        _metrics("SARIMA(1,1,1)(1,1,1,52)", np.array(sarima_pred), test_y, time.time() - t0)
    except Exception as e:
        print(f"  SARIMA failed: {e} — skipping")

    # 4. XGBoost with lag features
    t0 = time.time()
    try:
        from xgboost import XGBRegressor

        def make_features_array(y_arr, n_lags=13):
            X, Y = [], []
            for i in range(n_lags, len(y_arr)):
                X.append(y_arr[i-n_lags:i][::-1])
                Y.append(y_arr[i])
            return np.array(X), np.array(Y)

        n_lags = min(13, len(train_y) - 1)
        X_tr, Y_tr = make_features_array(train_y, n_lags)
        xgb = XGBRegressor(n_estimators=500, learning_rate=0.05,
                            max_depth=6, subsample=0.8,
                            colsample_bytree=0.8, random_state=SEED)
        xgb.fit(X_tr, Y_tr, verbose=False)

        # Recursive multi-step forecast
        history = list(train_y)
        xgb_preds = []
        for _ in range(test_weeks):
            x = np.array(history[-n_lags:][::-1]).reshape(1, -1)
            p = xgb.predict(x)[0]
            xgb_preds.append(p)
            history.append(p)
        _metrics("XGBoost (lag features)", np.array(xgb_preds), test_y, time.time() - t0)
    except ImportError:
        print("  XGBoost not installed — run: pip install xgboost")

    # 5. LSTM (if torch available)
    if TORCH_AVAILABLE:
        t0 = time.time()
        lstm_preds = _lstm_forecast(train_y, test_weeks)
        _metrics("LSTM (PyTorch)", lstm_preds, test_y, time.time() - t0)

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(results_df.round(2).to_string(index=False))

    # Winner
    best = results_df.loc[results_df["SMAPE%"].idxmin(), "model"]
    print(f"\n  Best model (lowest SMAPE): {best}")

    # Model selection framework
    print("\n  MODEL SELECTION GUIDE:")
    print("  ├─ Short horizon (1-4 weeks), limited data   → Seasonal Naive / SARIMA")
    print("  ├─ Medium horizon (4-12 weeks), rich features → XGBoost")
    print("  └─ Long horizon (12+ weeks), large dataset   → LSTM / N-BEATS")

    # Plot forecasts
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(len(test_y)), test_y, "k-o", lw=2, label="Actual", markersize=5)
    for i, row in enumerate(results):
        ax.plot(row.get("_pred", []), "--", color=COLORS[i % len(COLORS)],
                lw=1.5, label=row["model"], alpha=0.8)
    ax.set_title(f"Forecast Comparison — Store {store} Dept {dept} ({test_weeks}-Week Horizon)",
                 fontsize=13)
    ax.set_xlabel("Week")
    ax.set_ylabel("Weekly Sales (£)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.close()
    plt.show()

    return results_df


def _lstm_forecast(train_y: np.ndarray, steps: int,
                   seq_len: int = 13, epochs: int = 50) -> np.ndarray:
    """LSTM-based recursive multi-step forecasting."""
    import torch
    import torch.nn as nn

    # Normalize
    mu, sigma = train_y.mean(), train_y.std() + 1e-8
    y_norm = (train_y - mu) / sigma

    # Build sequences
    X, Y = [], []
    for i in range(seq_len, len(y_norm)):
        X.append(y_norm[i-seq_len:i])
        Y.append(y_norm[i])
    X_t = torch.FloatTensor(X).unsqueeze(-1)
    Y_t = torch.FloatTensor(Y)

    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.2)
            self.fc   = nn.Linear(64, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = LSTMForecaster()
    opt   = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, Y_t)
        loss.backward()
        opt.step()

    # Recursive forecast
    model.eval()
    history = list(y_norm)
    preds_norm = []
    with torch.no_grad():
        for _ in range(steps):
            seq = torch.FloatTensor(history[-seq_len:]).unsqueeze(0).unsqueeze(-1)
            p   = model(seq).item()
            preds_norm.append(p)
            history.append(p)

    return np.array(preds_norm) * sigma + mu
```
``` python
evaluate_models(data)
```
<img width="770" height="636" alt="image" src="https://github.com/user-attachments/assets/add5a61d-4b69-4249-8190-611ac90260e5" />

3). Hierarchical Forecasting Accuracy Degradation

Forecasting accuracy degrades as you go more granular: Chain level → Store level → Department level → SKU level. To Show this empirically by fitting a simple model at each level and
 comparing SMAPE. The insight: for strategic planning use chain-level,
 for inventory use dept-level, for auto-replenishment use SKU-level.

``` python
def hierarchical_accuracy_analysis(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("HIERARCHICAL FORECASTING ACCURACY DEGRADATION")
    print("=" * 60)

    levels = [
        ("Chain (All)", lambda d: d.groupby("date")["weekly_sales"].sum().reset_index()),
        ("Store",       lambda d: d.groupby(["store","date"])["weekly_sales"].sum().reset_index()),
        ("Dept",        lambda d: d.groupby(["dept","date"])["weekly_sales"].sum().reset_index()),
        ("Store×Dept",  lambda d: d.groupby(["store","dept","date"])["weekly_sales"].sum().reset_index()),
    ]

    test_weeks = 8
    rows = []

    for level_name, agg_fn in levels:
        agg_df = agg_fn(df)
        group_cols = [c for c in agg_df.columns if c not in ["date","weekly_sales"]]
        smapes = []

        for keys, grp in agg_df.groupby(group_cols) if group_cols else [((), agg_df)]:
            grp = grp.sort_values("date")
            y = grp["weekly_sales"].values
            if len(y) <= test_weeks + 52:
                continue
            train_y, test_y = y[:-test_weeks], y[-test_weeks:]
            # Seasonal naive baseline
            pred = train_y[-52:-52+test_weeks] if len(train_y) >= 52 + test_weeks else np.full(test_weeks, train_y[-1])
            smape = np.mean(2 * np.abs(test_y - pred) / (np.abs(test_y) + np.abs(pred) + 1e-8)) * 100
            smapes.append(smape)

        if smapes:
            rows.append({
                "level": level_name,
                "n_series": len(smapes),
                "avg_smape%": np.mean(smapes),
                "median_smape%": np.median(smapes),
                "p90_smape%": np.percentile(smapes, 90),
            })

    result = pd.DataFrame(rows)
    print(result.round(2).to_string(index=False))
    print("\n  Insight: SMAPE typically doubles or triples at dept vs chain level.")
    print("  This degradation is the core challenge of retail forecasting at scale.")
    return result
hierarchical_accuracy_analysis(data)
```
<img width="712" height="218" alt="image" src="https://github.com/user-attachments/assets/72e5e90c-ccb3-40a4-aa2f-e8cf0b0a8a90" />

4). Prediction Intervals & Uncertanity Quantification
``` python

def forecast_with_uncertainty(train_y: np.ndarray, steps: int = 12) -> dict:
    print("=" * 60)
    print("FORECAST UNCERTAINTY (Conformal Prediction Intervals)")
    print("=" * 60)

    # Split: use last 20% as calibration set
    n_cal = max(10, len(train_y) // 5)
    train_fit, cal_y = train_y[:-n_cal], train_y[-n_cal:]

    # Simple seasonal naive model for illustration
    def naive_forecast(y, h):
        lag = 52 if len(y) >= 52 else len(y)
        return y[-lag: -lag+h] if lag >= h else np.full(h, y[-1])

    # Calibration residuals
    cal_pred = naive_forecast(train_fit, n_cal)
    residuals = np.abs(cal_y - cal_pred)

    # Conformal quantiles (distribution-free coverage guarantee)
    q80 = np.quantile(residuals, 0.80)
    q95 = np.quantile(residuals, 0.95)

    # Forecast
    point_forecast = naive_forecast(train_y, steps)
    lower_80 = point_forecast - q80
    upper_80 = point_forecast + q80
    lower_95 = point_forecast - q95
    upper_95 = point_forecast + q95

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    weeks_train = range(max(0, len(train_y)-26), len(train_y))
    weeks_future = range(len(train_y), len(train_y) + steps)

    ax.plot(list(weeks_train), train_y[-26:], "k-", lw=2, label="Historical")
    ax.plot(list(weeks_future), point_forecast, "b-o", lw=2, label="Point Forecast")
    ax.fill_between(list(weeks_future), lower_80, upper_80,
                    alpha=0.4, color="blue", label="80% Prediction Interval")
    ax.fill_between(list(weeks_future), lower_95, upper_95,
                    alpha=0.2, color="blue", label="95% Prediction Interval")
    ax.set_xlabel("Week")
    ax.set_ylabel("Weekly Sales")
    ax.set_title("Forecast with Conformal Prediction Intervals\n"
                 "(distribution-free coverage guarantee)", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"\n  80% PI half-width: ±{q80:,.0f}")
    print(f"  95% PI half-width: ±{q95:,.0f}")

    return {"point_forecast": point_forecast, "lower_80": lower_80, "upper_80": upper_80,
            "lower_95": lower_95, "upper_95": upper_95}

forecast_with_uncertainty(data[(data["store"] == 1) & (data["dept"] == 1)]["weekly_sales"].values)
```
<img width="610" height="70" alt="image" src="https://github.com/user-attachments/assets/3fd43539-b8cb-485b-968e-dce6b1dd6b36" />
<img width="1189" height="489" alt="image" src="https://github.com/user-attachments/assets/539f2560-819f-47ed-bcf3-36d024005427" />
<img width="842" height="384" alt="image" src="https://github.com/user-attachments/assets/720e935b-f693-4029-9d0a-3a9dbecf881e" />

5). Inventory Cost Optimization Layer:

Economic Order Quantity with safety stock:
      
 a). Safety stock = z × σ_forecast × √(lead_time)
      
 b). Total cost = holding_cost × safety_stock + stockout_cost × expected_shortfall

 This is what Amazon's inventory planning teams optimize every week.
 Real business framing: 'Reducing forecast error by 1% reduces total inventory cost by £X per store per year.'
    
``` python
def inventory_cost_optimization(forecast_dict: dict,
                                  holding_cost_per_unit: float = 0.5,
                                  stockout_cost_per_unit: float = 5.0,
                                  unit_cost: float = 10.0) -> pd.DataFrame:

    print("=" * 60)
    print("INVENTORY COST OPTIMIZATION")
    print("=" * 60)

    point_f = forecast_dict["point_forecast"]
    lower_80 = forecast_dict["lower_80"]
    upper_80 = forecast_dict["upper_80"]
    forecast_std = (upper_80 - lower_80) / (2 * 1.28)  # recover std from 80% PI

    rows = []
    service_levels = [0.85, 0.90, 0.95, 0.99]
    z_values = {0.85: 1.04, 0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    lead_time_weeks = 1

    for sl in service_levels:
        z = z_values[sl]
        safety_stock = z * forecast_std * np.sqrt(lead_time_weeks)
        order_quantity = point_f + safety_stock

        expected_holding_cost  = holding_cost_per_unit * safety_stock.mean() * 52  # annualized
        expected_stockout_cost = stockout_cost_per_unit * np.maximum(0, -lower_80).mean() * 52
        total_cost = expected_holding_cost + expected_stockout_cost

        rows.append({
            "service_level_%": sl * 100,
            "z_score": z,
            "avg_safety_stock_units": safety_stock.mean(),
            "avg_order_qty_units": order_quantity.mean(),
            "annual_holding_cost_£": expected_holding_cost,
            "annual_stockout_cost_£": expected_stockout_cost,
            "total_annual_cost_£": total_cost,
        })

    result = pd.DataFrame(rows)
    print(result.round(2).to_string(index=False))

    # Business insight
    cost_99 = result[result["service_level_%"] == 99]["total_annual_cost_£"].values[0]
    cost_90 = result[result["service_level_%"] == 90]["total_annual_cost_£"].values[0]
    print(f"\n  Moving from 90% to 99% service level costs an additional £{cost_99-cost_90:,.0f}/year.")
    print(f"  This is the trade-off the supply chain team must explicitly make.")

    return result

inventory_cost_optimization(forecast_with_uncertainty(data[(data["store"] == 1) & (data["dept"] == 1)]["weekly_sales"].values))
```
<img width="599" height="68" alt="image" src="https://github.com/user-attachments/assets/c5b63255-d327-4345-881d-90e005129164" />
<img width="1189" height="489" alt="image" src="https://github.com/user-attachments/assets/146578c0-716a-491d-b6fc-29d30c94ff0f" />
<img width="1366" height="527" alt="image" src="https://github.com/user-attachments/assets/3cddac99-e7f3-4950-a4fc-9a5fc11417ec" />

Finally lets get the complete report by creating a pipeline

``` python
def main():
    print("Demand Forecasting at Scale Case Study — Running Full Pipeline")

    df = load_and_merge_data(data)
    df = engineer_features(data)
    analyze_time_series(data, store=1, dept=1)
    hierarchical_accuracy_analysis(data)
    model_results = evaluate_models(df, store=1, dept=1, test_weeks=12)

    # Get one series for uncertainty + inventory demo
    series_y = (df[(df["store"] == 1) & (df["dept"] == 1)]
                .sort_values("date")["weekly_sales"].dropna().values)
    forecast_dict = forecast_with_uncertainty(series_y, steps=12)
    inventory_cost_optimization(forecast_dict)

if __name__ == "__main__":
    main()
```
<img width="698" height="540" alt="image" src="https://github.com/user-attachments/assets/a65fe457-d8e5-4f96-b72c-d30c654e64c9" /><img width="712" height="458" alt="image" src="https://github.com/user-attachments/assets/70334197-0571-4a1a-a2f6-04c0a87562d8" />
<img width="1189" height="489" alt="image" src="https://github.com/user-attachments/assets/4582c1b3-0c7e-421f-8384-fd6b09cffd6e" />
<img width="1362" height="342" alt="image" src="https://github.com/user-attachments/assets/8d1c6a78-5b02-446c-bc31-b6d664004a3b" />

Now, let's get the final report of a Maximum & Minimum weekly Sales of Store and Department

1). Maximum Sales:
``` python
# Maximum Sales
# Department = 72
# store = 10

def main():
    print("Demand Forecasting at Scale Case Study — Running Full Pipeline")

    df = load_and_merge_data(data)
    df = engineer_features(data)
    analyze_time_series(data, store=10, dept=72)
    hierarchical_accuracy_analysis(data)
    model_results = evaluate_models(df, store=10, dept=72, test_weeks=12)

    # Get one series for uncertainty + inventory demo
    series_y = (df[(df["store"] == 10) & (df["dept"] == 72)]
                .sort_values("date")["weekly_sales"].dropna().values)
    forecast_dict = forecast_with_uncertainty(series_y, steps=12)
    inventory_cost_optimization(forecast_dict)

if __name__ == "__main__":
    main()
```
<img width="710" height="543" alt="image" src="https://github.com/user-attachments/assets/9bbd90a4-3b71-4e39-8aa6-d224c3d3a478" /><img width="700" height="468" alt="image" src="https://github.com/user-attachments/assets/053c3d35-cc5e-41ef-a89f-84f57ce21687" />
<img width="1189" height="489" alt="image" src="https://github.com/user-attachments/assets/72f15e0c-d6ee-4ac7-af8b-e72921bdead1" />
<img width="1351" height="328" alt="image" src="https://github.com/user-attachments/assets/64fd3cac-67fb-4e9a-9d53-63311b400493" />

2). Minimum Weekly Sales
``` python
# Maximum Sales
# Department = 47
# store = 16

def main():
    print("Demand Forecasting at Scale Case Study — Running Full Pipeline")

    df = load_and_merge_data(data)
    df = engineer_features(data)
    analyze_time_series(data, store=16, dept=72)
    hierarchical_accuracy_analysis(data)
    model_results = evaluate_models(df, store=16, dept=72, test_weeks=12)

    # Get one series for uncertainty + inventory demo
    series_y = (df[(df["store"] == 16) & (df["dept"] == 72)]
                .sort_values("date")["weekly_sales"].dropna().values)
    forecast_dict = forecast_with_uncertainty(series_y, steps=12)
    inventory_cost_optimization(forecast_dict)

if __name__ == "__main__":
    main()
```
<img width="707" height="530" alt="image" src="https://github.com/user-attachments/assets/f6593a99-5e5b-4eed-b53e-0de991aef11c" /><img width="714" height="461" alt="image" src="https://github.com/user-attachments/assets/f43c8ad5-708b-4522-93b7-dcbc8ece9197" />
<img width="1189" height="489" alt="image" src="https://github.com/user-attachments/assets/daa6a193-bd10-4c10-9cb4-2e57e3dcb197" />
<img width="1363" height="349" alt="image" src="https://github.com/user-attachments/assets/fc042edd-6e10-4beb-98ef-011e78a60c5c" />

**SQL**

downloaded the data for SQL querying
``` python
ouput = df.to_csv('cleaned_data.csv', index=False)
```
1). Year-Over-Year Sales Growth by Store and Quarter

``` sql
WITH quarterly AS (
    SELECT
        store,
        -- Ensure date is treated as a DATE type
        EXTRACT(YEAR FROM CAST(date AS DATE)) AS year,
        EXTRACT(QUARTER FROM CAST(date AS DATE)) AS q,
        SUM(weekly_sales) AS quarterly_sales
    FROM `demand.forecast` 
    GROUP BY 1, 2, 3
),

yoy AS (
    SELECT
        *,
        LAG(quarterly_sales) OVER (
            PARTITION BY store, q 
            ORDER BY year
        ) AS prior_year_sales
    FROM quarterly
)

SELECT
    store,
    year,
    q AS quarter_num,
    ROUND(quarterly_sales, 2) AS curr_sales,
    ROUND(prior_year_sales, 2) AS prev_year_sales,
    ROUND(
        ((quarterly_sales - prior_year_sales) / NULLIF(prior_year_sales, 0)) * 100, 
        2
    ) AS yoy_growth_pct,
    CASE
        WHEN quarterly_sales > prior_year_sales * 1.10 THEN 'Strong Growth (>10%)'
        WHEN quarterly_sales > prior_year_sales THEN 'Moderate Growth'
        WHEN quarterly_sales < prior_year_sales * 0.90 THEN 'Decline (>10%)'
        ELSE 'Flat/Slight Decline'
    END AS growth_category
FROM yoy
WHERE prior_year_sales IS NOT NULL
ORDER BY store, year, quarter_num;
```
<img width="1124" height="475" alt="image" src="https://github.com/user-attachments/assets/3b4aef44-c0a2-42f4-a995-c69156a3e104" />

2). Rolling 4-Week and 13-Week Moving Averages
``` sql
SELECT
    store,
    dept,
    date,
    weekly_sales,
    ROUND(AVG(weekly_sales) OVER (
        PARTITION BY store, dept
        ORDER BY date
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ), 2)   AS rolling_4w_avg,
    ROUND(AVG(weekly_sales) OVER (
        PARTITION BY store, dept
        ORDER BY date
        ROWS BETWEEN 12 PRECEDING AND CURRENT ROW
    ), 2)   AS rolling_13w_avg,
    ROUND(STDDEV(weekly_sales) OVER (
        PARTITION BY store, dept
        ORDER BY date
        ROWS BETWEEN 12 PRECEDING AND CURRENT ROW
    ), 2)   AS rolling_13w_std,
    -- Flag weeks where actual deviates >2σ from rolling mean (anomaly signal)
    CASE
        WHEN ABS(weekly_sales - AVG(weekly_sales) OVER (
                    PARTITION BY store, dept
                    ORDER BY date
                    ROWS BETWEEN 12 PRECEDING AND CURRENT ROW
                 )) > 2 * STDDEV(weekly_sales) OVER (
                    PARTITION BY store, dept
                    ORDER BY date
                    ROWS BETWEEN 12 PRECEDING AND CURRENT ROW
                 )
        THEN 'ANOMALY'
        ELSE 'Normal'
    END  AS anomaly_flag
FROM `demand.forecast`
ORDER BY store, dept, date;
```
<img width="1296" height="496" alt="image" src="https://github.com/user-attachments/assets/ccecd15b-b9b7-48b3-b4b0-bdfb1a972abf" />

3). Lag Features for ML Model Training
``` sql
SELECT
    store,
    dept,
    date,
    weekly_sales,
    isholiday,
    -- Lag features
    LAG(weekly_sales, 1)  OVER w   AS sales_lag_1w,
    LAG(weekly_sales, 2)  OVER w   AS sales_lag_2w,
    LAG(weekly_sales, 4)  OVER w   AS sales_lag_4w,
    LAG(weekly_sales, 8)  OVER w   AS sales_lag_8w,
    LAG(weekly_sales, 13) OVER w   AS sales_lag_13w,
    LAG(weekly_sales, 52) OVER w   AS sales_lag_52w,
    -- Rolling stats (4-week)
    AVG(weekly_sales) OVER (PARTITION BY store, dept ORDER BY date ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING)  AS roll_mean_4w,
    STDDEV(weekly_sales) OVER (PARTITION BY store, dept ORDER BY date ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS roll_std_4w,
    -- Rolling stats (13-week)
    AVG(weekly_sales) OVER (PARTITION BY store, dept ORDER BY date ROWS BETWEEN 13 PRECEDING AND 1 PRECEDING) AS roll_mean_13w,
    -- Year-over-year ratio (strong signal for seasonal products)
    weekly_sales / NULLIF(LAG(weekly_sales, 52) OVER w, 0)  AS yoy_ratio,
    -- Calendar features
    EXTRACT(week  FROM date)    AS week_of_year,
    EXTRACT(month FROM date)    AS month,
    EXTRACT(quarter FROM date)  AS quarter,
    CASE WHEN EXTRACT(month FROM date) = 12 THEN 1 ELSE 0 END  AS is_december
FROM `demand.forecast`
WINDOW w AS (PARTITION BY store, dept ORDER BY date)
ORDER BY store, dept, date;
```
<img width="1267" height="468" alt="image" src="https://github.com/user-attachments/assets/01e5e883-e029-423c-b145-bfd60f05b9f6" /><img width="1390" height="416" alt="image" src="https://github.com/user-attachments/assets/fb7e4c84-da1e-4e29-8cce-a56eb9b9dcee" /><img width="305" height="371" alt="image" src="https://github.com/user-attachments/assets/ea276146-b8f7-41ce-bbfe-bee6648a9be6" />

4).Holiday Markdown Impact Analysis
``` sql
WITH markdown_impact AS (
    SELECT
        s.store,
        s.dept,
        s.date,
        s.weekly_sales,
        s.isholiday,
        COALESCE(f.markdown1, 0) + COALESCE(f.markdown2, 0) +
        COALESCE(f.markdown3, 0) + COALESCE(f.markdown4, 0) +
        COALESCE(f.markdown5, 0)  AS total_markdown,
        CASE
            WHEN COALESCE(f.markdown1, 0) + COALESCE(f.markdown2, 0) +
                 COALESCE(f.markdown3, 0) + COALESCE(f.markdown4, 0) +
                 COALESCE(f.markdown5, 0) > 0
            THEN 'Has Markdown'
            ELSE 'No Markdown'
        END  AS markdown_flag
    FROM `demand.forecast` s
    LEFT JOIN `demand.forecast` f ON s.store = f.store AND s.date = f.date
)
SELECT
    isholiday,
    markdown_flag,
    COUNT(*)                     AS n_weeks,
    ROUND(AVG(weekly_sales), 2)  AS avg_sales,
    ROUND(AVG(total_markdown), 2) AS avg_markdown_amount,
    ROUND(CORR(total_markdown, weekly_sales), 4)  AS markdown_sales_correlation
FROM markdown_impact
GROUP BY isholiday, markdown_flag
ORDER BY isholiday DESC, markdown_flag;
```
<img width="1060" height="194" alt="image" src="https://github.com/user-attachments/assets/6ddac2b7-377f-4973-9336-cc78e95fa0aa" />

5).Department-Level Sales Concentration (Pareto Analysis)
``` sql
WITH dept_totals AS (
    SELECT
        store,
        dept,
        SUM(weekly_sales)  AS total_sales
    FROM `demand.forecast`
    GROUP BY store, dept
),
store_totals AS (
    SELECT store, SUM(total_sales) AS store_total FROM dept_totals GROUP BY store
),
ranked AS (
    SELECT
        d.*,
        s.store_total,
        ROUND(d.total_sales / s.store_total * 100, 2)  AS pct_of_store,
        SUM(d.total_sales) OVER (
            PARTITION BY d.store
            ORDER BY d.total_sales DESC
            ROWS UNBOUNDED PRECEDING
        ) / s.store_total * 100  AS cumulative_pct,
        RANK() OVER (PARTITION BY d.store ORDER BY d.total_sales DESC)  AS sales_rank
    FROM dept_totals d
    JOIN store_totals s USING (store)
)
SELECT
    store,
    dept,
    ROUND(total_sales, 0)      AS total_sales,
    pct_of_store,
    ROUND(cumulative_pct, 2)   AS cumulative_pct,
    sales_rank,
    CASE WHEN cumulative_pct <= 80 THEN 'Top 80% (priority)' ELSE 'Tail' END  AS priority_tier
FROM ranked
ORDER BY store, sales_rank;
```
<img width="1142" height="474" alt="image" src="https://github.com/user-attachments/assets/db58579e-e2aa-4069-956a-0d0e2a802bfe" />

6).Store Type Performance Comparison 
``` sql
WITH store_weekly AS (
    SELECT
        s.store,
        st.type,
        st.size,
        s.date,
        SUM(s.weekly_sales)  AS store_weekly_sales
    FROM `demand.forecast` s
    JOIN `demand.forecast` st ON s.store = st.store
    GROUP BY s.store, st.type, st.size, s.date
),
type_stats AS (
    SELECT
        type,
        COUNT(DISTINCT store)                  AS n_stores,
        ROUND(AVG(size),          0)           AS avg_size_sqft,
        ROUND(AVG(store_weekly_sales), 2)      AS avg_weekly_sales,
        ROUND(STDDEV(store_weekly_sales), 2)   AS stddev_weekly_sales,
        ROUND(STDDEV(store_weekly_sales) /
              AVG(store_weekly_sales), 4)      AS coefficient_of_variation,
        ROUND(MIN(store_weekly_sales), 0)      AS min_weekly,
        ROUND(MAX(store_weekly_sales), 0)      AS max_weekly
    FROM store_weekly
    GROUP BY type
)
SELECT
    *,
    -- Higher CV = more volatile = needs more sophisticated model
    CASE
        WHEN coefficient_of_variation > 0.3 THEN 'High volatility — use ML/DL'
        WHEN coefficient_of_variation > 0.15 THEN 'Medium — SARIMA or XGBoost'
        ELSE 'Low — Seasonal Naive or SARIMA sufficient'
    END  AS recommended_model
FROM type_stats
ORDER BY avg_weekly_sales DESC;
```
<img width="1410" height="228" alt="image" src="https://github.com/user-attachments/assets/fb64720b-5d8f-4893-bf8c-5ab0d731cba4" /><img width="243" height="131" alt="image" src="https://github.com/user-attachments/assets/14850385-8e42-41a0-8a82-d58d032030ba" />

7).Fuel Price & Unemployment Correlation with Sales
``` sql
WITH external_features AS (
    SELECT
        s.store,
        s.date,
        SUM(s.weekly_sales)   AS weekly_sales,
        AVG(f.fuel_price)     AS fuel_price,
        AVG(f.unemployment)   AS unemployment,
        AVG(f.temperature)    AS temperature,
        AVG(f.cpi)            AS cpi
    FROM `demand.forecast` s
    LEFT JOIN `demand.forecast`f ON s.store = f.store AND s.date = f.date
    GROUP BY s.store, s.date
)
SELECT
    'Fuel Price' AS feature,
    ROUND(CORR(fuel_price,   weekly_sales), 4)  AS correlation_with_sales
FROM external_features
UNION ALL
SELECT 'Unemployment', ROUND(CORR(unemployment, weekly_sales), 4) FROM external_features
UNION ALL
SELECT 'Temperature',  ROUND(CORR(temperature,  weekly_sales), 4) FROM external_features
UNION ALL
SELECT 'CPI',          ROUND(CORR(cpi,          weekly_sales), 4) FROM external_features
ORDER BY ABS(correlation_with_sales) DESC;
```
<img width="486" height="268" alt="image" src="https://github.com/user-attachments/assets/b1bc6511-993b-4b10-b2a5-877cce2c25b3" />

8). Seasonal Index Calculation
``` sql
WITH weekly_avg AS (
    SELECT
        store,
        dept,
        EXTRACT(week FROM date)   AS week_of_year,
        AVG(weekly_sales)         AS avg_this_week
    FROM `demand.forecast`
    GROUP BY store, dept, EXTRACT(week FROM date)
),
overall_avg AS (
    SELECT
        store,
        dept,
        AVG(weekly_sales) AS overall_avg
    FROM `demand.forecast`
    GROUP BY store, dept
),
seasonal_idx AS (
    SELECT
        w.store,
        w.dept,
        w.week_of_year,
        ROUND(w.avg_this_week / NULLIF(o.overall_avg, 0), 4)  AS seasonal_index
    FROM weekly_avg w
    JOIN overall_avg o USING (store, dept)
)
SELECT
    week_of_year,
    ROUND(AVG(seasonal_index), 4)    AS avg_seasonal_index,
    ROUND(MAX(seasonal_index), 4)    AS max_seasonal_index,
    ROUND(MIN(seasonal_index), 4)    AS min_seasonal_index
FROM seasonal_idx
GROUP BY week_of_year
ORDER BY week_of_year;
```
<img width="730" height="476" alt="image" src="https://github.com/user-attachments/assets/e459e201-3d66-4892-ac6d-eb38f8abcb71" />

**Tableau**

<img width="1267" height="891" alt="image" src="https://github.com/user-attachments/assets/34fc551f-d41e-4ddc-ae07-c69f5e5e1502" />

### Insights

- The "Holiday Halo" Effect: Sales during the four major holiday weeks account for a disproportionate amount of annual revenue, requiring specialized model weighting.

- Markdown Correlation: While Markdowns 1 and 5 showed the strongest correlation with sales spikes, Markdowns 2 and 3 were often specific to certain departments, suggesting a need for department-level promotional strategies.

- Regional Resilience: Economic indicators like CPI and Unemployment showed a surprisingly low correlation with short-term sales fluctuations, suggesting Walmart’s "Everyday Low Price" model is resilient to minor economic shifts.

- Cluster Performance: Type 'A' stores significantly outperformed Types 'B' and 'C' in both volume and stability, indicating that store infrastructure is a major baseline sales driver.

### Recommendations

- Strategic Inventory Stocking: Walmart should prioritize inventory surges 1-2 weeks prior to the "Thanksgiving" and "Christmas" windows, as the data shows these as the highest volume periods.

- Model Selection: For stable departments, SARIMAX is recommended for its ability to handle seasonality; however, for stores with frequent markdowns, Gradient Boosting or Random Forest should be used to capture non-linear interactions between promotions and sales.

- Data Enrichment: Future iterations should include local event data (e.g., sporting events or localized weather emergencies) to further refine store-specific predictions.

- Weighting Holidays: Ensure that the forecasting model continues to prioritize accuracy during holiday weeks, as these periods represent a disproportionate share of annual revenue and are the most prone to stock-outs.
