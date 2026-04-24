import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.features import engineer_features
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Demand Forecasting at Scale",
    page_icon="📦",
    layout="wide"
)

# ── Model loader ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.joblib")
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ── Data loader ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("clean_demand_data.csv")
        # Normalise column names (handle any capitalisation inconsistency)
        data.columns = [c.lower().replace(" ", "_") for c in data.columns]
        # Parse date column
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
        data = engineer_features(data)
        return data
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📦 Walmart Demand Forecasting")
st.caption("XGBoost model trained on 45 Walmart stores • Weekly sales forecast by store & department")
st.markdown("---")

model = load_model()
data  = load_data()

# ── Guard rails ──────────────────────────────────────────────────────────────
if model is None:
    st.warning(
        "⚠️ No trained model found (`model.joblib` is missing). "
        "Run `python train.py` locally and push the generated file to GitHub."
    )
    st.stop()

if data is None:
    st.warning(
        "⚠️ Data file `clean_demand_data.csv` not found. "
        "Ensure it is committed to the repository and pushed to GitHub."
    )
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🔎 Filter")

stores         = sorted(data["store"].unique())
selected_store = st.sidebar.selectbox("Store", stores)

store_data     = data[data["store"] == selected_store]
depts          = sorted(store_data["dept"].unique())
selected_dept  = st.sidebar.selectbox("Department", depts)

subset = store_data[store_data["dept"] == selected_dept].sort_values("date")

# ── Main content ──────────────────────────────────────────────────────────────
if subset.empty:
    st.warning("No data available for the selected Store and Department.")
    st.stop()

st.subheader(f"Store {selected_store} · Dept {selected_dept} — Historical Sales vs Forecast")

# Feature columns (exclude targets, date and raw type string)
features = [col for col in data.columns if col not in ["weekly_sales", "date", "type"]]

X      = subset[features]
y_true = subset["weekly_sales"]

try:
    preds = model.predict(X)

    plot_df = pd.DataFrame({
        "Date":            subset["date"].values,
        "Actual Sales":    y_true.values,
        "Predicted Sales": preds,
    }).set_index("Date")

    st.line_chart(plot_df)

    # ── Metrics row ───────────────────────────────────────────────────────
    mae  = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mape = np.mean(np.abs((y_true.values - preds) / (np.abs(y_true.values) + 1e-8))) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE",  f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("MAPE", f"{mape:.1f}%")

    st.markdown("#### Recent Predictions")
    st.dataframe(
        plot_df.reset_index().sort_values("Date", ascending=False).head(20),
        use_container_width=True,
    )

except Exception as e:
    st.error(f"Prediction error: {e}")
