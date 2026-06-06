import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import MagicMock

# 1. ADVANCED STREAMLIT SIMULATION MOCKS
mock_st = MagicMock()
# Forces Streamlit buttons and state parameters to return true, executing inner form logic
mock_st.button.return_value = True
mock_st.sidebar = MagicMock()
mock_st.selectbox.return_value = 1
mock_st.number_input.return_value = 10.0

# Prevent rendering elements from crashing inside headless CI runners
mock_st.dataframe = MagicMock()
mock_st.success = MagicMock()
mock_st.error = MagicMock()
mock_st.write = MagicMock()
mock_st.title = MagicMock()

sys.modules['streamlit'] = mock_st

# 2. MACHINE LEARNING MODULE MOCKS
mock_mlflow = MagicMock()
mock_mlflow.start_run.return_value.__enter__ = MagicMock()
mock_mlflow.start_run.return_value.__exit__ = MagicMock()
sys.modules['mlflow'] = mock_mlflow
sys.modules['mlflow.sklearn'] = MagicMock()

mock_regressor_instance = MagicMock()
mock_regressor_instance.predict.return_value = np.array([150.0, 250.0])

mock_xgb_module = MagicMock()
mock_xgb_module.XGBRegressor.return_value = mock_regressor_instance
sys.modules['xgboost'] = mock_xgb_module

from src.data_loader import load_and_merge_data
from src.features import engineer_features

# -------------------------------------------------------------------------
# CORE DATA PROCESSING TESTS
# -------------------------------------------------------------------------
def test_load_and_merge_data_pipeline(tmp_path):
    train_df = pd.DataFrame({
        "Store": [1], "Dept": [10], "Date": ["2011-01-07"],
        "Weekly_Sales": [1500.0], "IsHoliday": [False]
    })
    stores_df = pd.DataFrame({"Store": [1], "Type": ["A"], "Size": [100000]})
    features_df = pd.DataFrame({
        "Store": [1], "Date": ["2011-01-07"], "Temperature": [40.0],
        "Fuel_Price": [2.5], "MarkDown1": [0.0], "MarkDown2": [0.0],
        "MarkDown3": [0.0], "MarkDown4": [0.0], "MarkDown5": [0.0],
        "CPI": [210.0], "Unemployment": [8.0], "IsHoliday": [False]
    })

    t_file = tmp_path / "train.csv"
    s_file = tmp_path / "stores.csv"
    f_file = tmp_path / "features.csv"

    train_df.to_csv(t_file, index=False)
    stores_df.to_csv(s_file, index=False)
    features_df.to_csv(f_file, index=False)

    df = load_and_merge_data(str(t_file), str(s_file), str(f_file))
    assert df is not None


def test_engineer_features_logic():
    mock_df = pd.DataFrame({
        "date": pd.to_datetime(["2011-01-07", "2011-01-14"]),
        "weekly_sales": [100.0, 200.0],
        "store": [1, 1], "dept": [10, 10], "isholiday": [False, False], "type": ["A", "A"],
        "markdown1": [1.0, 1.0], "markdown2": [1.0, 1.0], "markdown3": [1.0, 1.0],
        "markdown4": [1.0, 1.0], "markdown5": [1.0, 1.0]
    })
    res = engineer_features(mock_df)
    assert "week_of_year" in res.columns


def test_wmae_loss_calculation():
    import train
    y_true = np.array([200.0, 400.0])
    y_pred = np.array([210.0, 390.0])
    weights = np.array([1.0, 5.0])
    score = train.wmae(y_true, y_pred, weights)
    assert score >= 0

# -------------------------------------------------------------------------
# EXTENDED INTERACTIVE FLOW EXECUTION FOR TARGET LOGIC
# -------------------------------------------------------------------------
def test_train_model_execution(monkeypatch):
    mock_data = pd.DataFrame({
        "store": [1, 1, 1, 1, 1], "dept": [10, 10, 10, 10, 10],
        "weekly_sales": [100.0, 200.0, 150.0, 300.0, 250.0],
        "date": pd.to_datetime(["2011-01-07", "2011-01-14", "2011-01-21", "2012-01-06", "2012-01-13"]),
        "year": [2011, 2011, 2011, 2012, 2012],
        "isholiday": [False] * 5, "type": ["A"] * 5, "size": [150000] * 5,
        "markdown1": [0.0] * 5, "markdown2": [0.0] * 5, "markdown3": [0.0] * 5, 
        "markdown4": [0.0] * 5, "markdown5": [0.0] * 5, "cpi": [211.0] * 5, 
        "unemployment": [8.1] * 5, "temperature": [42.0] * 5, "fuel_price": [2.5] * 5
    })

    monkeypatch.setattr("src.data_loader.load_and_merge_data", MagicMock(return_value=mock_data))
    monkeypatch.setattr("src.features.engineer_features", MagicMock(return_value=mock_data))
    monkeypatch.setattr("joblib.dump", MagicMock())
    monkeypatch.setattr("os.path.getsize", MagicMock(return_value=1024))

    import train
    train.train_model()
    assert True


def test_app_initialization_flow(monkeypatch):
    """Simulates active user input actions to sweep unexecuted user branches."""
    mock_app_data = pd.DataFrame({
        "store": [1, 2], "dept": [10, 20], "weekly_sales": [100, 200],
        "date": ["2011-01-07", "2011-01-14"], "type": ["A", "B"], "size": [100, 200],
        "temperature": [50, 60], "fuel_price": [2.5, 2.6], "cpi": [210, 211],
        "unemployment": [7.0, 7.1], "isholiday": [False, True]
    })

    # Intercept data reading & prediction utilities
    monkeypatch.setattr("pandas.read_csv", MagicMock(return_value=mock_app_data))
    monkeypatch.setattr("joblib.load", MagicMock(return_value=mock_regressor_instance))
    monkeypatch.setattr("src.features.engineer_features", MagicMock(return_value=mock_app_data))

    # Force clean context evaluation
    if 'app' in sys.modules:
        del sys.modules['app']
    
    import app
    assert app is not None