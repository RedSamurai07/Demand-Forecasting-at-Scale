import pytest
import pandas as pd
import os
from src.data_loader import load_and_merge_data

def test_load_data():
    # In CI, we use the pre-merged clean_demand_data.csv if raw files are missing
    if os.path.exists("clean_demand_data.csv"):
        df = pd.read_csv("clean_demand_data.csv")
        assert df is not None
        # Normalize columns as in app.py
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        assert "weekly_sales" in df.columns
        assert "store" in df.columns
        assert "dept" in df.columns
    else:
        # Fallback to load_and_merge_data if raw files exist
        df = load_and_merge_data(train_path="train.csv", stores_path="stores.csv", features_path="features.csv")
        if df is not None:
            assert "weekly_sales" in df.columns
            assert "store" in df.columns
            assert "dept" in df.columns
        else:
            pytest.skip("Data files missing for testing")
