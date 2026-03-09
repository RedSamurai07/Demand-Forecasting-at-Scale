import pytest
import pandas as pd
from src.data_loader import load_and_merge_data

def test_load_and_merge():
    # This assumes the CSV files are present in the root for the test
    # In a real CI, we might use a subset of data
    df = load_and_merge_data(train_path="train.csv", stores_path="stores.csv", features_path="features.csv")
    assert df is not None
    assert "weekly_sales" in df.columns
    assert "store" in df.columns
    assert "dept" in df.columns
