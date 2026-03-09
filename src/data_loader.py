import pandas as pd
import numpy as np

def load_and_merge_data(train_path="train.csv", stores_path="stores.csv", features_path="features.csv"):
    """
    Loads and merges the Walmart dataset.
    """
    print("--- Loading and Merging Datasets ---")
    try:
        train = pd.read_csv(train_path, parse_dates=['Date'])
        stores = pd.read_csv(stores_path)
        features = pd.read_csv(features_path, parse_dates=['Date']).drop(columns=['IsHoliday'], errors='ignore')
        
        dataset = train.merge(stores, how='left').merge(features, how='left')
        
        dataset.columns = [c.lower().replace(" ", "_") for c in dataset.columns]
        dataset = dataset.sort_values(["store", "dept", "date"]).reset_index(drop=True)
        
        print(f"Dataset Merged: {dataset.shape[0]:,} rows")
        return dataset
    except Exception as e:
        print(f"Error loading files: {e}")
        return None
