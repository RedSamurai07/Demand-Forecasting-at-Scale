import pandas as pd
from sklearn.preprocessing import LabelEncoder

def engineer_features(df):
    """
    Performs feature engineering on the combined dataset.
    """
    df = df.copy()
    
    # Calendar features
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter

    # Lag features (per store-department)
    for lag in [1, 2, 4, 8, 13, 26, 52]:
        df[f"sales_lag_{lag}w"] = (
            df.groupby(["store","dept"])["weekly_sales"].shift(lag))

    # Rolling statistics
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

    # Markdown total
    markdown_cols = [c for c in df.columns if "markdown" in c.lower()]
    if markdown_cols:
        df["total_markdown"] = df[markdown_cols].fillna(0).sum(axis=1)
        df["markdown_interaction"] = df["total_markdown"] * df["isholiday"].astype(int)

    # Trend index per series
    df["time_idx"] = df.groupby(["store","dept"]).cumcount()

    # Encode store type
    if "type" in df.columns:
        le = LabelEncoder()
        df["store_type_enc"] = le.fit_transform(df["type"].astype(str))

    df = df.fillna(0)
    return df
