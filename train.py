import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_loader import load_and_merge_data
from src.features import engineer_features
import os

def wmae(y_true, y_pred, weights):
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def train_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Demand_Forecasting_Walmart")
    
    with mlflow.start_run():
        # Load and process data
        data = load_and_merge_data()
        if data is None:
            return
            
        data = engineer_features(data)
        
        # Simple split for demonstration (in production, use TimeSeriesSplit)
        train_df = data[data['year'] < 2012]
        test_df = data[data['year'] == 2012]
        
        features = [col for col in data.columns if col not in ['weekly_sales', 'date', 'type']]
        X_train = train_df[features]
        y_train = train_df['weekly_sales']
        X_test = test_df[features]
        y_test = test_df['weekly_sales']
        
        # Weights for WMAE: 5x if it's a holiday
        weights = test_df['isholiday'].apply(lambda x: 5 if x else 1).values
        
        # Model parameters
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        
        mlflow.log_params(params)
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        wmae_score = wmae(y_test, preds, weights)
        
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("WMAE", wmae_score)
        
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Training complete. WMAE: {wmae_score:.2f}")

if __name__ == "__main__":
    train_model()
