import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_default_model():
    """Create a default linear regression model"""
    return LinearRegression()

def create_custom_model():
    """Create a custom random forest model"""
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

def create_tuned_model():
    """Create a tuned random forest model"""
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2)
    }

def train_model():
    """Train and evaluate different models for GK prediction"""
    with mlflow.start_run(run_name="model_training"):
        # Load processed data
        logger.info("Loading processed data...")
        X = pd.read_csv("data/processed/features.csv")
        y = pd.read_csv("data/processed/target.csv")
        
        # Handle any remaining NaN values
        if y['nilai'].isna().any():
            logger.warning("Found NaN values in target, filling with mean...")
            y['nilai'] = y['nilai'].fillna(y['nilai'].mean())
        
        # Convert to numpy arrays
        X_array = X.values
        y_array = y['nilai'].values
        
        logger.info(f"Data loaded - X shape: {X_array.shape}, y shape: {y_array.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array, test_size=0.2, random_state=42
        )
        
        # Train three different models
        models = {
            "default": create_default_model(),
            "custom": create_custom_model(),
            "tuned": create_tuned_model()
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            logger.info(f"Training {name} model...")
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test)
            results[name] = metrics
            
            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{name}_{metric_name}", value)
            
            # Log model
            mlflow.sklearn.log_model(model, f"{name}_model")
            
            # Save model locally
            model_dir = os.path.join(os.path.dirname(__file__), 'tuned_model.pkl')
            os.makedirs(os.path.dirname(model_dir), exist_ok=True)
            with open(model_dir, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"{name} model metrics: {metrics}")
        
        # Save metrics
        metrics_dir = os.path.join(os.path.dirname(__file__), '..', 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        with open(os.path.join(metrics_dir, 'all_metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

if __name__ == "__main__":
    train_model() 