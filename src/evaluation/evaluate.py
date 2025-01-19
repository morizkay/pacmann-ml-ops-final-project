import pandas as pd
import numpy as np
import mlflow
import json
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load a model from file"""
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def evaluate_single_model(model, model_name, X, y):
    """Evaluate a single model and return metrics"""
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate metrics
    valid_indices = ~np.isnan(y)  # Get indices where target is not NaN
    if not np.all(valid_indices):
        logger.warning(f"Found {np.sum(~valid_indices)} NaN values in target data")
        y = y[valid_indices]
        predictions = predictions[valid_indices]
    
    metrics = {
        "mse": mean_squared_error(y, predictions),
        "rmse": np.sqrt(mean_squared_error(y, predictions)),
        "mae": mean_absolute_error(y, predictions),
        "r2": r2_score(y, predictions)
    }
    
    # Add percentage error
    metrics["mape"] = np.mean(np.abs((y - predictions) / y)) * 100
    
    # Create feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        os.makedirs("metrics", exist_ok=True)
        feature_importance.to_csv(f"metrics/feature_importance_{model_name}.csv", index=False)
        mlflow.log_artifact(f"metrics/feature_importance_{model_name}.csv")
    
    return metrics, predictions

def evaluate_models():
    """
    Evaluate all trained models and compare their performance.
    Returns a dictionary of metrics for each model.
    """
    with mlflow.start_run(run_name="model_evaluation"):
        logger.info("Loading data...")
        # Load data
        X = pd.read_csv("data/processed/features.csv")
        y = pd.read_csv("data/processed/target.csv")['gk'].values
        
        # Handle missing values in features and target
        logger.info("Checking for missing values...")
        if X.isna().any().any():
            logger.warning(f"Found NaN values in features, filling with mean...")
            X = X.fillna(X.mean())
        
        if np.isnan(y).any():
            logger.warning(f"Found NaN values in target, filling with mean...")
            y = pd.Series(y).fillna(pd.Series(y).mean()).values
        
        logger.info(f"Data loaded - X shape: {X.shape}, y shape: {y.shape}")
        
        # Model names to evaluate
        model_names = ["default", "custom", "tuned"]
        all_metrics = {}
        all_predictions = {}
        
        # Evaluate each model
        for model_name in model_names:
            model_path = f"models/{model_name}_model.pkl"
            logger.info(f"Evaluating {model_name} model...")
            
            model = load_model(model_path)
            if model is None:
                continue
                
            metrics, predictions = evaluate_single_model(model, model_name, X, y)
            all_metrics[model_name] = metrics
            all_predictions[model_name] = predictions
            
            # Log metrics to MLflow
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", value)
            
            logger.info(f"{model_name} model metrics: {metrics}")
        
        # Save all metrics
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/all_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=4)
        
        # Find best model
        best_model = max(all_metrics.items(), key=lambda x: x[1]['r2'])
        logger.info(f"Best model: {best_model[0]} with R² score: {best_model[1]['r2']:.4f}")
        
        return all_metrics

if __name__ == "__main__":
    evaluate_models() 