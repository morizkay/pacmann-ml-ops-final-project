import pytest
import numpy as np
from src.models.train import (
    create_default_model,
    create_custom_model,
    create_tuned_model,
    train_model,
    evaluate_model
)

def test_model_creation():
    """Test if models can be created"""
    models = [
        create_default_model(),
        create_custom_model(),
        create_tuned_model()
    ]
    for model in models:
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

def test_model_training(processed_data):
    """Test if models can be trained"""
    X, y = processed_data
    models = [
        create_default_model(),
        create_custom_model(),
        create_tuned_model()
    ]
    for model in models:
        model.fit(X, y)
        assert hasattr(model, 'predict')

def test_model_prediction(processed_data):
    """Test if models can make predictions"""
    X, y = processed_data
    models = [
        create_default_model(),
        create_custom_model(),
        create_tuned_model()
    ]
    for model in models:
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert np.all(predictions > 0)  # GK predictions should be positive

def test_model_evaluation(processed_data):
    """Test model evaluation metrics"""
    X, y = processed_data
    model = create_default_model()
    model.fit(X, y)
    metrics = evaluate_model(model, X, y)
    
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert metrics['mse'] >= 0
    assert metrics['rmse'] >= 0
    assert metrics['r2'] <= 1 