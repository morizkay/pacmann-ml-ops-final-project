import numpy as np
import pandas as pd
from src.features.preprocessing import preprocess_data

def test_preprocessing_output_shape(sample_data):
    """Test if preprocessing returns correct shapes"""
    X, y = preprocess_data(sample_data)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] >= 5  # At least year_num, periode_num, and 3 dummy columns
    assert isinstance(y, pd.Series)

def test_preprocessing_scaling(sample_data):
    """Test if data is properly scaled"""
    X, _ = preprocess_data(sample_data)
    assert 'year_num' in X.columns
    assert 'periode_num' in X.columns
    assert X['year_num'].min() == 0
    assert X['periode_num'].isin([0, 1]).all()

def test_preprocessing_no_nulls(sample_data):
    """Test if preprocessing handles null values"""
    X, y = preprocess_data(sample_data)
    assert not X.isnull().any().any()
    assert not y.isnull().any()

def test_preprocessing_target_range(sample_data):
    """Test if target values are in expected range"""
    _, y = preprocess_data(sample_data)
    assert (y >= 0).all()  # GK values should be positive 