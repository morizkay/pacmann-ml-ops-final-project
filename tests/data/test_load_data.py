import pytest
import pandas as pd
import numpy as np
from src.data.load_data import load_data, preprocess_data

@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing"""
    data = {
        'provinsi': ['DKI Jakarta', 'Jawa Tengah'],
        'jenis': ['Makanan', 'Makanan'],
        'daerah': ['Perkotaan', 'Perkotaan'],
        'tahun': [2020, 2022],
        'periode': ['MARET', 'MARET'],
        'nilai': [500000, 600000]
    }
    return pd.DataFrame(data)

def test_load_data_shape(sample_data):
    """Test if data loading returns correct shape"""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert df.shape[1] >= 5  # At least year_num, periode_num, and 3 dummy columns

def test_load_data_columns(sample_data):
    """Test if required columns exist"""
    X, _ = load_data()
    required_columns = ['year_num', 'periode_num']
    for col in required_columns:
        assert col in X.columns
    assert any(col.startswith('jenis_') for col in X.columns)
    assert any(col.startswith('daerah_') for col in X.columns)

def test_load_data_types(sample_data):
    """Test if data types are correct"""
    X, _ = load_data()
    assert X['year_num'].dtype in ['int64', 'float64']
    assert X['periode_num'].dtype in ['int64', 'float64']
    assert all(X[col].dtype in ['int64', 'float64'] for col in X.columns)

def test_load_data(sample_data):
    """Test if data can be loaded correctly"""
    X, y = load_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert not X.empty
    assert not y.empty
    assert X.shape[0] == y.shape[0]

def test_preprocess_data(sample_data):
    """Test data preprocessing"""
    X, y = preprocess_data(sample_data)
    
    # Check if numerical features are created
    assert 'year_num' in X.columns
    assert 'periode_num' in X.columns
    
    # Check if categorical variables are encoded
    assert any(col.startswith('jenis_') for col in X.columns)
    assert any(col.startswith('daerah_') for col in X.columns)
    
    # Check if all values are numerical
    assert all(X[col].dtype in [np.float64, np.int64] for col in X.columns)

def test_data_splitting(sample_data):
    """Test if data can be split into features and target"""
    X, y = preprocess_data(sample_data)
    assert len(X) == len(y)
    assert all(X[col].dtype in [np.float64, np.int64] for col in X.columns)
    assert y.dtype in [np.float64, np.int64] 