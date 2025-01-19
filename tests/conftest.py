import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """
    Create a sample DataFrame for testing data loading and preprocessing functions.
    
    Returns:
        pd.DataFrame: A sample DataFrame with columns similar to the actual dataset
    """
    data = {
        'provinsi': ['DKI Jakarta', 'Jawa Barat', 'Jawa Tengah'] * 3,
        'jenis': ['Makanan', 'Non-Makanan', 'Makanan'] * 3,
        'daerah': ['Perkotaan', 'Perdesaan', 'Perkotaan'] * 3,
        'tahun': [2020, 2021, 2022] * 3,
        'periode': ['MARET', 'SEPTEMBER', 'MARET'] * 3,
        'nilai': np.random.uniform(100, 1000, 9)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def processed_data():
    """Create sample processed data"""
    X = np.array([
        [5, 1, 0, 1, 0],  # year_num, periode_num, jenis_NONMAKANAN, daerah_PERKOTAAN, daerah_PERDESAANPERKOTAAN
        [5, 0, 1, 0, 0],
        [6, 1, 0, 1, 0],
        [6, 0, 1, 0, 0]
    ])
    y = np.array([300000, 350000, 400000, 450000])
    
    return X, y

@pytest.fixture
def model_input():
    """Create sample model input"""
    return {
        "features": {
            "year_num": 5,
            "periode_num": 1,
            "jenis_NONMAKANAN": 0,
            "daerah_PERKOTAAN": 1,
            "daerah_PERDESAANPERKOTAAN": 0
        }
    } 