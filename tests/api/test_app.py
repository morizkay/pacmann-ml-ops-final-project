import pytest
import json
from src.api.app import app

@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'

def test_predict_endpoint(client, model_input):
    """Test the prediction endpoint"""
    response = client.post('/predict',
                         json=model_input,
                         content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert isinstance(data['prediction'], (int, float))
    assert data['prediction'] > 0  # GK predictions should be positive

def test_predict_endpoint_bad_input(client):
    """Test the prediction endpoint with invalid input"""
    response = client.post('/predict',
                         json={},
                         content_type='application/json')
    assert response.status_code == 400
    
    response = client.post('/predict',
                         json={'invalid': 'input'},
                         content_type='application/json')
    assert response.status_code == 400

def test_metadata_endpoint(client):
    """Test the metadata endpoint"""
    response = client.get('/metadata')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'model_type' in data
    assert 'features' in data
    assert 'version' in data
    assert isinstance(data['features'], list)
    assert len(data['features']) > 0 