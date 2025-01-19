from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model and metrics
try:
    # Load best model (tuned model)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tuned_model.pkl')
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load model metrics
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'metrics', 'all_metrics.json')
    with open(metrics_path, 'rb') as f:
        model_metrics = json.load(f)
    logger.info("Model and metrics loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or metrics: {str(e)}")
    model = None
    model_metrics = None

@app.route('/health')
def health():
    """Health check endpoint"""
    if model is None:
        return jsonify({
            "status": "unhealthy",
            "error": "Model not loaded"
        }), 503
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions for GK values.
    Expected input format:
    {
        "features": {
            "year_num": 5,
            "periode_num": 1,
            "jenis_NONMAKANAN": 0,
            "jenis_TOTAL": 0,
            "daerah_PERDESAANPERKOTAAN": 0,
            "daerah_PERKOTAAN": 1
        }
    }
    """
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "status": "error"
        }), 503
    
    try:
        # Get input data
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({
                "error": "Invalid input format. Expected 'features' object in request",
                "status": "error"
            }), 400
        
        # Validate features
        required_features = ['year_num', 'periode_num', 'jenis_NONMAKANAN', 'jenis_TOTAL', 
                           'daerah_PERDESAANPERKOTAAN', 'daerah_PERKOTAAN']
        missing_features = [f for f in required_features if f not in data['features']]
        if missing_features:
            return jsonify({
                "error": f"Missing required features: {missing_features}",
                "status": "error"
            }), 400
        
        # Convert to DataFrame
        features = pd.DataFrame([data['features']])
        
        # Make prediction
        prediction = model.predict(features)
        
        return jsonify({
            "prediction": float(prediction[0]),
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/metadata')
def metadata():
    """Get model metadata and performance metrics"""
    if model is None or model_metrics is None:
        return jsonify({
            "error": "Model or metrics not loaded",
            "status": "error"
        }), 503
    
    return jsonify({
        "model_type": "Random Forest Regressor (Tuned)",
        "metrics": model_metrics.get('tuned', {}),
        "features": ['year_num', 'periode_num', 'jenis_NONMAKANAN', 'jenis_TOTAL', 
                    'daerah_PERDESAANPERKOTAAN', 'daerah_PERKOTAAN'],
        "status": "success"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) 