# ml-service/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GasLeakPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 10
        self.feature_columns = ['methane', 'lpg', 'carbonMonoxide', 'hydrogenSulfide', 
                               'temperature', 'humidity', 'pressure']
        self.is_trained = False
        
        # Load or create model
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create a new one"""
        model_path = 'models/gas_leak_model.h5'
        scaler_path = 'models/scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                logger.info("Loaded existing model and scaler")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.create_model()
        else:
            self.create_model()
    
    def create_model(self):
        """Create a new LSTM model for gas leak prediction"""
        os.makedirs('models', exist_ok=True)
        
        # Create LSTM model
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.sequence_length, len(self.feature_columns))),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Generate synthetic training data for demonstration
        self.train_with_synthetic_data()
        
        logger.info("Created and trained new model")
    
    def train_with_synthetic_data(self):
        """Train model with synthetic data"""
        # Generate synthetic training data
        n_samples = 1000
        X_train, y_train = self.generate_synthetic_training_data(n_samples)
        
        # Fit scaler
        X_flat = X_train.reshape(-1, len(self.feature_columns))
        self.scaler.fit(X_flat)
        
        # Scale data
        X_train_scaled = self.scaler.transform(X_flat).reshape(X_train.shape)
        
        # Train model
        self.model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        
        # Save model and scaler
        self.model.save('models/gas_leak_model.h5')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        self.is_trained = True
        logger.info("Model training completed")
    
    def generate_synthetic_training_data(self, n_samples):
        """Generate synthetic training data"""
        X = []
        y = []
        
        for i in range(n_samples):
            # Generate sequence of gas readings
            sequence = []
            is_leak = np.random.choice([0, 1], p=[0.8, 0.2])  # 20% leak scenarios
            
            for j in range(self.sequence_length):
                if is_leak:
                    # Leak scenario - gradually increasing values
                    methane = 200 + (j * 50) + np.random.normal(0, 20)
                    lpg = 150 + (j * 30) + np.random.normal(0, 15)
                    co = 80 + (j * 20) + np.random.normal(0, 10)
                    h2s = 30 + (j * 10) + np.random.normal(0, 5)
                else:
                    # Normal scenario
                    methane = 245 + np.random.normal(0, 10)
                    lpg = 156 + np.random.normal(0, 8)
                    co = 89 + np.random.normal(0, 5)
                    h2s = 23 + np.random.normal(0, 3)
                
                # Environmental factors
                temp = 23.5 + np.random.normal(0, 2)
                humidity = 65.2 + np.random.normal(0, 5)
                pressure = 1013.25 + np.random.normal(0, 10)
                
                sequence.append([methane, lpg, co, h2s, temp, humidity, pressure])
            
            X.append(sequence)
            y.append(is_leak)
        
        return np.array(X), np.array(y)
    
    def predict_leak_probability(self, sensor_data):
        """Predict gas leak probability from sensor data"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        try:
            # Prepare input data
            features = self.extract_features(sensor_data)
            
            if len(features) < self.sequence_length:
                # Pad with last known values if insufficient data
                last_reading = features[-1] if features else [245, 156, 89, 23, 23.5, 65.2, 1013.25]
                while len(features) < self.sequence_length:
                    features.append(last_reading)
            
            # Take last sequence_length readings
            features = features[-self.sequence_length:]
            
            # Scale features
            features_array = np.array(features).reshape(1, self.sequence_length, len(self.feature_columns))
            features_scaled = self.scaler.transform(features_array.reshape(-1, len(self.feature_columns)))
            features_scaled = features_scaled.reshape(features_array.shape)
            
            # Make prediction
            prediction = self.model.predict(features_scaled, verbose=0)[0][0]
            
            # Calculate confidence and risk level
            confidence = min(95, max(75, 80 + (prediction * 15)))  # 75-95% confidence
            risk_level = self.calculate_risk_level(prediction)
            
            # Estimate time to threshold
            time_to_threshold = self.estimate_time_to_threshold(prediction, features[-1])
            
            return {
                "leakProbability": float(prediction * 100),
                "confidence": float(confidence),
                "riskLevel": risk_level,
                "timeToThreshold": time_to_threshold,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}
    
    def extract_features(self, sensor_data):
        """Extract features from sensor data"""
        features = []
        
        for reading in sensor_data:
            feature_row = [
                reading['gases']['methane'],
                reading['gases']['lpg'],
                reading['gases']['carbonMonoxide'],
                reading['gases']['hydrogenSulfide'],
                reading['environmental']['temperature'],
                reading['environmental']['humidity'],
                reading['environmental']['pressure']
            ]
            features.append(feature_row)
        
        return features
    
    def calculate_risk_level(self, probability):
        """Calculate risk level based on probability"""
        if probability > 0.7:
            return "CRITICAL"
        elif probability > 0.5:
            return "HIGH"
        elif probability > 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def estimate_time_to_threshold(self, probability, latest_reading):
        """Estimate time until dangerous threshold is reached"""
        if probability < 0.3:
            return "4.2+ hours"
        elif probability < 0.5:
            return "2.5-4 hours"
        elif probability < 0.7:
            return "1-2 hours"
        else:
            return "< 1 hour"

# Initialize predictor
predictor = GasLeakPredictor()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "service": "ML Prediction Engine",
        "model_trained": predictor.is_trained,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_gas_leak():
    try:
        data = request.get_json()
        
        if not data or 'sensorData' not in data:
            return jsonify({"error": "Invalid request data"}), 400
        
        sensor_data = data['sensorData']
        
        if not sensor_data:
            return jsonify({"error": "No sensor data provided"}), 400
        
        prediction = predictor.predict_leak_probability(sensor_data)
        
        if "error" in prediction:
            return jsonify(prediction), 500
        
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/model/retrain', methods=['POST'])
def retrain_model():
    try:
        predictor.train_with_synthetic_data()
        return jsonify({
            "success": True,
            "message": "Model retrained successfully",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify({
        "model_type": "LSTM Neural Network",
        "sequence_length": predictor.sequence_length,
        "features": predictor.feature_columns,
        "trained": predictor.is_trained,
        "architecture": {
            "layers": [
                "LSTM(50, return_sequences=True)",
                "Dropout(0.2)",
                "LSTM(50)",
                "Dropout(0.2)",
                "Dense(25)",
                "Dense(1, sigmoid)"
            ]
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False )