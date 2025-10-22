#!/usr/bin/env python3
"""
Hybrid Ransomware Detection Application
Combines Flask web app with Jupyter notebook functionality
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'ransomware_detection_secret_key_2024'

class HybridRansomwareDetector:
    """Enhanced detector with Jupyter-style analytics"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.training_data = None
        self.model_performance = {}
        self.detection_history = []
        self.analytics_data = {}
        
    def create_synthetic_dataset(self, n_samples=5000):
        """Create synthetic ransomware dataset"""
        print("Creating synthetic ransomware dataset...")
        
        np.random.seed(42)
        
        # Generate synthetic features
        data = {
            'file_access_count': np.random.poisson(50, n_samples),
            'entropy_change': np.random.normal(0, 2, n_samples),
            'system_calls': np.random.poisson(100, n_samples),
            'network_connections': np.random.poisson(20, n_samples),
            'file_modifications': np.random.poisson(30, n_samples),
            'cpu_usage': np.random.beta(2, 5, n_samples) * 100,
            'memory_usage': np.random.beta(2, 5, n_samples) * 100,
            'disk_io': np.random.poisson(200, n_samples),
            'process_count': np.random.poisson(150, n_samples),
            'registry_changes': np.random.poisson(10, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create labels based on feature combinations
        ransomware_indicators = (
            (df['file_access_count'] > 80) |
            (df['entropy_change'] > 2) |
            (df['system_calls'] > 150) |
            (df['file_modifications'] > 50) |
            (df['cpu_usage'] > 80)
        )
        
        df['label'] = ransomware_indicators.astype(int)
        
        # Add noise
        noise_mask = np.random.random(n_samples) < 0.1
        df.loc[noise_mask, 'label'] = 1 - df.loc[noise_mask, 'label']
        
        self.training_data = df
        self.feature_columns = [col for col in df.columns if col != 'label']
        
        return df
    
    def train_models(self):
        """Train machine learning models"""
        if self.training_data is None:
            self.create_synthetic_dataset()
        
        df = self.training_data
        X = df[self.feature_columns]
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        self.model_performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
        
        # Store analytics data
        self.analytics_data = {
            'test_predictions': y_pred.tolist(),
            'test_labels': y_test.tolist(),
            'feature_importance': self.model.feature_importances_.tolist(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return self.model_performance
    
    def predict(self, features):
        """Make prediction on new data"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet")
        
        feature_values = [features[col] for col in self.feature_columns]
        feature_array = np.array(feature_values).reshape(1, -1)
        feature_array_scaled = self.scaler.transform(feature_array)
        
        prediction = self.model.predict(feature_array_scaled)[0]
        probability = self.model.predict_proba(feature_array_scaled)[0]
        confidence = max(probability)
        
        result = {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'benign_probability': float(probability[0]),
            'ransomware_probability': float(probability[1]),
            'risk_level': self._get_risk_level(confidence),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Add to detection history
        self.detection_history.append({
            'features': features,
            'result': result
        })
        
        return result
    
    def _get_risk_level(self, confidence):
        """Determine risk level based on confidence"""
        if confidence > 0.8:
            return "HIGH"
        elif confidence > 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def create_feature_importance_chart(self):
        """Create feature importance visualization"""
        if not self.model_performance:
            return None
        
        plt.figure(figsize=(10, 6))
        features = list(self.model_performance['feature_importance'].keys())
        importances = list(self.model_performance['feature_importance'].values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_data)
        
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), [f.replace('_', ' ').title() for f in features])
        plt.xlabel('Importance Score')
        plt.title('Feature Importance for Ransomware Detection')
        plt.tight_layout()
        
        # Convert to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_str
    
    def create_confusion_matrix_chart(self):
        """Create confusion matrix visualization"""
        if not self.model_performance:
            return None
        
        plt.figure(figsize=(8, 6))
        cm = np.array(self.model_performance['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Benign', 'Ransomware'],
                   yticklabels=['Benign', 'Ransomware'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Convert to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_str
    
    def get_detection_history(self):
        """Get detection history"""
        return self.detection_history[-50:]
    
    def get_dataset_stats(self):
        """Get dataset statistics"""
        if self.training_data is None:
            return None
        
        df = self.training_data
        return {
            'total_samples': len(df),
            'features': len(self.feature_columns),
            'ransomware_samples': int(df['label'].sum()),
            'benign_samples': int(len(df) - df['label'].sum()),
            'feature_stats': df[self.feature_columns].describe().to_dict()
        }

# Initialize detector
detector = HybridRansomwareDetector()

# Auto-train model on startup
print("Initializing hybrid detector...")
detector.create_synthetic_dataset()
detector.train_models()
print("✅ Model trained and ready!")

@app.route('/')
def index():
    """Main dashboard page with Jupyter-style analytics"""
    return render_template('hybrid_index.html')

@app.route('/api/train', methods=['POST'])
def train_models():
    """Train the machine learning models"""
    try:
        performance = detector.train_models()
        return jsonify({
            'success': True,
            'message': 'Models trained successfully',
            'performance': performance
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Training failed: {str(e)}'
        })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on input features"""
    try:
        features = request.json
        result = detector.predict(features)
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Prediction failed: {str(e)}'
        })

@app.route('/api/dataset-stats')
def dataset_stats():
    """Get dataset statistics"""
    try:
        stats = detector.get_dataset_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to get stats: {str(e)}'
        })

@app.route('/api/detection-history')
def detection_history():
    """Get detection history"""
    try:
        history = detector.get_detection_history()
        return jsonify({
            'success': True,
            'history': history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to get history: {str(e)}'
        })

@app.route('/api/model-performance')
def model_performance():
    """Get model performance metrics"""
    try:
        if not detector.model_performance:
            detector.train_models()
        
        return jsonify({
            'success': True,
            'performance': detector.model_performance
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to get performance: {str(e)}'
        })

@app.route('/api/feature-chart')
def feature_chart():
    """Get feature importance chart"""
    try:
        # Ensure model is trained
        if not detector.model_performance:
            detector.train_models()
        
        chart_data = detector.create_feature_importance_chart()
        return jsonify({
            'success': True,
            'chart': chart_data,
            'performance': detector.model_performance
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to create chart: {str(e)}'
        })

@app.route('/api/confusion-matrix')
def confusion_matrix():
    """Get confusion matrix chart"""
    try:
        # Ensure model is trained
        if not detector.model_performance:
            detector.train_models()
        
        chart_data = detector.create_confusion_matrix_chart()
        return jsonify({
            'success': True,
            'chart': chart_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to create confusion matrix: {str(e)}'
        })

@app.route('/notebook')
def notebook_view():
    """Jupyter-style notebook view"""
    return render_template('notebook_view.html')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("Starting Hybrid Ransomware Detection Application...")
    print("Web Dashboard: http://localhost:5000")
    print("Notebook View: http://localhost:5000/notebook")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
