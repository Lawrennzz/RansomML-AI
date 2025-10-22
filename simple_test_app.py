#!/usr/bin/env python3
"""
Simple Test App for Visualizations
This will help us debug the chart loading issue
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class SimpleDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_performance = {}
        
    def create_data_and_train(self):
        """Create data and train model"""
        print("Creating synthetic dataset...")
        
        np.random.seed(42)
        n_samples = 5000
        
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
        
        # Create labels
        ransomware_indicators = (
            (df['file_access_count'] > 80) |
            (df['entropy_change'] > 2) |
            (df['system_calls'] > 150) |
            (df['file_modifications'] > 50) |
            (df['cpu_usage'] > 80)
        )
        
        df['label'] = ransomware_indicators.astype(int)
        
        self.feature_columns = [col for col in df.columns if col != 'label']
        X = df[self.feature_columns]
        y = df['label']
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        self.model_performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
        
        print("Model trained successfully!")
        return True

# Initialize detector
detector = SimpleDetector()

@app.route('/')
def index():
    return render_template('simple_test.html')

@app.route('/api/train')
def train():
    """Train the model"""
    try:
        detector.create_data_and_train()
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'performance': detector.model_performance
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Training failed: {str(e)}'
        })

@app.route('/api/charts')
def get_charts():
    """Get chart data"""
    try:
        if not detector.model_performance:
            detector.create_data_and_train()
        
        # Create feature importance chart
        plt.figure(figsize=(10, 6))
        features = list(detector.model_performance['feature_importance'].keys())
        importances = list(detector.model_performance['feature_importance'].values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_data)
        
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), [f.replace('_', ' ').title() for f in features])
        plt.xlabel('Importance Score')
        plt.title('Feature Importance for Ransomware Detection')
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'feature_chart': img_str,
            'performance': detector.model_performance
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Chart creation failed: {str(e)}'
        })

if __name__ == '__main__':
    print("Starting Simple Test App...")
    print("Dashboard: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
