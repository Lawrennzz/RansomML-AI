#!/usr/bin/env python3
"""
Ransomware Detection Web Application (Kagglehub version)
Loads Windows PE feature dataset from Kaggle via kagglehub and trains a model.
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import io
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except Exception as _:
    kagglehub = None
    KaggleDatasetAdapter = None

from typing import Optional

# Optional TensorFlow import for deep models
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    tf = None
    keras = None
    TF_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'ransomware_detection_secret_key_2025'

# Simple JSON storage for rules and settings
RULES_DB_FILE = 'rules.json'
SETTINGS_DB_FILE = 'settings.json'

def _read_json_file(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _write_json_file(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def _safe_float(value, default=0.0):
    try:
        if isinstance(value, str):
            v = value.strip().replace('%', '')
            return float(v)
        return float(value)
    except Exception:
        return float(default)

def _safe_int(value, default=0):
    try:
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ('ransomware', 'true', 'yes'):
                return 1
            if v == 'benign':
                return 0
            if v == '':
                return int(default)
            return int(float(v))
        return int(value)
    except Exception:
        return int(default)

# Role definitions based on UML Use Case Diagram
ROLES = {
    'cybersecurity_professional': {
        'name': 'Cybersecurity Professional',
        'permissions': ['detect_ransomware', 'monitor_system_behavior', 'view_detection_reports', 'train_ml_model', 'configure_detection_rules']
    },
    'it_administrator': {
        'name': 'IT Administrator',
        'permissions': ['train_ml_model', 'configure_detection_rules', 'monitor_system_performance', 'manage_system_settings']
    },
    'system_user': {
        'name': 'System User',
        'permissions': ['view_security_status', 'receive_protection', 'predict']  # Basic users can use prediction
    },
    'academic_researcher': {
        'name': 'Academic Researcher',
        'permissions': ['conduct_research', 'view_detection_reports', 'train_ml_model', 'view_visualizations', 'view_dataset_stats']
    }
}

# Simple user database (in production, use proper database)
USERS_DB_FILE = 'users.json'

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_DB_FILE):
        try:
            with open(USERS_DB_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def init_default_users():
    """Initialize default users if database is empty"""
    users = load_users()
    if not users:
        default_users = {
            'cyber_pro@example.com': {
                'username': 'cyber_pro',
                'password': generate_password_hash('cyber123'),
                'role': 'cybersecurity_professional',
                'full_name': 'Security Expert'
            },
            'admin@example.com': {
                'username': 'admin',
                'password': generate_password_hash('admin123'),
                'role': 'it_administrator',
                'full_name': 'IT Admin'
            },
            'user@example.com': {
                'username': 'user',
                'password': generate_password_hash('user123'),
                'role': 'system_user',
                'full_name': 'System User'
            },
            'researcher@example.com': {
                'username': 'researcher',
                'password': generate_password_hash('research123'),
                'role': 'academic_researcher',
                'full_name': 'Research Scholar'
            }
        }
        save_users(default_users)
        return default_users
    return users

# Initialize default users
USERS = init_default_users()

# Initialize rules and settings with sensible defaults
DEFAULT_RULES = [
    {
        'id': 'rule-1',
        'name': 'High crypto ops => Immediate',
        'conditions': {
            'BitcoinAddresses': { 'gt': 0 }
        },
        'when_prediction_is': 'ransomware',
        'recommendation': 'IMMEDIATE_ACTION',
        'enabled': True
    },
    {
        'id': 'rule-2',
        'name': 'High file modifications => Monitor',
        'conditions': {
            'ExportSize': { 'gt': 500 },
            'ResourceSize': { 'gt': 500 }
        },
        'when_prediction_is': 'ransomware',
        'recommendation': 'MONITOR',
        'enabled': True
    }
]

DEFAULT_SETTINGS = {
    'min_confidence_for_immediate': 0.80,
    'min_confidence_for_monitor': 0.60
}

RULES = _read_json_file(RULES_DB_FILE, DEFAULT_RULES)
SETTINGS = _read_json_file(SETTINGS_DB_FILE, DEFAULT_SETTINGS)


def require_role(*allowed_roles):
    """Decorator to require specific roles for routes"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': 'Authentication required'}), 401
            
            user = USERS.get(session.get('user_id'))
            if not user:
                return jsonify({'success': False, 'message': 'User not found'}), 401
            
            user_role = user.get('role')
            if user_role not in allowed_roles and 'all' not in allowed_roles:
                return jsonify({'success': False, 'message': f'Access denied. Required roles: {", ".join(allowed_roles)}'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_permission(permission):
    """Decorator to require specific permission for routes"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': 'Authentication required'}), 401
            
            user = USERS.get(session.get('user_id'))
            if not user:
                return jsonify({'success': False, 'message': 'User not found'}), 401
            
            user_role = user.get('role')
            role_permissions = ROLES.get(user_role, {}).get('permissions', [])
            
            if permission not in role_permissions:
                return jsonify({'success': False, 'message': f'Permission denied. Required permission: {permission}'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


class RansomwareDetector:
    """Detector using Kaggle PE features dataset.

    Label convention: dataset has 'Benign' (1=benign, 0=malicious).
    We predict 'ransomware' = 1 - Benign for clarity in UI.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.training_data = None
        self.model_performance = {}
        self.detection_history = []
        self.dataset_loaded = False
        self.model_type = 'rf'
        self.system_logs = []  # Store system behavior logs
        self.detection_logs = []  # Detailed detection event logs

    def load_kaggle_dataset(self, file_path: str = "data_file.csv"):
        """Load dataset from local CSV file first, fallback to kagglehub if needed"""
        # Try loading from local file first
        if os.path.exists(file_path):
            print(f"Loading dataset from local file: {file_path}")
            df = pd.read_csv(file_path)
        elif kagglehub is not None:
            print(f"Loading dataset from Kaggle via kagglehub...")
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "amdj3dax/ransomware-detection-data-set",
                file_path,
            )
        else:
            raise RuntimeError(
                f"Dataset file '{file_path}' not found locally and kagglehub not available. "
                "Either place data_file.csv in the project directory or install: pip install kagglehub[pandas-datasets]"
            )

        print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

        # Basic cleaning: drop identifiers, keep numeric columns
        drop_cols = [c for c in ['FileName', 'md5Hash'] if c in df.columns]
        df = df.drop(columns=drop_cols, errors='ignore')
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

        # Ensure Benign column exists
        if 'Benign' not in df.columns:
            raise ValueError("Dataset missing 'Benign' label column")

        # Define target as ransomware = 1 - Benign (1=benign, 0=malicious in dataset)
        df = df.copy()
        df['ransomware'] = 1 - df['Benign'].astype(int)

        # Feature selection: numeric columns excluding labels
        numeric_df = df.select_dtypes(include=[np.number])
        self.feature_columns = [c for c in numeric_df.columns if c not in ['Benign', 'ransomware']]

        # Keep only features + target
        self.training_data = numeric_df[self.feature_columns + ['ransomware']]
        self.dataset_loaded = True
        
        print(f"Processed dataset: {len(self.training_data)} samples, {len(self.feature_columns)} features")
        print(f"Features: {', '.join(self.feature_columns)}")
        
        return self.training_data

    def _build_mlp_keras(self, input_dim: int):
        """Build Keras MLP model"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras not installed for Neural Networks. Install with: pip install tensorflow")
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_cnn_lstm(self, timesteps: int):
        """Build CNN-LSTM model"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras not installed for CNN-LSTM. Install with: pip install tensorflow")
        model = keras.Sequential([
            keras.layers.Input(shape=(timesteps, 1)),
            keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            keras.layers.LSTM(64),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_models(self, model_type: Optional[str] = None):
        if not self.dataset_loaded:
            self.load_kaggle_dataset()
        
        df = self.training_data
        X = df[self.feature_columns]
        y = df['ransomware']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        chosen = (model_type or self.model_type or 'rf').lower()
        self.model_type = chosen

        # Start training timer
        training_start_time = time.time()

        if chosen == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced_subsample',
            )
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
        elif chosen == 'svm':
            self.model = SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
        elif chosen == 'nn':
            # Try Keras first, fallback to scikit-learn MLP
            if TF_AVAILABLE:
                try:
                    model = self._build_mlp_keras(X_train_scaled.shape[1])
                    model.fit(X_train_scaled, y_train, epochs=10, batch_size=256, verbose=0, validation_split=0.1)
                    self.model = model
                    y_prob = model.predict(X_test_scaled, verbose=0).reshape(-1)
                    y_pred = (y_prob >= 0.5).astype(int)
                except Exception as e:
                    # Fallback to scikit-learn MLP
                    self.model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                                               alpha=1e-4, batch_size=256, max_iter=100, random_state=42)
                    self.model.fit(X_train_scaled, y_train)
                    y_pred = self.model.predict(X_test_scaled)
            else:
                # Use scikit-learn MLP
                self.model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                                           alpha=1e-4, batch_size=256, max_iter=100, random_state=42)
                self.model.fit(X_train_scaled, y_train)
                y_pred = self.model.predict(X_test_scaled)
        elif chosen == 'cnn-lstm':
            if not TF_AVAILABLE:
                raise RuntimeError("CNN-LSTM requires TensorFlow. Install with: pip install tensorflow")
            Xtr_seq = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
            Xte_seq = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            model = self._build_cnn_lstm(Xtr_seq.shape[1])
            model.fit(Xtr_seq, y_train, epochs=8, batch_size=256, verbose=0, validation_split=0.1)
            self.model = model
            y_prob = model.predict(Xte_seq, verbose=0).reshape(-1)
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            raise ValueError(f"Unsupported model_type: {chosen}")
        
        # Calculate training time
        training_time = time.time() - training_start_time
        
        # Calculate feature importance
        feature_importance_dict = None
        try:
            # For Random Forest, use built-in feature_importances_
            if hasattr(self.model, 'feature_importances_') and not callable(getattr(self.model, 'feature_importances_', None)):
                feature_importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_.tolist()))
            else:
                # For other models (SVM, Neural Networks, CNN-LSTM), use permutation importance
                # This works with any model but takes longer to compute
                print(f"Calculating permutation importance for {chosen} model...")
                perm_importance = permutation_importance(
                    self.model, X_test_scaled, y_test,
                    n_repeats=10,  # Number of times to permute each feature
                    random_state=42,
                    n_jobs=-1  # Use all available cores
                )
                # Use mean importance scores
                feature_importance_dict = dict(zip(self.feature_columns, perm_importance.importances_mean.tolist()))
                print("Permutation importance calculation complete.")
        except Exception as e:
            print(f"Warning: Could not calculate feature importance: {str(e)}")
            feature_importance_dict = None
        
        self.model_performance = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': feature_importance_dict,
            'num_features': len(self.feature_columns),
            'num_samples': int(len(df)),
            'model_type': self.model_type,
            'training_time_seconds': float(training_time),
        }
        return self.model_performance
    
    def predict(self, features: dict):
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet")
        
        # Check if model is fitted (for scikit-learn models)
        if hasattr(self.model, 'support_vectors_'):
            # For SVM, check if model is fitted
            try:
                _ = len(self.model.support_vectors_)
            except AttributeError:
                raise ValueError("SVM model not properly fitted. Please retrain the model.")
        
        values = []
        for col in self.feature_columns:
            raw_val = features.get(col, 0)
            try:
                # robustly coerce to float; fallback to 0.0
                coerced = float(raw_val)
            except Exception:
                coerced = 0.0
            values.append(coerced)
        arr = np.array(values, dtype=float).reshape(1, -1)
        arr_scaled = self.scaler.transform(arr)

        # Predict based on model type
        try:
            model_type_lower = str(self.model_type).lower()
            
            if model_type_lower == 'cnn-lstm':
                if not TF_AVAILABLE:
                    raise ValueError("CNN-LSTM requires TensorFlow. Install with: pip install tensorflow")
                if not hasattr(self.model, 'predict'):
                    raise ValueError("CNN-LSTM model is not properly initialized")
                # Ensure correct shape for CNN-LSTM: (batch, timesteps, features)
                seq_len = arr_scaled.shape[1]
                arr_in = arr_scaled.reshape((1, seq_len, 1))
                prob_output = self.model.predict(arr_in, verbose=0)
                prob1 = float(np.array(prob_output).reshape(-1)[0])
                # Ensure probability is between 0 and 1
                prob1 = max(0.0, min(1.0, prob1))
                pred = int(prob1 >= 0.5)
                proba = np.array([1.0 - prob1, prob1])
                
            elif model_type_lower == 'nn':
                # Check if it's a Keras model
                is_keras_model = False
                if TF_AVAILABLE:
                    try:
                        is_keras_model = isinstance(self.model, keras.Model) or hasattr(self.model, '_keras_api_names')
                    except:
                        pass
                
                if is_keras_model:
                    # Keras MLP model
                    prob_output = self.model.predict(arr_scaled, verbose=0)
                    prob1 = float(np.array(prob_output).reshape(-1)[0])
                    prob1 = max(0.0, min(1.0, prob1))
                    pred = int(prob1 >= 0.5)
                    proba = np.array([1.0 - prob1, prob1])
                else:
                    # scikit-learn MLP
                    if not hasattr(self.model, 'predict'):
                        raise ValueError("Neural Network model is not properly initialized")
                    pred = int(self.model.predict(arr_scaled)[0])
                    try:
                        proba = self.model.predict_proba(arr_scaled)[0]
                    except:
                        # Fallback if predict_proba not available
                        proba = np.array([0.5, 0.5])
                        
            elif model_type_lower == 'svm':
                # SVM with probability support
                if not hasattr(self.model, 'predict'):
                    raise ValueError("SVM model is not properly initialized")
                try:
                    pred = int(self.model.predict(arr_scaled)[0])
                    proba_raw = self.model.predict_proba(arr_scaled)
                    # Handle both 1D and 2D outputs
                    if len(proba_raw.shape) == 2:
                        proba = proba_raw[0]
                    else:
                        proba = proba_raw
                    # Ensure we have 2 probabilities
                    if len(proba) < 2:
                        proba = np.array([1.0 - proba[0], proba[0]])
                except AttributeError as e:
                    # If support_vectors_ error, model might not be fitted
                    raise ValueError(f"SVM model not properly fitted: {str(e)}. Please retrain the model.")
                    
            elif model_type_lower == 'rf':
                # Random Forest with probability support
                if not hasattr(self.model, 'predict'):
                    raise ValueError("Random Forest model is not properly initialized")
                pred = int(self.model.predict(arr_scaled)[0])
                proba_raw = self.model.predict_proba(arr_scaled)
                # Handle both 1D and 2D outputs
                if len(proba_raw.shape) == 2:
                    proba = proba_raw[0]
                else:
                    proba = proba_raw
                # Ensure we have 2 probabilities
                if len(proba) < 2:
                    proba = np.array([1.0 - proba[0], proba[0]])
            else:
                # Default: try predict_proba first, then predict
                if not hasattr(self.model, 'predict'):
                    raise ValueError(f"Model type '{self.model_type}' is not properly initialized")
                pred = int(self.model.predict(arr_scaled)[0])
                if hasattr(self.model, 'predict_proba'):
                    try:
                        proba_raw = self.model.predict_proba(arr_scaled)
                        if len(proba_raw.shape) == 2:
                            proba = proba_raw[0]
                        else:
                            proba = proba_raw
                    except:
                        proba = np.array([0.5, 0.5])
                else:
                    proba = np.array([1.0 - float(pred), float(pred)])
            
            # Normalize probabilities to ensure 2 values
            if len(proba) < 2:
                if pred == 0:
                    proba = np.array([0.9, 0.1])
                else:
                    proba = np.array([0.1, 0.9])
            
            # Ensure probabilities sum to 1 and are valid
            proba = np.clip(proba, 0.0, 1.0)
            proba = proba / proba.sum()  # Normalize
            
            benign_prob = float(proba[0])
            ransom_prob = float(proba[1])
            confidence = float(max(benign_prob, ransom_prob))
            
        except ValueError:
            raise  # Re-raise ValueError as-is
        except Exception as e:
            raise ValueError(f"Prediction error for {self.model_type}: {str(e)}")
        
        result = {
            'prediction': pred,  # 1=ransomware, 0=benign
            'confidence': confidence,
            'benign_probability': benign_prob,
            'ransomware_probability': ransom_prob,
            'risk_level': self._risk_level(confidence),
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_type': self.model_type,
        }
        
        # Enhanced detection logging
        detection_log = {
            'timestamp': result['timestamp'],
            'prediction': 'Ransomware' if pred == 1 else 'Benign',
            'confidence': confidence,
            'risk_level': result['risk_level'],
            'model_type': self.model_type,
            'behavioral_features': self._extract_behavioral_indicators(features),
            'threat_classification': self._classify_threat(features, pred, confidence),
            'features_snapshot': {k: v for k, v in features.items() if k in self.feature_columns[:10]}  # Top 10 features
        }
        
        self.detection_history.append({'features': features, 'result': result})
        self.detection_logs.append(detection_log)
        
        # Keep only last 1000 logs to prevent memory issues
        if len(self.detection_logs) > 1000:
            self.detection_logs = self.detection_logs[-1000:]
        
        return result
    
    def _extract_behavioral_indicators(self, features: dict) -> dict:
        """Extract key behavioral indicators from features"""
        indicators = {
            'file_modifications': features.get('ExportSize', 0) + features.get('ResourceSize', 0),
            'system_calls': features.get('NumberOfSections', 0),
            'directory_access': features.get('DebugRVA', 0),
            'crypto_operations': features.get('BitcoinAddresses', 0),
            'suspicious_activity_score': 0.0
        }
        
        # Calculate suspicious activity score
        score = 0.0
        if indicators['crypto_operations'] > 0:
            score += 0.3  # Bitcoin addresses indicate crypto operations
        if indicators['file_modifications'] > 1000:
            score += 0.3  # High file modification activity
        if features.get('DllCharacteristics', 0) > 20000:
            score += 0.2  # Suspicious DLL characteristics
        if features.get('SizeOfStackReserve', 0) > 1048576:
            score += 0.2  # Large stack reserve
        
        indicators['suspicious_activity_score'] = min(score, 1.0)
        return indicators
    
    def _classify_threat(self, features: dict, prediction: int, confidence: float) -> str:
        """Classify the threat level based on behavioral patterns"""
        if prediction == 0:
            return 'Normal'
        
        # Analyze behavioral patterns
        crypto_ops = features.get('BitcoinAddresses', 0)
        file_mods = features.get('ExportSize', 0) + features.get('ResourceSize', 0)
        dll_chars = features.get('DllCharacteristics', 0)
        
        if crypto_ops > 0 and confidence > 0.8:
            return 'High-Risk Crypto Ransomware'
        elif file_mods > 5000 and confidence > 0.7:
            return 'High-Risk File Encryption Ransomware'
        elif dll_chars > 20000 and confidence > 0.6:
            return 'Medium-Risk Suspicious Behavior'
        elif confidence > 0.5:
            return 'Low-Risk Potential Threat'
        else:
            return 'Uncertain - Needs Review'
    
    def ingest_system_logs(self, logs: list):
        """Ingest and preprocess system behavior logs"""
        try:
            # Preprocess logs
            processed_logs = []
            for log in logs:
                # Clean and format log entry
                processed = {
                    'timestamp': log.get('timestamp', pd.Timestamp.now().isoformat()),
                    'event_type': log.get('event_type', 'unknown'),
                    'behavioral_data': self._preprocess_behavioral_data(log.get('behavioral_data', {})),
                    'raw_data': log
                }
                processed_logs.append(processed)
            
            self.system_logs.extend(processed_logs)
            
            # Keep only last 5000 logs
            if len(self.system_logs) > 5000:
                self.system_logs = self.system_logs[-5000:]
            
            return {'success': True, 'processed': len(processed_logs), 'total_logs': len(self.system_logs)}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _preprocess_behavioral_data(self, data: dict) -> dict:
        """Preprocess behavioral data: cleaning and formatting"""
        preprocessed = {}
        
        # Map common behavioral data to PE features
        feature_mapping = {
            'file_access_count': 'ExportSize',
            'file_modifications': 'ResourceSize',
            'system_calls': 'NumberOfSections',
            'directory_access': 'DebugRVA',
            'crypto_operations': 'BitcoinAddresses',
            'process_count': 'Machine',
            'registry_changes': 'IatVRA',
            'memory_usage': 'SizeOfStackReserve',
            'dll_characteristics': 'DllCharacteristics',
            'debug_info': 'DebugSize'
        }
        
        for key, value in data.items():
            # Clean numeric values
            if isinstance(value, (int, float)):
                preprocessed[key] = float(value)
            elif isinstance(value, str):
                try:
                    preprocessed[key] = float(value)
                except:
                    preprocessed[key] = 0.0
            else:
                preprocessed[key] = 0.0
        
        return preprocessed
    
    def get_detection_logs(self, limit: int = 100):
        """Get detailed detection event logs"""
        return self.detection_logs[-limit:] if limit else self.detection_logs
    
    def get_system_logs(self, limit: int = 100):
        """Get system behavior logs"""
        return self.system_logs[-limit:] if limit else self.system_logs
    
    def _risk_level(self, confidence: float) -> str:
        if confidence > 0.8:
            return 'HIGH'
        if confidence > 0.6:
            return 'MEDIUM'
        return 'LOW'
    
    def get_detection_history(self):
        return self.detection_history[-50:]
    
    def get_dataset_stats(self):
        if not self.dataset_loaded:
            return None
        df = self.training_data
        return {
            'total_samples': int(len(df)),
            'features': int(len(self.feature_columns)),
            'feature_stats': df[self.feature_columns].describe().to_dict(),
        }

    # Rule evaluation helper
    def evaluate_rules(self, features: dict, prediction: int, confidence: float):
        """Evaluate configured rules against features and model output.

        Returns tuple: (matched_rules: list, recommendation: str or None)
        """
        matched = []
        recommendation = None
        pred_str = 'ransomware' if prediction == 1 else 'benign'
        for rule in RULES:
            if not rule.get('enabled', True):
                continue
            if rule.get('when_prediction_is', pred_str) not in [pred_str, 'any']:
                continue
            conds = rule.get('conditions', {}) or {}
            ok = True
            for feat, comp in conds.items():
                val = _safe_float(features.get(feat, 0) or 0)
                if 'gt' in comp and not (val > _safe_float(comp.get('gt', 0))):
                    ok = False; break
                if 'gte' in comp and not (val >= _safe_float(comp.get('gte', 0))):
                    ok = False; break
                if 'lt' in comp and not (val < _safe_float(comp.get('lt', 0))):
                    ok = False; break
                if 'lte' in comp and not (val <= _safe_float(comp.get('lte', 0))):
                    ok = False; break
                if 'eq' in comp and not (val == _safe_float(comp.get('eq', 0))):
                    ok = False; break
            if ok:
                matched.append({'id': rule.get('id'), 'name': rule.get('name')})
                # first strong recommendation wins
                if recommendation is None:
                    recommendation = rule.get('recommendation')
        # Fallback to confidence thresholds if no rule matched
        if recommendation is None and prediction == 1:
            conf_val = _safe_float(confidence, 0.0)
            if conf_val >= _safe_float(SETTINGS.get('min_confidence_for_immediate', 0.80)):
                recommendation = 'IMMEDIATE_ACTION'
            elif conf_val >= _safe_float(SETTINGS.get('min_confidence_for_monitor', 0.60)):
                recommendation = 'MONITOR'
        if recommendation is None:
            recommendation = 'NORMAL'
        return matched, recommendation


detector = RansomwareDetector()

@app.route('/')
def index():
    """Main dashboard page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=USERS.get(session.get('user_id')), roles=ROLES)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        data = request.get_json() or {}
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        user = USERS.get(email)
        if user and check_password_hash(user['password'], password):
            session['user_id'] = email
            session['user_role'] = user['role']
            session['username'] = user['username']
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user': {
                    'email': email,
                    'username': user['username'],
                    'role': user['role'],
                    'role_name': ROLES[user['role']]['name']
                }
            })
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
    
    return render_template('login.html', roles=ROLES)

@app.route('/logout')
def logout():
    """Logout"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/current-user')
def current_user():
    """Get current logged in user"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    
    user = USERS.get(session.get('user_id'))
    if user:
        role_info = ROLES.get(user['role'], {})
        return jsonify({
            'success': True,
            'user': {
                'email': session['user_id'],
                'username': user['username'],
                'role': user['role'],
                'role_name': role_info.get('name', 'Unknown'),
                'permissions': role_info.get('permissions', [])
            }
        })
    return jsonify({'success': False, 'message': 'User not found'}), 404

@app.route('/api/train', methods=['POST'])
@require_permission('train_ml_model')
def train_models():
    """Train the machine learning models"""
    try:
        payload = request.get_json(silent=True) or {}
        requested = (payload.get('model_type') or '').lower().strip() or None
        try:
            performance = detector.train_models(model_type=requested)
            return jsonify({
                'success': True,
                'message': 'Models trained successfully',
                'performance': performance
            })
        except Exception as e:
            error_msg = str(e)
            # Provide helpful error message
            if 'TensorFlow' in error_msg or 'tensorflow' in error_msg.lower():
                return jsonify({
                    'success': False,
                    'message': f'Training failed: {error_msg}. Install TensorFlow with: pip install tensorflow'
                })
            return jsonify({
                'success': False,
                'message': f'Training failed: {error_msg}'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Training failed: {str(e)}'
        })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on input features - available to all authenticated users"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
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
@require_permission('view_detection_reports')
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

@app.route('/api/feature-columns')
def feature_columns():
    try:
        return jsonify({
            'success': True,
            'features': detector.feature_columns
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/ingest-logs', methods=['POST'])
@require_permission('monitor_system_behavior')
def ingest_logs():
    """Ingest and preprocess system behavior logs"""
    try:
        data = request.get_json()
        logs = data.get('logs', [])
        
        if not logs:
            return jsonify({'success': False, 'message': 'No logs provided'})
        
        result = detector.ingest_system_logs(logs)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ingestion failed: {str(e)}'})

@app.route('/api/classify-realtime', methods=['POST'])
def classify_realtime():
    """Real-time classification of system behavior"""
    try:
        behavioral_data = request.get_json()
        
        if not behavioral_data:
            return jsonify({'success': False, 'message': 'No behavioral data provided'})
        
        # Preprocess behavioral data to extract features
        preprocessed = detector._preprocess_behavioral_data(behavioral_data)
        
        # Map to expected feature format
        features = {}
        for col in detector.feature_columns:
            # Try direct mapping first
            if col in preprocessed:
                features[col] = preprocessed[col]
            else:
                # Use default mapping or 0
                features[col] = 0.0
        
        # Classify
        result = detector.predict(features)

        # Evaluate rules for recommendation
        matched_rules, recommendation = detector.evaluate_rules(
            features, result['prediction'], result['confidence']
        )

        # Return with behavioral indicators
        behavioral_indicators = detector._extract_behavioral_indicators(features)
        threat_classification = detector._classify_threat(features, result['prediction'], result['confidence'])
        
        return jsonify({
            'success': True,
            'result': result,
            'behavioral_indicators': behavioral_indicators,
            'threat_classification': threat_classification,
            'recommendation': recommendation,
            'matched_rules': matched_rules
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Classification failed: {str(e)}'})

@app.route('/api/detection-logs')
def get_detection_logs():
    """Get detailed detection event logs"""
    try:
        limit = request.args.get('limit', 100, type=int)
        logs = detector.get_detection_logs(limit)
        return jsonify({
            'success': True,
            'logs': logs,
            'total': len(logs)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/system-logs')
@require_permission('monitor_system_performance')
def get_system_logs():
    """Get system behavior logs"""
    try:
        limit = request.args.get('limit', 100, type=int)
        logs = detector.get_system_logs(limit)
        return jsonify({
            'success': True,
            'logs': logs,
            'total': len(logs)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/report/detection', methods=['GET'])
@require_permission('view_detection_reports')
def download_detection_report_pdf():
    try:
        if not REPORTLAB_AVAILABLE:
            return jsonify({'success': False, 'message': 'PDF generation not available. Install reportlab.'}), 501
        limit = request.args.get('limit', 50, type=int)
        logs = detector.get_detection_logs(limit)

        # Build PDF into memory
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, height - 50, "Ransomware Detection Report")
        c.setFont("Helvetica", 10)
        c.drawString(40, height - 68, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Summary
        y = height - 90
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Summary")
        y -= 16
        benign = sum(1 for x in logs if x.get('prediction') == 'Benign')
        ransom = sum(1 for x in logs if x.get('prediction') == 'Ransomware')
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Total events: {len(logs)} | Benign: {benign} | Ransomware: {ransom}")
        y -= 24

        # Table header
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "Time")
        c.drawString(200, y, "Prediction")
        c.drawString(290, y, "Risk")
        c.drawString(360, y, "Confidence")
        c.drawString(450, y, "Model")
        y -= 12
        c.line(40, y, width - 40, y)
        y -= 10

        c.setFont("Helvetica", 9)
        for item in reversed(logs):
            if y < 60:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica-Bold", 10)
                c.drawString(40, y, "Time")
                c.drawString(200, y, "Prediction")
                c.drawString(290, y, "Risk")
                c.drawString(360, y, "Confidence")
                c.drawString(450, y, "Model")
                y -= 12
                c.line(40, y, width - 40, y)
                y -= 10
                c.setFont("Helvetica", 9)

            c.drawString(40, y, str(item.get('timestamp', '') )[:19])
            c.drawString(200, y, str(item.get('prediction', '')))
            c.drawString(290, y, str(item.get('risk_level', '')))
            c.drawString(360, y, f"{_safe_float(item.get('confidence', 0))*100:.1f}%")
            c.drawString(450, y, str(item.get('model_type', '')))
            y -= 14

        c.showPage()
        c.save()
        buffer.seek(0)
        return send_file(buffer, mimetype='application/pdf', as_attachment=True,
                         download_name=f"detection_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/report/prediction', methods=['POST'])
def download_single_prediction_pdf():
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Authentication required'}), 401
        if not REPORTLAB_AVAILABLE:
            return jsonify({'success': False, 'message': 'PDF generation not available. Install reportlab.'}), 501
        payload = request.get_json() or {}
        result = payload.get('result', {}) or {}
        indicators = payload.get('behavioral_indicators', {})
        threat = payload.get('threat_classification', '')
        recommendation = payload.get('recommendation', 'NORMAL')
        matched_rules = payload.get('matched_rules', [])

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Header / branding
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, height - 50, "Ransomware Prediction Report")
        c.setFont("Helvetica", 10)
        c.drawString(40, height - 68, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        y = height - 100
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Summary")
        y -= 18
        c.setFont("Helvetica", 10)
        pred_val = _safe_int(result.get('prediction', 0))
        pred_txt = 'RANSOMWARE' if pred_val == 1 else 'BENIGN'
        benign_prob = _safe_float(result.get('benign_probability', 0))
        ransom_prob = _safe_float(result.get('ransomware_probability', 0))
        confidence_val = _safe_float(result.get('confidence', max(benign_prob, ransom_prob)))
        conf_pct = confidence_val * 100.0
        risk_txt = str(result.get('risk_level','') or '')
        c.drawString(40, y, f"Prediction: {pred_txt} | Confidence: {conf_pct:.2f}% | Risk: {risk_txt}")
        y -= 16
        c.drawString(40, y, f"Model: {str(result.get('model_type','') or '')} | Timestamp: {str(result.get('timestamp','') or '')[:19]}")
        y -= 24

        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Recommendation")
        y -= 16
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"{recommendation or ''}")
        y -= 24

        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Threat Classification")
        y -= 16
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"{threat or ''}")
        y -= 24

        # Probabilities
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Probabilities")
        y -= 16
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Benign: {benign_prob*100:.1f}%  |  Ransomware: {ransom_prob*100:.1f}%")
        y -= 24

        # Behavioral indicators
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Behavioral Indicators")
        y -= 16
        c.setFont("Helvetica", 10)
        for k in ['suspicious_activity_score','crypto_operations','file_modifications','system_calls']:
            if y < 60:
                c.showPage(); y = height - 50; c.setFont("Helvetica", 10)
            v = indicators.get(k, '')
            if k == 'suspicious_activity_score':
                v = f"{_safe_float(v, 0.0)*100:.1f}%"
            else:
                v = str(v) if v != '' else '0'
            c.drawString(40, y, f"- {k.replace('_',' ').title()}: {v}")
            y -= 14

        # Matched rules
        if matched_rules:
            if y < 80:
                c.showPage(); y = height - 50
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y, "Matched Rules")
            y -= 16
            c.setFont("Helvetica", 10)
            for r in matched_rules:
                if y < 60:
                    c.showPage(); y = height - 50; c.setFont("Helvetica", 10)
                c.drawString(40, y, f"- {r.get('name','')} ({r.get('id','')})")
                y -= 14

        c.showPage()
        c.save()
        buffer.seek(0)
        return send_file(buffer, mimetype='application/pdf', as_attachment=True,
                         download_name=f"prediction_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ------------------------ Detection Rules API ------------------------

@app.route('/api/rules', methods=['GET'])
@require_permission('configure_detection_rules')
def list_rules():
    try:
        return jsonify({'success': True, 'rules': RULES})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/rules', methods=['POST'])
@require_permission('configure_detection_rules')
def upsert_rule():
    try:
        data = request.get_json() or {}
        rid = (data.get('id') or f"rule-{int(time.time()*1000)}")
        exists = False
        for idx, r in enumerate(RULES):
            if r.get('id') == rid:
                RULES[idx] = {
                    'id': rid,
                    'name': data.get('name', r.get('name', 'Untitled Rule')),
                    'conditions': data.get('conditions', r.get('conditions', {})),
                    'when_prediction_is': data.get('when_prediction_is', r.get('when_prediction_is', 'any')),
                    'recommendation': data.get('recommendation', r.get('recommendation', 'MONITOR')),
                    'enabled': bool(data.get('enabled', r.get('enabled', True)))
                }
                exists = True
                break
        if not exists:
            RULES.append({
                'id': rid,
                'name': data.get('name', 'Untitled Rule'),
                'conditions': data.get('conditions', {}),
                'when_prediction_is': data.get('when_prediction_is', 'any'),
                'recommendation': data.get('recommendation', 'MONITOR'),
                'enabled': bool(data.get('enabled', True))
            })
        _write_json_file(RULES_DB_FILE, RULES)
        return jsonify({'success': True, 'rules': RULES, 'updated_id': rid})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/rules/<rule_id>', methods=['DELETE'])
@require_permission('configure_detection_rules')
def delete_rule(rule_id):
    try:
        global RULES
        RULES = [r for r in RULES if r.get('id') != rule_id]
        _write_json_file(RULES_DB_FILE, RULES)
        return jsonify({'success': True, 'rules': RULES})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ------------------------ System Settings API ------------------------

@app.route('/api/settings', methods=['GET'])
@require_permission('manage_system_settings')
def get_settings():
    try:
        return jsonify({'success': True, 'settings': SETTINGS})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
@require_permission('manage_system_settings')
def update_settings():
    try:
        data = request.get_json() or {}
        SETTINGS.update({
            'min_confidence_for_immediate': float(data.get('min_confidence_for_immediate', SETTINGS.get('min_confidence_for_immediate', 0.80))),
            'min_confidence_for_monitor': float(data.get('min_confidence_for_monitor', SETTINGS.get('min_confidence_for_monitor', 0.60)))
        })
        _write_json_file(SETTINGS_DB_FILE, SETTINGS)
        return jsonify({'success': True, 'settings': SETTINGS})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """Upload and analyze CSV dataset"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        # Read CSV
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        df_uploaded = pd.read_csv(stream)
        
        # Check for required features
        if not detector.feature_columns:
            # Load default dataset to get feature names
            detector.load_kaggle_dataset()
        
        # Find matching columns
        available_features = [f for f in detector.feature_columns if f in df_uploaded.columns]
        missing_features = [f for f in detector.feature_columns if f not in df_uploaded.columns]
        
        # Prepare preview data (first 10 rows, only available features)
        preview_data = df_uploaded[available_features].head(10).to_dict(orient='records')
        
        # Statistics
        stats = {
            'total_rows': int(len(df_uploaded)),
            'available_features': len(available_features),
            'missing_features': len(missing_features),
            'available_feature_names': available_features,
            'missing_feature_names': missing_features,
        }
        
        return jsonify({
            'success': True,
            'message': f'CSV uploaded: {len(df_uploaded)} rows, {len(available_features)}/{len(detector.feature_columns)} features found',
            'preview': preview_data,
            'stats': stats,
            'columns': list(df_uploaded.columns)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Upload failed: {str(e)}'})

@app.route('/api/feature-importance')
def feature_importance():
    try:
        imp = detector.model_performance.get('feature_importance') if detector.model_performance else None
        return jsonify({'success': True, 'importance': imp, 'model_type': detector.model_type})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/flow')
def flow_definitions():
    try:
        data = {
            'use_cases': [
                'Detect Ransomware', 'Monitor System Behaviour', 'View Detection Reports',
                'Train ML model', 'Configure Detection Rules', 'Monitor System Performance',
                'Manage System Settings', 'View Security Status', 'Receive Protection', 'Conduct Research'
            ],
            'activity_steps': [
                'start', 'use detection model', 'monitor file access', 'suspicious activity?',
                'analyze behaviours', 'apply detection rules', 'ransomware detected?',
                'generate the report', 'notify the user', 'display security status', 'end'
            ],
            'sequence': [
                {'from':'User','to':'DetectionEngine','action':'start detecting'},
                {'from':'DetectionEngine','to':'SystemMonitor','action':'start monitoring'},
                {'from':'SystemMonitor','to':'Activity','action':'monitoring file access'},
                {'from':'Activity','to':'SystemMonitor','action':'returns if file operations detected'},
                {'from':'DetectionEngine','to':'DetectionEngine','action':'apply detection rules'},
                {'from':'DetectionEngine','to':'ReportGenerator','action':'generates detection report'},
                {'from':'ReportGenerator','to':'DetectionEngine','action':'retrieve detection reports'},
                {'from':'DetectionEngine','to':'User','action':'notify users about the threat'},
                {'from':'User','to':'DetectionEngine','action':'views security status'}
            ]
        }
        return jsonify({'success': True, 'flow': data})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    # Ensure rules/settings files exist
    if not os.path.exists(RULES_DB_FILE):
        _write_json_file(RULES_DB_FILE, RULES)
    if not os.path.exists(SETTINGS_DB_FILE):
        _write_json_file(SETTINGS_DB_FILE, SETTINGS)
    
    print("Starting Ransomware Detection Web Application...")
    print("Dashboard available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
