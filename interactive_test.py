#!/usr/bin/env python3
"""
Interactive Ransomware Detection Tester
Test the system with your own custom inputs
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def train_model():
    """Train a quick model for testing"""
    print("🤖 Training detection model...")
    
    # Create training data
    np.random.seed(42)
    n_samples = 2000
    
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
    
    # Train model
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns]
    y = df['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    print("✅ Model trained successfully!")
    return rf, scaler, feature_columns

def get_user_input():
    """Get user input for testing"""
    print("\n📝 Enter system behavior metrics:")
    print("(Press Enter for default values)")
    
    metrics = {}
    defaults = {
        'file_access_count': 50,
        'entropy_change': 0.0,
        'system_calls': 100,
        'network_connections': 20,
        'file_modifications': 30,
        'cpu_usage': 50.0,
        'memory_usage': 50.0,
        'disk_io': 200,
        'process_count': 150,
        'registry_changes': 10
    }
    
    for feature, default in defaults.items():
        try:
            value = input(f"{feature.replace('_', ' ').title()} [{default}]: ").strip()
            if value:
                metrics[feature] = float(value) if 'usage' in feature else int(value)
            else:
                metrics[feature] = default
        except ValueError:
            print(f"Invalid input, using default: {default}")
            metrics[feature] = default
    
    return metrics

def predict_ransomware(model, scaler, feature_columns, metrics):
    """Make prediction"""
    feature_values = [metrics[col] for col in feature_columns]
    feature_array = np.array(feature_values).reshape(1, -1)
    feature_array_scaled = scaler.transform(feature_array)
    
    prediction = model.predict(feature_array_scaled)[0]
    probability = model.predict_proba(feature_array_scaled)[0]
    confidence = max(probability)
    
    return prediction, confidence, probability

def main():
    """Main interactive testing function"""
    print("🚨 INTERACTIVE RANSOMWARE DETECTION TESTER")
    print("=" * 50)
    
    # Train model
    model, scaler, feature_columns = train_model()
    
    while True:
        print("\n" + "="*50)
        print("Choose an option:")
        print("1. Test with custom input")
        print("2. Test with sample ransomware behavior")
        print("3. Test with sample benign behavior")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Custom input
            metrics = get_user_input()
            
        elif choice == '2':
            # Ransomware sample
            print("\n🚨 Testing with RANSOMWARE sample:")
            metrics = {
                'file_access_count': 150,
                'entropy_change': 3.5,
                'system_calls': 300,
                'network_connections': 50,
                'file_modifications': 100,
                'cpu_usage': 85.0,
                'memory_usage': 90.0,
                'disk_io': 500,
                'process_count': 200,
                'registry_changes': 25
            }
            
        elif choice == '3':
            # Benign sample
            print("\n✅ Testing with BENIGN sample:")
            metrics = {
                'file_access_count': 30,
                'entropy_change': 0.5,
                'system_calls': 80,
                'network_connections': 10,
                'file_modifications': 15,
                'cpu_usage': 25.0,
                'memory_usage': 40.0,
                'disk_io': 100,
                'process_count': 120,
                'registry_changes': 2
            }
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")
            continue
        
        # Make prediction
        prediction, confidence, probability = predict_ransomware(model, scaler, feature_columns, metrics)
        
        # Display results
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   Prediction: {'🚨 RANSOMWARE DETECTED!' if prediction == 1 else '✅ System appears BENIGN'}")
        print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"   Benign Probability: {probability[0]:.4f}")
        print(f"   Ransomware Probability: {probability[1]:.4f}")
        
        # Risk assessment
        if confidence > 0.8:
            risk_level = "🔴 HIGH RISK"
        elif confidence > 0.6:
            risk_level = "🟡 MEDIUM RISK"
        else:
            risk_level = "🟢 LOW RISK"
        
        print(f"   Risk Level: {risk_level}")
        
        # Feature analysis
        print(f"\n🔍 FEATURE ANALYSIS:")
        for feature, value in metrics.items():
            print(f"   {feature.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()
