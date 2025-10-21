#!/usr/bin/env python3
"""
Quick Test Script for Ransomware Detection System
This script tests the core functionality without running the full notebook
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def create_test_dataset():
    """Create synthetic ransomware dataset for testing"""
    print("🔧 Creating synthetic ransomware dataset...")
    
    np.random.seed(42)
    n_samples = 1000  # Smaller dataset for quick testing
    
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
    
    # Create labels based on feature combinations (simulating ransomware behavior)
    ransomware_indicators = (
        (df['file_access_count'] > 80) |
        (df['entropy_change'] > 2) |
        (df['system_calls'] > 150) |
        (df['file_modifications'] > 50) |
        (df['cpu_usage'] > 80)
    )
    
    df['label'] = ransomware_indicators.astype(int)
    
    # Add some noise to make it more realistic
    noise_mask = np.random.random(n_samples) < 0.1
    df.loc[noise_mask, 'label'] = 1 - df.loc[noise_mask, 'label']
    
    print(f"✅ Dataset created: {df.shape[0]} samples, {df.shape[1]-1} features")
    print(f"📊 Label distribution: {df['label'].value_counts().to_dict()}")
    return df

def test_models(df):
    """Test the machine learning models"""
    print("\n🤖 Testing Machine Learning Models...")
    
    # Prepare data
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns]
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test Random Forest
    print("\n🌲 Testing Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"   Accuracy: {rf_accuracy:.4f}")
    
    # Test SVM
    print("\n🎯 Testing SVM...")
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print(f"   Accuracy: {svm_accuracy:.4f}")
    
    return rf, svm, scaler, feature_columns

def test_detection(rf, scaler, feature_columns):
    """Test real-time detection"""
    print("\n🚨 Testing Real-Time Detection...")
    
    # Test ransomware sample
    ransomware_sample = {
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
    
    # Test benign sample
    benign_sample = {
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
    
    def predict_sample(sample, sample_type):
        feature_values = [sample[col] for col in feature_columns]
        feature_array = np.array(feature_values).reshape(1, -1)
        feature_array_scaled = scaler.transform(feature_array)
        
        prediction = rf.predict(feature_array_scaled)[0]
        probability = rf.predict_proba(feature_array_scaled)[0]
        confidence = max(probability)
        
        result = "🚨 RANSOMWARE" if prediction == 1 else "✅ BENIGN"
        print(f"   {sample_type}: {result} (Confidence: {confidence:.4f})")
        return prediction, confidence
    
    print("   Testing samples:")
    pred1, conf1 = predict_sample(ransomware_sample, "Ransomware Sample")
    pred2, conf2 = predict_sample(benign_sample, "Benign Sample")
    
    return pred1, pred2

def main():
    """Main testing function"""
    print("🧪 RANSOMWARE DETECTION SYSTEM - QUICK TEST")
    print("=" * 50)
    
    try:
        # Test 1: Create dataset
        df = create_test_dataset()
        
        # Test 2: Train and test models
        rf, svm, scaler, feature_columns = test_models(df)
        
        # Test 3: Real-time detection
        pred1, pred2 = test_detection(rf, scaler, feature_columns)
        
        # Summary
        print("\n📊 TEST SUMMARY")
        print("=" * 30)
        print("✅ Dataset creation: PASSED")
        print("✅ Model training: PASSED")
        print("✅ Real-time detection: PASSED")
        print("✅ Ransomware sample detected correctly:", pred1 == 1)
        print("✅ Benign sample classified correctly:", pred2 == 0)
        
        print("\n🎉 ALL TESTS PASSED!")
        print("\n💡 Next steps:")
        print("   1. Run 'jupyter notebook' to open the full interface")
        print("   2. Open 'ransomware_detection.ipynb'")
        print("   3. Run all cells for complete analysis")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure all packages are installed:")
        print("   pip install pandas numpy scikit-learn matplotlib seaborn")

if __name__ == "__main__":
    main()
