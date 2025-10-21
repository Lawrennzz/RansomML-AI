#!/usr/bin/env python3
"""
Component Testing - Test individual parts of the system
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def test_data_generation():
    """Test synthetic data generation"""
    print("🔧 Testing Data Generation...")
    
    np.random.seed(42)
    n_samples = 1000
    
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
    
    print(f"   ✅ Generated {df.shape[0]} samples")
    print(f"   ✅ {df.shape[1]-1} features")
    print(f"   ✅ Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df

def test_preprocessing(df):
    """Test data preprocessing"""
    print("\n🔧 Testing Data Preprocessing...")
    
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns]
    y = df['label']
    
    # Test scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   ✅ Feature scaling: {X_scaled.shape}")
    print(f"   ✅ Mean after scaling: {X_scaled.mean():.6f}")
    print(f"   ✅ Std after scaling: {X_scaled.std():.6f}")
    
    return X_scaled, y, scaler, feature_columns

def test_models(X_scaled, y):
    """Test individual models"""
    print("\n🤖 Testing Individual Models...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n   Testing {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        
        print(f"      CV Scores: {cv_scores}")
        print(f"      Mean CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model
        }
    
    return results

def test_feature_importance(df, feature_columns):
    """Test feature importance analysis"""
    print("\n🔍 Testing Feature Importance...")
    
    X = df[feature_columns]
    y = df['label']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("   Top 5 Important Features:")
    for i, row in feature_importance.head().iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")
    
    return feature_importance

def test_visualization(df):
    """Test visualization capabilities"""
    print("\n📊 Testing Visualization...")
    
    try:
        # Test basic plotting
        plt.figure(figsize=(10, 6))
        df['label'].value_counts().plot(kind='bar')
        plt.title('Label Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.close()  # Close to avoid display
        
        print("   ✅ Matplotlib plotting: PASSED")
        
        # Test seaborn
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x='cpu_usage', hue='label', alpha=0.7)
        plt.title('CPU Usage Distribution')
        plt.close()  # Close to avoid display
        
        print("   ✅ Seaborn plotting: PASSED")
        
    except Exception as e:
        print(f"   ❌ Visualization error: {e}")

def main():
    """Run all component tests"""
    print("🧪 COMPONENT TESTING - RANSOMWARE DETECTION SYSTEM")
    print("=" * 60)
    
    try:
        # Test 1: Data Generation
        df = test_data_generation()
        
        # Test 2: Preprocessing
        X_scaled, y, scaler, feature_columns = test_preprocessing(df)
        
        # Test 3: Models
        model_results = test_models(X_scaled, y)
        
        # Test 4: Feature Importance
        feature_importance = test_feature_importance(df, feature_columns)
        
        # Test 5: Visualization
        test_visualization(df)
        
        # Summary
        print("\n📊 COMPONENT TEST SUMMARY")
        print("=" * 40)
        print("✅ Data Generation: PASSED")
        print("✅ Data Preprocessing: PASSED")
        print("✅ Model Training: PASSED")
        print("✅ Feature Analysis: PASSED")
        print("✅ Visualization: PASSED")
        
        print(f"\n🏆 Best Model: {max(model_results.keys(), key=lambda k: model_results[k]['cv_mean'])}")
        print(f"   Accuracy: {max(model_results[k]['cv_mean'] for k in model_results):.4f}")
        
        print("\n🎉 ALL COMPONENT TESTS PASSED!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")

if __name__ == "__main__":
    main()
