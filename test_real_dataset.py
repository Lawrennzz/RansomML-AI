#!/usr/bin/env python3
"""
Test with Real Kaggle Dataset
Place your 'ransomware_dataset.csv' in the same directory and run this script
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def test_real_dataset():
    """Test with the actual Kaggle ransomware dataset"""
    try:
        print("🔍 Loading real ransomware dataset...")
        df = pd.read_csv('ransomware_dataset.csv')
        
        print(f"✅ Dataset loaded: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"📈 Label distribution: {df['label'].value_counts().to_dict()}")
        
        # Prepare data
        feature_columns = [col for col in df.columns if col != 'label']
        X = df[feature_columns]
        y = df['label']
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        # Test
        y_pred = rf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 Real Dataset Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Test samples: {len(y_test)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 5 Important Features:")
        for i, row in feature_importance.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Ransomware']))
        
        return True
        
    except FileNotFoundError:
        print("❌ 'ransomware_dataset.csv' not found!")
        print("💡 Download from: https://www.kaggle.com/datasets/amdj3dax/ransomware-detection-data-set")
        print("💡 Or use the synthetic data in the notebook")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_real_dataset()
