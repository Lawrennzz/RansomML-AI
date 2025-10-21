#!/usr/bin/env python3
"""
Ransomware Detection System - Core Version (Without TensorFlow)
This version works with just scikit-learn models
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import sklearn

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')  # Changed from seaborn-v0_8

# Interactive widgets
import ipywidgets as widgets
from IPython.display import display, clear_output

# Model persistence
import joblib

# Performance measurement
import time
from datetime import datetime

print("All libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

# Create synthetic dataset
print("\n🔧 Creating synthetic ransomware dataset...")

np.random.seed(42)
n_samples = 10000

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

# Display sample data
print("\n📋 First 5 rows of the dataset:")
print(df.head())

print("\n📊 Dataset statistics:")
print(df.describe())

# Prepare features and labels
feature_columns = [col for col in df.columns if col != 'label']
X = df[feature_columns]
y = df['label']

print(f"\n🔍 Feature columns: {feature_columns}")
print(f"📈 Number of features: {len(feature_columns)}")
print(f"📊 Number of samples: {len(X)}")
print(f"🚨 Number of ransomware samples: {y.sum()}")
print(f"✅ Number of benign samples: {len(y) - y.sum()}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📚 Training set size: {X_train.shape}")
print(f"🧪 Test set size: {X_test.shape}")

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, 'ransomware_scaler.pkl')
print("\n✅ Features normalized and scaler saved to 'ransomware_scaler.pkl'")

# Train Random Forest to get feature importance
print("\n🌲 Training Random Forest for feature importance...")
rf_feature_analysis = RandomForestClassifier(n_estimators=100, random_state=42)
rf_feature_analysis.fit(X_train_scaled, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_feature_analysis.feature_importances_
}).sort_values('importance', ascending=False)

print("🔍 Feature Importance Ranking:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Feature Importance for Ransomware Detection')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Initialize models (without Neural Network for now)
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        probability=True
    ),
}

# Cross-validation setup
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate models with cross-validation
cv_results = {}
trained_models = {}

print("\n🤖 Training models with 5-fold cross-validation...")
print("=" * 50)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='f1')
    cv_results[name] = cv_scores
    
    # Train on full training set
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\n✅ Cross-validation completed!")

# Save the best model
best_model_name = 'Random Forest'  # Random Forest typically performs well
best_model = trained_models[best_model_name]

# Save models
joblib.dump(best_model, 'best_ransomware_model.pkl')
joblib.dump(trained_models['SVM'], 'ransomware_svm_model.pkl')

print(f"\n💾 Models saved successfully!")
print(f"🏆 Best model: {best_model_name}")
print("📁 Files created:")
print("- best_ransomware_model.pkl")
print("- ransomware_svm_model.pkl")
print("- ransomware_scaler.pkl")

# Evaluate all models on test set
evaluation_results = {}

print("\n📊 Model Evaluation on Test Set")
print("=" * 40)

for name, model in trained_models.items():
    print(f"\nEvaluating {name}...")
    
    # Traditional ML models
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    evaluation_results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Predictions': y_pred
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Check if accuracy target is met
    if accuracy >= 0.95:
        print(f"✅ {name} meets the >95% accuracy target!")
    else:
        print(f"❌ {name} does not meet the >95% accuracy target.")

print("\n✅ Evaluation completed!")

# Create comparison table
results_df = pd.DataFrame({
    name: {
        'Accuracy': results['Accuracy'],
        'Precision': results['Precision'],
        'Recall': results['Recall'],
        'F1-Score': results['F1-Score']
    }
    for name, results in evaluation_results.items()
}).T

print("\n📊 Model Performance Comparison:")
print(results_df.round(4))

# Visualize model comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    results_df[metric].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_title(f'{metric} Comparison')
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for j, v in enumerate(results_df[metric]):
        ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, (name, results) in enumerate(evaluation_results.items()):
    cm = confusion_matrix(y_test, results['Predictions'])
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        ax=axes[i],
        xticklabels=['Benign', 'Ransomware'],
        yticklabels=['Benign', 'Ransomware']
    )
    
    axes[i].set_title(f'{name} Confusion Matrix')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Detailed classification reports
for name, results in evaluation_results.items():
    print(f"\n📋 {name} Classification Report:")
    print("=" * 40)
    print(classification_report(y_test, results['Predictions'], 
                              target_names=['Benign', 'Ransomware']))

print("\n🎉 RANSOMWARE DETECTION SYSTEM - CORE VERSION COMPLETED!")
print("=" * 60)
print("✅ Dataset created and preprocessed")
print("✅ Models trained and evaluated")
print("✅ Performance metrics calculated")
print("✅ Visualizations generated")
print("✅ Models saved for future use")
print("\n💡 Note: This version uses Random Forest and SVM only.")
print("💡 TensorFlow/Neural Network can be added later if needed.")
