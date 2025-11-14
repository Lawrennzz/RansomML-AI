#!/usr/bin/env python3
"""
Create a test sample CSV dataset from the original data_file.csv structure
This will be used for testing the upload and training functionality
"""

import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, 'data_file.csv')

print("Loading original dataset...")
print(f"Looking for data file at: {data_file_path}")
df_original = pd.read_csv(data_file_path)

print(f"Original dataset: {len(df_original)} rows, {len(df_original.columns)} columns")
print(f"Benign distribution: {df_original['Benign'].value_counts().to_dict()}")

# Get feature columns (excluding FileName, md5Hash, Benign)
feature_cols = [col for col in df_original.columns if col not in ['FileName', 'md5Hash', 'Benign']]

print(f"\nFeature columns: {feature_cols}")
print(f"\nFeature statistics:")
print(df_original[feature_cols].describe())

# Create test sample with balanced classes
# Sample 100 benign and 100 ransomware samples
benign_samples = df_original[df_original['Benign'] == 1].sample(n=min(100, len(df_original[df_original['Benign'] == 1])), random_state=42)
ransomware_samples = df_original[df_original['Benign'] == 0].sample(n=min(100, len(df_original[df_original['Benign'] == 0])), random_state=42)

# Combine samples
test_df = pd.concat([benign_samples, ransomware_samples], ignore_index=True)

# Shuffle the dataset
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTest dataset created: {len(test_df)} rows")
print(f"Test dataset Benign distribution: {test_df['Benign'].value_counts().to_dict()}")

# Save to new CSV file
output_file = os.path.join(script_dir, 'test_data_file.csv')
test_df.to_csv(output_file, index=False)

print(f"\nTest dataset saved to: {output_file}")
print(f"Columns: {list(test_df.columns)}")
print(f"\nFirst 5 rows preview:")
print(test_df.head())

print("\n[SUCCESS] Test dataset created successfully!")
print("You can now upload this file via the 'Upload CSV' section in the web interface.")

