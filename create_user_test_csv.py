#!/usr/bin/env python3
"""
Create a short test CSV file for System User testing
Small dataset for quick testing
"""

import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, 'data_file.csv')

print("Loading original dataset...")
df_original = pd.read_csv(data_file_path)

print(f"Original dataset: {len(df_original)} rows")

# Create a small test sample with mix of benign and ransomware
# Sample 5 benign and 5 ransomware samples
benign_samples = df_original[df_original['Benign'] == 1].sample(n=5, random_state=42)
ransomware_samples = df_original[df_original['Benign'] == 0].sample(n=5, random_state=42)

# Combine and shuffle
test_df = pd.concat([benign_samples, ransomware_samples], ignore_index=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTest dataset created: {len(test_df)} rows")
print(f"Benign: {(test_df['Benign']==1).sum()}, Ransomware: {(test_df['Benign']==0).sum()}")

# Save to new CSV file
output_file = os.path.join(script_dir, 'user_test_file.csv')
test_df.to_csv(output_file, index=False)

print(f"\n[SUCCESS] User test file created: {output_file}")
print(f"File contains {len(test_df)} samples - perfect for quick testing!")

