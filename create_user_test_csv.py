#!/usr/bin/env python3
"""
Create test CSV files for System User testing
Generates datasets with different sizes, all using Sample_0001 format
"""

import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, 'data_file.csv')
test_data_file_path = os.path.join(script_dir, 'test_data_file.csv')

print("Loading original dataset...")
df_original = pd.read_csv(data_file_path)

print(f"Original dataset: {len(df_original)} rows")

df_benign = df_original[df_original['Benign'] == 1]
df_ransom = df_original[df_original['Benign'] == 0]

# Configuration for 10 and 50 row datasets
configs = [
    {
        'filename': 'user_test_file_10.csv',
        'benign_n': 5,
        'ransom_n': 5,
        'shuffle_seed': 42,
        'sample_seed': 42
    },
    {
        'filename': 'user_test_file_50.csv',
        'benign_n': 25,
        'ransom_n': 25,
        'shuffle_seed': 99,
        'sample_seed': 84
    }
]

# Generate 10 and 50 row datasets
for cfg in configs:
    benign_samples = df_benign.sample(
        n=cfg['benign_n'],
        random_state=cfg['sample_seed']
    )
    ransomware_samples = df_ransom.sample(
        n=cfg['ransom_n'],
        random_state=cfg['sample_seed']
    )

    combined = pd.concat([benign_samples, ransomware_samples], ignore_index=True)
    combined = combined.sample(frac=1, random_state=cfg['shuffle_seed']).reset_index(drop=True)

    benign_count = int((combined['Benign'] == 1).sum())
    ransomware_count = int((combined['Benign'] == 0).sum())

    # Use Sample_0001 format (no prefix)
    sanitized = combined.copy()
    sanitized['FileName'] = [f"Sample_{i+1:04d}" for i in range(len(sanitized))]

    output_df = sanitized.drop(columns=['Benign'])
    output_path = os.path.join(script_dir, cfg['filename'])
    output_df.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] Created {cfg['filename']}")
    print(f" - Rows: {len(output_df)} (Benign source rows: {benign_count}, Ransomware source rows: {ransomware_count})")
    print(f" - Saved to: {output_path}")

# Create 200 row dataset from test_data_file.csv
if os.path.exists(test_data_file_path):
    print("\n" + "="*60)
    print("Creating user_test_file_200.csv from test_data_file.csv")
    print("="*60)
    
    df_test = pd.read_csv(test_data_file_path)
    
    # Rename FileName to Sample_0001 format (remove _BENIGN/_RANSOM suffix)
    df_test['FileName'] = [f"Sample_{i+1:04d}" for i in range(len(df_test))]
    
    # Remove Benign column
    output_df_200 = df_test.drop(columns=['Benign'])
    output_path_200 = os.path.join(script_dir, 'user_test_file_200.csv')
    output_df_200.to_csv(output_path_200, index=False)
    
    print(f"\n[SUCCESS] Created user_test_file_200.csv")
    print(f" - Rows: {len(output_df_200)}")
    print(f" - Saved to: {output_path_200}")
else:
    print(f"\n[WARNING] test_data_file.csv not found. Skipping user_test_file_200.csv creation.")

print("\n" + "="*60)
print("All user test files created successfully!")
print("="*60)
