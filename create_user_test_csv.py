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

df_benign = df_original[df_original['Benign'] == 1]
df_ransom = df_original[df_original['Benign'] == 0]

configs = [
    {
        'filename': 'user_test_file.csv',
        'benign_n': 5,
        'ransom_n': 5,
        'prefix': 'SampleShort',
        'shuffle_seed': 42,
        'sample_seed': 42
    },
    {
        'filename': 'user_test_file_large.csv',
        'benign_n': 25,
        'ransom_n': 25,
        'prefix': 'SampleLarge',
        'shuffle_seed': 99,
        'sample_seed': 84
    }
]

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

    sanitized = combined.copy()
    sanitized['FileName'] = [f"{cfg['prefix']}_{i+1:04d}" for i in range(len(sanitized))]

    output_df = sanitized.drop(columns=['Benign'])
    output_path = os.path.join(script_dir, cfg['filename'])
    output_df.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] Created {cfg['filename']}")
    print(f" - Rows: {len(output_df)} (Benign source rows: {benign_count}, Ransomware source rows: {ransomware_count})")
    print(f" - Saved to: {output_path}")

