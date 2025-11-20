# Test Dataset Guide

## Overview
A test sample dataset (`test_data_file.csv`) has been created from the original `data_file.csv` for testing the upload and training functionality.

## Test Dataset Details

| File | Rows | Purpose | Notes |
|------|------|---------|-------|
| `test_data_file.csv` | 200 | Primary balanced dataset used for training demos | Includes the `Benign` label column |
| `user_test_file.csv` | 10 | Ultra-light test file for System User uploads | Balanced 5 benign / 5 ransomware, `Benign` column removed |

All four CSVs share the same feature columns as the original Kaggle dataset. The `FileName` values were sanitized to neutral identifiers such as `Sample_0007_RANSOM` so nothing in the UI exposes vendor-specific names like “VirusShare”.

## How to Use

### Step 1: Start the Application
```bash
python app.py
```

### Step 2: Login
- Use an account with `train_ml_model` permission:
  - **Cybersecurity Professional**: `cyber_pro@example.com` / `cyber123`
  - **IT Administrator**: `admin@example.com` / `admin123`
  - **Academic Researcher**: `researcher@example.com` / `research123`

### Step 3: Upload Test Dataset (Training Flow)
1. Navigate to **"Upload CSV"** section in the navbar
2. Click **"Choose File"** and select `test_data_file.csv`
3. Click **"Upload & Analyze"** button
4. The system will:
   - Validate the CSV format
   - Process and load the dataset
   - Display statistics and preview
   - Show message: "Ready for training!"

### Step 4: Train Model
## System User Testing (Prediction-Only Flow)

If you simply want to demonstrate the System User interface (upload → detect ransomware, no training required), use the smaller CSVs:

1. Log in as the **System User** account (`user@example.com` / `user123`).
2. In the simplified interface, upload either:
   - `user_test_file.csv` (quick 10-row file), or
   - `user_dataset_without_labels.csv` (20 rows).
3. The system:
   - Automatically runs batch prediction on all rows.
   - Displays total counts, analysis time, and the exact sample names that were flagged.
   - Stores the upload in the history table so users can trace previous runs.

Because `user_test_file.csv` is unlabeled, it mirrors the exact experience a production System User has—no exposure to the `Benign` column or any hint about which rows are malicious. Admin roles should continue using `test_data_file.csv` (or another labeled dataset) when training or retraining.

## Dataset Structure
1. Navigate to **"Train Model"** section
2. Select a model type:
   - **Random Forest** (~30-60s) - Recommended for quick testing
   - **SVM** (~2-5 min)
   - **Neural Networks** (~1-3 min)
   - **CNN-LSTM** (~3-8 min)
3. Click **"Train model"** button
4. The model will train on the uploaded test dataset
5. View training performance metrics and visualizations

## Dataset Structure

The test dataset contains the same columns as the original:

```
FileName, md5Hash, Machine, DebugSize, DebugRVA, MajorImageVersion, 
MajorOSVersion, ExportRVA, ExportSize, IatVRA, MajorLinkerVersion, 
MinorLinkerVersion, NumberOfSections, SizeOfStackReserve, 
DllCharacteristics, ResourceSize, BitcoinAddresses, Benign
```

**Note**: `FileName` and `md5Hash` are dropped during processing (identifiers only).

## Creating Your Own Test Dataset

To create a new test dataset, run:

```bash
python create_test_dataset.py
```

This script will:
- Load the original `data_file.csv`
- Sample 100 benign and 100 ransomware samples
- Create a balanced test dataset
- Save to `test_data_file.csv`

You can modify the script to:
- Change the number of samples
- Adjust the class distribution
- Add custom filtering criteria

## Technical Details

### Code Changes Made

1. **Added `load_dataset_from_dataframe()` method** in `RansomwareDetector` class
   - Allows loading dataset from uploaded CSV DataFrame
   
2. **Added `_process_dataset()` method** 
   - Centralized dataset processing logic
   - Used by both `load_kaggle_dataset()` and `load_dataset_from_dataframe()`

3. **Updated `/api/upload-csv` endpoint**
   - Now loads uploaded CSV into detector
   - Validates required columns
   - Returns training-ready status
   - Shows processed statistics

### Workflow

```
Upload CSV → Validate → Process → Load into Detector → Ready for Training
```

When you upload a CSV:
1. System validates it has `Benign` column
2. Processes and cleans the data
3. Loads it into `detector.training_data`
4. When you click "Train Model", it uses the uploaded dataset

## Verification

After uploading and training, you can verify:
- **Dataset Stats**: Check dashboard shows correct sample count (200)
- **Model Performance**: Training metrics reflect the test dataset size
- **Predictions**: Can make predictions using the trained model

## Troubleshooting

**Issue**: "Dataset must contain 'Benign' column"
- **Solution**: Ensure your CSV has a `Benign` column with values 0 or 1

**Issue**: "Failed to process dataset"
- **Solution**: Check that all feature columns are numeric and contain valid values

**Issue**: Training fails after upload
- **Solution**: Ensure dataset has at least 20 samples for train/test split

## Files Created

- `test_data_file.csv` - Main balanced dataset (200 samples, labeled)
- `user_test_file.csv` - Mini dataset for demos (10 samples, unlabeled)
- `create_test_dataset.py` - Script to generate large balanced dataset
- `create_user_test_csv.py` - Script to regenerate the unlabeled mini dataset
- `TEST_DATASET_README.md` - This documentation

