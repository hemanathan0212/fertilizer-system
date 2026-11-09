"""
================================================================================
COURSE: CAF7104 - Mini Project
PROJECT: AI Agent-Based Fertilizer Recommendation System
MODULE 1: Data Preprocessing (ml/module1_preprocessing.py)
================================================================================
Purpose:
  1. Load raw dataset (fertilizer_dataset.csv).
  2. Perform Exploratory Data Analysis (EDA) sanity checks: shapes and missing counts.
  3. Impute missing numeric values using column mean.
  4. Label-encode categorical features (Soil_Type, Crop_Type) and target (Fertilizer_Name).
  5. Save trained encoders to 'ml/saved_models/' using joblib.
  6. Perform Stratified 80/20 Train-Test split.
  7. Export processed splits (X_train, X_test, y_train, y_test) as CSV files.
================================================================================
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def run_preprocessing(
    input_csv_path="data/fertilizer_dataset.csv",
    models_dir="ml/saved_models",
    processed_dir="data"
):
    """
    Executes the complete preprocessing pipeline.
    
    Parameters:
      input_csv_path (str): Path to input raw CSV.
      models_dir (str): Directory where encoders are saved.
      processed_dir (str): Directory where train/test splits are saved.
      
    Returns:
      tuple: (X_train, X_test, y_train, y_test, encoders_dict)
    """
    print("\n" + "="*70)
    print("MODULE 1: DATA PREPROCESSING PIPELINE INITIATED")
    print("="*70)

    # Ensure output directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # Step 1: Load Raw Dataset
    # --------------------------------------------------------------------------
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Dataset not found at {input_csv_path}. Please generate it first.")

    df = pd.read_csv(input_csv_path)
    print(f"\n[Step 1] Loaded dataset from '{input_csv_path}'.")
    print(f"         Total records: {df.shape[0]}, Total attributes: {df.shape[1]}")

    # --------------------------------------------------------------------------
    # Step 2: Missing Values Inspection (Before Imputation)
    # --------------------------------------------------------------------------
    print("\n[Step 2] Missing Values Before Cleaning:")
    missing_before = df.isnull().sum()
    print(missing_before[missing_before > 0])
    if missing_before.sum() == 0:
        print("         No missing values detected.")

    # --------------------------------------------------------------------------
    # Step 3: Mean Imputation for Numeric Features
    # --------------------------------------------------------------------------
    numeric_columns = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    imputation_means = {}

    print("\n[Step 3] Imputing missing numeric values using column mean...")
    for col in numeric_columns:
        col_mean = df[col].mean()
        imputation_means[col] = float(col_mean)
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(col_mean)
            print(f"         - Imputed {missing_count} missing values in '{col}' with mean: {col_mean:.2f}")

    # Save imputation means for inference time consistency
    joblib.dump(imputation_means, os.path.join(models_dir, "imputation_means.pkl"))
    print(f"         Saved imputation statistics to '{models_dir}/imputation_means.pkl'")

    # Sanity check after imputation
    missing_after = df.isnull().sum().sum()
    print(f"\n[Step 3 Check] Total missing values after cleaning: {missing_after}")

    # --------------------------------------------------------------------------
    # Step 4: Categorical & Target Encoding
    # --------------------------------------------------------------------------
    print("\n[Step 4] Label Encoding Categorical Features & Target...")
    encoders = {}

    # Encode Soil_Type
    soil_encoder = LabelEncoder()
    df['Soil_Type'] = soil_encoder.fit_transform(df['Soil_Type'])
    encoders['Soil_Type'] = soil_encoder
    joblib.dump(soil_encoder, os.path.join(models_dir, "soil_encoder.pkl"))
    print(f"         - Soil_Type classes encoded: {list(soil_encoder.classes_)}")

    # Encode Crop_Type
    crop_encoder = LabelEncoder()
    df['Crop_Type'] = crop_encoder.fit_transform(df['Crop_Type'])
    encoders['Crop_Type'] = crop_encoder
    joblib.dump(crop_encoder, os.path.join(models_dir, "crop_encoder.pkl"))
    print(f"         - Crop_Type classes encoded: {list(crop_encoder.classes_)}")

    # Encode Target: Fertilizer_Name
    fert_encoder = LabelEncoder()
    df['Fertilizer_Name'] = fert_encoder.fit_transform(df['Fertilizer_Name'])
    encoders['Fertilizer_Name'] = fert_encoder
    joblib.dump(fert_encoder, os.path.join(models_dir, "fertilizer_encoder.pkl"))
    print(f"         - Target Fertilizer_Name classes encoded: {list(fert_encoder.classes_)}")

    # Save master dictionary with all encoders
    joblib.dump(encoders, os.path.join(models_dir, "encoders.pkl"))

    # Feature columns specification
    feature_cols = [
        'Temperature', 'Humidity', 'Moisture', 'Soil_Type',
        'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous'
    ]
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_columns.pkl"))

    # --------------------------------------------------------------------------
    # Step 5: Train-Test Split (80/20 Stratified)
    # --------------------------------------------------------------------------
    print("\n[Step 5] Performing 80/20 Stratified Train-Test Split...")
    X = df[feature_cols]
    y = df['Fertilizer_Name']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"         - X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")
    print(f"         - X_test Shape:  {X_test.shape}, y_test Shape:  {y_test.shape}")

    # --------------------------------------------------------------------------
    # Step 6: Export Processed Splits as CSVs
    # --------------------------------------------------------------------------
    print("\n[Step 6] Saving train and test splits as CSV files...")
    x_train_path = os.path.join(processed_dir, "X_train.csv")
    x_test_path = os.path.join(processed_dir, "X_test.csv")
    y_train_path = os.path.join(processed_dir, "y_train.csv")
    y_test_path = os.path.join(processed_dir, "y_test.csv")

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    print(f"         - Saved: {x_train_path}")
    print(f"         - Saved: {x_test_path}")
    print(f"         - Saved: {y_train_path}")
    print(f"         - Saved: {y_test_path}")

    print("\n" + "="*70)
    print("MODULE 1: DATA PREPROCESSING COMPLETED SUCCESSFULLY")
    print("="*70)

    return X_train, X_test, y_train, y_test, encoders


if __name__ == "__main__":
    run_preprocessing()
