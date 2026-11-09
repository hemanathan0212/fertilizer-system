"""
================================================================================
COURSE: CAF7104 - Mini Project
PROJECT: AI Agent-Based Fertilizer Recommendation System
MODULE 2: ML Prediction Engine (ml/module2_model_training.py)
================================================================================
Purpose:
  1. Load processed training and testing splits (X_train, X_test, y_train, y_test).
  2. Train three machine learning classification algorithms:
       - Random Forest Classifier (Primary Ensemble Model)
       - Decision Tree Classifier (Baseline Tree Model)
       - Support Vector Classifier - SVM (Kernel-based Baseline)
  3. Evaluate each model using:
       - Accuracy
       - Precision (Weighted & Macro)
       - Recall (Weighted & Macro)
       - F1-Score (Weighted & Macro)
       - Confusion Matrix
  4. Print comprehensive comparison table.
  5. Select the best performing model (Random Forest) and save as 'fertilizer_model.pkl'.
================================================================================
"""

import os
import json
import warnings
import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def train_and_evaluate_models(
    data_dir="data",
    models_dir="ml/saved_models"
):
    """
    Trains RF, DT, and SVM models, evaluates their performance,
    prints comparison reports, and saves the best model.
    """
    print("\n" + "="*75)
    print("MODULE 2: MACHINE LEARNING MODEL TRAINING & COMPARISON")
    print("="*75)

    # --------------------------------------------------------------------------
    # Step 1: Load Processed Datasets
    # --------------------------------------------------------------------------
    x_train_path = os.path.join(data_dir, "X_train.csv")
    x_test_path = os.path.join(data_dir, "X_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")

    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing split file: {p}. Run 'ml/module1_preprocessing.py' first."
            )

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Load target classes for readable reports
    fert_encoder_path = os.path.join(models_dir, "fertilizer_encoder.pkl")
    if os.path.exists(fert_encoder_path):
        fert_encoder = joblib.load(fert_encoder_path)
        class_names = [str(c) for c in fert_encoder.classes_]
    else:
        class_names = [str(c) for c in np.unique(y_train)]

    print(f"\n[Step 1] Loaded preprocessed data successfully:")
    print(f"         - Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"         - Testing set:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"         - Classes ({len(class_names)}): {class_names}")

    # --------------------------------------------------------------------------
    # Step 2: Define Models to Compare
    # --------------------------------------------------------------------------
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),
        "Support Vector Machine (SVM)": SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=42
        )
    }

    results = {}
    best_model_name = None
    best_f1_score = -1.0
    best_model_instance = None

    # --------------------------------------------------------------------------
    # Step 3: Train and Evaluate Each Model
    # --------------------------------------------------------------------------
    print("\n[Step 2 & 3] Training & Evaluating Models...")

    for name, model in models.items():
        print(f"\n---> Training: {name}...")
        model.fit(X_train, y_train)

        # Predictions on Test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

        cm = confusion_matrix(y_test, y_pred)
        clf_rep = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

        results[name] = {
            "model_instance": model,
            "accuracy": acc,
            "precision_weighted": prec_weighted,
            "recall_weighted": rec_weighted,
            "f1_weighted": f1_weighted,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "confusion_matrix": cm.tolist(),
            "classification_report": clf_rep
        }

        print(f"     Accuracy:  {acc*100:.2f}%")
        print(f"     Precision: {prec_weighted*100:.2f}% (Weighted)")
        print(f"     Recall:    {rec_weighted*100:.2f}% (Weighted)")
        print(f"     F1-Score:  {f1_weighted*100:.2f}% (Weighted)")

        # Track best model by F1-Score (robust against minor class imbalances)
        if f1_weighted > best_f1_score:
            best_f1_score = f1_weighted
            best_model_name = name
            best_model_instance = model

    # --------------------------------------------------------------------------
    # Step 4: Print Comparative Performance Table
    # --------------------------------------------------------------------------
    print("\n" + "="*85)
    print("MODEL COMPARISON TABLE (EVALUATED ON INDEPENDENT TEST SET - 133 SAMPLES)")
    print("="*85)
    header = f"{'Model Name':<30} | {'Accuracy (%)':<14} | {'Precision (%)':<14} | {'Recall (%)':<14} | {'F1-Score (%)':<14}"
    print(header)
    print("-" * 85)

    summary_metrics = {}
    for name, m in results.items():
        row = f"{name:<30} | {m['accuracy']*100:<14.2f} | {m['precision_weighted']*100:<14.2f} | {m['recall_weighted']*100:<14.2f} | {m['f1_weighted']*100:<14.2f}"
        print(row)
        summary_metrics[name] = {
            "accuracy": round(m['accuracy'] * 100, 2),
            "precision": round(m['precision_weighted'] * 100, 2),
            "recall": round(m['recall_weighted'] * 100, 2),
            "f1_score": round(m['f1_weighted'] * 100, 2)
        }
    print("="*85)

    print(f"\n[Detailed Report for Winner: {best_model_name}]")
    print(results[best_model_name]["classification_report"])

    print(f"[Confusion Matrix for {best_model_name}]:")
    print(np.array(results[best_model_name]["confusion_matrix"]))

    # --------------------------------------------------------------------------
    # Step 5: Save the Best Model
    # --------------------------------------------------------------------------
    save_path = os.path.join(models_dir, "fertilizer_model.pkl")
    joblib.dump(best_model_instance, save_path)
    print(f"\n[Step 5] Best Model '{best_model_name}' saved to: {save_path}")

    # Also save metadata and evaluation summary for easy reference
    meta_path = os.path.join(models_dir, "model_comparison_metrics.json")
    with open(meta_path, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "metrics": summary_metrics,
            "target_classes": class_names
        }, f, indent=4)
    print(f"         Evaluation metrics exported to: {meta_path}")

    print("\n" + "="*75)
    print("MODULE 2: MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("="*75)

    return best_model_name, results


if __name__ == "__main__":
    train_and_evaluate_models()
