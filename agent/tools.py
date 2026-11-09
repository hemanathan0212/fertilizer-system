"""
================================================================================
COURSE: CAF7104 - Mini Project
PROJECT: AI Agent-Based Fertilizer Recommendation System
MODULE 4 (Part A): LangChain Tools (agent/tools.py)
================================================================================
Purpose:
  Defines three LangChain Tool objects:
    1. predict_fertilizer: Wraps the trained Random Forest model to predict
       fertilizer class from environmental and soil parameters.
    2. lookup_fertilizer_info: Queries SQLite knowledge base for standard
       dosage, application method, and NPK composition.
    3. adjust_for_crop: Queries SQLite knowledge base for crop-specific
       agronomic adjustment notes (split doses, timing, precautions).
================================================================================
"""

import os
import json
import sqlite3
import joblib
import pandas as pd
from typing import Dict, Any, Optional, Union
from langchain_core.tools import tool

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "saved_models")
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
DB_PATH = os.path.join(KB_DIR, "fertilizer_kb.db")
MODEL_PATH = os.path.join(MODELS_DIR, "fertilizer_model.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
COLS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
MEANS_PATH = os.path.join(MODELS_DIR, "imputation_means.pkl")

# Cached model resources
_MODEL = None
_ENCODERS = None
_FEATURE_COLS = None
_MEANS = None


def _load_ml_resources():
    """Lazy loader for ML model, encoders, and feature columns."""
    global _MODEL, _ENCODERS, _FEATURE_COLS, _MEANS
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. Run 'ml/module2_model_training.py' first."
            )
        _MODEL = joblib.load(MODEL_PATH)
        _ENCODERS = joblib.load(ENCODERS_PATH)
        _FEATURE_COLS = joblib.load(COLS_PATH)
        _MEANS = joblib.load(MEANS_PATH) if os.path.exists(MEANS_PATH) else {}
    return _MODEL, _ENCODERS, _FEATURE_COLS, _MEANS


def run_raw_prediction(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Direct Python helper to invoke the ML model with input dictionary.
    
    Accepts keys (case-insensitive):
      temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous
    """
    model, encoders, feature_cols, means = _load_ml_resources()

    # Normalize incoming parameter keys
    p = {k.lower().strip(): v for k, v in params.items()}

    temp = float(p.get("temperature", means.get("Temperature", 28.0)))
    humidity = float(p.get("humidity", means.get("Humidity", 60.0)))
    moisture = float(p.get("moisture", means.get("Moisture", 40.0)))
    nitrogen = float(p.get("nitrogen", means.get("Nitrogen", 25.0)))
    potassium = float(p.get("potassium", means.get("Potassium", 25.0)))
    phosphorous = float(p.get("phosphorous", means.get("Phosphorous", 20.0)))

    raw_soil = str(p.get("soil_type", "Loamy")).strip().title()
    raw_crop = str(p.get("crop_type", "Wheat")).strip().title()

    soil_enc = encoders["Soil_Type"]
    crop_enc = encoders["Crop_Type"]
    fert_enc = encoders["Fertilizer_Name"]

    # Safe encoding with fallback if unknown label
    if raw_soil in soil_enc.classes_:
        soil_code = soil_enc.transform([raw_soil])[0]
    else:
        soil_code = 0
        raw_soil = soil_enc.classes_[0]

    # Normalize crop name variations (e.g., 'Groundnut' -> 'Ground Nuts', 'Oilseeds' -> 'Oil seeds')
    crop_aliases = {
        "Groundnut": "Ground Nuts",
        "Groundnuts": "Ground Nuts",
        "Oilseed": "Oil seeds",
        "Oilseeds": "Oil seeds",
        "Paddy Rice": "Paddy",
        "Rice": "Paddy"
    }
    raw_crop = crop_aliases.get(raw_crop, raw_crop)

    if raw_crop in crop_enc.classes_:
        crop_code = crop_enc.transform([raw_crop])[0]
    else:
        crop_code = 0
        raw_crop = crop_enc.classes_[0]

    # Build input DataFrame
    input_dict = {
        "Temperature": temp,
        "Humidity": humidity,
        "Moisture": moisture,
        "Soil_Type": soil_code,
        "Crop_Type": crop_code,
        "Nitrogen": nitrogen,
        "Potassium": potassium,
        "Phosphorous": phosphorous
    }
    input_df = pd.DataFrame([input_dict])[feature_cols]

    # Predict class & confidence
    pred_idx = model.predict(input_df)[0]
    fert_name = str(fert_enc.inverse_transform([pred_idx])[0])

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(probabilities[pred_idx])

    return {
        "fertilizer_name": fert_name,
        "confidence": round(confidence * 100, 2),
        "soil_type": raw_soil,
        "crop_type": raw_crop,
        "temperature": temp,
        "humidity": humidity,
        "moisture": moisture,
        "nitrogen": nitrogen,
        "potassium": potassium,
        "phosphorous": phosphorous
    }


# ------------------------------------------------------------------------------
# LangChain Tool 1: predict_fertilizer
# ------------------------------------------------------------------------------
@tool
def predict_fertilizer(
    temperature: float = 28.0,
    humidity: float = 65.0,
    moisture: float = 40.0,
    soil_type: str = "Loamy",
    crop_type: str = "Wheat",
    nitrogen: float = 25.0,
    potassium: float = 25.0,
    phosphorous: float = 20.0
) -> str:
    """
    Predicts the most suitable fertilizer using the trained Machine Learning classifier.
    Inputs:
      - temperature: Environmental temperature in degrees Celsius (e.g. 28.5)
      - humidity: Relative humidity percentage (e.g. 65.0)
      - moisture: Soil moisture percentage (e.g. 42.0)
      - soil_type: Type of soil ('Sandy', 'Loamy', 'Black', 'Red', 'Clayey')
      - crop_type: Name of the crop (e.g. 'Wheat', 'Paddy', 'Sugarcane', 'Maize', etc.)
      - nitrogen: Soil Nitrogen ratio/value (e.g. 18.0)
      - potassium: Soil Potassium ratio/value (e.g. 24.0)
      - phosphorous: Soil Phosphorous ratio/value (e.g. 30.0)
    Returns:
      JSON string with predicted fertilizer name, confidence level, and input context.
    """
    params = {
        "temperature": temperature,
        "humidity": humidity,
        "moisture": moisture,
        "soil_type": soil_type,
        "crop_type": crop_type,
        "nitrogen": nitrogen,
        "potassium": potassium,
        "phosphorous": phosphorous
    }
    result = run_raw_prediction(params)
    return json.dumps(result, indent=2)


# ------------------------------------------------------------------------------
# LangChain Tool 2: lookup_fertilizer_info
# ------------------------------------------------------------------------------
@tool
def lookup_fertilizer_info(fertilizer_name: str) -> str:
    """
    Queries the SQLite fertilizer knowledge base for standard application guidelines.
    Input:
      - fertilizer_name: Name of the fertilizer (e.g. 'Urea', 'DAP', '14-35-14', etc.)
    Returns:
      JSON string with dosage_per_acre, application_method, and N-P-K nutrient composition.
    """
    fert_name = fertilizer_name.strip()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name, dosage_per_acre, application_method, n_content, p_content, k_content FROM fertilizers WHERE LOWER(name) = LOWER(?)",
        (fert_name,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        data = dict(row)
        return json.dumps({
            "status": "found",
            "fertilizer": data["name"],
            "dosage_per_acre": data["dosage_per_acre"],
            "application_method": data["application_method"],
            "npk_composition": f"N: {data['n_content']}%, P: {data['p_content']}%, K: {data['k_content']}%"
        }, indent=2)
    else:
        return json.dumps({
            "status": "generic",
            "fertilizer": fert_name,
            "dosage_per_acre": "45-50 kg/acre (General agronomic guideline)",
            "application_method": "Apply as basal or top dress with irrigation depending on crop growth stage.",
            "npk_composition": "Standard commercial compound grade"
        }, indent=2)


# ------------------------------------------------------------------------------
# LangChain Tool 3: adjust_for_crop
# ------------------------------------------------------------------------------
@tool
def adjust_for_crop(fertilizer_name: str, crop_type: str) -> str:
    """
    Queries crop-specific agronomic management adjustments and split-dosage tips.
    Inputs:
      - fertilizer_name: The recommended fertilizer name (e.g. 'Urea', 'DAP')
      - crop_type: The target crop type (e.g. 'Sugarcane', 'Cotton', 'Paddy', 'Wheat')
    Returns:
      JSON string with crop-specific adjustment notes, split dosing, and precautions.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT adjustment_notes FROM crop_adjustments WHERE LOWER(crop_type) = LOWER(?) AND LOWER(fertilizer_name) = LOWER(?)",
        (crop_type.strip(), fertilizer_name.strip())
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        notes = row[0]
    else:
        notes = f"For {crop_type}, apply {fertilizer_name} according to standard regional stage intervals with optimal soil moisture."

    return json.dumps({
        "crop_type": crop_type,
        "fertilizer_name": fertilizer_name,
        "adjustment_notes": notes
    }, indent=2)


# Expose list of tools for LangChain agents
ALL_TOOLS = [predict_fertilizer, lookup_fertilizer_info, adjust_for_crop]
