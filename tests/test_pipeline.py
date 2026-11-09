"""
================================================================================
COURSE: CAF7104 - Mini Project
PROJECT: AI Agent-Based Fertilizer Recommendation System
MODULE 7: Automated End-to-End Pipeline Sanity Tests (tests/test_pipeline.py)
================================================================================
Test Suite covers:
  1. Module 1: Preprocessing outputs exist and have correct shapes/dimensions.
  2. Module 2: Model artifacts exist, load cleanly, and execute inference without error.
  3. Module 3: SQLite knowledge base tables and records are properly seeded.
  4. Module 4: LangChain tools and AI Agent orchestrate and produce non-empty explanations.
  5. Module 5: FastAPI REST endpoints (/predict and /recommend) validate inputs and return 200 OK.
================================================================================
"""

import os
import sys
import json
import sqlite3
import joblib
import pandas as pd
from fastapi.testclient import TestClient

# Ensure root workspace is on python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.tools import predict_fertilizer, lookup_fertilizer_info, adjust_for_crop, run_raw_prediction
from agent.agent import get_agent_instance
from backend.main import app


def test_module1_outputs():
    """Verify that Module 1 created valid data splits and encoders."""
    print("\n[TEST 1] Checking Module 1 Preprocessing Outputs...")

    files = [
        "data/fertilizer_dataset.csv",
        "data/X_train.csv",
        "data/X_test.csv",
        "data/y_train.csv",
        "data/y_test.csv",
        "ml/saved_models/soil_encoder.pkl",
        "ml/saved_models/crop_encoder.pkl",
        "ml/saved_models/fertilizer_encoder.pkl",
        "ml/saved_models/encoders.pkl"
    ]

    for rel_path in files:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        assert os.path.exists(full_path), f"Missing file: {rel_path}"

    X_train = pd.read_csv(os.path.join(PROJECT_ROOT, "data/X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROJECT_ROOT, "data/X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROJECT_ROOT, "data/y_train.csv"))
    y_test = pd.read_csv(os.path.join(PROJECT_ROOT, "data/y_test.csv"))

    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train row count mismatch"
    assert X_test.shape[0] == y_test.shape[0], "X_test and y_test row count mismatch"
    assert X_train.shape[1] == 8, f"Expected 8 feature columns, found {X_train.shape[1]}"
    assert X_test.shape[1] == 8, f"Expected 8 feature columns, found {X_test.shape[1]}"

    print(f"  --> PASSED: Train split {X_train.shape}, Test split {X_test.shape}")


def test_module2_model():
    """Verify that trained model exists, loads, and executes predictions."""
    print("\n[TEST 2] Checking Module 2 Trained Model...")

    model_path = os.path.join(PROJECT_ROOT, "ml/saved_models/fertilizer_model.pkl")
    assert os.path.exists(model_path), "Model file missing: fertilizer_model.pkl"

    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Loaded object does not implement .predict()"

    # Test raw prediction helper
    sample = {
        "temperature": 28.0,
        "humidity": 65.0,
        "moisture": 45.0,
        "soil_type": "Loamy",
        "crop_type": "Wheat",
        "nitrogen": 22.0,
        "potassium": 25.0,
        "phosphorous": 28.0
    }
    pred_res = run_raw_prediction(sample)
    assert "fertilizer_name" in pred_res, "Prediction result missing fertilizer_name"
    assert "confidence" in pred_res, "Prediction result missing confidence"
    assert isinstance(pred_res["confidence"], (int, float)), "Confidence must be a numeric value"

    print(f"  --> PASSED: Model predicted '{pred_res['fertilizer_name']}' with {pred_res['confidence']}% confidence")


def test_module3_database():
    """Verify that SQLite Knowledge Base exists and contains required records."""
    print("\n[TEST 3] Checking Module 3 SQLite Knowledge Base...")

    db_path = os.path.join(PROJECT_ROOT, "knowledge_base/fertilizer_kb.db")
    assert os.path.exists(db_path), "Database file missing: fertilizer_kb.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM fertilizers")
    fert_count = cursor.fetchone()[0]
    assert fert_count == 7, f"Expected 7 fertilizer classes in DB, found {fert_count}"

    cursor.execute("SELECT COUNT(*) FROM crop_adjustments")
    adj_count = cursor.fetchone()[0]
    assert adj_count >= 70, f"Expected at least 70 crop adjustments, found {adj_count}"

    conn.close()
    print(f"  --> PASSED: Knowledge base verified with {fert_count} fertilizers and {adj_count} crop adjustments")


def test_module4_tools_and_agent():
    """Verify that LangChain tools and Agent generate a complete explanation."""
    print("\n[TEST 4] Checking Module 4 LangChain Tools & Agent Layer...")

    # Tool 1: predict_fertilizer
    p_out = predict_fertilizer.invoke({
        "temperature": 30.0,
        "humidity": 70.0,
        "moisture": 45.0,
        "soil_type": "Loamy",
        "crop_type": "Sugarcane",
        "nitrogen": 15.0,
        "potassium": 30.0,
        "phosphorous": 35.0
    })
    p_data = json.loads(p_out)
    assert "fertilizer_name" in p_data, "Tool 1 missing fertilizer_name"

    # Tool 2: lookup_fertilizer_info
    l_out = lookup_fertilizer_info.invoke({"fertilizer_name": "Urea"})
    l_data = json.loads(l_out)
    assert "dosage_per_acre" in l_data, "Tool 2 missing dosage_per_acre"

    # Tool 3: adjust_for_crop
    a_out = adjust_for_crop.invoke({"fertilizer_name": "Urea", "crop_type": "Sugarcane"})
    a_data = json.loads(a_out)
    assert "adjustment_notes" in a_data, "Tool 3 missing adjustment_notes"

    # Agent execution
    agent = get_agent_instance()
    agent_res = agent.run({
        "temperature": 30.0,
        "humidity": 70.0,
        "moisture": 45.0,
        "soil_type": "Loamy",
        "crop_type": "Sugarcane",
        "nitrogen": 15.0,
        "potassium": 30.0,
        "phosphorous": 35.0
    })

    assert agent_res.get("success") is True, "Agent execution failed"
    assert len(agent_res.get("explanation", "")) > 50, "Agent explanation is too short or empty"
    assert "Urea" in agent_res["fertilizer_name"], "Expected Urea recommendation for low Nitrogen"

    print(f"  --> PASSED: Agent produced {len(agent_res['explanation'])} char explanation for {agent_res['fertilizer_name']}")


def test_module5_fastapi_endpoints():
    """Verify that FastAPI REST API responds correctly."""
    print("\n[TEST 5] Checking Module 5 FastAPI Endpoints...")

    client = TestClient(app)

    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    sample_payload = {
        "temperature": 29.0,
        "humidity": 65.0,
        "moisture": 42.0,
        "soil_type": "Loamy",
        "crop_type": "Sugarcane",
        "nitrogen": 15.0,
        "potassium": 28.0,
        "phosphorous": 32.0
    }

    # /predict endpoint
    res_pred = client.post("/predict", json=sample_payload)
    assert res_pred.status_code == 200
    pred_data = res_pred.json()
    assert pred_data["success"] is True
    assert "fertilizer_name" in pred_data

    # /recommend endpoint
    res_rec = client.post("/recommend", json=sample_payload)
    assert res_rec.status_code == 200
    rec_data = res_rec.json()
    assert rec_data["success"] is True
    assert len(rec_data["explanation"]) > 50
    assert "dosage_per_acre" in rec_data

    print(f"  --> PASSED: Endpoints /health, /predict, and /recommend responded with HTTP 200 OK")


def run_all_tests():
    """Executes the full test suite."""
    print("="*75)
    print("RUNNING END-TO-END SANITY TEST SUITE (CAF7104 MINI PROJECT)")
    print("="*75)

    test_module1_outputs()
    test_module2_model()
    test_module3_database()
    test_module4_tools_and_agent()
    test_module5_fastapi_endpoints()

    print("\n" + "="*75)
    print("[SUCCESS] ALL 5 SANITY TEST SUITES PASSED PERFECTLY (100% SUCCESS)")
    print("="*75)


if __name__ == "__main__":
    run_all_tests()
