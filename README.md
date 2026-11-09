# AI Agent-Based Fertilizer Recommendation System using LangChain

> **Course:** MCA Mini Project (Course Code: CAF7104)  
> **Student Implementation:** Fully functional, end-to-end, reproducible agronomic recommendation platform.

---

## 📌 Project Overview

Modern precision agriculture requires not only accurate soil nutrient classification but also transparent, actionable advice that farmers and agronomists can understand.

This project implements an **AI Agent-Based Fertilizer Recommendation System** combining:
1. **Machine Learning Classifier:** A trained **Random Forest** model (benchmarked against **Decision Tree** and **SVM**) that predicts the optimal fertilizer formulation from 8 environmental and soil inputs.
2. **Relational Knowledge Base:** An SQLite database storing application methods, per-acre dosages, and agronomic crop-adjustment notes across 11 crop types and 7 fertilizer formulations.
3. **LangChain AI Agent:** An autonomous reasoning agent coordinating three tools (`predict_fertilizer`, `lookup_fertilizer_info`, and `adjust_for_crop`) to synthesize structured, natural language explanations.
4. **FastAPI REST Backend:** High-performance RESTful API endpoints (`/predict`, `/recommend`, `/options`, `/health`).
5. **Interactive Frontend:** Responsive, clean HTML5/CSS3 interface with one-click faculty viva presets and a dedicated report view.

---

## 🏗️ System Architecture

```
+-------------------------------------------------------------------------+
|                              Frontend UI                                |
|        (frontend/index.html & frontend/result.html with style.css)       |
+-------------------------------------------------------------------------+
                                    │  HTTP POST /recommend
                                    ▼
+-------------------------------------------------------------------------+
|                         FastAPI REST Backend                            |
|                          (backend/main.py)                              |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
|                        LangChain AI Agent Layer                         |
|                           (agent/agent.py)                              |
|                                                                         |
|   ┌─────────────────────────────────────────────────────────────────┐   |
|   │ LangChain Tool 1: predict_fertilizer()                          │   |
|   │ ──► Invokes Trained Random Forest Model (ml/saved_models/)       │   |
|   ├─────────────────────────────────────────────────────────────────┤   |
|   │ LangChain Tool 2: lookup_fertilizer_info()                      │   |
|   │ ──► Queries fertilizers table in SQLite (knowledge_base/)       │   |
|   ├─────────────────────────────────────────────────────────────────┤   |
|   │ LangChain Tool 3: adjust_for_crop()                             │   |
|   │ ──► Queries crop_adjustments table in SQLite (knowledge_base/)  │   |
|   └─────────────────────────────────────────────────────────────────┘   |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
|                  Natural Language Agronomic Synthesizer                 |
|       (Generates diagnosis, rationale, dosage & crop guidelines)        |
+-------------------------------------------------------------------------+
```

---

## 📂 Project Structure

```
Fertlizer_system/
├── data/
│   ├── generate_dataset.py           # Synthetic dataset generator with agronomic rules
│   ├── fertilizer_dataset.csv        # 665 records, 8 features, 7 target classes
│   ├── X_train.csv                   # 80% Stratified training features (532 rows)
│   ├── X_test.csv                    # 20% Stratified testing features (133 rows)
│   ├── y_train.csv                   # Training target labels
│   └── y_test.csv                    # Testing target labels
├── ml/
│   ├── module1_preprocessing.py      # Mean imputation, label encoding, 80/20 stratified split
│   ├── module2_model_training.py     # RF vs DT vs SVM comparison & saving best model
│   └── saved_models/
│       ├── fertilizer_model.pkl      # Best trained Random Forest model
│       ├── encoders.pkl              # Master dictionary of trained label encoders
│       ├── soil_encoder.pkl          # Soil_Type LabelEncoder
│       ├── crop_encoder.pkl          # Crop_Type LabelEncoder
│       ├── fertilizer_encoder.pkl    # Target Fertilizer_Name LabelEncoder
│       ├── feature_columns.pkl       # Feature schema order
│       └── model_comparison_metrics.json # Precision, Recall, F1 comparison report
├── agent/
│   ├── tools.py                      # 3 LangChain Tools (predict, lookup, crop_adjust)
│   └── agent.py                      # LangChain Agent orchestrator + explanation generator
├── knowledge_base/
│   ├── schema.sql                    # SQLite tables: fertilizers & crop_adjustments
│   ├── seed_data.py                  # DDL execution & population of 7 ferts & 77 crop notes
│   └── fertilizer_kb.db              # SQLite relational database
├── backend/
│   └── main.py                       # FastAPI REST API (/predict, /recommend, CORS, UI mount)
├── frontend/
│   ├── index.html                    # Input form with live faculty demo presets
│   ├── style.css                     # Modern agricultural design system (Vanilla CSS)
│   └── result.html                   # Detailed recommendation report view
├── tests/
│   └── test_pipeline.py              # Automated 5-step end-to-end sanity test suite
├── requirements.txt                  # Pinned Python package dependencies
└── README.md                         # Complete project documentation & viva guide
```

---

## 📊 Dataset Schema

The system analyzes 8 predictive features to determine the target fertilizer:

| Feature Name | Type | Description / Range |
|---|---|---|
| `Temperature` | Numeric | Ambient air temperature (20.0°C – 38.0°C) |
| `Humidity` | Numeric | Relative air humidity percentage (45% – 85%) |
| `Moisture` | Numeric | Soil moisture percentage (25% – 65%) |
| `Soil_Type` | Categorical | Sandy, Loamy, Black, Red, Clayey (5 types) |
| `Crop_Type` | Categorical | Maize, Sugarcane, Cotton, Tobacco, Paddy, Barley, Wheat, Millets, Oil seeds, Pulses, Ground Nuts (11 types) |
| `Nitrogen` | Numeric | Available soil Nitrogen level (ppm / ratio) |
| `Potassium` | Numeric | Available soil Potash level (ppm / ratio) |
| `Phosphorous` | Numeric | Available soil Phosphorus level (ppm / ratio) |
| **`Fertilizer_Name`** | **Target** | **Urea, DAP, 14-35-14, 28-28, 17-17-17, 20-20, 10-26-26 (7 classes)** |

---

## 🚀 Step-by-Step Setup & Execution Guide

### Step 1: Create and Activate Virtual Environment
Open PowerShell or Command Prompt in the project folder:
```powershell
# Navigate to project directory
cd d:\Fertlizer_system

# Create virtual environment (Python 3.10+)
python -m venv venv

# Activate virtual environment on Windows
.\venv\Scripts\activate
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Module 1 — Data Preprocessing
Loads the dataset, inspects missing values, imputes numeric missing values with column means, encodes categorical features, and performs an 80/20 stratified split:
```powershell
python ml/module1_preprocessing.py
```
*Outputs: `data/X_train.csv`, `data/X_test.csv`, `ml/saved_models/encoders.pkl`.*

### Step 4: Run Module 2 — Model Training & Comparison
Trains Random Forest, Decision Tree, and SVM. Evaluates accuracy, precision, recall, and F1-score, prints a comparison table, and saves the winner:
```powershell
python ml/module2_model_training.py
```
*Outputs: `ml/saved_models/fertilizer_model.pkl`.*

### Step 5: Run Module 3 — Seed SQLite Knowledge Base
Creates the SQLite database and seeds 7 master fertilizer classes and 77 crop adjustment notes:
```powershell
python knowledge_base/seed_data.py
```
*Outputs: `knowledge_base/fertilizer_kb.db`.*

### Step 6: Run Module 7 — End-to-End Sanity Tests
Validates all 5 system layers:
```powershell
python tests/test_pipeline.py
```
*Expected: `[SUCCESS] ALL 5 SANITY TEST SUITES PASSED PERFECTLY (100% SUCCESS)`.*

### Step 7: Launch FastAPI Backend
```powershell
python backend/main.py
```
*Backend API starts at: `http://127.0.0.1:8000`*  
*Interactive Swagger Documentation: `http://127.0.0.1:8000/docs`*

### Step 8: Open the Frontend
You can interact with the system in either of two ways:
1. **Via Browser URL:** Navigate to `http://127.0.0.1:8000/ui/index.html` (hosted directly by FastAPI).
2. **Direct File Open:** Double-click `frontend/index.html` in Windows Explorer or open it in Google Chrome / Edge.

---

## 🎯 Viva & Faculty Demo Presets

The frontend includes **one-click demo presets** designed specifically for presenting during your viva:

| Preset Name | Crop | Soil | Key Deficiency | Expected Recommendation | Agronomic Rationale |
|---|---|---|---|---|---|
| **Sugarcane (Low N)** | Sugarcane | Loamy | N: 14, P: 35, K: 30 | **Urea** | High Nitrogen demand; split into 3 doses. |
| **Cotton (Low P)** | Cotton | Black | N: 28, P: 12, K: 26 | **DAP** | Phosphorus deficient; applied at sowing. |
| **Wheat (Balanced)** | Wheat | Loamy | N: 22, P: 22, K: 22 | **17-17-17** | Balanced multi-nutrient requirement. |
| **Groundnut (High K)** | Ground Nuts | Sandy | N: 35, P: 15, K: 12 | **10-26-26** | Potash critical for pod filling & drought resistance. |

---

## 💡 Viva Presentation Talking Points (For Faculty Panel)

1. **Why combine Machine Learning with a LangChain Agent?**
   - Pure ML classifiers only output a categorical string (e.g. `"Urea"`). A farmer cannot safely act on a label alone without knowing **dosage per acre**, **application timing**, **placement method**, and **split-dose management**.
   - The LangChain agent decouples classification from domain knowledge retrieval, translating raw model predictions into actionable agronomic guidance.

2. **Why Random Forest won the benchmark?**
   - Soil and environmental features exhibit non-linear interactions (e.g., Nitrogen utilization depends on soil moisture and temperature).
   - Random Forest ensembles 100 de-correlated decision trees using bagging (bootstrap aggregation), significantly reducing overfitting and variance compared to a single Decision Tree.

3. **How does the LangChain Agent coordinate tools?**
   - Tool 1 (`predict_fertilizer`): Executes ML inference using `fertilizer_model.pkl`.
   - Tool 2 (`lookup_fertilizer_info`): Queries the SQLite `fertilizers` table for dosage and NPK grade.
   - Tool 3 (`adjust_for_crop`): Queries `crop_adjustments` for crop-specific growth-stage scheduling.
   - The orchestrator synthesizes a complete report combining soil diagnostic, rationale, application protocol, and crop management.

4. **Does the system work offline?**
   - **Yes, 100% offline.** The system includes an intelligent Agronomic Synthesis Engine that generates detailed natural language explanations without requiring a paid OpenAI API key or internet access. If an `OPENAI_API_KEY` is provided, it seamlessly switches to live LLM generation.

---

## 📄 License & Attribution
Developed for Course **CAF7104** (MCA Mini Project). Built with Python, scikit-learn, LangChain, FastAPI, SQLite, and HTML/CSS.
