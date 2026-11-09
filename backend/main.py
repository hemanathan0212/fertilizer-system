"""
================================================================================
COURSE: CAF7104 - Mini Project
PROJECT: AI Agent-Based Fertilizer Recommendation System
MODULE 5: FastAPI REST Backend (backend/main.py)
================================================================================
Endpoints:
  - GET  /             : API metadata and health status
  - GET  /health       : Simple health check
  - GET  /options      : Returns available Soil Types, Crop Types, and Fertilizers
  - POST /predict      : Pure ML classifier prediction (returns fertilizer & confidence)
  - POST /recommend    : Full LangChain AI Agent workflow (prediction + KB lookup
                         + crop adjustment + natural language explanation)
================================================================================
"""

import os
import sys
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.tools import run_raw_prediction
from agent.agent import get_agent_instance

# Initialize FastAPI application
app = FastAPI(
    title="AI Agent-Based Fertilizer Recommendation System",
    description="MCA Mini Project (CAF7104) combining Random Forest classifier and LangChain Agent.",
    version="1.0.0"
)

# Enable CORS for local and cross-origin frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Pydantic Schemas for Request & Response Validation
# ------------------------------------------------------------------------------
class SoilDataInput(BaseModel):
    temperature: float = Field(..., ge=-10.0, le=60.0, description="Ambient temperature in °C", example=28.5)
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative humidity percentage", example=65.0)
    moisture: float = Field(..., ge=0.0, le=100.0, description="Soil moisture percentage", example=42.0)
    soil_type: str = Field(..., description="Soil Type: Sandy, Loamy, Black, Red, Clayey", example="Loamy")
    crop_type: str = Field(..., description="Crop Type: Maize, Sugarcane, Cotton, Tobacco, Paddy, Barley, Wheat, Millets, Oil seeds, Pulses, Ground Nuts", example="Wheat")
    nitrogen: float = Field(..., ge=0.0, le=200.0, description="Soil Nitrogen level", example=22.0)
    potassium: float = Field(..., ge=0.0, le=200.0, description="Soil Potassium level", example=26.0)
    phosphorous: float = Field(..., ge=0.0, le=200.0, description="Soil Phosphorous level", example=28.0)


class PredictResponse(BaseModel):
    success: bool
    fertilizer_name: str
    confidence: float
    input_parameters: Dict[str, Any]


class RecommendResponse(BaseModel):
    success: bool
    fertilizer_name: str
    confidence: float
    dosage_per_acre: str
    application_method: str
    npk_composition: str
    crop_type: str
    crop_adjustments: str
    explanation: str
    input_parameters: Dict[str, Any]
    execution_mode: str


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/", tags=["Status"])
def root_status(request: Request):
    """Returns project details or redirects browser to UI."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url="/ui/index.html")
    return {
        "project": "AI Agent-Based Fertilizer Recommendation System",
        "course": "CAF7104 Mini Project",
        "status": "Operational",
        "endpoints": {
            "predict": "POST /predict",
            "recommend": "POST /recommend",
            "options": "GET /options",
            "docs": "/docs",
            "ui": "/ui/index.html"
        }
    }


@app.get("/health", tags=["Status"])
def health_check():
    """Health check for service monitoring."""
    return {"status": "healthy", "service": "Fertilizer Recommendation API"}


@app.get("/options", tags=["Metadata"])
def get_dropdown_options():
    """Returns valid options for soil types, crop types, and supported fertilizers."""
    return {
        "soil_types": ["Sandy", "Loamy", "Black", "Red", "Clayey"],
        "crop_types": [
            "Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy",
            "Barley", "Wheat", "Millets", "Oil seeds", "Pulses", "Ground Nuts"
        ],
        "fertilizers": [
            "Urea", "DAP", "14-35-14", "28-28", "17-17-17", "20-20", "10-26-26"
        ]
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_fertilizer_raw(payload: SoilDataInput):
    """
    POST /predict
    Raw ML prediction endpoint. Invokes the Random Forest classifier directly
    and returns predicted fertilizer and classification confidence.
    """
    try:
        data_dict = payload.model_dump()
        prediction = run_raw_prediction(data_dict)
        return PredictResponse(
            success=True,
            fertilizer_name=prediction["fertilizer_name"],
            confidence=prediction["confidence"],
            input_parameters=data_dict
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/recommend", response_model=RecommendResponse, tags=["Inference"])
def recommend_fertilizer_agent(payload: SoilDataInput):
    """
    POST /recommend
    Full LangChain AI Agent pipeline. Invokes ML prediction, looks up SQLite KB
    for dosage and method, fetches crop-specific notes, and generates an
    expert-level natural language explanation.
    """
    try:
        data_dict = payload.model_dump()
        agent = get_agent_instance()
        recommendation = agent.run(data_dict)
        return RecommendResponse(**recommendation)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation agent error: {str(e)}"
        )


# Serve frontend static assets if directory exists
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print("\nStarting Fertilizer Recommendation Backend at http://127.0.0.1:8000 ...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
