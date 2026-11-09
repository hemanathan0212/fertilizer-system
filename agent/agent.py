"""
================================================================================
COURSE: CAF7104 - Mini Project
PROJECT: AI Agent-Based Fertilizer Recommendation System
MODULE 4 (Part B): LangChain AI Agent Orchestrator (agent/agent.py)
================================================================================
Purpose:
  Orchestrates the three LangChain tools (ML Prediction, DB Lookup, Crop Adjustment).
  Takes soil and crop inputs, coordinates tool execution, and synthesizes a
  fluent, expert-level natural language explanation.
  
  Features:
    1. Supports live LLM execution via LangChain if an API key (e.g., OPENAI_API_KEY)
       is provided.
    2. Includes a comprehensive, domain-aware Agronomic Explanation Generator fallback
       so that the agent functions 100% offline for college evaluations, vivas,
       and demos without requiring paid API credits.
================================================================================
"""

import os
import sys
import json
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.tools import (
    ALL_TOOLS,
    predict_fertilizer,
    lookup_fertilizer_info,
    adjust_for_crop,
    run_raw_prediction
)


class FertilizerRecommendationAgent:
    """
    AI Agent that combines machine learning inference with agronomic domain knowledge
    via LangChain tools to deliver transparent, actionable fertilizer recommendations.
    """

    def __init__(self, api_key: Optional[str] = None):
        # Check for API key in constructor or environment variables
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        self.tools = ALL_TOOLS
        self.llm_available = bool(self.api_key)

    def _diagnose_soil_nutrients(self, n: float, p: float, k: float) -> Dict[str, str]:
        """Provides qualitative deficiency classification for N, P, K."""
        status = {}
        # Nitrogen benchmarks
        if n < 20:
            status["n"] = "Low (Deficient)"
        elif n <= 35:
            status["n"] = "Moderate"
        else:
            status["n"] = "Optimal / High"

        # Phosphorous benchmarks
        if p < 18:
            status["p"] = "Low (Deficient)"
        elif p <= 32:
            status["p"] = "Moderate"
        else:
            status["p"] = "Optimal / High"

        # Potassium benchmarks
        if k < 20:
            status["k"] = "Low (Deficient)"
        elif k <= 35:
            status["k"] = "Moderate"
        else:
            status["k"] = "Optimal / High"

        return status

    def _generate_expert_explanation(
        self,
        pred_data: Dict[str, Any],
        kb_info: Dict[str, Any],
        crop_adj: Dict[str, Any]
    ) -> str:
        """
        Generates a rich, structured natural language explanation synthesized
        from ML predictions and knowledge base parameters.
        """
        fert = pred_data.get("fertilizer_name", "Fertilizer")
        conf = pred_data.get("confidence", 0.0)
        crop = pred_data.get("crop_type", "Crop")
        soil = pred_data.get("soil_type", "Soil")
        n = pred_data.get("nitrogen", 0.0)
        p = pred_data.get("phosphorous", 0.0)
        k = pred_data.get("potassium", 0.0)
        temp = pred_data.get("temperature", 0.0)
        humidity = pred_data.get("humidity", 0.0)
        moisture = pred_data.get("moisture", 0.0)

        dosage = kb_info.get("dosage_per_acre", "40-50 kg/acre")
        method = kb_info.get("application_method", "Apply during soil preparation or active growth.")
        npk = kb_info.get("npk_composition", "Balanced grade")
        adj_notes = crop_adj.get("adjustment_notes", "Follow standard growth-stage scheduling.")

        diag = self._diagnose_soil_nutrients(n, p, k)

        explanation = (
            f"Based on our AI diagnostic pipeline, **{fert}** is recommended for your **{crop}** "
            f"cultivated in **{soil}** soil (Model Confidence: {conf}%).\n\n"
            f"### 1. Soil & Climate Assessment\n"
            f"- **Nutrient Levels:** Nitrogen is {diag['n']} ({n} ppm), Phosphorous is {diag['p']} ({p} ppm), "
            f"and Potassium is {diag['k']} ({k} ppm).\n"
            f"- **Growing Conditions:** Ambient temperature is {temp}°C with {humidity}% relative humidity "
            f"and {moisture}% soil moisture, providing favorable vegetative absorption conditions.\n\n"
            f"### 2. Fertilizer Rationale & Nutrient Profile\n"
            f"- **Selection:** {fert} was identified as the optimal compound ({npk}) to restore "
            f"soil nutrient equilibrium and satisfy the high nutrient uptake demands of {crop}.\n\n"
            f"### 3. Application Protocol\n"
            f"- **Dosage:** {dosage}.\n"
            f"- **Method:** {method}\n\n"
            f"### 4. Crop-Specific Management ({crop})\n"
            f"- {adj_notes}"
        )
        return explanation

    def run(self, soil_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the recommendation workflow by invoking the 3 LangChain tools.
        """
        # Step 1: Tool 1 - Predict Fertilizer via ML Model
        pred_tool_output = predict_fertilizer.invoke(soil_params)
        if isinstance(pred_tool_output, str):
            pred_data = json.loads(pred_tool_output)
        else:
            pred_data = pred_tool_output

        fert_name = pred_data.get("fertilizer_name")
        crop_type = pred_data.get("crop_type")

        # Step 2: Tool 2 - Lookup Fertilizer Knowledge Base Info
        kb_tool_output = lookup_fertilizer_info.invoke({"fertilizer_name": fert_name})
        if isinstance(kb_tool_output, str):
            kb_data = json.loads(kb_tool_output)
        else:
            kb_data = kb_tool_output

        # Step 3: Tool 3 - Adjust for Specific Crop
        adj_tool_output = adjust_for_crop.invoke({
            "fertilizer_name": fert_name,
            "crop_type": crop_type
        })
        if isinstance(adj_tool_output, str):
            adj_data = json.loads(adj_tool_output)
        else:
            adj_data = adj_tool_output

        # Step 4: Synthesize Explanation (Offline Expert or Live LLM)
        mode = "LangChain Expert Engine (Offline Mode)"
        explanation = self._generate_expert_explanation(pred_data, kb_data, adj_data)

        # If LLM key is configured, optional enhancement via ChatOpenAI
        if self.llm_available:
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import SystemMessage, HumanMessage

                llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
                prompt = (
                    f"You are an expert agricultural scientist. Summarize and enhance this fertilizer recommendation:\n"
                    f"Prediction: {json.dumps(pred_data)}\n"
                    f"Knowledge Base: {json.dumps(kb_data)}\n"
                    f"Crop Adjustment: {json.dumps(adj_data)}"
                )
                response = llm.invoke([
                    SystemMessage(content="You are a senior agronomist advising a farmer."),
                    HumanMessage(content=prompt)
                ])
                explanation = response.content
                mode = "LangChain Agent with Live LLM"
            except Exception as e:
                # Graceful fallback to expert template if network/key failure
                mode = f"LangChain Expert Engine (LLM fallback: {str(e)[:40]}...)"

        return {
            "success": True,
            "fertilizer_name": fert_name,
            "confidence": pred_data.get("confidence", 0.0),
            "dosage_per_acre": kb_data.get("dosage_per_acre"),
            "application_method": kb_data.get("application_method"),
            "npk_composition": kb_data.get("npk_composition"),
            "crop_type": crop_type,
            "crop_adjustments": adj_data.get("adjustment_notes"),
            "explanation": explanation,
            "input_parameters": {
                "temperature": pred_data.get("temperature"),
                "humidity": pred_data.get("humidity"),
                "moisture": pred_data.get("moisture"),
                "soil_type": pred_data.get("soil_type"),
                "crop_type": crop_type,
                "nitrogen": pred_data.get("nitrogen"),
                "potassium": pred_data.get("potassium"),
                "phosphorous": pred_data.get("phosphorous")
            },
            "execution_mode": mode
        }


# Global agent instance for re-use
_AGENT_INSTANCE = None


def get_agent_instance() -> FertilizerRecommendationAgent:
    """Singleton getter for the AI Recommendation Agent."""
    global _AGENT_INSTANCE
    if _AGENT_INSTANCE is None:
        _AGENT_INSTANCE = FertilizerRecommendationAgent()
    return _AGENT_INSTANCE


def recommend_fertilizer(soil_params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper entry point for backend and tests."""
    agent = get_agent_instance()
    return agent.run(soil_params)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODULE 4: TESTING AI AGENT LAYER")
    print("="*70)

    sample_input = {
        "temperature": 29.5,
        "humidity": 68.0,
        "moisture": 48.0,
        "soil_type": "Loamy",
        "crop_type": "Sugarcane",
        "nitrogen": 14.0,
        "potassium": 28.0,
        "phosphorous": 32.0
    }

    print("\n[Input Soil & Crop Parameters]")
    print(json.dumps(sample_input, indent=2))

    agent = get_agent_instance()
    result = agent.run(sample_input)

    print("\n[Agent Output]")
    print(f"Recommended Fertilizer: {result['fertilizer_name']} (Confidence: {result['confidence']}%)")
    print(f"Dosage:                 {result['dosage_per_acre']}")
    print(f"Application:            {result['application_method']}")
    print(f"Crop Adjustments:       {result['crop_adjustments']}")
    print(f"Execution Mode:         {result['execution_mode']}")
    print("\n[Generated Natural Language Explanation]:")
    print(result["explanation"])
    print("="*70)
