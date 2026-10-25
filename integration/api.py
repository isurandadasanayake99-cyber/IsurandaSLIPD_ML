"""
Phase 6: Lightweight FastAPI Service
Serves the trained PyTorch RUL regression model and passes predictions to Ollama LLM
for real-time dashboard notifications.
"""

import sys
import uvicorn
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR / "database"))
sys.path.append(str(ROOT_DIR / "pipeline"))
sys.path.append(str(ROOT_DIR / "data"))
sys.path.append(str(ROOT_DIR / "models"))
sys.path.append(str(ROOT_DIR / "integration"))

from inference_service import FleetInferenceEngine
from seed_data import generate_mock_fleet, generate_telemetry_series
import pandas as pd

# Pydantic Schemas
class SensorReading(BaseModel):
    recorded_at: Optional[str] = None
    engine_rpm: float = Field(..., example=2150.0)
    engine_temp_c: float = Field(..., example=94.5)
    oil_pressure_psi: float = Field(..., example=32.0)
    vibration_amplitude_g: float = Field(..., example=0.065)
    vehicle_speed_kmh: float = Field(..., example=75.0)
    fuel_flow_rate_lph: float = Field(..., example=24.2)
    coolant_level_pct: float = Field(..., example=88.0)
    intake_air_temp_c: float = Field(..., example=28.0)
    battery_voltage: float = Field(..., example=14.1)

class TelemetryBatchRequest(BaseModel):
    vehicle_id: str = Field(..., example="VEH-1001")
    odometer_km: float = Field(..., example=85000.0)
    readings: List[SensorReading]

class PredictionResponse(BaseModel):
    prediction_id: str
    vehicle_id: str
    predicted_rul_hours: float
    severity: str
    latest_telemetry_snapshot: Dict[str, Any]
    dashboard_alert: str
    model_version: str

# App Initialization
app = FastAPI(
    title="Fleet Predictive Maintenance & Telemetry Engine",
    description="Deep Learning RUL Forecast API integrated with Ollama GenAI alerts.",
    version="1.0.0"
)

# Enable CORS for web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singleton engine
_engine: Optional[FleetInferenceEngine] = None

def get_engine() -> FleetInferenceEngine:
    global _engine
    if _engine is None:
        _engine = FleetInferenceEngine()
    return _engine

@app.get("/", tags=["General"])
def read_root():
    return {
        "system": "Fleet Predictive Maintenance & Telemetry Engine",
        "status": "ONLINE",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health", tags=["Health"])
def healthcheck():
    engine = get_engine()
    ollama_live = engine.ollama.check_connection()
    return {
        "status": "HEALTHY",
        "model_loaded": engine.model is not None,
        "scaler_loaded": engine.scaler is not None,
        "ollama_connected": ollama_live,
        "ollama_endpoint": engine.ollama.api_endpoint
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_telemetry_stream(payload: TelemetryBatchRequest):
    """
    Accepts a time-series stream of sensor readings for a vehicle,
    extracts rolling physics features, evaluates the PyTorch ANN,
    and returns an Ollama-generated natural language dashboard alert.
    """
    if len(payload.readings) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Readings array cannot be empty. Minimum 1 reading required."
        )

    engine = get_engine()

    # Convert incoming Pydantic readings to DataFrame
    records = []
    for r in payload.readings:
        d = r.dict()
        d["vehicle_id"] = payload.vehicle_id
        d["current_odometer_km"] = payload.odometer_km
        d["manufacture_year"] = 2022
        d["engine_type"] = "ICE_DIESEL"
        d["log_id"] = 1
        if not d.get("recorded_at"):
            d["recorded_at"] = pd.Timestamp.now()
        records.append(d)

    df = pd.DataFrame(records)
    result = engine.predict_from_telemetry_df(df, vehicle_id=payload.vehicle_id)
    return result

@app.get("/simulate/{vehicle_id}", response_model=PredictionResponse, tags=["Simulation"])
def simulate_and_predict(vehicle_id: str, wear_level: float = 0.85):
    """
    Convenience endpoint: Simulates a vehicle with degradation physics and runs the full pipeline.
    Useful for demonstration without manually crafting JSON payloads.
    """
    engine = get_engine()
    # Generate realistic degraded vehicle stream
    v = generate_mock_fleet(1)[0]
    v["vehicle_id"] = vehicle_id
    stream = generate_telemetry_series(vehicle_id, num_records=30)
    
    # Adjust wear on the tail of the stream
    for s in stream:
        s["manufacture_year"] = v["manufacture_year"]
        s["current_odometer_km"] = v["current_odometer_km"]
        s["engine_type"] = v["engine_type"]
        s["log_id"] = 1

    df = pd.DataFrame(stream)
    result = engine.predict_from_telemetry_df(df, vehicle_id=vehicle_id)
    return result

if __name__ == "__main__":
    print("[SERVER] Starting FastAPI server on http://localhost:8000...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
