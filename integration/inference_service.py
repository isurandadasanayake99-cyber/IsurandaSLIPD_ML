"""
Phase 6: Inference & Diagnostics Service
Loads serialized PyTorch model checkpoint and StandardScaler,
processes live telemetry, predicts Remaining Useful Life, and generates GenAI alerts.
"""

import os
import sys
import uuid
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR / "database"))
sys.path.append(str(ROOT_DIR / "pipeline"))
sys.path.append(str(ROOT_DIR / "data"))
sys.path.append(str(ROOT_DIR / "models"))
sys.path.append(str(ROOT_DIR / "integration"))

from ann_model import FleetRULPredictor
from feature_pipeline import TelemetryPipeline
from ollama_client import OllamaAlertGenerator

class FleetInferenceEngine:
    """
    Production inference engine orchestrating PyTorch forward scoring,
    risk thresholding, and Ollama natural language alert synthesis.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        db_manager = None
    ):
        self.device = torch.device("cpu")
        self.model_path = model_path or str(ROOT_DIR / "models" / "best_model.pt")
        self.scaler_path = scaler_path or str(ROOT_DIR / "models" / "scaler.joblib")
        self.db_manager = db_manager

        # Initialize helper pipelines
        self.pipeline = TelemetryPipeline(db_manager=db_manager)
        self.ollama = OllamaAlertGenerator()

        # Load fitted StandardScaler
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler artifact not found at: {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)

        # Load trained PyTorch ANN model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        input_dim = self.scaler.n_features_in_
        self.model = FleetRULPredictor(input_dim=input_dim)
        
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print(f"[INFERENCE ENGINE] Initialized PyTorch model ({input_dim} features) and Scaler.")

    def determine_severity(self, rul_hours: float) -> str:
        """Determines operational risk tier based on forecasted RUL."""
        if rul_hours <= 35.0:
            return "CRITICAL"
        elif rul_hours <= 85.0:
            return "WARNING"
        return "NORMAL"

    @torch.no_grad()
    def predict_from_telemetry_df(
        self,
        telemetry_df: pd.DataFrame,
        vehicle_id: str
    ) -> Dict[str, Any]:
        """
        Runs live or recent time-series telemetry for a specific vehicle through
        the feature engineering pipeline, standard scaler, and PyTorch model.
        """
        # 1. Feature Engineering
        imputed_df = self.pipeline.handle_missing_values(telemetry_df)
        featured_df = self.pipeline.engineer_features(imputed_df)

        non_feature_cols = [
            "log_id", "vehicle_id", "recorded_at", "manufacture_year",
            "engine_type", "target_rul_hours"
        ]
        feature_cols = [c for c in featured_df.columns if c not in non_feature_cols]

        # Extract latest point-in-time state
        latest_row = featured_df[featured_df["vehicle_id"] == vehicle_id].iloc[-1]
        raw_feature_vector = latest_row[feature_cols].values.reshape(1, -1)

        # 2. Scale features with persisted scaler
        scaled_features = self.scaler.transform(raw_feature_vector)
        tensor_input = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)

        # 3. PyTorch Forward Pass
        prediction_tensor = self.model(tensor_input)
        predicted_rul = float(prediction_tensor.item())
        predicted_rul = max(0.0, predicted_rul)  # RUL is non-negative

        severity = self.determine_severity(predicted_rul)

        # 4. Synthesize diagnostic context for Ollama
        diagnostic_context = {
            "engine_rpm": float(latest_row.get("engine_rpm", 0)),
            "engine_temp_c": float(latest_row.get("engine_temp_c", 0)),
            "oil_pressure_psi": float(latest_row.get("oil_pressure_psi", 0)),
            "vibration_amplitude_g": float(latest_row.get("vibration_amplitude_g", 0)),
            "vibration_roll_std_5": float(latest_row.get("vibration_roll_std_5", 0)),
            "oil_pressure_rpm_ratio": float(latest_row.get("oil_pressure_rpm_ratio", 0)),
        }

        # 5. Generate Natural Language Alert via Ollama LLM
        llm_alert = self.ollama.generate_alert(
            vehicle_id=vehicle_id,
            predicted_rul_hours=predicted_rul,
            severity=severity,
            telemetry=diagnostic_context
        )

        result = {
            "prediction_id": str(uuid.uuid4()),
            "vehicle_id": vehicle_id,
            "predicted_rul_hours": round(predicted_rul, 2),
            "severity": severity,
            "latest_telemetry_snapshot": diagnostic_context,
            "dashboard_alert": llm_alert,
            "model_version": "PyTorch-Fleet-ANN-v1"
        }

        # 6. Audit to MySQL if db_manager is active
        self._record_to_db(result)

        return result

    def _record_to_db(self, result: Dict[str, Any]):
        """Persists inference record and GenAI alert into model_predictions table."""
        if not self.db_manager:
            return
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            sql = """
                INSERT INTO model_predictions (
                    prediction_id, vehicle_id, predicted_rul_hours, predicted_cost,
                    alert_severity, generated_llm_alert
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                result["prediction_id"],
                result["vehicle_id"],
                result["predicted_rul_hours"],
                0.0,
                result["severity"],
                result["dashboard_alert"]
            ))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"[WARN] Failed to write inference audit to MySQL: {e}")

if __name__ == "__main__":
    print("Testing FleetInferenceEngine End-to-End...")
    from seed_data import generate_mock_fleet, generate_telemetry_series

    # Synthesize test stream
    v = generate_mock_fleet(1)[0]
    stream = generate_telemetry_series(v["vehicle_id"], num_records=40)
    for s in stream:
        s["manufacture_year"] = v["manufacture_year"]
        s["current_odometer_km"] = v["current_odometer_km"]
        s["engine_type"] = v["engine_type"]
        s["log_id"] = 1

    df = pd.DataFrame(stream)
    engine = FleetInferenceEngine()
    prediction_result = engine.predict_from_telemetry_df(df, vehicle_id=v["vehicle_id"])

    print("\n[PREDICTION RESULT]:")
    print(f"Vehicle:   {prediction_result['vehicle_id']}")
    print(f"Pred RUL:  {prediction_result['predicted_rul_hours']} Hours")
    print(f"Severity:  {prediction_result['severity']}")
    print(f"\n[DASHBOARD ALERT]:\n{prediction_result['dashboard_alert']}")
