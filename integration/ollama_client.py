"""
Phase 6: Ollama GenAI Client
Communicates with the local Ollama daemon to transform numerical PyTorch RUL predictions
and sensor telemetry into natural language fleet maintenance alerts.
Includes automatic fallback if the Ollama service is offline.
"""

import os
import json
import requests
from typing import Dict, Any, Optional

class OllamaAlertGenerator:
    """
    Manages prompt composition and inference requests to local Ollama LLM instances.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 15
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3")
        self.timeout = timeout
        self.api_endpoint = f"{self.base_url.rstrip('/')}/api/generate"

    def check_connection(self) -> bool:
        """Pings Ollama daemon to check service availability."""
        try:
            resp = requests.get(f"{self.base_url.rstrip('/')}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def build_system_prompt(
        self,
        vehicle_id: str,
        predicted_rul_hours: float,
        severity: str,
        telemetry: Dict[str, Any]
    ) -> str:
        """Composes a high-context prompt for the LLM."""
        return f"""You are the Chief Reliability Engineer and AI Diagnostic Specialist for an enterprise commercial fleet.

A deep learning predictive maintenance engine has analyzed vehicle telemetry and issued an alert:

================ VEHICLE DIAGNOSTIC TELEMETRY ================
Vehicle ID: {vehicle_id}
Alert Severity: {severity.upper()}
Forecasted Remaining Useful Life (RUL): {predicted_rul_hours:.1f} Operating Hours
Key Sensor Telemetry:
  - Engine RPM: {telemetry.get('engine_rpm', 'N/A')} RPM
  - Engine Temperature: {telemetry.get('engine_temp_c', 'N/A')} °C
  - Oil Pressure: {telemetry.get('oil_pressure_psi', 'N/A')} PSI
  - Vibration Amplitude: {telemetry.get('vibration_amplitude_g', 'N/A')} G
  - Rolling Vibration Jitter: {telemetry.get('vibration_roll_std_5', 'N/A')} G
  - Lubrication Ratio (Oil Press / RPM): {telemetry.get('oil_pressure_rpm_ratio', 'N/A')}
==============================================================

TASK:
Write a concise, professional fleet maintenance alert for the dispatch dashboard.
Format your response with the following sections:
1. SUMMARY OF RISK: Direct statement of the operational danger and estimated time to failure.
2. ROOT CAUSE HYPOTHESIS: Specific physical component suspected (e.g., bearing raceway fatigue, oil pump cavitation, cooling fan failure) based on the telemetry.
3. IMMEDIATE ACTION: Exact instructions for fleet operators (e.g., immediate pull-over, route diversion to nearest depot, scheduled inspection).

Keep the language direct, technical, and urgent."""

    def generate_alert(
        self,
        vehicle_id: str,
        predicted_rul_hours: float,
        severity: str,
        telemetry: Dict[str, Any]
    ) -> str:
        """
        Sends generation request to Ollama.
        Falls back to rule-based template if Ollama is unreachable.
        """
        prompt = self.build_system_prompt(vehicle_id, predicted_rul_hours, severity, telemetry)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,  # Low temperature for deterministic diagnostic precision
                "num_predict": 250
            }
        }

        try:
            response = requests.post(self.api_endpoint, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return self._fallback_alert(vehicle_id, predicted_rul_hours, severity, telemetry, reason=f"Ollama HTTP {response.status_code}")
        except Exception as e:
            return self._fallback_alert(vehicle_id, predicted_rul_hours, severity, telemetry, reason=str(e))

    def _fallback_alert(
        self,
        vehicle_id: str,
        predicted_rul: float,
        severity: str,
        telemetry: Dict[str, Any],
        reason: str
    ) -> str:
        """Deterministic diagnostic alert generated when Ollama daemon is offline."""
        vib = telemetry.get("vibration_amplitude_g", 0.05)
        temp = telemetry.get("engine_temp_c", 85.0)
        oil = telemetry.get("oil_pressure_psi", 40.0)

        return (
            f"[DIAGNOSTIC ENGINE ALERT - SEVERITY: {severity.upper()}]\n"
            f"Vehicle {vehicle_id} forecasted RUL: {predicted_rul:.1f} hours remaining.\n\n"
            f"1. SUMMARY OF RISK: High probability of premature component wear. Immediate attention required.\n"
            f"2. ROOT CAUSE HYPOTHESIS: Vibration reading ({vib:.3f}G) and temperature ({temp:.1f}°C) indicate "
            f"micro-bearing degradation and frictional thermal rise. Oil pressure at {oil:.1f} PSI.\n"
            f"3. IMMEDIATE ACTION: Route vehicle to nearest maintenance depot within {max(1.0, predicted_rul * 0.8):.0f} operating hours "
            f"for lubricant analysis and bearing mount inspection.\n"
            f"(Note: Generated via deterministic rule engine; Ollama service offline: {reason})"
        )
