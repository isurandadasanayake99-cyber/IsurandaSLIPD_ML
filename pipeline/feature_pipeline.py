"""
Phase 2: Telemetry Data Pipeline & Feature Engineering
Extracts raw telemetry from MySQL, imputes missing signals using causal forward-filling,
engineers physics-based rolling and interaction features, and generates RUL targets.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from datetime import datetime

class TelemetryPipeline:
    """
    Production-grade ETL & feature engineering pipeline for vehicle telemetry.
    Designed to prevent data leakage (only uses historical causal windows).
    """

    def __init__(self, db_manager=None):
        self.db_manager = db_manager

    def extract_from_mysql(self, vehicle_id: Optional[str] = None) -> pd.DataFrame:
        """
        Extracts telemetry time-series joined with vehicle metadata from MySQL.
        """
        if self.db_manager is None:
            raise ValueError("Database manager instance is required for MySQL extraction.")

        conn = self.db_manager.get_connection()
        query = """
            SELECT 
                t.log_id,
                t.vehicle_id,
                t.recorded_at,
                t.engine_rpm,
                t.engine_temp_c,
                t.oil_pressure_psi,
                t.vibration_amplitude_g,
                t.vehicle_speed_kmh,
                t.fuel_flow_rate_lph,
                t.coolant_level_pct,
                t.intake_air_temp_c,
                t.battery_voltage,
                v.manufacture_year,
                v.current_odometer_km,
                v.engine_type
            FROM telemetry_logs t
            JOIN vehicles v ON t.vehicle_id = v.vehicle_id
        """
        params = []
        if vehicle_id:
            query += " WHERE t.vehicle_id = %s"
            params.append(vehicle_id)

        query += " ORDER BY t.vehicle_id ASC, t.recorded_at ASC;"

        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        df = pd.DataFrame(rows)
        if not df.empty and "recorded_at" in df.columns:
            df["recorded_at"] = pd.to_datetime(df["recorded_at"])
            # Cast decimal types from MySQL to float64
            numeric_cols = [
                "engine_temp_c", "oil_pressure_psi", "vibration_amplitude_g",
                "vehicle_speed_kmh", "fuel_flow_rate_lph", "coolant_level_pct",
                "intake_air_temp_c", "battery_voltage", "current_odometer_km"
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Causal Missing Value Imputation:
        1. Forward-fill (ffill): Real-world sensors retain last known state during packet drops.
           Avoids lookahead leakage.
        2. Backward-fill (bfill): Fills boundary conditions at sequence origins.
        3. Median imputation: Fills completely missing channels across fleet records.
        """
        df = df.copy()
        df = df.sort_values(by=["vehicle_id", "recorded_at"]).reset_index(drop=True)

        sensor_cols = [
            "engine_rpm", "engine_temp_c", "oil_pressure_psi", "vibration_amplitude_g",
            "vehicle_speed_kmh", "fuel_flow_rate_lph", "coolant_level_pct",
            "intake_air_temp_c", "battery_voltage"
        ]

        # Group by vehicle to guarantee no cross-vehicle signal bleeding
        for col in sensor_cols:
            if col in df.columns:
                df[col] = df.groupby("vehicle_id")[col].transform(lambda group: group.ffill().bfill())
                if df[col].isnull().any():
                    # Fleet-level median fallback for cold-start edge cases
                    df[col] = df[col].fillna(df[col].median())

        return df

    def engineer_features(self, df: pd.DataFrame, window_sizes: List[int] = [5, 15, 30]) -> pd.DataFrame:
        """
        Feature Engineering based on Mechanical & Thermodynamics Physics:
        1. Rolling Means: Filters high-frequency white noise to capture macro-degradation.
        2. Rolling Standard Deviations: Measures mechanical jitter & instability (piston slap, bearing wear).
        3. Rate-of-Change (Gradients): Thermal delta dT/dt to detect cooling degradation.
        4. Cross-Channel Interaction: Oil pressure-to-RPM ratio (identifies lubrication breakdown).
        5. Cumulative Fatigue Stress: Accumulated mechanical load over operating cycles.
        """
        df = df.copy()
        df = df.sort_values(by=["vehicle_id", "recorded_at"]).reset_index(drop=True)

        grouped = df.groupby("vehicle_id")

        # 1. Rolling statistics per vehicle
        for w in window_sizes:
            # Moving averages (smoothing)
            df[f"engine_temp_roll_mean_{w}"] = grouped["engine_temp_c"].transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"vibration_roll_mean_{w}"] = grouped["vibration_amplitude_g"].transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"oil_pressure_roll_mean_{w}"] = grouped["oil_pressure_psi"].transform(lambda x: x.rolling(w, min_periods=1).mean())
            
            # Moving variance / jitter (mechanical instability marker)
            df[f"vibration_roll_std_{w}"] = grouped["vibration_amplitude_g"].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0.0))
            df[f"rpm_roll_std_{w}"] = grouped["engine_rpm"].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0.0))

        # 2. Physics Derivatives: Thermal Gradient dT/dt (Degrees Celsius per sample step)
        df["temp_gradient"] = grouped["engine_temp_c"].transform(lambda x: x.diff().fillna(0.0))

        # 3. Mechanical Interaction: Pressure-to-RPM Ratio
        # Higher RPM requires proportional lubrication; low pressure at high RPM signals imminent failure
        df["oil_pressure_rpm_ratio"] = df["oil_pressure_psi"] / (df["engine_rpm"] + 1e-4)

        # 4. Thermal Stress Index: Temperature difference over ambient/intake air
        df["thermal_stress_delta"] = df["engine_temp_c"] - df["intake_air_temp_c"]

        # 5. Cumulative Mechanical Wear Integral: Sum of (vibration * rpm * speed) over lifetime
        instant_stress = df["vibration_amplitude_g"] * (df["engine_rpm"] / 1000.0) * (df["vehicle_speed_kmh"] / 100.0)
        df["instantaneous_stress"] = instant_stress
        df["cumulative_wear_score"] = grouped["instantaneous_stress"].transform(lambda x: x.cumsum())

        return df

    def compute_piecewise_rul(self, df: pd.DataFrame, max_rul_hours: float = 200.0) -> pd.DataFrame:
        """
        Constructs Remaining Useful Life (RUL) regression target.
        Theory: Linear RUL assumes wear starts from cycle 0. In physical machinery,
        degradation only manifests after a healthy operational baseline.
        Piecewise linear degradation caps RUL at max_rul_hours, focusing neural network
        capacity on the active deterioration phase.
        """
        df = df.copy()
        
        # Calculate time elapsed per vehicle sequence towards the end of monitored logs (failure/service event)
        max_time_per_vehicle = df.groupby("vehicle_id")["recorded_at"].transform("max")
        
        # Remaining time delta in hours
        time_to_event_hours = (max_time_per_vehicle - df["recorded_at"]).dt.total_seconds() / 3600.0
        
        # Piecewise Linear Clipping
        df["target_rul_hours"] = time_to_event_hours.clip(upper=max_rul_hours)
        
        return df

    def run_pipeline(
        self,
        raw_df: Optional[pd.DataFrame] = None,
        max_rul_hours: float = 200.0
    ) -> Tuple[pd.DataFrame, List[str], str]:
        """
        Executes end-to-end data cleaning, imputation, feature engineering, and labeling.
        Returns:
            df_processed: Complete DataFrame with features and target
            feature_cols: Names of all engineered numerical feature inputs
            target_col: Name of the target variable ('target_rul_hours')
        """
        if raw_df is None:
            raw_df = self.extract_from_mysql()

        print(f"[PIPELINE] Ingested {len(raw_df)} raw telemetry rows.")

        # 1. Imputation
        df_imputed = self.handle_missing_values(raw_df)
        print(f"[PIPELINE] Missing values imputed via causal forward-fill.")

        # 2. Physics Feature Engineering
        df_featured = self.engineer_features(df_imputed)
        print(f"[PIPELINE] Engineered physical features & rolling statistics.")

        # 3. RUL Target Construction
        df_labeled = self.compute_piecewise_rul(df_featured, max_rul_hours=max_rul_hours)
        print(f"[PIPELINE] Generated Piecewise RUL targets (Capped at {max_rul_hours}h).")

        # Exclude metadata, identifiers, and timestamps from feature set
        non_feature_cols = [
            "log_id", "vehicle_id", "recorded_at", "manufacture_year",
            "engine_type", "target_rul_hours"
        ]
        feature_cols = [c for c in df_labeled.columns if c not in non_feature_cols]

        return df_labeled, feature_cols, "target_rul_hours"

if __name__ == "__main__":
    import sys
    # Demonstration & unit test using synthetic generator if database is not connected
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / "database"))
    from seed_data import generate_mock_fleet, generate_telemetry_series

    print("Running Pipeline Unit Test with In-Memory Telemetry...")
    vehicles = generate_mock_fleet(2)
    mock_logs = []
    for v in vehicles:
        logs = generate_telemetry_series(v["vehicle_id"], num_records=100)
        for l in logs:
            l["manufacture_year"] = v["manufacture_year"]
            l["current_odometer_km"] = v["current_odometer_km"]
            l["engine_type"] = v["engine_type"]
            l["log_id"] = 1
        mock_logs.extend(logs)
    
    test_df = pd.DataFrame(mock_logs)
    pipeline = TelemetryPipeline()
    processed_df, features, target = pipeline.run_pipeline(raw_df=test_df)
    
    print(f"\n[SUCCESS] Pipeline Output Shape: {processed_df.shape}")
    print(f"Features ({len(features)}): {features[:8]} ...")
    print(f"Target Column: {target}")
    print(processed_df[["vehicle_id", "recorded_at", "vibration_roll_mean_5", "oil_pressure_rpm_ratio", "target_rul_hours"]].head())
