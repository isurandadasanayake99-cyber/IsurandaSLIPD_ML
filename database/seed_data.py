"""
Synthetic Telemetry & Fleet Data Generator
Generates realistic degradation physics (vibration drift, thermal stress, oil pressure drops)
along with occasional missing values to validate the Phase 2 pipeline.
"""

import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
try:
    from database.db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager

def generate_mock_fleet(num_vehicles: int = 3) -> List[Dict[str, Any]]:
    vehicles = []
    models = [("Volvo", "FH16", "ICE_DIESEL"), ("Scania", "R500", "ICE_DIESEL"), ("Mercedes", "Actros", "ICE_DIESEL")]
    for i in range(num_vehicles):
        make, model, engine_type = models[i % len(models)]
        v_id = f"VEH-{1001 + i}"
        vehicles.append({
            "vehicle_id": v_id,
            "vin": f"1M8GDM9A_{1001 + i}_{random.randint(1000, 9999)}",
            "make": make,
            "model": model,
            "manufacture_year": 2021 + (i % 3),
            "engine_type": engine_type,
            "current_odometer_km": round(random.uniform(45000, 120000), 2),
            "status": "ACTIVE"
        })
    return vehicles

def generate_components(vehicles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    components = []
    types = [
        ("ENGINE", 12000),
        ("TRANSMISSION", 15000),
        ("BRAKE_SYSTEM", 4000),
        ("COOLING_SYSTEM", 8000)
    ]
    for v in vehicles:
        for comp_type, lifespan in types:
            components.append({
                "component_id": str(uuid.uuid4()),
                "vehicle_id": v["vehicle_id"],
                "component_type": comp_type,
                "serial_number": f"SN-{comp_type[:3]}-{random.randint(100000, 999999)}",
                "installed_at": datetime.now() - timedelta(days=random.randint(200, 500)),
                "expected_lifespan_hours": lifespan
            })
    return components

def generate_telemetry_series(vehicle_id: str, num_records: int = 300) -> List[Dict[str, Any]]:
    """
    Generates time-series sensor data demonstrating component degradation over time.
    Injects realistic null values (~3%) to test cleaning pipelines.
    """
    records = []
    current_time = datetime.now() - timedelta(hours=num_records)
    
    # Baseline health variables
    baseline_temp = 85.0       # Celsius
    baseline_pressure = 45.0   # PSI
    baseline_vibration = 0.025 # G-force amplitude
    
    for t in range(num_records):
        current_time += timedelta(minutes=60)
        wear_factor = (t / num_records) ** 1.8  # Non-linear physical wear progression
        
        # Operational variations
        rpm = random.randint(1200, 2400)
        speed = round(random.uniform(40, 95), 2)
        
        # Physics degradation simulation:
        # As wear increases: vibration rises, temp climbs, oil pressure drops
        vibration = baseline_vibration + (0.070 * wear_factor) + random.gauss(0, 0.004)
        engine_temp = baseline_temp + (25.0 * wear_factor) + (rpm / 500) + random.gauss(0, 1.2)
        oil_pressure = max(10.0, baseline_pressure - (18.0 * wear_factor) + random.gauss(0, 0.8))
        
        # Inject ~3% probability of missing telemetry readings (sensor glitch / packet loss)
        records.append({
            "vehicle_id": vehicle_id,
            "recorded_at": current_time,
            "engine_rpm": None if random.random() < 0.03 else rpm,
            "engine_temp_c": None if random.random() < 0.03 else round(engine_temp, 2),
            "oil_pressure_psi": None if random.random() < 0.03 else round(oil_pressure, 2),
            "vibration_amplitude_g": None if random.random() < 0.03 else round(max(0.001, vibration), 4),
            "vehicle_speed_kmh": speed,
            "fuel_flow_rate_lph": round(15.0 + (rpm / 150) + random.uniform(0, 3), 2),
            "coolant_level_pct": round(max(20.0, 95.0 - (15.0 * wear_factor)), 2),
            "intake_air_temp_c": round(25.0 + random.uniform(-3, 8), 2),
            "battery_voltage": round(random.uniform(13.8, 14.4), 2)
        })
    return records

def seed_database():
    db = DatabaseManager()
    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        print("[INFO] Seeding Vehicles...")
        vehicles = generate_mock_fleet(3)
        v_sql = """
            INSERT INTO vehicles (vehicle_id, vin, make, model, manufacture_year, engine_type, current_odometer_km, status)
            VALUES (%(vehicle_id)s, %(vin)s, %(make)s, %(model)s, %(manufacture_year)s, %(engine_type)s, %(current_odometer_km)s, %(status)s)
            ON DUPLICATE KEY UPDATE current_odometer_km=VALUES(current_odometer_km);
        """
        cursor.executemany(v_sql, vehicles)

        print("[INFO] Seeding Monitored Components...")
        components = generate_components(vehicles)
        c_sql = """
            INSERT INTO components (component_id, vehicle_id, component_type, serial_number, installed_at, expected_lifespan_hours)
            VALUES (%(component_id)s, %(vehicle_id)s, %(component_type)s, %(serial_number)s, %(installed_at)s, %(expected_lifespan_hours)s)
            ON DUPLICATE KEY UPDATE serial_number=VALUES(serial_number);
        """
        cursor.executemany(c_sql, components)

        print("[INFO] Seeding Telemetry Logs with Degradation Physics...")
        t_sql = """
            INSERT INTO telemetry_logs (
                vehicle_id, recorded_at, engine_rpm, engine_temp_c, oil_pressure_psi,
                vibration_amplitude_g, vehicle_speed_kmh, fuel_flow_rate_lph,
                coolant_level_pct, intake_air_temp_c, battery_voltage
            ) VALUES (
                %(vehicle_id)s, %(recorded_at)s, %(engine_rpm)s, %(engine_temp_c)s, %(oil_pressure_psi)s,
                %(vibration_amplitude_g)s, %(vehicle_speed_kmh)s, %(fuel_flow_rate_lph)s,
                %(coolant_level_pct)s, %(intake_air_temp_c)s, %(battery_voltage)s
            )
        """
        for v in vehicles:
            logs = generate_telemetry_series(v["vehicle_id"], num_records=250)
            cursor.executemany(t_sql, logs)

        print("[INFO] Seeding Maintenance Logs...")
        m_sql = """
            INSERT INTO maintenance_logs (
                maintenance_id, vehicle_id, component_id, service_date,
                maintenance_type, service_description, replaced_component,
                labor_hours, parts_cost, labor_cost, operating_hours_at_service, technician_notes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        for comp in components[:3]:
            cursor.execute(m_sql, (
                str(uuid.uuid4()),
                comp["vehicle_id"],
                comp["component_id"],
                datetime.now() - timedelta(days=30),
                "UNSCHEDULED",
                "High vibration detected in engine block; bearing inspection and lubricant flush.",
                False,
                4.5,
                350.00,
                450.00,
                3800,
                "Bearing races exhibited early micro-pitting. Flushed lubricant and recalibrated mounts."
            ))

        conn.commit()
        print("[SUCCESS] Fleet database seeded successfully with mock telemetry & maintenance events.")

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Seeding failed: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    seed_database()
