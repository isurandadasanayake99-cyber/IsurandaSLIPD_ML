-- =====================================================================
-- Predictive Maintenance & Fleet Telemetry Engine
-- Database Schema: 3NF Relational Structure (MySQL 8.0+)
-- =====================================================================

CREATE DATABASE IF NOT EXISTS fleet_telemetry_db
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE fleet_telemetry_db;

-- 1. VEHICLE FLEET MASTER TABLE (1NF, 2NF, 3NF compliant)
-- Independent entity containing stationary vehicle metadata.
CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id VARCHAR(36) NOT NULL,              -- UUID or structured fleet ID
    vin VARCHAR(17) NOT NULL UNIQUE,              -- Vehicle Identification Number
    make VARCHAR(50) NOT NULL,
    model VARCHAR(50) NOT NULL,
    manufacture_year SMALLINT UNSIGNED NOT NULL,
    engine_type ENUM('ICE_DIESEL', 'ICE_GASOLINE', 'HYBRID', 'BEV') NOT NULL,
    current_odometer_km DECIMAL(10, 2) NOT NULL DEFAULT 0.00,
    status ENUM('ACTIVE', 'IN_MAINTENANCE', 'DECOMMISSIONED') NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (vehicle_id),
    INDEX idx_vehicles_status (status),
    INDEX idx_vehicles_vin (vin)
) ENGINE=InnoDB;

-- 2. VEHICLE COMPONENT INVENTORY TABLE
-- Isolates component-level tracking from vehicle-level tracking (prevents partial & transitive dependency)
CREATE TABLE IF NOT EXISTS components (
    component_id VARCHAR(36) NOT NULL,
    vehicle_id VARCHAR(36) NOT NULL,
    component_type ENUM('ENGINE', 'TRANSMISSION', 'BRAKE_SYSTEM', 'COOLING_SYSTEM', 'TURBOCHARGER', 'BATTERY') NOT NULL,
    serial_number VARCHAR(100) NOT NULL UNIQUE,
    installed_at DATETIME NOT NULL,
    expected_lifespan_hours INT UNSIGNED NOT NULL, -- Design baseline specification
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (component_id),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id) ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_components_vehicle (vehicle_id),
    INDEX idx_components_type (component_type)
) ENGINE=InnoDB;

-- 3. TELEMETRY LOGS (High-frequency time-series data)
-- Normalized: Sensor readings depend exclusively on (log_id) or composite key (vehicle_id, recorded_at).
CREATE TABLE IF NOT EXISTS telemetry_logs (
    log_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    vehicle_id VARCHAR(36) NOT NULL,
    recorded_at DATETIME(3) NOT NULL,             -- Millisecond precision for IoT timestamps
    engine_rpm SMALLINT UNSIGNED NULL,            -- 0 to 65535 RPM
    engine_temp_c DECIMAL(5, 2) NULL,             -- Coolant / Engine temp in Celsius (-50.00 to 999.99)
    oil_pressure_psi DECIMAL(5, 2) NULL,          -- Oil pressure in PSI
    vibration_amplitude_g DECIMAL(6, 4) NULL,     -- Vibration acceleration in Gs (key wear indicator)
    vehicle_speed_kmh DECIMAL(5, 2) NULL,         -- Vehicle speed
    fuel_flow_rate_lph DECIMAL(5, 2) NULL,        -- Fuel flow rate (liters per hour)
    coolant_level_pct DECIMAL(5, 2) NULL,         -- Percentage (0.00 - 100.00)
    intake_air_temp_c DECIMAL(5, 2) NULL,         -- Intake manifold air temp
    battery_voltage DECIMAL(4, 2) NULL,           -- Direct current voltage (e.g. 12.60V, 24.50V)
    PRIMARY KEY (log_id, recorded_at),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id) ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_telemetry_vehicle_time (vehicle_id, recorded_at DESC)
) ENGINE=InnoDB
-- Partitioning strategy note: For enterprise workloads, range partition on recorded_at by month
PARTITION BY RANGE (YEAR(recorded_at) * 100 + MONTH(recorded_at)) (
    PARTITION p202601 VALUES LESS THAN (202602),
    PARTITION p202602 VALUES LESS THAN (202603),
    PARTITION p202603 VALUES LESS THAN (202604),
    PARTITION p202604 VALUES LESS THAN (202605),
    PARTITION p202605 VALUES LESS THAN (202606),
    PARTITION p202606 VALUES LESS THAN (202607),
    PARTITION p202607 VALUES LESS THAN (202608),
    PARTITION p202608 VALUES LESS THAN (202609),
    PARTITION p202609 VALUES LESS THAN (202610),
    PARTITION p202610 VALUES LESS THAN (202611),
    PARTITION p202611 VALUES LESS THAN (202612),
    PARTITION p202612 VALUES LESS THAN (202701),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- 4. MAINTENANCE LOGS (Interventions, component replacements, RUL ground-truth targets)
-- Records historical repairs, failure milestones, and service costs.
CREATE TABLE IF NOT EXISTS maintenance_logs (
    maintenance_id VARCHAR(36) NOT NULL,
    vehicle_id VARCHAR(36) NOT NULL,
    component_id VARCHAR(36) NULL,                -- Optional: NULL if general vehicle service
    service_date DATETIME NOT NULL,
    maintenance_type ENUM('SCHEDULED', 'UNSCHEDULED', 'BREAKDOWN', 'INSPECTION') NOT NULL,
    service_description TEXT NOT NULL,
    replaced_component BOOLEAN NOT NULL DEFAULT FALSE,
    labor_hours DECIMAL(5, 2) NOT NULL DEFAULT 0.00,
    parts_cost DECIMAL(10, 2) NOT NULL DEFAULT 0.00,
    labor_cost DECIMAL(10, 2) NOT NULL DEFAULT 0.00,
    total_cost DECIMAL(10, 2) GENERATED ALWAYS AS (parts_cost + labor_cost) STORED, -- Computed 3NF column
    operating_hours_at_service INT UNSIGNED NOT NULL,
    technician_notes TEXT NULL,
    PRIMARY KEY (maintenance_id),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (component_id) REFERENCES components(component_id) ON DELETE SET NULL ON UPDATE CASCADE,
    INDEX idx_maintenance_vehicle (vehicle_id, service_date DESC),
    INDEX idx_maintenance_component (component_id)
) ENGINE=InnoDB;

-- 5. MODEL PREDICTIONS / INFERENCE AUDIT LOG
-- Retains Deep Learning RUL and cost forecasts along with GenAI alerts
CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id VARCHAR(36) NOT NULL,
    vehicle_id VARCHAR(36) NOT NULL,
    component_id VARCHAR(36) NULL,
    predicted_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    predicted_rul_hours DECIMAL(8, 2) NOT NULL,   -- Remaining Useful Life target
    predicted_cost DECIMAL(10, 2) NOT NULL,       -- Forecasted cost target
    confidence_lower_bound DECIMAL(8, 2) NULL,
    confidence_upper_bound DECIMAL(8, 2) NULL,
    alert_severity ENUM('NORMAL', 'WARNING', 'CRITICAL') NOT NULL DEFAULT 'NORMAL',
    generated_llm_alert TEXT NULL,                -- Ollama synthesized summary
    PRIMARY KEY (prediction_id),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (component_id) REFERENCES components(component_id) ON DELETE SET NULL ON UPDATE CASCADE,
    INDEX idx_predictions_vehicle (vehicle_id, predicted_at DESC)
) ENGINE=InnoDB;
