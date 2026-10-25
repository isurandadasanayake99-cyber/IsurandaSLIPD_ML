"""
Database Connection & Management Module
Handles MySQL schema initialization, connection pooling, and transactional operations.
"""

import os
import mysql.connector
from mysql.connector import pooling, Error
from typing import Optional, Dict, Any, List

# Default connection settings (can be overridden via environment variables)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "fleet_telemetry_db"),
}

class DatabaseManager:
    """Manages MySQL connection pooling and DDL/DML query execution."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DB_CONFIG
        self.pool: Optional[pooling.MySQLConnectionPool] = None

    def initialize_pool(self, pool_size: int = 5) -> None:
        """Initializes a thread-safe MySQL connection pool."""
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name="fleet_pool",
                pool_size=pool_size,
                pool_reset_session=True,
                **self.config
            )
            print(f"[INFO] Initialized MySQL Connection Pool: {self.config['host']}:{self.config['port']}/{self.config['database']}")
        except Error as e:
            print(f"[ERROR] Failed to initialize connection pool: {e}")
            raise

    def get_connection(self):
        """Retrieves an active connection from the pool."""
        if not self.pool:
            self.initialize_pool()
        return self.pool.get_connection()

    def execute_schema_file(self, schema_filepath: str) -> bool:
        """
        Executes raw DDL statements from schema.sql.
        Connects without selecting database initially to ensure CREATE DATABASE succeeds.
        """
        initial_config = {k: v for k, v in self.config.items() if k != "database"}
        try:
            conn = mysql.connector.connect(**initial_config)
            cursor = conn.cursor()

            with open(schema_filepath, "r", encoding="utf-8") as f:
                schema_sql = f.read()

            # Execute multi-statement SQL script
            for statement in cursor.execute(schema_sql, multi=True):
                if statement.with_rows:
                    statement.fetchall()

            conn.commit()
            cursor.close()
            conn.close()
            print("[SUCCESS] Schema executed successfully.")
            return True
        except Error as e:
            print(f"[ERROR] Error executing schema file: {e}")
            return False

if __name__ == "__main__":
    import pathlib
    base_dir = pathlib.Path(__file__).parent.resolve()
    schema_path = base_dir / "schema.sql"
    
    db = DatabaseManager()
    print("Initializing Database and Tables...")
    db.execute_schema_file(str(schema_path))
