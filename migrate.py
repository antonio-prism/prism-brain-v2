"""
PRISM Brain Database Migration Script v3
Adds missing id columns, renames mismatched columns, adds Phase 4B-4E columns.
Run as pre-deploy command on Railway.
"""
import os
import sys
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set")
    sys.exit(1)

engine = create_engine(DATABASE_URL)

# Step 0: Add missing id (primary key) columns to tables that were created without them
ADD_ID_COLUMNS = [
    ("risk_events", "id"),
    ("risk_probabilities", "id"),
    ("indicator_weights", "id"),
    ("indicator_values", "id"),
    ("data_source_health", "id"),
    ("calculation_logs", "id"),
]

# Step 1: Rename columns where models.py was out of sync with main.py
RENAME_COLUMNS = [
    ("calculation_logs", "started_at", "start_time"),
    ("calculation_logs", "completed_at", "end_time"),
    ("calculation_logs", "events_calculated", "events_processed"),
    ("calculation_logs", "total_duration_seconds", "duration_seconds"),
    ("calculation_logs", "error_message", "errors"),
    ("indicator_values", "fetched_at", "timestamp"),
    ("indicator_values", "source", "data_source"),
    ("data_source_health", "checked_at", "check_time"),
]

# Step 2: Add all columns that should exist (IF NOT EXISTS = safe to re-run)
ADD_COLUMNS = [
    # risk_events
    "ALTER TABLE risk_events ADD COLUMN IF NOT EXISTS baseline_1_5 FLOAT DEFAULT 3.0",
    # risk_probabilities: Phase 4B signal columns
    "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS signal FLOAT",
    "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS momentum FLOAT",
    "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS trend VARCHAR(50)",
    "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS is_anomaly BOOLEAN DEFAULT FALSE",
    # risk_probabilities: Phase 4C ML columns
    "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS ensemble_method VARCHAR(50)",
    "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS ml_probability_pct FLOAT",
    # risk_probabilities: Phase 4D explainability columns
    "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS attribution JSON",
    "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS explanation TEXT",
    "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS recommendation VARCHAR(100)",
    "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS previous_probability_pct FLOAT",
    "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS probability_change_pct FLOAT",
    # risk_probabilities: Phase 4E dependency columns
    "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS dependency_adjustment FLOAT",
    "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS dependency_details JSON",
    # indicator_values: ensure correct column names exist
    "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS data_source VARCHAR(100)",
    "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP DEFAULT NOW()",
    "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS quality_score FLOAT",
    "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS historical_mean FLOAT",
    "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS historical_std FLOAT",
    # indicator_values: Phase 4B signal columns
    "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS signal FLOAT",
    "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS momentum FLOAT",
    "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS trend VARCHAR(20)",
    "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS is_anomaly BOOLEAN DEFAULT FALSE",
    # calculation_logs: ensure correct column names exist
    "ALTER TABLE calculation_logs ADD COLUMN IF NOT EXISTS calculation_id VARCHAR(50)",
    "ALTER TABLE calculation_logs ADD COLUMN IF NOT EXISTS start_time TIMESTAMP DEFAULT NOW()",
    "ALTER TABLE calculation_logs ADD COLUMN IF NOT EXISTS end_time TIMESTAMP",
    "ALTER TABLE calculation_logs ADD COLUMN IF NOT EXISTS events_processed INTEGER DEFAULT 0",
    "ALTER TABLE calculation_logs ADD COLUMN IF NOT EXISTS events_succeeded INTEGER DEFAULT 0",
    "ALTER TABLE calculation_logs ADD COLUMN IF NOT EXISTS duration_seconds FLOAT",
    "ALTER TABLE calculation_logs ADD COLUMN IF NOT EXISTS errors TEXT",
    "ALTER TABLE calculation_logs ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'running'",
    "ALTER TABLE calculation_logs ADD COLUMN IF NOT EXISTS trigger VARCHAR(50) DEFAULT 'manual'",
    "ALTER TABLE calculation_logs ADD COLUMN IF NOT EXISTS method VARCHAR(50) DEFAULT 'bayesian'",
    # data_source_health: ensure correct column names exist
    "ALTER TABLE data_source_health ADD COLUMN IF NOT EXISTS check_time TIMESTAMP DEFAULT NOW()",
    "ALTER TABLE data_source_health ADD COLUMN IF NOT EXISTS success_rate_24h FLOAT",
]

print("=== PRISM Brain Database Migration v3 ===")
print()

with engine.connect() as conn:
    # Step 0: Add missing id columns
    print("Step 0: Adding missing id (primary key) columns...")
    for table, col in ADD_ID_COLUMNS:
        try:
            # Check if column exists
            result = conn.execute(text(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{table}' AND column_name = '{col}'"
            ))
            if result.fetchone() is None:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} SERIAL"))
                print(f"  ADDED: {table}.{col} (SERIAL)")
            else:
                print(f"  SKIP: {table}.{col} (already exists)")
        except Exception as e:
            print(f"  ERROR: {table}.{col}: {e}")
            conn.rollback()
    conn.commit()

    # Step 1: Try column renames (safe: fails silently if old name doesn't exist)
    print()
    print("Step 1: Renaming mismatched columns...")
    for table, old_col, new_col in RENAME_COLUMNS:
        try:
            conn.execute(text(f"ALTER TABLE {table} RENAME COLUMN {old_col} TO {new_col}"))
            print(f"  RENAMED: {table}.{old_col} -> {new_col}")
        except Exception as e:
            print(f"  SKIP: {table}.{old_col} -> {new_col} (already correct or not found)")
            conn.rollback()
    conn.commit()

    # Step 2: Add all columns (IF NOT EXISTS = safe to re-run)
    print()
    print("Step 2: Adding missing columns...")
    for stmt in ADD_COLUMNS:
        try:
            conn.execute(text(stmt))
            print(f"  OK: {stmt[:70]}...")
        except Exception as e:
            print(f"  SKIP: {stmt[:70]}... ({e})")
    conn.commit()

print()
print("Migration complete.")

# Also run init_db to create any entirely missing tables
from database.connection import init_db
init_db()
print("Database tables verified.")
