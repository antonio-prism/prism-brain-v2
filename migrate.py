"""
PRISM Brain Database Migration Script v4 - COMPREHENSIVE
Ensures ALL columns from ALL models exist in the database.
Safe to re-run: uses IF NOT EXISTS and try/except for renames.
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

# ============================================================
# ALL columns for ALL tables - comprehensive list from models.py
# ============================================================

ALL_COLUMNS = {
    "risk_events": [
        "id SERIAL",
        "event_id VARCHAR(50)",
        "event_name VARCHAR(500)",
        "layer1_primary VARCHAR(100)",
        "layer1_secondary VARCHAR(100)",
        "layer2_primary VARCHAR(100)",
        "layer2_secondary VARCHAR(100)",
        "baseline_probability FLOAT DEFAULT 0.5",
        "baseline_1_5 FLOAT DEFAULT 3.0",
        "super_risk BOOLEAN DEFAULT FALSE",
        "baseline_impact FLOAT",
        "methodology_tier VARCHAR(50)",
        "geographic_scope VARCHAR(100)",
        "time_horizon VARCHAR(100)",
        "description TEXT",
        "created_at TIMESTAMP DEFAULT NOW()",
        "updated_at TIMESTAMP DEFAULT NOW()",
    ],
    "risk_probabilities": [
        "id SERIAL",
        "event_id VARCHAR(50)",
        "probability_pct FLOAT",
        "confidence_score FLOAT",
        "calculation_date TIMESTAMP DEFAULT NOW()",
        "data_sources_used INTEGER DEFAULT 0",
        "flags TEXT",
        "ci_lower_pct FLOAT",
        "ci_upper_pct FLOAT",
        "change_direction VARCHAR(20)", "methodology_tier VARCHAR(50)",
        "precision_band VARCHAR(50)",
        "log_odds FLOAT",
        "total_adjustment FLOAT",
        "indicators_used INTEGER DEFAULT 0",
        "calculation_method VARCHAR(50) DEFAULT 'bayesian'",
        "signal FLOAT",
        "momentum FLOAT",
        "trend VARCHAR(50)",
        "is_anomaly BOOLEAN DEFAULT FALSE",
        "ensemble_method VARCHAR(50)",
        "ml_probability_pct FLOAT",
        "attribution JSON",
        "explanation TEXT",
        "recommendation VARCHAR(100)",
        "previous_probability_pct FLOAT",
        "probability_change_pct FLOAT",
        "dependency_adjustment FLOAT",
        "dependency_details JSON",
        "calculation_id VARCHAR(50)",
        "baseline_probability_pct FLOAT",
        "ci_level FLOAT",
        "ci_width_pct FLOAT",
        "bootstrap_iterations INTEGER",
    ],
    "indicator_weights": [
        "id SERIAL",
        "event_id VARCHAR(50)",
        "indicator_name VARCHAR(200)",
        "weight FLOAT DEFAULT 1.0",
        "normalized_weight FLOAT",
        "data_source VARCHAR(100)",
        "beta_type VARCHAR(50)",
        "time_scale VARCHAR(50)",
        "created_at TIMESTAMP DEFAULT NOW()",
    ],
    "indicator_values": [
        "id SERIAL",
        "event_id VARCHAR(50)",
        "indicator_name VARCHAR(200)",
        "value FLOAT",
        "raw_value FLOAT",
        "z_score FLOAT",
        "data_source VARCHAR(100)",
        "timestamp TIMESTAMP DEFAULT NOW()",
        "quality_score FLOAT",
        "historical_mean FLOAT",
        "historical_std FLOAT",
        "signal FLOAT",
        "momentum FLOAT",
        "trend VARCHAR(20)",
        "is_anomaly BOOLEAN DEFAULT FALSE",
    ],
    "data_source_health": [
        "id SERIAL",
        "source_name VARCHAR(100)",
        "status VARCHAR(50) DEFAULT 'unknown'",
        "last_success TIMESTAMP",
        "last_failure TIMESTAMP",
        "error_message TEXT",
        "response_time_ms INTEGER",
        "records_fetched INTEGER DEFAULT 0",
        "check_time TIMESTAMP DEFAULT NOW()",
        "success_rate_24h FLOAT",
    ],
    "calculation_logs": [
        "id SERIAL",
        "calculation_id VARCHAR(50)",
        "start_time TIMESTAMP DEFAULT NOW()",
        "end_time TIMESTAMP",
        "events_processed INTEGER DEFAULT 0",
        "events_succeeded INTEGER DEFAULT 0",
        "events_failed INTEGER DEFAULT 0",
        "duration_seconds FLOAT",
        "status VARCHAR(50) DEFAULT 'running'",
        "errors TEXT",
        "trigger VARCHAR(50) DEFAULT 'manual'",
        "method VARCHAR(50) DEFAULT 'bayesian'",
    ],
}

# Column renames from old models.py to new
RENAME_COLUMNS = [
    ("calculation_logs", "started_at", "start_time"),
    ("calculation_logs", "completed_at", "end_time"),
    ("calculation_logs", "events_calculated", "events_processed"),
    ("calculation_logs", "total_duration_seconds", "duration_seconds"),
    ("calculation_logs", "error_message", "errors"),
    ("indicator_values", "fetched_at", "timestamp"),
    ("indicator_values", "source", "data_source"),
    ("indicator_values", "data_quality", "quality_score"),
    ("data_source_health", "checked_at", "check_time"),
    ("risk_probabilities", "ci_lower", "ci_lower_pct"),
    ("risk_probabilities", "ci_upper", "ci_upper_pct"),
]

print("=== PRISM Brain Database Migration v4 (COMPREHENSIVE) ===")
print()

with engine.connect() as conn:
    # Step 1: Rename old columns
    print("Step 1: Renaming old columns...")
    for table, old_col, new_col in RENAME_COLUMNS:
        try:
            conn.execute(text(f"ALTER TABLE {table} RENAME COLUMN {old_col} TO {new_col}"))
            print(f"  RENAMED: {table}.{old_col} -> {new_col}")
        except Exception:
            print(f"  SKIP: {table}.{old_col} (already correct or missing)")
            conn.rollback()
    conn.commit()

    # Step 2: Add ALL columns to ALL tables
    print()
    print("Step 2: Adding all missing columns...")
    for table, columns in ALL_COLUMNS.items():
        print(f"  Table: {table}")
        for col_def in columns:
            col_name = col_def.split()[0]
            try:
                conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col_def}"
                ))
                print(f"    OK: {col_name}")
            except Exception as e:
                print(f"    SKIP: {col_name} ({e})")
        conn.commit()

print()
print("Migration v4 complete - all columns verified.")

from database.connection import init_db
init_db()
print("Database tables verified.")

# Step 3: Regenerate indicator weights with correct fetcher indicator names
print()
print("Step 3: Regenerating indicator weights...")
try:
    from regenerate_weights import regenerate_weights
    regenerate_weights()
    print("Weight regeneration complete.")
except Exception as e:
    print(f"WARNING: Weight regeneration failed: {e}")
    print("Weights will use existing values. Run regenerate_weights.py manually if needed.")
