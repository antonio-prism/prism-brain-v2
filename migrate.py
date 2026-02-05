"""
PRISM Brain Database Migration Script
Adds all Phase 4B-4E columns to existing tables.
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

ALTER_STATEMENTS = [
    # risk_events: new columns added in v3.0.0
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
    # indicator_values: Phase 4B signal columns
    "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS signal FLOAT",
    "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS momentum FLOAT",
    "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS trend VARCHAR(20)",
    "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS is_anomaly BOOLEAN DEFAULT FALSE",
]

print("Running PRISM Brain database migration...")
with engine.connect() as conn:
    for stmt in ALTER_STATEMENTS:
        try:
            conn.execute(text(stmt))
            print(f"  OK: {stmt[:60]}...")
        except Exception as e:
            print(f"  SKIP: {stmt[:60]}... ({e})")
    conn.commit()
print("Migration complete.")

# Also run init_db to create any missing tables
from database.connection import init_db
init_db()
print("Database tables verified.")
