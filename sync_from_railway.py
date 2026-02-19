"""
Sync risk events and data from Railway production database to local database.
Connects directly to both databases and copies the data.

Usage: python sync_from_railway.py
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load local .env
load_dotenv()

# Railway production database (public URL for external access)
RAILWAY_URL = "postgresql://postgres:GuDIefSxVNllIGCEHlyYIgwGQfYPHRVp@switchyard.proxy.rlwy.net:16186/railway"

# Local database from .env
LOCAL_URL = os.getenv("DATABASE_URL", "postgresql://prism_user:prism2026@127.0.0.1:5432/prism_brain_v2")

# Tables to sync (in order, respecting foreign keys)
TABLES_TO_SYNC = [
    "risk_events",
    "risk_probabilities",
    "indicator_weights",
    "indicator_values",
    "data_source_health",
    "calculation_logs",
]


def sync():
    print("=" * 60)
    print("PRISM Brain: Sync from Railway Production Database")
    print("=" * 60)

    # Connect to both databases
    print("\n[1/3] Connecting to databases...")
    try:
        railway_engine = create_engine(RAILWAY_URL, pool_pre_ping=True)
        with railway_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("  Railway (production): Connected")
    except Exception as e:
        print(f"  ERROR connecting to Railway: {e}")
        return

    try:
        local_engine = create_engine(LOCAL_URL, pool_pre_ping=True)
        with local_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"  Local database: Connected")
    except Exception as e:
        print(f"  ERROR connecting to local database: {e}")
        return

    # Check what's on Railway
    print("\n[2/3] Checking Railway data...")
    with railway_engine.connect() as conn:
        for table in TABLES_TO_SYNC:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                print(f"  {table}: {count} rows")
            except Exception:
                print(f"  {table}: table not found (skipping)")

    # Sync each table
    print("\n[3/3] Syncing data to local database...")
    with railway_engine.connect() as r_conn:
        with local_engine.connect() as l_conn:
            for table in TABLES_TO_SYNC:
                try:
                    # Check if table exists on Railway
                    count_result = r_conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = count_result.scalar()

                    if count == 0:
                        print(f"  {table}: empty on Railway, skipping")
                        continue

                    # Get all columns for this table from Railway
                    columns_result = r_conn.execute(text(
                        f"SELECT column_name FROM information_schema.columns "
                        f"WHERE table_name = '{table}' ORDER BY ordinal_position"
                    ))
                    railway_columns = [row[0] for row in columns_result]

                    # Get local columns too
                    local_columns_result = l_conn.execute(text(
                        f"SELECT column_name FROM information_schema.columns "
                        f"WHERE table_name = '{table}' ORDER BY ordinal_position"
                    ))
                    local_columns = [row[0] for row in local_columns_result]

                    # Only sync columns that exist in both
                    shared_columns = [c for c in railway_columns if c in local_columns and c != 'id']

                    if not shared_columns:
                        print(f"  {table}: no matching columns, skipping")
                        continue

                    col_list = ", ".join(shared_columns)

                    # Fetch all rows from Railway
                    rows = r_conn.execute(text(f"SELECT {col_list} FROM {table}")).fetchall()

                    if not rows:
                        print(f"  {table}: no data, skipping")
                        continue

                    # Clear local table and insert
                    l_conn.execute(text(f"DELETE FROM {table}"))

                    # Build insert statement
                    placeholders = ", ".join([f":{c}" for c in shared_columns])
                    insert_sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"

                    # Insert in batches
                    batch_size = 100
                    for i in range(0, len(rows), batch_size):
                        batch = rows[i:i + batch_size]
                        batch_dicts = [dict(zip(shared_columns, row)) for row in batch]
                        l_conn.execute(text(insert_sql), batch_dicts)

                    l_conn.commit()
                    print(f"  {table}: synced {len(rows)} rows")

                except Exception as e:
                    print(f"  {table}: ERROR - {e}")
                    try:
                        l_conn.rollback()
                    except:
                        pass

    # Verify local data
    print("\n" + "=" * 60)
    print("SYNC COMPLETE! Local database now contains:")
    print("=" * 60)
    with local_engine.connect() as conn:
        for table in TABLES_TO_SYNC:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                print(f"  {table}: {count} rows")
            except:
                pass

    print(f"\nNext steps:")
    print(f"  1. Run: python regenerate_weights.py")
    print(f"  2. Start the app: python -m uvicorn main:app --host 0.0.0.0 --port 8000")
    print(f"  3. Visit: http://localhost:8000/docs")


if __name__ == "__main__":
    sync()
