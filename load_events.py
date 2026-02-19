"""
Load risk events from risk_database.json into the local PostgreSQL database.
This populates the backend API with all the risk event data.

Usage: python load_events.py
"""

import json
import os
from pathlib import Path
from sqlalchemy import create_engine, text

# Try to load .env file for DATABASE_URL
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("  Note: python-dotenv not installed, reading DATABASE_URL from environment directly")

LOCAL_URL = os.getenv("DATABASE_URL", "postgresql://prism_user:prism2026@127.0.0.1:5432/prism_brain_v2")
SEEDS_DIR = Path(__file__).parent / "frontend" / "data" / "seeds"
SEED_FILES = [
    "physical_domain_seed.json",
    "structural_domain_seed.json",
    "digital_domain_seed.json",
    "operational_domain_seed.json",
]


def load_events():
    print("=" * 60)
    print("PRISM Brain: Load Risk Events into Local Database")
    print("=" * 60)

    # Read all seed JSON files
    print(f"\n[1/3] Reading seed files from {SEEDS_DIR.name}/...")
    events = []
    for seed_file in SEED_FILES:
        seed_path = SEEDS_DIR / seed_file
        if not seed_path.exists():
            print(f"  WARNING: Seed file not found: {seed_path}")
            continue
        with open(seed_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Handle both flat array and metadata-wrapped formats
        file_events = data if isinstance(data, list) else data.get('events', [])
        print(f"  {seed_file}: {len(file_events)} events")
        events.extend(file_events)

    if not events:
        print("  ERROR: No events found in seed files!")
        return

    print(f"  Total: {len(events)} risk events loaded from {len(SEED_FILES)} seed files")

    # Show domains
    domains = {}
    for e in events:
        dom = e.get("domain", "Unknown")
        domains[dom] = domains.get(dom, 0) + 1
    for dom, count in sorted(domains.items()):
        print(f"    {dom}: {count} events")

    # Connect to local database
    print(f"\n[2/3] Connecting to local database...")
    try:
        engine = create_engine(LOCAL_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"  Connected successfully")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # Insert events
    print(f"\n[3/3] Loading events into database...")
    with engine.connect() as conn:
        # Check current count
        current = conn.execute(text("SELECT COUNT(*) FROM risk_events")).scalar()
        print(f"  Current events in database: {current}")

        inserted = 0
        updated = 0
        errors = 0

        for e in events:
            try:
                # Support both seed format (event_id) and database format (Event_ID)
                event_id = e.get("Event_ID") or e.get("event_id", "")
                if not event_id:
                    continue

                # Check if exists
                exists = conn.execute(
                    text("SELECT event_id FROM risk_events WHERE event_id = :eid"),
                    {"eid": event_id}
                ).fetchone()

                super_risk = True if (e.get("Super_Risk") or e.get("super_risk") or "NO") == "YES" else False

                # Support both formats for all fields
                event_name = e.get("Event_Name") or e.get("event_name", "")
                description = e.get("Event_Description") or e.get("description", "")
                layer1_primary = e.get("Layer_1_Primary") or e.get("domain", "")
                layer1_secondary = e.get("Layer_1_Secondary", "")
                layer2_primary = e.get("Layer_2_Primary") or e.get("family_name", "")
                layer2_secondary = e.get("Layer_2_Secondary", "") or e.get("family_code", "")
                geographic_scope = e.get("Geographic_Scope") or e.get("geographic_scope", "")
                time_horizon = e.get("Time_Horizon") or e.get("base_rate_frequency", "")

                # Convert base_rate_pct (percentage 0-100) to decimal (0-1)
                # Seed data uses base_rate_pct (e.g. 0.12 means 0.12%, 6.5 means 6.5%)
                # Legacy data may use base_probability directly as 0-1
                if "base_rate_pct" in e:
                    base_probability = float(e.get("base_rate_pct", 0)) / 100.0
                else:
                    base_probability = float(e.get("base_probability", 0.5))

                base_impact = float(e.get("base_impact", 0.5))

                params = {
                    "event_id": event_id,
                    "event_name": event_name,
                    "description": description,
                    "layer1_primary": layer1_primary.upper() if layer1_primary else "",
                    "layer1_secondary": layer1_secondary,
                    "layer2_primary": layer2_primary,
                    "layer2_secondary": layer2_secondary,
                    "super_risk": super_risk,
                    "baseline_probability": base_probability,
                    "baseline_impact": base_impact,
                    "geographic_scope": geographic_scope,
                    "time_horizon": time_horizon,
                }

                if exists:
                    conn.execute(text("""
                        UPDATE risk_events SET
                            event_name = :event_name,
                            description = :description,
                            layer1_primary = :layer1_primary,
                            layer1_secondary = :layer1_secondary,
                            layer2_primary = :layer2_primary,
                            layer2_secondary = :layer2_secondary,
                            super_risk = :super_risk,
                            baseline_probability = :baseline_probability,
                            baseline_impact = :baseline_impact,
                            geographic_scope = :geographic_scope,
                            time_horizon = :time_horizon
                        WHERE event_id = :event_id
                    """), params)
                    updated += 1
                else:
                    conn.execute(text("""
                        INSERT INTO risk_events
                            (event_id, event_name, description, layer1_primary,
                             layer1_secondary, layer2_primary, layer2_secondary,
                             super_risk, baseline_probability, baseline_impact,
                             geographic_scope, time_horizon)
                        VALUES
                            (:event_id, :event_name, :description, :layer1_primary,
                             :layer1_secondary, :layer2_primary, :layer2_secondary,
                             :super_risk, :baseline_probability, :baseline_impact,
                             :geographic_scope, :time_horizon)
                    """), params)
                    inserted += 1

            except Exception as ex:
                errors += 1
                if errors <= 3:
                    print(f"  Warning: Error on {e.get('Event_ID', '?')}: {ex}")

        conn.commit()

        # Final count
        final = conn.execute(text("SELECT COUNT(*) FROM risk_events")).scalar()
        super_count = conn.execute(text("SELECT COUNT(*) FROM risk_events WHERE super_risk = true")).scalar()

    print(f"\n{'=' * 60}")
    print(f"LOAD COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Inserted: {inserted} new events")
    print(f"  Updated:  {updated} existing events")
    print(f"  Errors:   {errors}")
    print(f"  Total in database: {final}")
    print(f"  Super risks: {super_count}")
    print(f"\nNext steps:")
    print(f"  1. Run: python regenerate_weights.py")
    print(f"  2. Start the API: python -m uvicorn main:app --host 0.0.0.0 --port 8000")
    print(f"  3. Start the dashboard: cd frontend && streamlit run Welcome.py")


if __name__ == "__main__":
    load_events()
