"""
PRISM Brain - Indicator Weight Regeneration Script

Rebuilds the indicator_weights table so that each event's weights
use the CORRECT indicator names (matching DataFetcher output).

HOW IT WORKS:
1. Loads all 905 events from the database
2. For each event, extracts its category prefix (e.g., GEO from GEO_001)
3. Looks up the correct indicators in CATEGORY_INDICATOR_MAP
4. Deletes old weights for that event
5. Inserts new weights with correct fetcher indicator names
6. Weights sum to 1.0 per event (already normalized in config)

USAGE:
  python regenerate_weights.py
  Or called automatically during Railway pre-deploy via migrate.py.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config.category_indicators import (
    CATEGORY_INDICATOR_MAP,
    get_category_prefix,
    get_indicators_for_event,
    validate_weights,
)


def get_database_url():
    """Get database URL from environment."""
    url = os.getenv("DATABASE_URL", "")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if not url:
        logger.error("DATABASE_URL not set!")
        sys.exit(1)
    return url


def regenerate_weights():
    """
    Main function: delete old weights, insert new ones based on category mapping.
    """
    issues = validate_weights()
    if issues:
        logger.warning(f"Weight validation issues (weights don't sum to 1.0): {issues}")
        logger.warning("Continuing anyway.")

    db_url = get_database_url()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Step 1: Get all events
        result = session.execute(text("SELECT event_id, baseline_probability FROM risk_events"))
        events = result.fetchall()
        logger.info(f"Found {len(events)} events in database")

        # Step 2: Count current weights
        count_result = session.execute(text("SELECT COUNT(*) FROM indicator_weights"))
        old_count = count_result.scalar()
        logger.info(f"Current indicator_weights rows: {old_count}")

        # Step 3: Delete ALL old weights
        session.execute(text("DELETE FROM indicator_weights"))
        logger.info(f"Deleted {old_count} old weight rows")

        # Step 4: Insert new weights for each event
        new_count = 0
        unmapped_categories = set()
        mapped_categories = {}

        for event_row in events:
            event_id = event_row[0]
            prefix = get_category_prefix(event_id)
            indicators = get_indicators_for_event(event_id)

            if not indicators:
                unmapped_categories.add(prefix)
                continue

            if prefix not in mapped_categories:
                mapped_categories[prefix] = 0
            mapped_categories[prefix] += 1

            for ind in indicators:
                session.execute(
                    text("""
                        INSERT INTO indicator_weights
                            (event_id, indicator_name, weight, data_source, beta_type, time_scale)
                        VALUES
                            (:event_id, :indicator_name, :weight, :data_source, :beta_type, :time_scale)
                    """),
                    {
                        "event_id": event_id,
                        "indicator_name": ind["name"],
                        "weight": ind["weight"],
                        "data_source": ind["source"],
                        "beta_type": ind["beta"],
                        "time_scale": ind["time_scale"],
                    }
                )
                new_count += 1

        session.commit()

        # Step 5: Summary
        logger.info("=" * 60)
        logger.info("WEIGHT REGENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Events processed: {len(events)}")
        logger.info(f"Old weights deleted: {old_count}")
        logger.info(f"New weights inserted: {new_count}")
        logger.info(f"Categories mapped: {mapped_categories}")
        if unmapped_categories:
            logger.warning(f"Unmapped categories: {unmapped_categories}")

        # Verify
        verify_result = session.execute(text("SELECT COUNT(*) FROM indicator_weights"))
        final_count = verify_result.scalar()
        logger.info(f"Final indicator_weights count: {final_count}")

        # Show sample
        sample = session.execute(
            text("SELECT event_id, indicator_name, weight, data_source FROM indicator_weights LIMIT 10")
        ).fetchall()
        logger.info("Sample weights:")
        for row in sample:
            logger.info(f"  {row[0]} | {row[1]} | weight={row[2]} | source={row[3]}")

    except Exception as e:
        session.rollback()
        logger.error(f"Error during weight regeneration: {e}")
        raise
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    logger.info("Starting PRISM Brain weight regeneration...")
    regenerate_weights()
    logger.info("Done!")
