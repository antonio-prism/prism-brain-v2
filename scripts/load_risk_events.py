#!/usr/bin/env python3
"""
Load Risk Events from JSON

Loads the 900 risk events from risks_complete.json into the database.
"""

import json
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import init_db, get_session_context, create_tables
from database.models import RiskEvent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Tier classification based on event characteristics
TIER_1_PREFIXES = ['CYBER']  # High-frequency, data-rich events
TIER_3_KEYWORDS = [
    'nuclear', 'asteroid', 'agi', 'extinction', 'civilizational',
    'pandemic novel', 'superintelligence'
]


def classify_methodology_tier(event: dict) -> str:
    """
    Classify event into methodology tier based on characteristics.

    Returns:
        TIER_1_ML_ENHANCED: Data-rich events with ML support
        TIER_2_ANALOG: Historical analog-based reasoning
        TIER_3_SCENARIO: Scenario decomposition for unprecedented events
    """
    event_id = event.get('event_id', '')
    event_name = event.get('event_name', '').lower()
    description = event.get('description', '').lower()

    # Tier 1: Data-rich categories with quantitative indicators
    prefix = event_id.split('-')[0] if '-' in event_id else ''
    if prefix in TIER_1_PREFIXES:
        return 'TIER_1_ML_ENHANCED'

    # Tier 3: Black swan / unprecedented events
    combined_text = f"{event_name} {description}"
    for keyword in TIER_3_KEYWORDS:
        if keyword in combined_text:
            return 'TIER_3_SCENARIO'

    # Default: Tier 2 analog-based
    return 'TIER_2_ANALOG'


def load_risk_events(filepath: str, dry_run: bool = False) -> int:
    """
    Load risk events from JSON file into database.

    Args:
        filepath: Path to risks_complete.json
        dry_run: If True, validate but don't commit

    Returns:
        Number of events loaded
    """
    logger.info(f"Loading risk events from {filepath}")

    # Read JSON file
    with open(filepath, 'r', encoding='utf-8') as f:
        events_data = json.load(f)

    logger.info(f"Found {len(events_data)} events in file")

    # Initialize database
    init_db()
    create_tables()

    loaded_count = 0
    errors = []

    with get_session_context() as session:
        for event_data in events_data:
            try:
                # Map JSON fields to database columns
                event = RiskEvent(
                    event_id=event_data['event_id'],
                    event_name=event_data['event_name'],
                    description=event_data.get('description'),
                    layer1_primary=event_data.get('layer1_primary'),
                    layer1_secondary=event_data.get('layer1_secondary'),
                    layer2_primary=event_data.get('layer2_primary'),
                    layer2_secondary=event_data.get('layer2_secondary'),
                    super_risk=event_data.get('super_risk') == 'YES',
                    affected_industries=event_data.get('affected_industries'),
                    geographic_scope=event_data.get('geographic_scope'),
                    time_horizon=event_data.get('time_horizon'),
                    baseline_probability=event_data.get('baseline_probability'),
                    baseline_impact=event_data.get('baseline_impact'),
                    source_category=event_data.get('source_category'),
                    methodology_tier=classify_methodology_tier(event_data)
                )

                # Merge (insert or update)
                session.merge(event)
                loaded_count += 1

                if loaded_count % 100 == 0:
                    logger.info(f"Processed {loaded_count} events...")

            except Exception as e:
                errors.append({
                    'event_id': event_data.get('event_id', 'UNKNOWN'),
                    'error': str(e)
                })
                logger.error(f"Error loading event {event_data.get('event_id')}: {e}")

        if dry_run:
            logger.info("Dry run - rolling back changes")
            session.rollback()
        else:
            session.commit()
            logger.info(f"Committed {loaded_count} events to database")

    # Summary
    logger.info("=" * 50)
    logger.info(f"Load complete: {loaded_count} events loaded")

    if errors:
        logger.warning(f"Errors encountered: {len(errors)}")
        for err in errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {err['event_id']}: {err['error']}")

    # Count by tier
    with get_session_context() as session:
        tier_counts = {}
        for tier in ['TIER_1_ML_ENHANCED', 'TIER_2_ANALOG', 'TIER_3_SCENARIO']:
            count = session.query(RiskEvent).filter(
                RiskEvent.methodology_tier == tier
            ).count()
            tier_counts[tier] = count

        logger.info("Events by methodology tier:")
        for tier, count in tier_counts.items():
            logger.info(f"  {tier}: {count}")

    return loaded_count


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Load risk events into database')
    parser.add_argument('filepath', help='Path to risks_complete.json')
    parser.add_argument('--dry-run', action='store_true', help='Validate without committing')

    args = parser.parse_args()

    if not Path(args.filepath).exists():
        logger.error(f"File not found: {args.filepath}")
        sys.exit(1)

    count = load_risk_events(args.filepath, dry_run=args.dry_run)
    logger.info(f"Successfully loaded {count} risk events")


if __name__ == '__main__':
    main()
