#!/usr/bin/env python3
"""
Load Indicator Weights from JSON

Loads the weight derivations for all 900 events from event_indicator_weights.json.
"""

import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import init_db, get_session_context, create_tables
from database.models import IndicatorWeight
from config.settings import BETA_PARAMETERS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_weights(filepath: str, dry_run: bool = False) -> int:
    """
    Load indicator weights from JSON file into database.

    Args:
        filepath: Path to event_indicator_weights.json
        dry_run: If True, validate but don't commit

    Returns:
        Number of weight records loaded
    """
    logger.info(f"Loading indicator weights from {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        weights_data = json.load(f)

    logger.info(f"Found weights for {len(weights_data)} events")

    init_db()
    create_tables()

    loaded_count = 0
    total_indicators = 0

    with get_session_context() as session:
        # Clear existing weights
        session.query(IndicatorWeight).delete()
        logger.info("Cleared existing weight records")

        for event_id, event_weights in weights_data.items():
            indicators = event_weights.get('indicators', [])

            for ind in indicators:
                try:
                    # Get beta value from type
                    beta_type = ind.get('beta_type', 'moderate_correlation')
                    beta_value = BETA_PARAMETERS.get(beta_type, 0.7)

                    weight = IndicatorWeight(
                        event_id=event_id,
                        indicator_name=ind.get('indicator_name', ind.get('name', 'unknown')),
                        data_source=ind.get('data_source', ind.get('source', 'unknown')),
                        causal_proximity_score=ind.get('causal_proximity_score'),
                        data_quality_score=ind.get('data_quality_score'),
                        timeliness_score=ind.get('timeliness_score'),
                        predictive_lead_score=ind.get('predictive_lead_score'),
                        raw_score=ind.get('raw_score'),
                        normalized_weight=ind.get('normalized_weight', ind.get('weight', 0.2)),
                        beta_type=beta_type,
                        beta_value=beta_value,
                        time_scale=ind.get('time_scale', 'medium'),
                        justification=ind.get('justification')
                    )
                    session.add(weight)
                    total_indicators += 1

                except Exception as e:
                    logger.error(f"Error loading weight for {event_id}/{ind.get('indicator_name')}: {e}")

            loaded_count += 1
            if loaded_count % 100 == 0:
                logger.info(f"Processed {loaded_count} events ({total_indicators} indicators)...")

        if dry_run:
            logger.info("Dry run - rolling back")
            session.rollback()
        else:
            session.commit()
            logger.info(f"Committed weights for {loaded_count} events")

    logger.info(f"Load complete: {loaded_count} events, {total_indicators} indicator weights")
    return total_indicators


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Load indicator weights into database')
    parser.add_argument('filepath', help='Path to event_indicator_weights.json')
    parser.add_argument('--dry-run', action='store_true', help='Validate without committing')
    args = parser.parse_args()

    if not Path(args.filepath).exists():
        logger.error(f"File not found: {args.filepath}")
        sys.exit(1)

    load_weights(args.filepath, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
