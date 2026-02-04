#!/usr/bin/env python3
"""
Load Causal Dependencies from YAML

Loads the dependency network from dependency_network.yaml into the database.
"""

import sys
import logging
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import init_db, get_session_context, create_tables
from database.models import CausalDependency

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dependencies(filepath: str, dry_run: bool = False) -> int:
    """
    Load causal dependencies from YAML file into database.

    Args:
        filepath: Path to dependency_network.yaml
        dry_run: If True, validate but don't commit

    Returns:
        Number of dependencies loaded
    """
    logger.info(f"Loading dependencies from {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    dependencies = data.get('dependencies', [])
    logger.info(f"Found {len(dependencies)} dependencies in file")

    init_db()
    create_tables()

    loaded_count = 0
    errors = []

    with get_session_context() as session:
        # Clear existing dependencies
        session.query(CausalDependency).delete()
        logger.info("Cleared existing dependency records")

        for dep in dependencies:
            try:
                dependency = CausalDependency(
                    driver_event_id=dep.get('driver_event'),
                    dependent_event_id=dep.get('dependent_event'),
                    relationship_type=dep.get('relationship_type', 'CORRELATED'),
                    multiplier=float(dep.get('multiplier', 1.5)),
                    confidence=dep.get('confidence', 'MEDIUM'),
                    bidirectional=dep.get('bidirectional', False),
                    rationale=dep.get('rationale')
                )
                session.add(dependency)
                loaded_count += 1

                if loaded_count % 500 == 0:
                    logger.info(f"Processed {loaded_count} dependencies...")

            except Exception as e:
                errors.append({
                    'driver': dep.get('driver_event'),
                    'dependent': dep.get('dependent_event'),
                    'error': str(e)
                })

        if dry_run:
            logger.info("Dry run - rolling back")
            session.rollback()
        else:
            session.commit()
            logger.info(f"Committed {loaded_count} dependencies")

    # Summary by type
    with get_session_context() as session:
        logger.info("Dependencies by relationship type:")
        for rtype in ['CAUSAL', 'ENABLING', 'CORRELATED', 'WEAK']:
            count = session.query(CausalDependency).filter(
                CausalDependency.relationship_type == rtype
            ).count()
            logger.info(f"  {rtype}: {count}")

    if errors:
        logger.warning(f"Errors: {len(errors)}")

    return loaded_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Load dependencies into database')
    parser.add_argument('filepath', help='Path to dependency_network.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if not Path(args.filepath).exists():
        logger.error(f"File not found: {args.filepath}")
        sys.exit(1)

    load_dependencies(args.filepath, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
