"""
Celery Application Configuration

Handles background task execution and scheduled calculations.
"""

from celery import Celery
from celery.schedules import crontab
import asyncio
import logging
from typing import List, Optional

from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    'prism_brain',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Beat schedule for weekly calculations
celery_app.conf.beat_schedule = {
    'weekly-probability-calculation': {
        'task': 'tasks.celery_app.run_calculation_task',
        'schedule': crontab(
            day_of_week=settings.calculation_schedule_day,
            hour=settings.calculation_schedule_hour,
            minute=settings.calculation_schedule_minute
        ),
        'args': (None,),  # Calculate all events
    },
    'hourly-health-check': {
        'task': 'tasks.celery_app.check_data_sources_health',
        'schedule': crontab(minute=0),  # Every hour
    },
}


@celery_app.task(bind=True, max_retries=3)
def run_calculation_task(self, event_ids: Optional[List[str]] = None):
    """
    Run probability calculation for specified events.

    Args:
        event_ids: List of event IDs to calculate (None = all events)
    """
    from probability_engine.calculation import ProbabilityOrchestrator

    logger.info(f"Starting calculation task. Events: {event_ids or 'all'}")

    try:
        orchestrator = ProbabilityOrchestrator()

        # Run the async calculation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            batch = loop.run_until_complete(
                orchestrator.run_batch_calculation(event_ids=event_ids)
            )
        finally:
            loop.close()

        result = {
            'calculation_id': batch.calculation_id,
            'events_processed': batch.events_processed,
            'events_succeeded': batch.events_succeeded,
            'events_failed': batch.events_failed,
            'duration_seconds': (batch.end_time - batch.start_time).total_seconds()
            if batch.end_time else None
        }

        logger.info(f"Calculation complete: {result}")
        return result

    except Exception as e:
        logger.error(f"Calculation task failed: {e}")
        self.retry(exc=e, countdown=60 * 5)  # Retry in 5 minutes


@celery_app.task
def check_data_sources_health():
    """Check health of all configured data sources."""
    from probability_engine.data_acquisition.base_client import BaseAPIClient
    from database.connection import get_session_context
    from database.models import DataSourceHealth
    from datetime import datetime
    import time

    logger.info("Running data source health checks")

    # Import all clients
    from probability_engine.data_acquisition.clients import (
        acled_client, world_bank_client, fred_client, gdelt_client,
        noaa_client, eia_client, otx_client, nvd_client, imf_client,
        nasa_client
    )

    clients = [
        acled_client.ACLEDClient(),
        world_bank_client.WorldBankClient(),
        fred_client.FREDClient(),
        gdelt_client.GDELTClient(),
        noaa_client.NOAAClient(),
        eia_client.EIAClient(),
        otx_client.OTXClient(),
        nvd_client.NVDClient(),
        imf_client.IMFClient(),
        nasa_client.NASAClient(),
    ]

    results = []
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        for client in clients:
            try:
                health = loop.run_until_complete(client.health_check())
                results.append(health)
            except Exception as e:
                results.append({
                    'source': client.source_name,
                    'status': 'ERROR',
                    'error': str(e)
                })
    finally:
        loop.close()

    # Store results
    with get_session_context() as session:
        for result in results:
            health_record = DataSourceHealth(
                source_name=result.get('source', 'UNKNOWN'),
                check_time=datetime.utcnow(),
                status=result.get('status', 'UNKNOWN'),
                response_time_ms=result.get('response_time_ms'),
                error_message=result.get('error')
            )
            session.add(health_record)

    logger.info(f"Health checks complete: {len(results)} sources checked")
    return results


@celery_app.task
def cleanup_old_data(days_to_keep: int = 365):
    """
    Clean up old calculation data to manage database size.

    Args:
        days_to_keep: Number of days of history to retain
    """
    from database.connection import get_session_context
    from database.models import RiskProbability, IndicatorValue, DataSourceHealth
    from datetime import datetime, timedelta

    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
    logger.info(f"Cleaning up data older than {cutoff_date}")

    with get_session_context() as session:
        # Delete old probability records (keep at least weekly snapshots)
        old_probs = session.query(RiskProbability).filter(
            RiskProbability.calculation_date < cutoff_date
        ).delete()

        # Delete old indicator values
        old_indicators = session.query(IndicatorValue).filter(
            IndicatorValue.timestamp < cutoff_date
        ).delete()

        # Delete old health checks
        old_health = session.query(DataSourceHealth).filter(
            DataSourceHealth.check_time < cutoff_date
        ).delete()

        logger.info(
            f"Cleanup complete: {old_probs} probabilities, "
            f"{old_indicators} indicators, {old_health} health records deleted"
        )

    return {
        'probabilities_deleted': old_probs,
        'indicators_deleted': old_indicators,
        'health_records_deleted': old_health
    }


@celery_app.task
def recalculate_event(event_id: str):
    """Recalculate a single event's probability."""
    return run_calculation_task([event_id])


# Allow running celery worker from command line
if __name__ == '__main__':
    celery_app.start()
