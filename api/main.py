"""
PRISM Brain FastAPI Application

Main entry point for the REST API.
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import datetime
import logging

from config.settings import get_settings
from database.connection import init_db, get_session_context
from database.models import (
    RiskEvent, RiskProbability, IndicatorWeight,
    DataSourceHealth, CalculationLog
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Probability calculation engine for 900 risk events",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup."""
    logger.info("Starting PRISM Brain API...")
    init_db()
    logger.info("Database initialized")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/v1/events")
async def list_events(
    layer1: Optional[str] = None,
    layer2: Optional[str] = None,
    super_risk: Optional[bool] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500)
):
    """
    List all risk events with optional filtering.

    - **layer1**: Filter by Layer 1 classification
    - **layer2**: Filter by Layer 2 classification
    - **super_risk**: Filter for super-risk events only
    """
    with get_session_context() as session:
        query = session.query(RiskEvent)

        if layer1:
            query = query.filter(RiskEvent.layer1_primary == layer1)
        if layer2:
            query = query.filter(RiskEvent.layer2_primary == layer2)
        if super_risk is not None:
            query = query.filter(RiskEvent.super_risk == super_risk)

        total = query.count()
        events = query.offset(skip).limit(limit).all()

        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "events": [
                {
                    "event_id": e.event_id,
                    "event_name": e.event_name,
                    "layer1_primary": e.layer1_primary,
                    "layer2_primary": e.layer2_primary,
                    "super_risk": e.super_risk,
                    "baseline_probability": e.baseline_probability,
                    "methodology_tier": e.methodology_tier
                }
                for e in events
            ]
        }


@app.get("/api/v1/events/{event_id}")
async def get_event(event_id: str):
    """Get detailed information about a specific event."""
    with get_session_context() as session:
        event = session.query(RiskEvent).filter(
            RiskEvent.event_id == event_id
        ).first()

        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        # Get latest probability
        latest_prob = session.query(RiskProbability).filter(
            RiskProbability.event_id == event_id
        ).order_by(RiskProbability.calculation_date.desc()).first()

        # Get indicator weights
        weights = session.query(IndicatorWeight).filter(
            IndicatorWeight.event_id == event_id
        ).all()

        return {
            "event": {
                "event_id": event.event_id,
                "event_name": event.event_name,
                "description": event.description,
                "layer1_primary": event.layer1_primary,
                "layer1_secondary": event.layer1_secondary,
                "layer2_primary": event.layer2_primary,
                "layer2_secondary": event.layer2_secondary,
                "super_risk": event.super_risk,
                "baseline_probability": event.baseline_probability,
                "baseline_impact": event.baseline_impact,
                "geographic_scope": event.geographic_scope,
                "time_horizon": event.time_horizon,
                "methodology_tier": event.methodology_tier
            },
            "latest_probability": {
                "probability_pct": latest_prob.probability_pct,
                "ci_lower_pct": latest_prob.ci_lower_pct,
                "ci_upper_pct": latest_prob.ci_upper_pct,
                "precision_band": latest_prob.precision_band,
                "confidence_score": latest_prob.confidence_score,
                "calculation_date": latest_prob.calculation_date.isoformat(),
                "change_direction": latest_prob.change_direction,
                "flags": latest_prob.flags
            } if latest_prob else None,
            "indicators": [
                {
                    "name": w.indicator_name,
                    "data_source": w.data_source,
                    "normalized_weight": w.normalized_weight,
                    "beta_type": w.beta_type,
                    "time_scale": w.time_scale
                }
                for w in weights
            ]
        }


@app.get("/api/v1/probabilities")
async def list_probabilities(
    calculation_id: Optional[str] = None,
    min_probability: Optional[float] = None,
    flags: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500)
):
    """
    List probability calculations.

    - **calculation_id**: Filter by specific calculation run
    - **min_probability**: Filter for probabilities above threshold
    - **flags**: Filter by flag (e.g., "BLACK_SWAN", "CONFLICTING_SIGNALS")
    """
    with get_session_context() as session:
        query = session.query(RiskProbability)

        if calculation_id:
            query = query.filter(RiskProbability.calculation_id == calculation_id)
        if min_probability:
            query = query.filter(RiskProbability.probability_pct >= min_probability)

        # Order by most recent first
        query = query.order_by(
            RiskProbability.calculation_date.desc(),
            RiskProbability.probability_pct.desc()
        )

        total = query.count()
        probabilities = query.offset(skip).limit(limit).all()

        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "probabilities": [
                {
                    "event_id": p.event_id,
                    "calculation_id": p.calculation_id,
                    "calculation_date": p.calculation_date.isoformat(),
                    "probability_pct": p.probability_pct,
                    "ci_lower_pct": p.ci_lower_pct,
                    "ci_upper_pct": p.ci_upper_pct,
                    "precision_band": p.precision_band,
                    "confidence_score": p.confidence_score,
                    "methodology_tier": p.methodology_tier,
                    "change_direction": p.change_direction,
                    "flags": p.flags
                }
                for p in probabilities
            ]
        }


@app.get("/api/v1/probabilities/{event_id}/history")
async def get_probability_history(
    event_id: str,
    limit: int = Query(52, ge=1, le=200)  # Default to 1 year
):
    """Get probability history for a specific event."""
    with get_session_context() as session:
        # Verify event exists
        event = session.query(RiskEvent).filter(
            RiskEvent.event_id == event_id
        ).first()

        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        # Get history
        history = session.query(RiskProbability).filter(
            RiskProbability.event_id == event_id
        ).order_by(
            RiskProbability.calculation_date.desc()
        ).limit(limit).all()

        return {
            "event_id": event_id,
            "event_name": event.event_name,
            "history": [
                {
                    "calculation_id": p.calculation_id,
                    "calculation_date": p.calculation_date.isoformat(),
                    "probability_pct": p.probability_pct,
                    "ci_lower_pct": p.ci_lower_pct,
                    "ci_upper_pct": p.ci_upper_pct,
                    "change_direction": p.change_direction
                }
                for p in history
            ]
        }


@app.get("/api/v1/calculations")
async def list_calculations(
    status: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """List calculation runs."""
    with get_session_context() as session:
        query = session.query(CalculationLog)

        if status:
            query = query.filter(CalculationLog.status == status)

        query = query.order_by(CalculationLog.start_time.desc())
        calculations = query.limit(limit).all()

        return {
            "calculations": [
                {
                    "calculation_id": c.calculation_id,
                    "start_time": c.start_time.isoformat(),
                    "end_time": c.end_time.isoformat() if c.end_time else None,
                    "duration_seconds": c.duration_seconds,
                    "events_processed": c.events_processed,
                    "events_succeeded": c.events_succeeded,
                    "events_failed": c.events_failed,
                    "status": c.status
                }
                for c in calculations
            ]
        }


@app.get("/api/v1/data-sources/health")
async def get_data_source_health():
    """Get health status of all data sources."""
    with get_session_context() as session:
        # Get most recent health check for each source
        from sqlalchemy import func

        subquery = session.query(
            DataSourceHealth.source_name,
            func.max(DataSourceHealth.check_time).label('latest')
        ).group_by(DataSourceHealth.source_name).subquery()

        health_records = session.query(DataSourceHealth).join(
            subquery,
            (DataSourceHealth.source_name == subquery.c.source_name) &
            (DataSourceHealth.check_time == subquery.c.latest)
        ).all()

        return {
            "data_sources": [
                {
                    "source_name": h.source_name,
                    "status": h.status,
                    "check_time": h.check_time.isoformat(),
                    "response_time_ms": h.response_time_ms,
                    "success_rate_24h": h.success_rate_24h,
                    "error_message": h.error_message
                }
                for h in health_records
            ]
        }


@app.post("/api/v1/calculations/trigger")
async def trigger_calculation(
    event_ids: Optional[List[str]] = None,
    background: bool = True
):
    """
    Trigger a probability calculation.

    - **event_ids**: Optional list of specific events (None = all events)
    - **background**: Run in background (True) or synchronous (False)
    """
    from tasks.celery_app import run_calculation_task

    if background:
        task = run_calculation_task.delay(event_ids)
        return {
            "status": "queued",
            "task_id": task.id,
            "message": "Calculation triggered in background"
        }
    else:
        # Synchronous execution (for small batches)
        from probability_engine.calculation import run_weekly_calculation
        import asyncio

        batch = asyncio.run(run_weekly_calculation())
        return {
            "status": "completed",
            "calculation_id": batch.calculation_id,
            "events_processed": batch.events_processed,
            "events_succeeded": batch.events_succeeded
        }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
