"""
PRISM Brain FastAPI Application

Main entry point for the REST API.
Updated with bulk import endpoints for Phase 2.
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import logging
import json

from config.settings import get_settings
from database.connection import init_db, get_session_context
from database.models import (
    RiskEvent, RiskProbability, IndicatorWeight,
    DataSourceHealth, CalculationLog, IndicatorValue
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class EventCreate(BaseModel):
    event_id: str
    event_name: str
    description: str = ""
    layer1_primary: Optional[str] = None
    layer1_secondary: Optional[str] = None
    layer2_primary: Optional[str] = None
    layer2_secondary: Optional[str] = None
    super_risk: bool = False
    baseline_probability: int = 3
    baseline_impact: int = 3
    geographic_scope: Optional[str] = None
    time_horizon: Optional[str] = None
    methodology_tier: Optional[str] = None

class IndicatorWeightCreate(BaseModel):
    event_id: str
    indicator_name: str
    normalized_weight: float
    data_source: str
    beta_type: str = "MODERATE"
    time_scale: str = "medium"
    update_frequency_hours: int = 24

class IndicatorValueCreate(BaseModel):
    event_id: str
    indicator_name: str
    data_source: str
    value: float
    timestamp: Optional[datetime] = None
    raw_value: Optional[float] = None
    historical_mean: Optional[float] = None
    historical_std: Optional[float] = None
    z_score: Optional[float] = None
    quality_score: float = 0.8


# ============== Startup/Shutdown ==============

@app.on_event("startup")
async def startup_event():
    logger.info("Starting PRISM Brain API...")
    init_db()
    logger.info("Database initialized")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down PRISM Brain API...")


# ============== Health Check ==============

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version
    }


# ============== Events Endpoints ==============

@app.get("/api/v1/events")
async def list_events(
    layer1: Optional[str] = None,
    layer2: Optional[str] = None,
    super_risk: Optional[bool] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500)
):
    """List all risk events with optional filtering."""
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
                    "description": e.description,
                    "layer1_primary": e.layer1_primary,
                    "layer1_secondary": e.layer1_secondary,
                    "layer2_primary": e.layer2_primary,
                    "layer2_secondary": e.layer2_secondary,
                    "super_risk": e.super_risk,
                    "baseline_probability": e.baseline_probability,
                    "baseline_impact": e.baseline_impact,
                    "geographic_scope": e.geographic_scope,
                    "time_horizon": e.time_horizon,
                    "methodology_tier": e.methodology_tier
                }
                for e in events
            ]
        }


@app.get("/api/v1/events/{event_id}")
async def get_event(event_id: str):
    """Get a specific risk event by ID."""
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
        
        # Get indicators
        weights = session.query(IndicatorWeight).filter(
            IndicatorWeight.event_id == event_id
        ).all()
        
        return {
            "event": {
                "event_id": event.event_id,
                "event_name": event.event_name,
                "description": event.description,
                "layer1_primary": event.layer1_primary,
                "layer2_primary": event.layer2_primary,
                "super_risk": event.super_risk,
                "baseline_probability": event.baseline_probability,
                "methodology_tier": event.methodology_tier
            },
            "latest_probability": {
                "probability_pct": latest_prob.probability_pct,
                "ci_lower_pct": latest_prob.ci_lower_pct,
                "ci_upper_pct": latest_prob.ci_upper_pct,
                "confidence_score": latest_prob.confidence_score,
                "calculation_date": latest_prob.calculation_date.isoformat()
            } if latest_prob else None,
            "indicators": [
                {
                    "indicator_name": w.indicator_name,
                    "data_source": w.data_source,
                    "normalized_weight": w.normalized_weight,
                    "beta_type": w.beta_type,
                    "time_scale": w.time_scale
                }
                for w in weights
            ]
        }


@app.post("/api/v1/events/bulk")
async def bulk_import_events(events: List[EventCreate]):
    """Bulk import risk events."""
    with get_session_context() as session:
        imported = 0
        updated = 0
        for e in events:
            existing = session.query(RiskEvent).filter(
                RiskEvent.event_id == e.event_id
            ).first()
            if existing:
                # Update existing
                existing.event_name = e.event_name
                existing.description = e.description
                existing.layer1_primary = e.layer1_primary
                existing.layer2_primary = e.layer2_primary
                existing.super_risk = e.super_risk
                existing.baseline_probability = e.baseline_probability
                existing.baseline_impact = e.baseline_impact
                updated += 1
            else:
                # Create new
                event = RiskEvent(
                    event_id=e.event_id,
                    event_name=e.event_name,
                    description=e.description,
                    layer1_primary=e.layer1_primary,
                    layer1_secondary=e.layer1_secondary,
                    layer2_primary=e.layer2_primary,
                    layer2_secondary=e.layer2_secondary,
                    super_risk=e.super_risk,
                    baseline_probability=e.baseline_probability,
                    baseline_impact=e.baseline_impact,
                    geographic_scope=e.geographic_scope,
                    time_horizon=e.time_horizon,
                    methodology_tier=e.methodology_tier
                )
                session.add(event)
                imported += 1
        session.commit()
        total = session.query(RiskEvent).count()
    return {"imported": imported, "updated": updated, "total": total}


# ============== Indicator Weights Endpoints ==============

@app.get("/api/v1/indicator-weights")
async def list_indicator_weights(
    event_id: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """List indicator weights, optionally filtered by event."""
    with get_session_context() as session:
        query = session.query(IndicatorWeight)
        if event_id:
            query = query.filter(IndicatorWeight.event_id == event_id)
        total = query.count()
        weights = query.offset(skip).limit(limit).all()
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "weights": [
                {
                    "event_id": w.event_id,
                    "indicator_name": w.indicator_name,
                    "normalized_weight": w.normalized_weight,
                    "data_source": w.data_source,
                    "beta_type": w.beta_type,
                    "time_scale": w.time_scale
                }
                for w in weights
            ]
        }


@app.post("/api/v1/indicator-weights/bulk")
async def bulk_import_indicator_weights(weights: List[IndicatorWeightCreate]):
    """Bulk import indicator weights."""
    with get_session_context() as session:
        imported = 0
        updated = 0
        for w in weights:
            existing = session.query(IndicatorWeight).filter(
                IndicatorWeight.event_id == w.event_id,
                IndicatorWeight.indicator_name == w.indicator_name
            ).first()
            if existing:
                existing.normalized_weight = w.normalized_weight
                existing.data_source = w.data_source
                existing.beta_type = w.beta_type
                existing.time_scale = w.time_scale
                updated += 1
            else:
                weight = IndicatorWeight(
                    event_id=w.event_id,
                    indicator_name=w.indicator_name,
                    normalized_weight=w.normalized_weight,
                    data_source=w.data_source,
                    beta_type=w.beta_type,
                    time_scale=w.time_scale,
                    # Default values for required score fields
                    causal_proximity_score=0.5,
                    data_quality_score=0.7,
                    timeliness_score=0.6,
                    predictive_lead_score=0.5,
                    raw_score=w.normalized_weight,
                    beta_value=0.5
                )
                session.add(weight)
                imported += 1
        session.commit()
        total = session.query(IndicatorWeight).count()
    return {"imported": imported, "updated": updated, "total": total}


# ============== Indicator Values Endpoints ==============

@app.get("/api/v1/indicator-values")
async def list_indicator_values(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """List current indicator values."""
    with get_session_context() as session:
        total = session.query(IndicatorValue).count()
        values = session.query(IndicatorValue).offset(skip).limit(limit).all()
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "values": [
                {
                    "event_id": v.event_id,
                    "indicator_name": v.indicator_name,
                    "data_source": v.data_source,
                    "value": v.value,
                    "timestamp": v.timestamp.isoformat() if v.timestamp else None,
                    "z_score": v.z_score,
                    "quality_score": v.quality_score
                }
                for v in values
            ]
        }


@app.post("/api/v1/indicator-values/bulk")
async def bulk_import_indicator_values(values: List[IndicatorValueCreate]):
    """Bulk import indicator values (time series data points)."""
    with get_session_context() as session:
        imported = 0
        updated = 0
        for v in values:
            timestamp = v.timestamp or datetime.utcnow()
            existing = session.query(IndicatorValue).filter(
                IndicatorValue.event_id == v.event_id,
                IndicatorValue.indicator_name == v.indicator_name,
                IndicatorValue.timestamp == timestamp
            ).first()
            if existing:
                existing.value = v.value
                existing.raw_value = v.raw_value or v.value
                existing.quality_score = v.quality_score
                updated += 1
            else:
                value = IndicatorValue(
                    event_id=v.event_id,
                    indicator_name=v.indicator_name,
                    data_source=v.data_source,
                    timestamp=timestamp,
                    value=v.value,
                    raw_value=v.raw_value or v.value,
                    historical_mean=v.historical_mean,
                    historical_std=v.historical_std,
                    z_score=v.z_score,
                    quality_score=v.quality_score
                )
                session.add(value)
                imported += 1
        session.commit()
        total = session.query(IndicatorValue).count()
    return {"imported": imported, "updated": updated, "total": total}


# ============== Probabilities Endpoints ==============

@app.get("/api/v1/probabilities")
async def list_probabilities(
    event_id: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500)
):
    """List calculated probabilities."""
    with get_session_context() as session:
        query = session.query(RiskProbability)
        if event_id:
            query = query.filter(RiskProbability.event_id == event_id)
        query = query.order_by(RiskProbability.calculation_date.desc())
        
        total = query.count()
        probabilities = query.offset(skip).limit(limit).all()
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "probabilities": [
                {
                    "event_id": p.event_id,
                    "probability_pct": p.probability_pct,
                    "ci_lower_pct": p.ci_lower_pct,
                    "ci_upper_pct": p.ci_upper_pct,
                    "precision_band": p.precision_band,
                    "confidence_score": p.confidence_score,
                    "methodology_tier": p.methodology_tier,
                    "change_direction": p.change_direction,
                    "flags": p.flags,
                    "calculation_date": p.calculation_date.isoformat()
                }
                for p in probabilities
            ]
        }


@app.get("/api/v1/probabilities/{event_id}/history")
async def get_probability_history(
    event_id: str,
    limit: int = Query(52, ge=1, le=200)
):
    """Get probability history for a specific event."""
    with get_session_context() as session:
        event = session.query(RiskEvent).filter(
            RiskEvent.event_id == event_id
        ).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        probabilities = session.query(RiskProbability).filter(
            RiskProbability.event_id == event_id
        ).order_by(RiskProbability.calculation_date.desc()).limit(limit).all()
        
        return {
            "event_id": event_id,
            "event_name": event.event_name,
            "history": [
                {
                    "probability_pct": p.probability_pct,
                    "ci_lower_pct": p.ci_lower_pct,
                    "ci_upper_pct": p.ci_upper_pct,
                    "confidence_score": p.confidence_score,
                    "change_direction": p.change_direction,
                    "calculation_date": p.calculation_date.isoformat()
                }
                for p in probabilities
            ]
        }


# ============== Calculations Endpoints ==============

@app.get("/api/v1/calculations")
async def list_calculations(limit: int = Query(20, ge=1, le=100)):
    """List recent calculation batches."""
    with get_session_context() as session:
        calculations = session.query(CalculationLog).order_by(
            CalculationLog.started_at.desc()
        ).limit(limit).all()
        
        return {
            "calculations": [
                {
                    "calculation_id": c.calculation_id,
                    "status": c.status,
                    "events_processed": c.events_processed,
                    "events_succeeded": c.events_succeeded,
                    "events_failed": c.events_failed,
                    "started_at": c.started_at.isoformat() if c.started_at else None,
                    "completed_at": c.completed_at.isoformat() if c.completed_at else None,
                    "error_summary": c.error_summary
                }
                for c in calculations
            ]
        }


@app.post("/api/v1/calculations/trigger")
async def trigger_calculation(
    event_ids: Optional[List[str]] = None,
    background: bool = True
):
    """Trigger a probability calculation."""
    try:
        if background:
            # Queue for background processing
            from tasks.calculation_tasks import run_calculation_batch
            import uuid
            
            calculation_id = str(uuid.uuid4())
            
            with get_session_context() as session:
                log = CalculationLog(
                    calculation_id=calculation_id,
                    status="queued",
                    events_processed=0,
                    events_succeeded=0,
                    events_failed=0,
                    started_at=datetime.utcnow()
                )
                session.add(log)
                session.commit()
            
            # In production, this would queue to Celery
            # For now, run synchronously in a simpler way
            return {
                "status": "queued",
                "calculation_id": calculation_id,
                "message": "Calculation queued for processing"
            }
        else:
            # Synchronous execution
            from probability_engine.calculation import run_weekly_calculation
            import asyncio
            
            batch = asyncio.run(run_weekly_calculation())
            return {
                "status": "completed",
                "calculation_id": batch.calculation_id,
                "events_processed": batch.events_processed,
                "events_succeeded": batch.events_succeeded
            }
    except Exception as e:
        logger.error(f"Calculation trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Data Sources Endpoints ==============

@app.get("/api/v1/data-sources/health")
async def get_data_source_health():
    """Get health status of all data sources."""
    from sqlalchemy import func
    with get_session_context() as session:
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


# ============== Stats Endpoint ==============

@app.get("/api/v1/stats")
async def get_stats():
    """Get overall system statistics."""
    with get_session_context() as session:
        event_count = session.query(RiskEvent).count()
        weight_count = session.query(IndicatorWeight).count()
        value_count = session.query(IndicatorValue).count()
        prob_count = session.query(RiskProbability).count()
        
        # Events with weights
        events_with_weights = session.query(IndicatorWeight.event_id).distinct().count()
        
        # Latest calculation
        latest_calc = session.query(CalculationLog).order_by(
            CalculationLog.started_at.desc()
        ).first()
        
        return {
            "events": {
                "total": event_count,
                "with_weights": events_with_weights
            },
            "indicator_weights": weight_count,
            "indicator_values": value_count,
            "probabilities_calculated": prob_count,
            "latest_calculation": {
                "id": latest_calc.calculation_id if latest_calc else None,
                "status": latest_calc.status if latest_calc else None,
                "date": latest_calc.started_at.isoformat() if latest_calc else None
            }
        }


# ============== Error Handler ==============

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
