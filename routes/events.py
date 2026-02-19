"""
Event listing endpoints (V1 API).

Kept endpoints:
- GET /api/v1/events (list events) - used by frontend
"""

from fastapi import Query, HTTPException
from typing import Optional
from database.models import RiskEvent
from database.connection import get_session_context


def register_events_routes(app, get_session_fn):
    """Register event endpoints on the FastAPI app."""

    @app.get("/api/v1/events")
    async def list_events(
        layer1: Optional[str] = None,
        layer2: Optional[str] = None,
        super_risk: Optional[bool] = None,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500)
    ):
        """List all risk events with optional filtering."""
        with get_session_fn() as session:
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
