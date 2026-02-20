"""
PRISM Engine — FastAPI route integration.

Adds endpoints that expose the probability engine to the existing app:
  GET  /api/v2/engine/compute/{event_id}  — Compute single event
  GET  /api/v2/engine/compute-all         — Compute all 174 events
  GET  /api/v2/engine/compute-phase1      — Compute 10 Phase 1 events only
  GET  /api/v2/engine/status              — Engine health/status
  GET  /api/v2/engine/fallback-rates      — List all fallback rates
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query

logger = logging.getLogger(__name__)


def register_engine_routes(app: FastAPI):
    """Register prism_engine API endpoints on the FastAPI app."""

    @app.get("/api/v2/engine/compute/{event_id}")
    async def compute_event(event_id: str):
        """Compute the probability for a single event using the engine."""
        from prism_engine.engine import compute
        result = compute(event_id)
        return result

    @app.get("/api/v2/engine/compute-all")
    async def compute_all_events(domain: Optional[str] = Query(None)):
        """Compute all 174 events. Optionally filter by domain."""
        from prism_engine.engine import compute_all
        results = compute_all()

        # Optional domain filter
        if domain:
            domain_lower = domain.lower()
            results = {
                eid: r for eid, r in results.items()
                if r.get("domain", "").lower() == domain_lower
            }

        # Compute summary metrics
        methods = {"A": 0, "B": 0, "C": 0, "FALLBACK": 0}
        for r in results.values():
            m = r.get("layer1", {}).get("method", "FALLBACK")
            methods[m] = methods.get(m, 0) + 1

        return {
            "computed_at": datetime.utcnow().isoformat() + "Z",
            "event_count": len(results),
            "methods": methods,
            "domain_filter": domain,
            "events": results,
        }

    @app.get("/api/v2/engine/compute-phase1")
    async def compute_phase1():
        """Compute only the 10 Phase 1 prototype events."""
        from prism_engine.engine import compute_all_phase1
        results = compute_all_phase1()
        return {
            "computed_at": datetime.utcnow().isoformat() + "Z",
            "event_count": len(results),
            "events": results,
        }

    @app.get("/api/v2/engine/status")
    async def engine_status():
        """Check the engine's health and data source availability."""
        from prism_engine.config.credentials import check_all_credentials, NO_KEY_REQUIRED
        from prism_engine.fallback import load_fallback_rates
        from prism_engine.config.event_mapping import get_phase1_event_ids, get_all_event_ids

        credentials = check_all_credentials()
        fallback_rates = load_fallback_rates()
        all_ids = get_all_event_ids()

        return {
            "engine_version": "2.0.0",
            "spec_version": "2.3",
            "phase": "2 (all 174 events)",
            "total_events": len(all_ids),
            "phase1_events": get_phase1_event_ids(),
            "fallback_rates_loaded": len(fallback_rates),
            "api_credentials": credentials,
            "no_key_sources": NO_KEY_REQUIRED,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    @app.get("/api/v2/engine/fallback-rates")
    async def get_fallback_rates():
        """List all 174 fallback rates from the seed files."""
        from prism_engine.fallback import load_fallback_rates
        rates = load_fallback_rates()
        return {
            "count": len(rates),
            "rates": rates,
        }

    logger.info("Registered prism_engine API routes at /api/v2/engine/*")
