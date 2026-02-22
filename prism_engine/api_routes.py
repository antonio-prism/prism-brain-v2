"""
PRISM Engine — FastAPI route integration.

Adds endpoints that expose the probability engine to the existing app:
  GET  /api/v2/engine/compute/{event_id}  — Compute single event
  GET  /api/v2/engine/compute-all         — Compute all 174 events
  GET  /api/v2/engine/compute-phase1      — Compute 10 Phase 1 events only
  GET  /api/v2/engine/status              — Engine health/status
  GET  /api/v2/engine/fallback-rates      — List all fallback rates
  GET  /api/v2/engine/annual-data         — Get current annual update data
  PUT  /api/v2/engine/annual-data         — Save annual update data
  GET  /api/v2/engine/method-c-status     — Method C integration status
  POST /api/v2/engine/method-c-integrate  — Integrate research from file
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query, Request

logger = logging.getLogger(__name__)


def register_engine_routes(app: FastAPI):
    """Register prism_engine API endpoints on the FastAPI app."""

    @app.get("/api/v2/engine/compute/{event_id}")
    async def compute_event(event_id: str):
        """Compute the probability for a single event using the engine."""
        from prism_engine.engine import compute
        result = await asyncio.to_thread(compute, event_id)
        return result

    @app.get("/api/v2/engine/compute-all")
    async def compute_all_events(domain: Optional[str] = Query(None)):
        """Compute all 174 events. Optionally filter by domain.

        Runs in a thread pool so it doesn't block the event loop — other
        endpoints (health, events, etc.) remain responsive while this runs.
        """
        from prism_engine.engine import compute_all
        results = await asyncio.to_thread(compute_all)

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

    # ── Annual data management endpoints ────────────────────────────────

    @app.get("/api/v2/engine/annual-data")
    async def get_annual_data():
        """Get current annual update data (DBIR rates, Dragos stats, dark figures)."""
        from prism_engine.annual_data import load_annual_data
        return load_annual_data()

    @app.get("/api/v2/engine/era5-calibration")
    async def era5_calibration():
        """Run ERA5 temperature scaling regression and return results."""
        from prism_engine.computation.era5_calibration import run_scaling_regression
        return run_scaling_regression()

    @app.put("/api/v2/engine/annual-data")
    async def save_annual_data_endpoint(request: Request):
        """Save annual update data from the manual entry page."""
        from prism_engine.annual_data import save_annual_data
        body = await request.json()
        success = save_annual_data(body)
        if success:
            return {"status": "saved", "message": "Annual data updated successfully"}
        return {"status": "error", "message": "Failed to save annual data"}

    # ── Method C research integration ─────────────────────────────────

    @app.get("/api/v2/engine/method-c-status")
    async def method_c_status():
        """Check Method C research integration status."""
        from prism_engine.method_c_loader import OVERRIDES_PATH, FULL_RESEARCH_PATH
        import json
        from pathlib import Path

        result = {"overrides_loaded": False, "override_count": 0}

        if OVERRIDES_PATH.exists():
            try:
                with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                overrides = data.get("overrides", {})
                metadata = data.get("metadata", {})
                result["overrides_loaded"] = True
                result["override_count"] = len(overrides)
                result["schema_version"] = metadata.get("version", "unknown")
                result["overrides_file"] = str(OVERRIDES_PATH)

                # Confidence distribution
                conf_dist = {}
                for ev in overrides.values():
                    c = ev.get("confidence", "Unknown")
                    conf_dist[c] = conf_dist.get(c, 0) + 1
                result["confidence_distribution"] = conf_dist
            except Exception as e:
                result["error"] = str(e)

        result["full_research_available"] = FULL_RESEARCH_PATH.exists()
        result["timestamp"] = datetime.utcnow().isoformat() + "Z"
        return result

    @app.post("/api/v2/engine/method-c-integrate")
    async def method_c_integrate(path: Optional[str] = Query(None)):
        """Integrate Method C research from a file on disk.

        Uses the default research file (method_c_v3_complete.json) unless
        a custom path is provided via query parameter.
        """
        from prism_engine.method_c_loader import load_and_integrate
        stats = await asyncio.to_thread(load_and_integrate, path)
        status = "success" if stats["integrated"] > 0 else "error"
        return {"status": status, "stats": stats}

    @app.post("/api/v2/engine/load-method-c-research")
    async def load_method_c_research(request: Request):
        """Load and integrate Method C research output JSON.

        Accepts the JSON body directly (same schema as method_c_research_output.json).
        Validates, integrates, and returns stats.
        """
        from prism_engine.method_c_loader import load_research_output, integrate_research
        import tempfile
        from pathlib import Path

        body = await request.json()

        # Write to temp file and validate
        tmp = Path(tempfile.mktemp(suffix=".json"))
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                import json
                json.dump(body, f)
            data, errors = load_research_output(tmp)
        finally:
            tmp.unlink(missing_ok=True)

        if data is None:
            return {"status": "error", "errors": errors}

        if errors:
            return {
                "status": "warning",
                "message": f"Loaded with {len(errors)} validation warnings",
                "errors": errors[:20],
                "stats": integrate_research(data),
            }

        return {
            "status": "success",
            "stats": integrate_research(data),
        }

    logger.info("Registered prism_engine API routes at /api/v2/engine/*")
