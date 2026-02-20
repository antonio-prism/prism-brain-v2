"""
PRISM Brain FastAPI Application - v3.0.0

Slim entry point for the REST API.
Phase 1-3: Database, API, Railway deployment, 905 events, probability engine.
Phase 4A-4E: Signal extraction, ML enhancement, explainability, risk modeling.

Removed endpoints (not used by frontend):
- Single event detail, bulk events, bulk weights, bulk values
- Indicator weights listing, indicator values listing
- Probability history, attribution, explanation, dependencies (endpoints removed, but model kept)
- Dashboard summary, calculations listing, legacy trigger
- Data sources listing (redundant with data-sources/health)
- Dashboard HTML page (static file)

Kept endpoints (used by frontend):
- GET /health
- GET /api/v1/events (list events)
- GET /api/v1/probabilities (list probabilities)
- GET /api/v1/data-sources/health
- POST /api/v1/data/refresh
- POST /api/v1/calculations/trigger-full
- GET /api/v1/stats
"""

import logging

# Load .env file BEFORE any other imports that use environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on system environment variables

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

from config.settings import get_settings
from database.connection import init_db, get_session_context
from routes.events import register_events_routes
from routes.calculations import register_calculations_routes
from routes.data_sources import register_data_sources_routes
from client_routes import register_client_routes
from v2_routes import register_v2_routes
from prism_engine.api_routes import register_engine_routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version="3.0.0",
    description="Probability calculation engine for 900 risk events with signal extraction, ML enhancement, and explainability",
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

# Register all route modules
register_client_routes(app, get_session_context)
register_v2_routes(app, get_session_context)
register_events_routes(app, get_session_context)
register_calculations_routes(app, get_session_context)
register_data_sources_routes(app, get_session_context)
register_engine_routes(app)


# ============== Health Check ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0",
        "features": [
            "signal_extraction", "explainability",
            "dependency_modeling", "enhanced_confidence"
        ]
    }


# ============== Schema Migration ==============

def ensure_schema_updates(session):
    """
    Add new columns to database tables if they don't exist.
    Uses ALTER TABLE with IF NOT EXISTS logic.
    """
    alter_statements = [
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS attribution JSON",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS explanation TEXT",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS recommendation JSON",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS dependency_adjustment FLOAT",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS dependency_details JSON",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS ensemble_method VARCHAR(50)",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS ml_probability_pct FLOAT",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS previous_probability_pct FLOAT",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS probability_change_pct FLOAT",
        "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS signal FLOAT",
        "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS momentum FLOAT",
        "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS trend VARCHAR(20)",
        "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS is_anomaly BOOLEAN DEFAULT FALSE",
    ]
    for stmt in alter_statements:
        try:
            session.execute(stmt)
        except Exception as e:
            # Column might already exist or DB might not support IF NOT EXISTS
            logger.debug(f"Schema update note: {e}")
    try:
        session.commit()
    except Exception:
        session.rollback()


# ============== Startup/Shutdown ==============

@app.on_event("startup")
async def startup_event():
    """Initialize database and run schema migrations on startup."""
    logger.info("Starting PRISM Brain API v3.0.0...")
    init_db()
    logger.info("Database initialized")
    # Run schema migrations
    try:
        with get_session_context() as session:
            ensure_schema_updates(session)
            logger.info("Schema updates applied")
    except Exception as e:
        logger.warning(f"Schema update warning (may be OK on first run): {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown."""
    logger.info("Shutting down PRISM Brain API...")


# ============== Error Handler ==============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
