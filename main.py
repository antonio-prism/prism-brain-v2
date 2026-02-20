"""
PRISM Brain FastAPI Application - v3.1.0

Slim entry point for the REST API.

Active route groups:
- GET  /health                     — Health check
- /api/v1/clients/*                — Client CRUD (client_routes.py)
- /api/v2/*                        — V2 taxonomy, events, probabilities (v2_routes.py)
- /api/v2/engine/*                 — Probability engine (prism_engine/api_routes.py)

Legacy V1 data/calculation routes have been retired — the probability engine
(prism_engine) replaces them entirely.
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
    version="3.1.0",
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
register_engine_routes(app)


# ============== Health Check ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.1.0",
        "features": [
            "probability_engine", "annual_data_updates",
            "method_c_research", "era5_calibration"
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
    logger.info("Starting PRISM Brain API v3.1.0...")
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
