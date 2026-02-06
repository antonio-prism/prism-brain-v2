"""
PRISM Brain Database Models
SQLAlchemy ORM models for PostgreSQL.
Updated with Phase 4B-4E columns for signals, explainability, and dependencies.
Column names aligned with main.py code expectations.
"""

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, Boolean, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class RiskEvent(Base):
    """Risk event with 905 events across multiple categories."""
    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(50), unique=True, nullable=False, index=True)
    event_name = Column(String(500), nullable=False)
    layer1_primary = Column(String(100))
    layer1_secondary = Column(String(100))
    layer2_primary = Column(String(100))
    layer2_secondary = Column(String(100))
    baseline_probability = Column(Float, default=0.5)
    baseline_1_5 = Column(Float, default=3.0)
    super_risk = Column(Boolean, default=False)
    baseline_impact = Column(Float)
    methodology_tier = Column(String(50))
    geographic_scope = Column(String(100))
    time_horizon = Column(String(100))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RiskProbability(Base):
    """Calculated probability for a risk event - one row per calculation run."""
    __tablename__ = "risk_probabilities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(50), nullable=False, index=True)
    probability_pct = Column(Float)
    confidence_score = Column(Float)
    calculation_date = Column(DateTime, default=datetime.utcnow)
    data_sources_used = Column(Integer, default=0)
    flags = Column(Text)
    ci_lower_pct = Column(Float)
    ci_upper_pct = Column(Float)
    change_direction = Column(String(20))
    methodology_tier = Column(String(50))
    precision_band = Column(String(50))
    log_odds = Column(Float)
    total_adjustment = Column(Float)
    indicators_used = Column(Integer, default=0)
    calculation_method = Column(String(50), default="bayesian")
    calculation_id = Column(String(50))
    baseline_probability_pct = Column(Float)
    ci_level = Column(Float)
    ci_width_pct = Column(Float)
    bootstrap_iterations = Column(Integer)

    # Phase 4B: Signal extraction columns
    signal = Column(Float)
    momentum = Column(Float)
    trend = Column(String(50))
    is_anomaly = Column(Boolean, default=False)

    # Phase 4C: ML enhancement columns
    ensemble_method = Column(String(50))
    ml_probability_pct = Column(Float)

    # Phase 4D: Explainability columns
    attribution = Column(JSON)
    explanation = Column(Text)
    recommendation = Column(JSON)
    previous_probability_pct = Column(Float)
    probability_change_pct = Column(Float)

    # Phase 4E: Dependency columns
    dependency_adjustment = Column(Float)
    dependency_details = Column(JSON)


class IndicatorWeight(Base):
    """Weight configuration for each indicator per event."""
    __tablename__ = "indicator_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(50), nullable=False, index=True)
    indicator_name = Column(String(200), nullable=False)
    weight = Column(Float, default=1.0)
    normalized_weight = Column(Float)
    data_source = Column(String(100))
    beta_type = Column(String(50))
    time_scale = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class IndicatorValue(Base):
    """Time-series storage for indicator values - appended, never overwritten."""
    __tablename__ = "indicator_values"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(50), nullable=False, index=True)
    indicator_name = Column(String(200), nullable=False)
    value = Column(Float)
    raw_value = Column(Float)
    z_score = Column(Float)
    data_source = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)
    quality_score = Column(Float)
    historical_mean = Column(Float)
    historical_std = Column(Float)

    # Phase 4B: Signal extraction columns
    signal = Column(Float)
    momentum = Column(Float)
    trend = Column(String(20))
    is_anomaly = Column(Boolean, default=False)


class DataSourceHealth(Base):
    """Health tracking for external data sources."""
    __tablename__ = "data_source_health"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_name = Column(String(100), nullable=False, index=True)
    status = Column(String(50), default="unknown")
    last_success = Column(DateTime)
    last_failure = Column(DateTime)
    error_message = Column(Text)
    response_time_ms = Column(Integer)
    records_fetched = Column(Integer, default=0)
    check_time = Column(DateTime, default=datetime.utcnow)
    success_rate_24h = Column(Float)


class CalculationLog(Base):
    """Audit log for probability calculation runs."""
    __tablename__ = "calculation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    calculation_id = Column(String(50), unique=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    events_processed = Column(Integer, default=0)
    events_succeeded = Column(Integer, default=0)
    events_failed = Column(Integer, default=0)
    duration_seconds = Column(Float)
    status = Column(String(50), default="running")
    errors = Column(Text)
    trigger = Column(String(50), default="manual")
    method = Column(String(50), default="bayesian")
