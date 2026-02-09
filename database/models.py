"""
PRISM Brain Database Models
SQLAlchemy ORM models for PostgreSQL.
Updated with Phase 4B-4E columns for signals, explainability, and dependencies.
Column names aligned with main.py code expectations.
Phase 5 (Phase 2 Integration): Client data models for shared database.
Phase 3: Enhanced Dashboard models (trends, alerts, profiles, reports).
"""

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, Boolean, JSON, ForeignKey, Index
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


# =============================================================================
# Phase 2 Integration: Client Data Models
# =============================================================================

class Client(Base):
    """
    Client company profile.
    Stores company details needed for risk assessment workflows.
    """
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(500), nullable=False)
    location = Column(String(200))
    industry = Column(String(200))
    revenue = Column(Float)
    employees = Column(Integer)
    currency = Column(String(10), default='EUR')
    export_percentage = Column(Float, default=0)
    primary_markets = Column(Text)
    sectors = Column(Text)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ClientProcess(Base):
    """
    Business process assigned to a client.
    Uses APQC process framework codes (e.g., '4.1.2').
    criticality_per_day = euros lost per day of disruption.
    """
    __tablename__ = "client_processes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(Integer, ForeignKey('clients.id', ondelete='CASCADE'), nullable=False, index=True)
    process_id = Column(String(50), nullable=False)
    process_name = Column(String(500), nullable=False)
    custom_name = Column(String(500))
    category = Column(String(100))
    criticality_per_day = Column(Float, default=0)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_client_process', 'client_id', 'process_id'),
    )


class ClientRisk(Base):
    """
    Risk event selected for a client.
    Links to the 905 risk events in risk_events table via risk_id.
    """
    __tablename__ = "client_risks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(Integer, ForeignKey('clients.id', ondelete='CASCADE'), nullable=False, index=True)
    risk_id = Column(String(50), nullable=False)
    risk_name = Column(String(500), nullable=False)
    domain = Column(String(100))
    category = Column(String(200))
    probability = Column(Float, default=0.5)
    is_prioritized = Column(Boolean, default=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_client_risk', 'client_id', 'risk_id'),
    )


class ClientRiskAssessment(Base):
    """
    Assessment of a risk's impact on a specific client process.

    PRISM formula: Exposure = Criticality x Vulnerability x (1-Resilience) x Downtime x Probability
    """
    __tablename__ = "client_risk_assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(Integer, ForeignKey('clients.id', ondelete='CASCADE'), nullable=False, index=True)
    process_id = Column(Integer, ForeignKey('client_processes.id', ondelete='CASCADE'), nullable=False)
    risk_id = Column(Integer, ForeignKey('client_risks.id', ondelete='CASCADE'), nullable=False)
    vulnerability = Column(Float, default=0.5)
    resilience = Column(Float, default=0.3)
    expected_downtime = Column(Integer, default=5)
    notes = Column(Text)
    assessed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_assessment_combo', 'client_id', 'process_id', 'risk_id', unique=True),
    )


# =============================================================================
# Phase 3: Enhanced Dashboard Models
# =============================================================================

class ProbabilitySnapshot(Base):
    """Periodic snapshots of probability data for trend tracking and charting."""
    __tablename__ = 'probability_snapshots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(50), nullable=False, index=True)
    probability_pct = Column(Float)
    confidence_score = Column(Float)
    signal = Column(Float)
    momentum = Column(Float)
    trend = Column(String(50))
    snapshot_date = Column(DateTime, default=datetime.utcnow)
    calculation_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_event_snapshot_date', 'event_id', 'snapshot_date'),
    )


class ProbabilityAlert(Base):
    """Alert rules for probability threshold crossing."""
    __tablename__ = 'probability_alerts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(Integer, ForeignKey('clients.id', ondelete='CASCADE'), nullable=True, index=True)
    event_id = Column(String(50), nullable=False)
    alert_name = Column(String(200), nullable=False)
    threshold_pct = Column(Float, nullable=False)
    direction = Column(String(20), nullable=False)
    severity = Column(String(20), default='MEDIUM')
    is_active = Column(Boolean, default=True)
    notification_email = Column(String(500), nullable=True)
    last_triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_event_active', 'event_id', 'is_active'),
    )


class AlertEvent(Base):
    """Log of triggered alerts."""
    __tablename__ = 'alert_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(Integer, ForeignKey('probability_alerts.id', ondelete='CASCADE'), nullable=False, index=True)
    event_id = Column(String(50), nullable=False)
    triggered_value = Column(Float)
    previous_value = Column(Float)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_alert_triggered_date', 'alert_id', 'triggered_at'),
    )


class IndustryProfile(Base):
    """Pre-configured risk profiles by industry."""
    __tablename__ = 'industry_profiles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    industry = Column(String(200), nullable=False, index=True)
    profile_name = Column(String(200), nullable=False)
    description = Column(Text)
    is_template = Column(Boolean, default=True)
    created_by = Column(String(200), default='system')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_industry_template', 'industry', 'is_template'),
    )


class ProfileRiskEvent(Base):
    """Links industry profiles to relevant risk events."""
    __tablename__ = 'profile_risk_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey('industry_profiles.id', ondelete='CASCADE'), nullable=False, index=True)
    event_id = Column(String(50), nullable=False)
    relevance_score = Column(Float, default=1.0)
    weight_multiplier = Column(Float, default=1.0)
    category = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_profile_event', 'profile_id', 'event_id'),
    )


class ReportSchedule(Base):
    """Scheduled report configurations."""
    __tablename__ = 'report_schedules'

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(Integer, ForeignKey('clients.id', ondelete='CASCADE'), nullable=True, index=True)
    report_name = Column(String(200), nullable=False)
    report_type = Column(String(50), default='WEEKLY')
    report_format = Column(String(20), default='PDF')
    recipients = Column(Text)
    is_active = Column(Boolean, default=True)
    include_trends = Column(Boolean, default=True)
    include_alerts = Column(Boolean, default=True)
    include_recommendations = Column(Boolean, default=True)
    last_generated_at = Column(DateTime, nullable=True)
    next_scheduled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_client_active', 'client_id', 'is_active'),
    )


class ReportHistory(Base):
    """Log of generated reports."""
    __tablename__ = 'report_histories'

    id = Column(Integer, primary_key=True, autoincrement=True)
    schedule_id = Column(Integer, ForeignKey('report_schedules.id', ondelete='CASCADE'), nullable=True, index=True)
    client_id = Column(Integer, nullable=True)
    report_type = Column(String(50))
    report_format = Column(String(20))
    generated_at = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String(500))
    sent_to = Column(Text)
    delivery_status = Column(String(50), default='GENERATED')
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_schedule_generated', 'schedule_id', 'generated_at'),
    )
