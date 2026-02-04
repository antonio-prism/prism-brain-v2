"""
PRISM Brain Database Models

SQLAlchemy ORM models for the probability engine.
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, JSON, Text,
    Boolean, ForeignKey, Index, Enum, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class MethodologyTier(enum.Enum):
    """Methodology tier classification."""
    TIER_1_ML_ENHANCED = "TIER_1_ML_ENHANCED"
    TIER_2_ANALOG = "TIER_2_ANALOG"
    TIER_3_SCENARIO = "TIER_3_SCENARIO"


class PrecisionBand(enum.Enum):
    """Confidence interval precision classification."""
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class ChangeDirection(enum.Enum):
    """Probability change direction."""
    INCREASE = "INCREASE"
    DECREASE = "DECREASE"
    STABLE = "STABLE"


class RelationshipType(enum.Enum):
    """Causal dependency relationship type."""
    CAUSAL = "CAUSAL"
    ENABLING = "ENABLING"
    CORRELATED = "CORRELATED"
    WEAK = "WEAK"


class ConfidenceLevel(enum.Enum):
    """Confidence level for dependencies."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RiskEvent(Base):
    """
    Master table of all 900 risk events.
    This is static reference data loaded from risks_complete.json.
    """
    __tablename__ = 'risk_events'

    event_id = Column(String(20), primary_key=True)  # e.g., "GEO-001"
    event_name = Column(String(500), nullable=False)
    description = Column(Text)

    # Classification
    layer1_primary = Column(String(50))  # STRUCTURAL, PHYSICAL, DIGITAL, OPERATIONAL
    layer1_secondary = Column(String(255))
    layer2_primary = Column(String(200))
    layer2_secondary = Column(String(500))

    # Risk properties
    super_risk = Column(Boolean, default=False)
    affected_industries = Column(Text)
    geographic_scope = Column(String(100))
    time_horizon = Column(String(100))

    # Baseline values (1-5 scale from original data)
    baseline_probability = Column(Integer)  # 1-5 scale
    baseline_impact = Column(Integer)  # 1-5 scale

    # Categorization
    source_category = Column(String(200))

    # Methodology
    methodology_tier = Column(String(30), default="TIER_2_ANALOG")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    probabilities = relationship("RiskProbability", back_populates="event", cascade="all, delete-orphan")
    indicator_values = relationship("IndicatorValue", back_populates="event", cascade="all, delete-orphan")
    indicator_weights = relationship("IndicatorWeight", back_populates="event", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_risk_events_super_risk', 'super_risk'),
        Index('idx_risk_events_layer2_primary', 'layer2_primary'),
        Index('idx_risk_events_methodology_tier', 'methodology_tier'),
    )


class IndicatorWeight(Base):
    """
    Indicator weight derivations for each event.
    Loaded from event_indicator_weights.json.
    """
    __tablename__ = 'indicator_weights'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(20), ForeignKey('risk_events.event_id'), nullable=False)
    indicator_name = Column(String(100), nullable=False)
    data_source = Column(String(50), nullable=False)

    # Raw scores (1-5 scale)
    causal_proximity_score = Column(Float)
    data_quality_score = Column(Float)
    timeliness_score = Column(Float)
    predictive_lead_score = Column(Float)

    # Derived weight
    raw_score = Column(Float)
    normalized_weight = Column(Float, nullable=False)  # Must sum to 1.0 per event

    # Beta parameter type
    beta_type = Column(String(50))  # direct_causal, strong_correlation, etc.
    beta_value = Column(Float)

    # Time scale classification
    time_scale = Column(String(20))  # fast, medium, slow

    # Metadata
    justification = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    event = relationship("RiskEvent", back_populates="indicator_weights")

    __table_args__ = (
        UniqueConstraint('event_id', 'indicator_name', name='uq_event_indicator'),
        Index('idx_indicator_weights_event', 'event_id'),
    )


class CausalDependency(Base):
    """
    Causal dependency network between events.
    Loaded from dependency_network.yaml.
    """
    __tablename__ = 'causal_dependencies'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Relationship
    driver_event_id = Column(String(20), ForeignKey('risk_events.event_id'), nullable=False)
    dependent_event_id = Column(String(20), ForeignKey('risk_events.event_id'), nullable=False)

    # Dependency properties
    relationship_type = Column(String(20), nullable=False)  # CAUSAL, ENABLING, CORRELATED, WEAK
    multiplier = Column(Float, nullable=False)  # 1.0 to 10.0
    confidence = Column(String(10), nullable=False)  # HIGH, MEDIUM, LOW
    bidirectional = Column(Boolean, default=False)

    # Documentation
    rationale = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('driver_event_id', 'dependent_event_id', name='uq_dependency'),
        Index('idx_dependencies_driver', 'driver_event_id'),
        Index('idx_dependencies_dependent', 'dependent_event_id'),
    )


class RiskProbability(Base):
    """
    Calculated probabilities for each event (time series).
    New records are created each weekly calculation run.
    """
    __tablename__ = 'risk_probabilities'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(20), ForeignKey('risk_events.event_id'), nullable=False)
    calculation_id = Column(String(50), nullable=False)  # e.g., "2026-W06"
    calculation_date = Column(DateTime, nullable=False, index=True)

    # Probability values
    probability_pct = Column(Float, nullable=False)  # Final probability (0.1 to 99.9)
    log_odds = Column(Float)  # Log-odds value before conversion
    baseline_probability_pct = Column(Float)  # Poisson-derived baseline

    # Confidence interval (from bootstrap)
    ci_lower_pct = Column(Float)
    ci_upper_pct = Column(Float)
    ci_level = Column(Float, default=0.80)  # 80% CI
    ci_width_pct = Column(Float)
    precision_band = Column(String(20))  # HIGH, MODERATE, LOW, VERY_LOW
    bootstrap_iterations = Column(Integer, default=1000)

    # Confidence scoring
    confidence_score = Column(Float)  # 0.0 to 1.0
    data_completeness = Column(Float)
    source_agreement = Column(Float)
    data_recency = Column(Float)

    # Methodology
    methodology_tier = Column(String(30))
    ml_contribution_pct = Column(Float, default=0)  # % from ML vs Bayesian

    # Change tracking
    previous_probability_pct = Column(Float)
    probability_change_pct = Column(Float)
    change_direction = Column(String(20))  # INCREASE, DECREASE, STABLE
    change_significance = Column(String(20))  # MATERIAL, MINOR, NONE

    # Quality indicators
    flags = Column(JSON)  # ['BLACK_SWAN', 'CONFLICTING_SIGNALS', 'LOW_CONFIDENCE']
    data_quality = Column(JSON)  # Detailed quality metrics

    # Attribution (explainability)
    attribution = Column(JSON)  # Breakdown of contributing factors
    dependencies_applied = Column(JSON)  # List of dependency adjustments

    # Performance tracking
    calculation_duration_ms = Column(Integer)
    calculation_version = Column(String(20), default="2.0.0")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    event = relationship("RiskEvent", back_populates="probabilities")

    __table_args__ = (
        Index('idx_probabilities_calc_date', 'calculation_date'),
        Index('idx_probabilities_event_date', 'event_id', 'calculation_date'),
        Index('idx_probabilities_high_prob', 'probability_pct'),
    )


class IndicatorValue(Base):
    """
    Time series of indicator values extracted from data sources.
    """
    __tablename__ = 'indicator_values'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(20), ForeignKey('risk_events.event_id'), nullable=False)
    indicator_name = Column(String(100), nullable=False)
    data_source = Column(String(50), nullable=False)

    # Timestamp
    timestamp = Column(DateTime, nullable=False, index=True)

    # Values
    value = Column(Float, nullable=False)
    raw_value = Column(Float)  # Before any normalization

    # Statistical context
    historical_mean = Column(Float)
    historical_std = Column(Float)
    z_score = Column(Float)

    # Signal calculation
    signal = Column(Float)  # -1.0 to +1.0
    momentum = Column(Float)

    # Data quality
    quality_score = Column(Float)
    is_interpolated = Column(Boolean, default=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    event = relationship("RiskEvent", back_populates="indicator_values")

    __table_args__ = (
        Index('idx_indicator_values_event_time', 'event_id', 'timestamp'),
        Index('idx_indicator_values_source_time', 'data_source', 'timestamp'),
    )


class DataSourceHealth(Base):
    """
    Monitor health and performance of external data sources.
    """
    __tablename__ = 'data_source_health'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_name = Column(String(50), nullable=False, index=True)
    check_time = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Status
    status = Column(String(20), nullable=False)  # OPERATIONAL, DEGRADED, DOWN
    response_time_ms = Column(Integer)
    error_message = Column(Text)

    # Rate limiting
    requests_made = Column(Integer)
    rate_limit_remaining = Column(Integer)
    rate_limit_reset_time = Column(DateTime)

    # Quality metrics
    success_rate_24h = Column(Float)
    data_freshness_hours = Column(Float)

    __table_args__ = (
        Index('idx_health_source_time', 'source_name', 'check_time'),
    )


class CalculationLog(Base):
    """
    Audit trail of probability calculation runs.
    """
    __tablename__ = 'calculation_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    calculation_id = Column(String(50), nullable=False, unique=True, index=True)

    # Timing
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    duration_seconds = Column(Integer)

    # Results summary
    events_processed = Column(Integer)
    events_succeeded = Column(Integer)
    events_failed = Column(Integer)

    # Tier breakdown
    tier_1_count = Column(Integer)
    tier_2_count = Column(Integer)
    tier_3_count = Column(Integer)

    # Data source summary
    sources_queried = Column(JSON)
    total_api_calls = Column(Integer)
    failed_api_calls = Column(Integer)
    cache_hit_rate = Column(Float)

    # Flags summary
    black_swan_events = Column(Integer)
    conflicting_signal_events = Column(Integer)
    low_confidence_events = Column(Integer)

    # Errors
    errors = Column(JSON)

    # Status
    status = Column(String(20))  # RUNNING, COMPLETED, FAILED

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)


class ValidationOutcome(Base):
    """
    Record actual event outcomes for calibration tracking.
    """
    __tablename__ = 'validation_outcomes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(20), ForeignKey('risk_events.event_id'), nullable=False)

    # Outcome
    outcome_date = Column(DateTime, nullable=False)
    outcome_occurred = Column(Boolean, nullable=False)
    outcome_description = Column(Text)
    severity = Column(Integer)  # 1-5 scale if occurred

    # Pre-event prediction (most recent before outcome)
    prediction_date = Column(DateTime)
    predicted_probability_pct = Column(Float)
    predicted_ci_lower_pct = Column(Float)
    predicted_ci_upper_pct = Column(Float)

    # Analysis
    days_before_outcome = Column(Integer)
    was_surprising = Column(Boolean)  # Was prediction significantly wrong?
    notes = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_validation_event_date', 'event_id', 'outcome_date'),
    )
