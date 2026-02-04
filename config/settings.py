"""
PRISM Brain Configuration Settings

Centralized configuration management using Pydantic Settings.
All configuration is loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "PRISM Brain Probability Engine"
    app_version: str = "2.0.0"
    debug: bool = False

    # Database
    database_url: str = "postgresql://prism:password@localhost:5432/prism_brain"
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_default: int = 3600  # 1 hour

    # API Rate Limiting
    api_rate_limit_requests: int = 100
    api_rate_limit_window: int = 60  # seconds

    # Calculation Settings
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.80
    max_probability: float = 0.999
    min_probability: float = 0.001

    # Data Source API Keys (Free Tier)
    acled_api_key: Optional[str] = None
    acled_email: Optional[str] = None
    noaa_token: Optional[str] = None
    eia_api_key: Optional[str] = None
    otx_api_key: Optional[str] = None
    nvd_api_key: Optional[str] = None
    nasa_earthdata_token: Optional[str] = None

    # Premium API Keys (Stubs - Phase 3)
    bloomberg_api_key: Optional[str] = None
    recorded_future_api_key: Optional[str] = None
    iea_api_key: Optional[str] = None
    platts_api_key: Optional[str] = None

    # Celery Settings
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Weekly Calculation Schedule (cron format)
    calculation_schedule_day: str = "sunday"
    calculation_schedule_hour: int = 0
    calculation_schedule_minute: int = 0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Beta parameters for indicator types (log-odds shift magnitude)
BETA_PARAMETERS = {
    "direct_causal": 1.5,      # Can roughly quadruple odds
    "strong_correlation": 1.0,  # Can roughly double odds
    "moderate_correlation": 0.7,  # ~2x odds at maximum
    "weak_contextual": 0.4     # ~1.5x odds at maximum
}

# Indicator time scale classifications
TIME_SCALE_WEIGHTS = {
    "fast": {"z_score": 0.4, "momentum": 0.6},      # Cyber, market sentiment
    "medium": {"z_score": 0.5, "momentum": 0.5},    # Diplomatic, political
    "slow": {"z_score": 0.7, "momentum": 0.3}       # Economic, climate
}

# Tier classification thresholds
TIER_THRESHOLDS = {
    "ml_enhanced_min_analogs": 20,
    "analog_min_similarity": 0.6,
    "scenario_max_historical_events": 5
}

# Precision band definitions (based on CI width)
PRECISION_BANDS = {
    "HIGH": 10,        # CI width < 10 percentage points
    "MODERATE": 25,    # CI width 10-25 percentage points
    "LOW": 50,         # CI width 25-50 percentage points
    "VERY_LOW": 100    # CI width > 50 percentage points
}

# Data source configurations
DATA_SOURCES = {
    "acled": {
        "name": "ACLED - Armed Conflict Location & Event Data",
        "type": "FREE",
        "rate_limit": 500,  # per hour
        "cache_ttl": 3600,
        "time_scale": "medium",
        "quality_score": 5,
        "beta_type": "direct_causal"
    },
    "gdelt": {
        "name": "GDELT - Global Database of Events, Language, and Tone",
        "type": "FREE",
        "rate_limit": 999999,  # No strict limit
        "cache_ttl": 900,  # 15 minutes
        "time_scale": "fast",
        "quality_score": 3,
        "beta_type": "moderate_correlation",
        "requires_filtering": True
    },
    "world_bank": {
        "name": "World Bank API",
        "type": "FREE",
        "rate_limit": 999999,
        "cache_ttl": 86400,  # 24 hours
        "time_scale": "slow",
        "quality_score": 5,
        "beta_type": "weak_contextual"
    },
    "fred": {
        "name": "FRED - Federal Reserve Economic Data",
        "type": "FREE",
        "rate_limit": 120,  # per minute
        "cache_ttl": 3600,
        "time_scale": "slow",
        "quality_score": 5,
        "beta_type": "moderate_correlation"
    },
    "noaa": {
        "name": "NOAA Climate Data Online",
        "type": "FREE",
        "rate_limit": 1000,  # per day
        "cache_ttl": 86400,
        "time_scale": "slow",
        "quality_score": 5,
        "beta_type": "direct_causal"
    },
    "eia": {
        "name": "U.S. Energy Information Administration",
        "type": "FREE",
        "rate_limit": 5000,  # per hour
        "cache_ttl": 3600,
        "time_scale": "medium",
        "quality_score": 5,
        "beta_type": "strong_correlation"
    },
    "otx": {
        "name": "AlienVault Open Threat Exchange",
        "type": "FREE",
        "rate_limit": 10000,  # per day
        "cache_ttl": 3600,
        "time_scale": "fast",
        "quality_score": 4,
        "beta_type": "direct_causal"
    },
    "nvd": {
        "name": "National Vulnerability Database",
        "type": "FREE",
        "rate_limit": 50,  # per 30 seconds without key
        "cache_ttl": 3600,
        "time_scale": "fast",
        "quality_score": 5,
        "beta_type": "direct_causal"
    },
    "imf": {
        "name": "IMF Data API",
        "type": "FREE",
        "rate_limit": 999999,
        "cache_ttl": 86400,
        "time_scale": "slow",
        "quality_score": 5,
        "beta_type": "moderate_correlation"
    },
    "nasa_earthdata": {
        "name": "NASA Earthdata",
        "type": "FREE",
        "rate_limit": 999999,
        "cache_ttl": 86400,
        "time_scale": "slow",
        "quality_score": 5,
        "beta_type": "strong_correlation"
    }
}
