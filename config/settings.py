"""
PRISM Brain Configuration Settings
Centralized configuration using environment variables.
"""

import os
from functools import lru_cache


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        # App settings
        self.app_name = "PRISM Brain API"
        self.app_version = "3.0.0"
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

        # Database
        self.database_url = os.getenv("DATABASE_URL", "")
        # Railway uses postgres:// but SQLAlchemy needs postgresql://
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace(
                "postgres://", "postgresql://", 1
            )

        # API Keys for data sources
        self.fred_api_key = os.getenv("FRED_API_KEY", "")
        self.noaa_api_key = os.getenv("NOAA_API_KEY", "")
        self.nvd_api_key = os.getenv("NVD_API_KEY", "")
        self.eia_api_key = os.getenv("EIA_API_KEY", "")
        self.otx_api_key = os.getenv("OTX_API_KEY", "")
        self.acled_email = os.getenv("ACLED_EMAIL", "")
        self.acled_password = os.getenv("ACLED_PASSWORD", "")

        # Server settings
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))

        # Calculation settings
        self.max_probability = 0.999
        self.min_probability = 0.001
        self.default_confidence = 0.5
        self.bayesian_weight = 0.6
        self.ml_weight = 0.4
        self.dependency_damping = 0.3


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
