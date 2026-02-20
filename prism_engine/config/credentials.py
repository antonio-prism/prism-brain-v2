"""
PRISM Engine — API credential management.

All API keys are read from environment variables. Never hardcoded.
"""

import os
import logging

logger = logging.getLogger(__name__)


def get_credential(service: str) -> str | None:
    """Get an API credential from environment variables."""
    env_map = {
        "cds": "CDS_API_KEY",
        "fred": "FRED_API_KEY",
        "nvd": "NVD_API_KEY",
        "acled_email": "ACLED_EMAIL",
        "acled_password": "ACLED_PASSWORD",
        "entsoe": "ENTSOE_API_KEY",
    }
    var_name = env_map.get(service)
    if not var_name:
        return None
    value = os.getenv(var_name, "").strip()
    if not value:
        logger.warning(f"API key not configured: {var_name}")
        return None
    return value


def check_all_credentials() -> dict[str, bool]:
    """Check which API credentials are configured."""
    services = ["cds", "fred", "nvd", "acled_email", "acled_password", "entsoe"]
    return {svc: get_credential(svc) is not None for svc in services}


# Services that require NO API key
NO_KEY_REQUIRED = ["usgs", "cisa_kev", "gpr", "world_bank", "noaa_cpc", "who_don", "ucdp"]
