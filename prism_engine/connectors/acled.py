"""
PRISM Engine — ACLED connector (Source A07).

Fetches armed conflict event data from ACLED.
Used for STR-GEO-001 (armed conflict in supplier country).

Requires ACLED_EMAIL and ACLED_PASSWORD environment variables.
ACLED switched from API keys to OAuth tokens in 2025.
Tokens are valid for 24 hours and cached automatically.

Note: ACLED comprehensive coverage starts from 2018, so observation window
is 2018-2024 (7 years) for conflict events.
"""

import logging
import time

import requests

from .base import ConnectorResult, get_cached, save_cache
from ..config.credentials import get_credential
from ..config.regions import REGIONS

logger = logging.getLogger(__name__)

ACLED_AUTH_URL = "https://acleddata.com/oauth/token"
ACLED_API_URL = "https://acleddata.com/api/acled/read"

# Module-level token cache (avoids re-authenticating on every call)
_token_cache = {"token": None, "expires_at": 0}


def _get_access_token() -> str | None:
    """
    Get a valid ACLED OAuth access token.
    Caches the token for 23 hours (tokens are valid for 24h).
    """
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"]:
        return _token_cache["token"]

    email = get_credential("acled_email")
    password = get_credential("acled_password")

    if not email or not password:
        logger.warning("ACLED_EMAIL or ACLED_PASSWORD not configured")
        return None

    try:
        resp = requests.post(
            ACLED_AUTH_URL,
            data={
                "username": email,
                "password": password,
                "grant_type": "password",
                "client_id": "acled",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )

        if resp.status_code != 200:
            logger.error(f"ACLED auth failed: HTTP {resp.status_code}")
            return None

        data = resp.json()
        token = data.get("access_token")
        if not token:
            logger.error("ACLED auth response missing access_token")
            return None

        # Cache for 23 hours (1 hour safety margin)
        _token_cache["token"] = token
        _token_cache["expires_at"] = now + (23 * 3600)
        logger.info("ACLED OAuth token acquired successfully")
        return token

    except Exception as e:
        logger.error(f"ACLED auth error: {e}")
        return None


def count_conflict_years(countries: list[str] | None = None,
                         start_year: int = 2018, end_year: int = 2024,
                         event_type: str = "Battles") -> ConnectorResult:
    """
    Count years with at least 1 conflict event in any of the listed countries.

    Uses shorter observation window (2018-2024) because ACLED comprehensive
    coverage starts from 2018.
    """
    if countries is None:
        countries = REGIONS["TOP20_SUPPLIERS"]

    cache_params = {"countries": sorted(countries), "start": start_year,
                    "end": end_year, "type": event_type}
    cached = get_cached("acled", cache_params, max_age_hours=168)
    if cached:
        return ConnectorResult(source_id="A07", success=True, data=cached, cached=True)

    token = _get_access_token()
    if not token:
        return ConnectorResult(
            source_id="A07", success=False,
            error="ACLED authentication failed — check ACLED_EMAIL and ACLED_PASSWORD in .env"
        )

    headers = {"Authorization": f"Bearer {token}"}
    years_with_conflict = set()

    for year in range(start_year, end_year + 1):
        params = {
            "event_type": event_type,
            "year": year,
            "limit": 1,
            "_format": "json",
        }

        try:
            resp = requests.get(ACLED_API_URL, params=params,
                                headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                count = data.get("count", 0)
                if isinstance(count, str):
                    count = int(count)
                if count > 0:
                    years_with_conflict.add(year)
            else:
                logger.warning(f"ACLED query failed for year {year}: HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"ACLED request error for year {year}: {e}")

    total_years = end_year - start_year + 1
    conflict_count = len(years_with_conflict)

    data = {
        "event_type": event_type,
        "observation_window": f"{start_year}-{end_year} ({total_years}yr)",
        "total_years": total_years,
        "years_with_conflict": sorted(years_with_conflict),
        "conflict_year_count": conflict_count,
        "prior": round(conflict_count / total_years, 4) if total_years > 0 else 0,
        "formula": f"{conflict_count} conflict-years / {total_years} total years",
        "note": "ACLED comprehensive coverage from 2018, shorter window used",
    }

    save_cache("acled", cache_params, data)
    return ConnectorResult(source_id="A07", success=True, data=data)
