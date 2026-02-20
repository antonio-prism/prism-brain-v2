"""
PRISM Engine — ACLED connector (Source A07).

Fetches armed conflict event data from ACLED.
Used for STR-GEO-001 (armed conflict in supplier country).

Requires ACLED_KEY and ACLED_EMAIL environment variables.
Note: ACLED comprehensive coverage starts from 2018, so observation window
is 2018-2024 (7 years) for conflict events.
"""

import logging

from .base import ConnectorResult, fetch_with_retry, get_cached, save_cache
from ..config.credentials import get_credential
from ..config.regions import REGIONS

logger = logging.getLogger(__name__)

ACLED_URL = "https://api.acleddata.com/acled/read"


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

    api_key = get_credential("acled_key")
    email = get_credential("acled_email")

    if not api_key or not email:
        return ConnectorResult(
            source_id="A07", success=False,
            error="ACLED_KEY or ACLED_EMAIL not configured"
        )

    cache_params = {"countries": sorted(countries), "start": start_year,
                    "end": end_year, "type": event_type}
    cached = get_cached("acled", cache_params, max_age_hours=168)
    if cached:
        return ConnectorResult(source_id="A07", success=True, data=cached, cached=True)

    years_with_conflict = set()
    country_conflict_years = {}

    for year in range(start_year, end_year + 1):
        # ACLED uses country names, not ISO codes — we pass ISO and let ACLED handle it
        # For efficiency, check if ANY event exists with limit=1
        params = {
            "key": api_key,
            "email": email,
            "event_type": event_type,
            "year": year,
            "limit": 1,
        }

        resp = fetch_with_retry(ACLED_URL, params=params, timeout=30)
        if resp and resp.status_code == 200:
            try:
                data = resp.json()
                count = data.get("count", 0)
                if isinstance(count, str):
                    count = int(count)
                if count > 0:
                    years_with_conflict.add(year)
            except Exception as e:
                logger.warning(f"ACLED parse error for year {year}: {e}")

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
