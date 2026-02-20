"""
PRISM Engine — World Bank API connector (Source A09).

Fetches GDP growth and recession data for OECD economies.
Used for STR-ECO-001 (recession prior) and other economic events.

No API key required.
"""

import logging

from .base import ConnectorResult, fetch_with_retry, get_cached, save_cache
from ..config.regions import REGIONS

logger = logging.getLogger(__name__)

WB_BASE = "https://api.worldbank.org/v2"


G7_COUNTRIES = ["US", "GB", "DE", "JP", "FR", "IT", "CA"]


def fetch_gdp_growth(countries: list[str] | None = None,
                     start_year: int = 2000, end_year: int = 2024) -> ConnectorResult:
    """
    Fetch annual GDP growth for specified countries.
    Indicator: NY.GDP.MKTP.KD.ZG (GDP growth, annual %)

    Default: G7 countries only (for recession detection).
    Using all OECD countries would overcount because small economies
    frequently have single-year contractions that don't constitute
    a systemic recession risk.
    """
    if countries is None:
        countries = G7_COUNTRIES

    cache_params = {"countries": sorted(countries), "start": start_year, "end": end_year}
    cached = get_cached("world_bank", cache_params, max_age_hours=168)
    if cached:
        return ConnectorResult(source_id="A09", success=True, data=cached, cached=True)

    # World Bank accepts semicolon-separated country codes
    country_str = ";".join(countries)
    url = f"{WB_BASE}/country/{country_str}/indicator/NY.GDP.MKTP.KD.ZG"
    params = {
        "format": "json",
        "per_page": 1000,
        "date": f"{start_year}:{end_year}",
    }

    resp = fetch_with_retry(url, params=params, timeout=30)
    if not resp or resp.status_code != 200:
        logger.error("World Bank GDP growth query failed")
        return ConnectorResult(source_id="A09", success=False, error="HTTP request failed")

    try:
        json_data = resp.json()
        if not isinstance(json_data, list) or len(json_data) < 2 or not json_data[1]:
            return ConnectorResult(source_id="A09", success=False, error="Unexpected response structure")
    except Exception as e:
        return ConnectorResult(source_id="A09", success=False, error=f"Parse error: {e}")

    records = json_data[1]

    # Organize by year: did ANY major economy have negative GDP growth?
    recession_years = set()
    year_data = {}

    for record in records:
        year = record.get("date")
        value = record.get("value")
        country = record.get("country", {}).get("id", "??")

        if year and value is not None:
            year = int(year)
            if year not in year_data:
                year_data[year] = []
            year_data[year].append({"country": country, "gdp_growth": value})
            if value < 0:
                recession_years.add(year)

    total_years = end_year - start_year + 1
    recession_count = len(recession_years)

    data = {
        "indicator": "NY.GDP.MKTP.KD.ZG",
        "countries_queried": len(countries),
        "observation_window": f"{start_year}-{end_year}",
        "total_years": total_years,
        "recession_years": sorted(recession_years),
        "recession_count": recession_count,
        "prior": round(recession_count / total_years, 4) if total_years > 0 else 0,
        "formula": f"{recession_count} recession-years / {total_years} total years",
    }

    save_cache("world_bank", cache_params, data)
    return ConnectorResult(source_id="A09", success=True, data=data)
