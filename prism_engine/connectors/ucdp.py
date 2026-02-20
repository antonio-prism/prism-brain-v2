"""
PRISM Engine — UCDP connector (Source A07, replaces ACLED).

Fetches armed conflict data from the Uppsala Conflict Data Program (UCDP).
Used for STR-GEO-001 (armed conflict in supplier country).

No API key required — completely free and open.
Uses the UCDP/PRIO Armed Conflict Dataset (country-year level).
Academic gold standard for armed conflict measurement (used by UN, World Bank).

Coverage: 1946-2024 (version 25.1). We use 2000-2024 (25-year window).
Filters for intensity_level=2 (war: 1000+ battle deaths/year) to capture
conflicts that would actually disrupt supply chains.

Rate limit: 5,000 requests/day (resets midnight UTC).
"""

import logging

import requests

from .base import ConnectorResult, get_cached, save_cache
from ..config.regions import REGIONS

logger = logging.getLogger(__name__)

UCDP_API_URL = "https://ucdpapi.pcr.uu.se/api/ucdpprioconflict/25.1"

# ISO-2 → Gleditsch-Ward numeric codes (used by UCDP API)
ISO2_TO_GW = {
    "CN": 710, "US": 2, "GB": 200, "CH": 225, "RU": 365,
    "NO": 385, "JP": 740, "KR": 732, "TR": 640, "IN": 750,
    "BR": 140, "VN": 816, "TW": 713, "TH": 800, "SA": 670,
    "ID": 850, "MY": 820, "MX": 70, "UA": 369, "ZA": 560,
    # EU27 + EEA (for broader queries)
    "AT": 305, "BE": 211, "BG": 355, "HR": 344, "CY": 352,
    "CZ": 316, "DK": 390, "EE": 366, "FI": 375, "FR": 220,
    "DE": 255, "GR": 350, "HU": 310, "IE": 205, "IT": 325,
    "LV": 367, "LT": 368, "LU": 212, "MT": 338, "NL": 210,
    "PL": 290, "PT": 235, "RO": 360, "SK": 317, "SI": 349,
    "ES": 230, "SE": 380, "IS": 395, "LI": 223,
    "CA": 20, "AU": 900,
}


def _gw_loc_matches(gwno_loc: str, gw_codes: set[str]) -> bool:
    """Check if gwno_loc matches any of our country codes.

    UCDP uses comma-separated codes for interstate conflicts
    (e.g. "365, 369" for Russia-Ukraine war).
    """
    for code in gwno_loc.split(","):
        if code.strip() in gw_codes:
            return True
    return False


def count_conflict_years(countries: list[str] | None = None,
                         start_year: int = 2000, end_year: int = 2024,
                         min_intensity: int = 2) -> ConnectorResult:
    """
    Count years with at least 1 armed conflict in any of the listed countries.

    Uses UCDP/PRIO Armed Conflict Dataset (country-year level).
    25-year observation window (2000-2024) for robust priors.

    min_intensity: 1 = any conflict (25+ deaths), 2 = war (1000+ deaths).
    Default is 2 (war-level) to capture supply-chain-disrupting conflicts.
    """
    if countries is None:
        countries = REGIONS["TOP20_SUPPLIERS"]

    cache_params = {"countries": sorted(countries), "start": start_year,
                    "end": end_year, "source": "ucdp", "min_intensity": min_intensity}
    cached = get_cached("ucdp", cache_params, max_age_hours=168)  # 7-day cache
    if cached:
        return ConnectorResult(source_id="A07", success=True, data=cached, cached=True)

    # Convert ISO-2 codes to Gleditsch-Ward codes
    gw_codes = set()
    for iso2 in countries:
        gw = ISO2_TO_GW.get(iso2)
        if gw:
            gw_codes.add(str(gw))
        else:
            logger.debug(f"No GW code for {iso2}, skipping")

    if not gw_codes:
        return ConnectorResult(
            source_id="A07", success=False,
            error="No valid country codes for UCDP query"
        )

    years_with_conflict = set()

    try:
        # Paginate through all UCDP conflict records
        page = 0
        total_pages = 1

        while page < total_pages:
            params = {
                "pagesize": 1000,
                "page": page,
            }

            resp = requests.get(UCDP_API_URL, params=params, timeout=30)
            if resp.status_code != 200:
                logger.error(f"UCDP API returned HTTP {resp.status_code}")
                return ConnectorResult(
                    source_id="A07", success=False,
                    error=f"UCDP API returned HTTP {resp.status_code}"
                )

            data = resp.json()
            total_pages = data.get("TotalPages", 1)

            for record in data.get("Result", []):
                year = int(record.get("year", 0))
                if year < start_year or year > end_year:
                    continue

                intensity = int(record.get("intensity_level", 0))
                if intensity < min_intensity:
                    continue

                gwno_loc = str(record.get("gwno_loc", ""))
                gwno_a = str(record.get("gwno_a", ""))

                # Check gwno_loc (comma-separated for interstate) and gwno_a
                if _gw_loc_matches(gwno_loc, gw_codes) or gwno_a in gw_codes:
                    years_with_conflict.add(year)

            page += 1

    except Exception as e:
        logger.error(f"UCDP request error: {e}")
        return ConnectorResult(source_id="A07", success=False, error=str(e))

    total_years = end_year - start_year + 1
    conflict_count = len(years_with_conflict)
    intensity_label = "war (1000+ battle deaths/yr)" if min_intensity == 2 else "armed conflict (25+ deaths/yr)"

    result_data = {
        "intensity_filter": intensity_label,
        "observation_window": f"{start_year}-{end_year} ({total_years}yr)",
        "total_years": total_years,
        "years_with_conflict": sorted(years_with_conflict),
        "conflict_year_count": conflict_count,
        "prior": round(conflict_count / total_years, 4) if total_years > 0 else 0,
        "formula": f"{conflict_count} conflict-years / {total_years} total years",
        "data_source": "UCDP/PRIO Armed Conflict Dataset v25.1",
        "note": f"Uppsala Conflict Data Program — filtered for {intensity_label}",
        "countries_queried": len(gw_codes),
    }

    save_cache("ucdp", cache_params, result_data)
    return ConnectorResult(source_id="A07", success=True, data=result_data)
