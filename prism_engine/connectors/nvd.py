"""
PRISM Engine — NIST NVD connector (Source A04).

Queries the National Vulnerability Database for ICS/OT-related CVEs.
Used as a proxy (C02) for OT attack surface expansion.

Requires NVD_API_KEY environment variable (higher rate limits).
"""

import logging
import time

from .base import ConnectorResult, fetch_with_retry, get_cached, save_cache
from ..config.credentials import get_credential

logger = logging.getLogger(__name__)

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# ICS/OT keyword queries — run separately and deduplicate by CVE ID.
# NVD keywordSearch does substring matching, so these cover the ICS landscape.
ICS_KEYWORD_QUERIES = ["SCADA", "ICS industrial control", "PLC HMI"]


def _fetch_all_ics_cves() -> list[dict]:
    """Fetch all ICS-related CVEs from NVD (no date filter).

    Returns list of CVE items with at minimum a 'published' date field.
    NVD date parameters have reliability issues, so we fetch all matching
    CVEs and filter by year client-side.
    """
    api_key = get_credential("nvd")
    if not api_key:
        return []
    headers = {"apiKey": api_key}

    seen_ids = set()
    all_cves = []

    for keyword in ICS_KEYWORD_QUERIES:
        start_index = 0
        while True:
            params = {
                "keywordSearch": keyword,
                "resultsPerPage": 2000,
                "startIndex": start_index,
            }
            resp = fetch_with_retry(NVD_URL, params=params, headers=headers, timeout=60)
            if not resp or resp.status_code != 200:
                break
            try:
                data = resp.json()
                vulns = data.get("vulnerabilities", [])
                for v in vulns:
                    cve = v.get("cve", {})
                    cve_id = cve.get("id", "")
                    if cve_id and cve_id not in seen_ids:
                        seen_ids.add(cve_id)
                        all_cves.append(cve)
                total = data.get("totalResults", 0)
                start_index += len(vulns)
                if start_index >= total or not vulns:
                    break
            except Exception:
                break
            time.sleep(0.7)
        time.sleep(0.7)

    return all_cves


def fetch_ics_cve_timeseries(start_year: int = 2015, end_year: int = 2025) -> ConnectorResult:
    """
    Build annual ICS CVE count time series.
    Used as proxy C02 for OT connectivity / attack surface expansion.
    """
    api_key = get_credential("nvd")
    if not api_key:
        return ConnectorResult(
            source_id="A04", success=False,
            error="NVD_API_KEY not configured — skipping ICS CVE timeseries"
        )

    cache_params = {"source": "nvd_ics_v2", "start": start_year, "end": end_year}
    cached = get_cached("nvd", cache_params, max_age_hours=168)
    if cached:
        return ConnectorResult(source_id="A04", success=True, data=cached, cached=True)

    all_cves = _fetch_all_ics_cves()
    if not all_cves:
        return ConnectorResult(
            source_id="A04", success=False,
            error="No ICS CVEs retrieved from NVD"
        )

    # Count CVEs by publication year
    counts = {y: 0 for y in range(start_year, end_year + 1)}
    for cve in all_cves:
        pub_date = cve.get("published", "")
        if pub_date:
            try:
                year = int(pub_date[:4])
                if start_year <= year <= end_year:
                    counts[year] = counts.get(year, 0) + 1
            except (ValueError, IndexError):
                pass

    logger.info(f"NVD ICS CVEs by year: {counts}")

    # Compute YoY growth rate (two most recent complete years)
    years_sorted = sorted(y for y in counts if counts[y] > 0)
    if len(years_sorted) >= 2:
        latest = counts[years_sorted[-1]]
        previous = counts[years_sorted[-2]]
        yoy_growth = latest / previous if previous > 0 else 1.0
    else:
        yoy_growth = 1.0

    total = sum(counts.values())
    n_years = sum(1 for v in counts.values() if v > 0) or 1
    avg_annual = total / n_years

    data = {
        "annual_counts": counts,
        "total_unique_cves": len(all_cves),
        "total": total,
        "avg_annual": round(avg_annual, 1),
        "yoy_growth": round(yoy_growth, 2),
        "latest_year": years_sorted[-1] if years_sorted else None,
        "latest_count": counts.get(years_sorted[-1]) if years_sorted else 0,
    }

    save_cache("nvd", cache_params, data)
    return ConnectorResult(source_id="A04", success=True, data=data)


def get_ics_cve_modifier() -> dict:
    """
    Compute ICS vulnerability trend modifier (proxy C02).
    YoY growth > 1.0 → expanding attack surface → higher risk.
    """
    result = fetch_ics_cve_timeseries()
    if not result.success:
        return {"modifier": 1.0, "status": "FALLBACK", "error": result.error}

    yoy = result.data.get("yoy_growth", 1.0)
    modifier = round(max(0.50, min(2.00, yoy)), 2)

    return {
        "name": "ICS CVE growth rate (proxy C02)",
        "source_id": "A04",
        "modifier": modifier,
        "status": "COMPUTED",
        "indicator_value": result.data.get("latest_count", 0),
        "indicator_unit": "ICS-related CVEs per year",
        "yoy_growth": round(yoy, 2),
        "avg_annual": result.data.get("avg_annual", 0),
        "proxy": "C02",
    }
