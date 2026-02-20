"""
PRISM Engine — CISA KEV connector (Source A05).

Fetches the CISA Known Exploited Vulnerabilities catalog and computes:
- Annual addition counts (for prior derivation trend)
- ICS-relevant vulnerability counts (for DIG-CIC modifier)
- Recent addition rate (for real-time modifier)

No API key required.
"""

import logging
from collections import Counter
from datetime import datetime

from .base import ConnectorResult, fetch_with_retry, get_cached, save_cache

logger = logging.getLogger(__name__)

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

ICS_VENDOR_KEYWORDS = [
    "siemens", "schneider", "rockwell", "honeywell", "abb",
    "emerson", "yokogawa", "ge", "mitsubishi", "omron",
    "aveva", "codesys", "moxa", "phoenix contact", "wago",
]


def fetch_kev_catalog() -> ConnectorResult:
    """
    Fetch and parse the full CISA KEV catalog.

    Returns yearly counts, ICS-relevant entries, and recent addition rate.
    """
    cache_params = {"source": "kev_full"}
    cached = get_cached("cisa_kev", cache_params, max_age_hours=24)
    if cached:
        return ConnectorResult(source_id="A05", success=True, data=cached, cached=True)

    resp = fetch_with_retry(KEV_URL, timeout=30)
    if not resp or resp.status_code != 200:
        logger.error("Failed to fetch CISA KEV catalog")
        return ConnectorResult(source_id="A05", success=False, error="HTTP request failed")

    try:
        kev_data = resp.json()
    except Exception as e:
        return ConnectorResult(source_id="A05", success=False, error=f"JSON parse error: {e}")

    vulns = kev_data.get("vulnerabilities", [])

    # Count additions per year
    yearly_counts = Counter()
    for v in vulns:
        try:
            year = datetime.strptime(v["dateAdded"], "%Y-%m-%d").year
            yearly_counts[year] += 1
        except (KeyError, ValueError):
            pass

    # Identify ICS-relevant entries
    ics_vulns = []
    for v in vulns:
        vendor = v.get("vendorProject", "").lower()
        product = v.get("product", "").lower()
        combined = f"{vendor} {product}"
        if any(kw in combined for kw in ICS_VENDOR_KEYWORDS):
            ics_vulns.append({
                "cveID": v.get("cveID"),
                "vendor": v.get("vendorProject"),
                "product": v.get("product"),
                "dateAdded": v.get("dateAdded"),
            })

    # ICS counts per year
    ics_yearly = Counter()
    for v in ics_vulns:
        try:
            year = datetime.strptime(v["dateAdded"], "%Y-%m-%d").year
            ics_yearly[year] += 1
        except (KeyError, ValueError):
            pass

    # Recent rate: entries in last 90 days
    cutoff_90d = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    from datetime import timedelta
    cutoff_90d = cutoff_90d - timedelta(days=90)
    recent_count = sum(
        1 for v in vulns
        if _parse_date(v.get("dateAdded", "")) and _parse_date(v["dateAdded"]) >= cutoff_90d
    )

    # Compute growth rate for modifier
    # Use the two most recent COMPLETE years to avoid partial-year bias
    # (current year may only have 1-2 months of data)
    current_year = datetime.utcnow().year
    complete_years = sorted(y for y in yearly_counts.keys() if y < current_year)
    if len(complete_years) >= 2:
        latest_complete = complete_years[-1]
        prev_complete = complete_years[-2]
        yoy_growth = (yearly_counts[latest_complete] / yearly_counts[prev_complete]
                      if yearly_counts[prev_complete] > 0 else 1.0)
    else:
        yoy_growth = 1.0

    data = {
        "total_vulns": len(vulns),
        "yearly_counts": dict(sorted(yearly_counts.items())),
        "ics_total": len(ics_vulns),
        "ics_yearly": dict(sorted(ics_yearly.items())),
        "recent_90d_count": recent_count,
        "yoy_growth_rate": round(yoy_growth, 2),
        "catalog_date": kev_data.get("catalogVersion", "unknown"),
    }

    save_cache("cisa_kev", cache_params, data)
    return ConnectorResult(source_id="A05", success=True, data=data)


def get_kev_modifier() -> dict:
    """
    Compute CISA KEV growth rate modifier for DIG-RDE events.
    Growth > baseline → increased threat activity.
    """
    result = fetch_kev_catalog()
    if not result.success:
        return {"modifier": 1.0, "status": "FALLBACK", "error": result.error}

    yoy_growth = result.data.get("yoy_growth_rate", 1.0)
    # Modifier = growth rate, clipped to [0.50, 2.00]
    modifier = round(max(0.50, min(2.00, yoy_growth)), 2)

    return {
        "modifier": modifier,
        "status": "COMPUTED",
        "yoy_growth": yoy_growth,
        "total_kev": result.data.get("total_vulns", 0),
        "recent_90d": result.data.get("recent_90d_count", 0),
    }


def _parse_date(date_str: str) -> datetime | None:
    """Safely parse a date string."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
