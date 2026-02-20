"""
PRISM Engine — USGS ComCat connector (Source A02).

Queries the USGS earthquake catalog for seismic event counts
by magnitude threshold and geographic zone.

No API key required.
"""

import logging
from ..config.regions import SEISMIC_ZONES
from .base import ConnectorResult, fetch_with_retry, get_cached, save_cache

logger = logging.getLogger(__name__)

COMCAT_COUNT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/count"


def count_earthquakes(zone_name: str, zone_bbox: dict,
                      start: str = "2000-01-01", end: str = "2024-12-31",
                      min_mag: float = 6.0) -> dict:
    """
    Count earthquakes in a given zone for the observation window.

    Returns: {"zone": str, "count": int, "years": int, "prior": float}
    """
    params = {
        "format": "text",
        "starttime": start,
        "endtime": end,
        "minmagnitude": min_mag,
        "minlatitude": zone_bbox["south"],
        "maxlatitude": zone_bbox["north"],
        "minlongitude": zone_bbox["west"],
        "maxlongitude": zone_bbox["east"],
    }

    # Check cache (24h for historical data)
    cache_params = {"zone": zone_name, "start": start, "end": end, "min_mag": min_mag}
    cached = get_cached("usgs", cache_params, max_age_hours=168)  # 7 days for historical
    if cached:
        return cached

    resp = fetch_with_retry(COMCAT_COUNT_URL, params=params, timeout=30)
    if resp and resp.status_code == 200:
        try:
            count = int(resp.text.strip())
        except ValueError:
            logger.error(f"USGS returned non-integer: {resp.text[:100]}")
            count = 0

        # Calculate years from date range
        start_year = int(start[:4])
        end_year = int(end[:4])
        years = end_year - start_year + 1

        result = {
            "zone": zone_name,
            "count": count,
            "years": years,
            "prior": round(count / years, 4) if years > 0 else 0,
        }
        save_cache("usgs", cache_params, result)
        return result

    logger.error(f"USGS query failed for zone {zone_name}")
    return {"zone": zone_name, "count": 0, "years": 25, "prior": 0}


def compute_earthquake_prior(start: str = "2000-01-01", end: str = "2024-12-31",
                             min_mag: float = 6.0) -> ConnectorResult:
    """
    Compute PHY-GEO-001 prior: P(at least 1 M6.0+ earthquake in any major seismic zone).

    USGS /count returns TOTAL events, not distinct years. We convert:
      annual_rate λ = count / years
      P(≥1 per year in zone) = 1 - e^(-λ)  (Poisson approximation)
      P(≥1 in ANY zone) = 1 - ∏(1 - P_zone_i)

    Note: This gives a very high prior (~0.95) because M6.0+ earthquakes
    happen somewhere every year. This is correct for Layer 1 (global).
    The old fallback rate (3.5%) was a Layer-2-adjusted value.
    """
    import math

    zone_results = {}
    product_no_event = 1.0

    for zone_name, zone_bbox in SEISMIC_ZONES.items():
        result = count_earthquakes(zone_name, zone_bbox, start, end, min_mag)
        # result["prior"] is actually annual_rate (count/years), can be >1
        annual_rate = result["prior"]
        p_at_least_one = 1.0 - math.exp(-annual_rate)
        result["annual_rate"] = annual_rate
        result["p_at_least_one"] = round(p_at_least_one, 4)
        zone_results[zone_name] = result
        product_no_event *= (1.0 - p_at_least_one)

    combined_prior = round(1.0 - product_no_event, 4)

    return ConnectorResult(
        source_id="A02",
        success=True,
        data={
            "zones": zone_results,
            "combined_prior": combined_prior,
            "formula": "1 - ∏(1 - P_zone_i) where P_zone = 1 - e^(-λ)",
            "min_magnitude": min_mag,
            "observation_window": f"{start} to {end}",
            "note": (
                "Layer 1 prior is high (~0.95) because M6.0+ earthquakes occur "
                "somewhere every year. The old rate (3.5%) included material impact "
                "probability, which belongs in Layer 2."
            ),
        },
    )


def get_recent_seismicity_modifier(days: int = 90) -> ConnectorResult:
    """
    Compute seismicity modifier: recent activity rate vs long-term rate.
    Uses last 90 days vs 25-year average.
    """
    from datetime import datetime, timedelta

    end = datetime.utcnow()
    start_recent = end - timedelta(days=days)

    total_recent = 0
    total_longterm = 0

    for zone_name, zone_bbox in SEISMIC_ZONES.items():
        # Recent period
        recent = count_earthquakes(
            f"{zone_name}_recent", zone_bbox,
            start_recent.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            min_mag=5.0,
        )
        longterm = count_earthquakes(
            zone_name, zone_bbox,
            "2000-01-01", "2024-12-31",
            min_mag=5.0,
        )
        total_recent += recent["count"]
        total_longterm += longterm["count"]

    # Annualize recent count
    recent_annual = total_recent * (365.25 / days)
    longterm_annual = total_longterm / 25.0 if total_longterm > 0 else 1.0

    ratio = recent_annual / longterm_annual if longterm_annual > 0 else 1.0
    modifier = round(max(0.50, min(3.00, ratio)), 2)

    return ConnectorResult(
        source_id="A02",
        success=True,
        data={
            "recent_count": total_recent,
            "recent_days": days,
            "recent_annualized": round(recent_annual, 1),
            "longterm_annual": round(longterm_annual, 1),
            "ratio": round(ratio, 2),
            "modifier": modifier,
        },
    )
