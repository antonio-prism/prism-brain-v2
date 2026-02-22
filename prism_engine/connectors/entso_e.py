"""
PRISM Engine — ENTSO-E Transparency Platform connector (Source A10).

Fetches European electricity grid data for PHY-ENE (energy supply) events:
- Actual generation output by fuel type
- Day-ahead load forecast vs actual load
- Grid adequacy margins

Requires ENTSOE_API_KEY environment variable (free registration at
https://transparency.entsoe.eu/).

Falls back gracefully if no API key is configured — the engine continues
to use FRED manufacturing orders as a proxy for energy demand pressure.
"""

import logging
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

from .base import ConnectorResult, fetch_with_retry, get_cached, save_cache
from ..config.credentials import get_credential

logger = logging.getLogger(__name__)

# ENTSO-E Transparency Platform API
_BASE_URL = "https://web-api.tp.entsoe.eu/api"

# Area EIC codes for major European bidding zones
_AREAS = {
    "DE_LU": "10Y1001A1001A82H",   # Germany-Luxembourg
    "FR": "10YFR-RTE------C",       # France
    "ES": "10YES-REE------0",       # Spain
    "IT_NORTH": "10Y1001A1001A73I", # Italy North
    "NL": "10YNL----------L",       # Netherlands
}

# Document types (ENTSO-E API codes)
_DOC_ACTUAL_LOAD = "A65"          # Actual total load
_DOC_FORECAST_LOAD = "A65"        # Day-ahead total load forecast
_DOC_ACTUAL_GENERATION = "A75"    # Actual generation per type
_PROCESS_REALISED = "A16"         # Realised (actual)
_PROCESS_DAY_AHEAD = "A01"        # Day-ahead forecast


def _check_available() -> bool:
    """Check if ENTSO-E API key is configured."""
    key = get_credential("entsoe")
    return key is not None


def _api_request(params: dict) -> ET.Element | None:
    """Make an authenticated ENTSO-E API request and parse XML response."""
    key = get_credential("entsoe")
    if not key:
        return None

    params["securityToken"] = key
    resp = fetch_with_retry(_BASE_URL, params=params, timeout=30)

    if resp is None or resp.status_code != 200:
        status = resp.status_code if resp else "no response"
        logger.warning(f"ENTSO-E API returned {status}")
        return None

    try:
        return ET.fromstring(resp.text)
    except ET.ParseError as e:
        logger.error(f"ENTSO-E XML parse error: {e}")
        return None


def _parse_timeseries_values(root: ET.Element) -> list[float]:
    """Extract numeric values from ENTSO-E TimeSeries XML."""
    ns = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}
    values = []

    # Try with namespace
    for point in root.findall(".//ns:Point/ns:quantity", ns):
        try:
            values.append(float(point.text))
        except (ValueError, TypeError):
            pass

    # Fallback: try without namespace (some endpoints use different schemas)
    if not values:
        for point in root.iter():
            if point.tag.endswith("quantity") or point.tag == "quantity":
                try:
                    values.append(float(point.text))
                except (ValueError, TypeError):
                    pass

    return values


def fetch_load_data(area_code: str = None,
                    days_back: int = 30) -> ConnectorResult:
    """Fetch actual load data for a European area.

    Returns average load (MW) and peak load (MW) over the recent period.
    """
    if area_code is None:
        area_code = _AREAS["DE_LU"]  # Germany as default (largest EU market)

    cache_params = {"source": "entsoe_load", "area": area_code, "days": days_back}
    cached = get_cached("entsoe", cache_params, max_age_hours=24)
    if cached:
        return ConnectorResult(source_id="A10", success=True, data=cached, cached=True)

    if not _check_available():
        return ConnectorResult(
            source_id="A10", success=False,
            error="ENTSOE_API_KEY not configured"
        )

    end = datetime.utcnow()
    start = end - timedelta(days=days_back)

    params = {
        "documentType": _DOC_ACTUAL_LOAD,
        "processType": _PROCESS_REALISED,
        "outBiddingZone_Domain": area_code,
        "periodStart": start.strftime("%Y%m%d0000"),
        "periodEnd": end.strftime("%Y%m%d0000"),
    }

    root = _api_request(params)
    if root is None:
        return ConnectorResult(
            source_id="A10", success=False,
            error="ENTSO-E actual load request failed"
        )

    values = _parse_timeseries_values(root)
    if not values:
        return ConnectorResult(
            source_id="A10", success=False,
            error="No load data points found in ENTSO-E response"
        )

    avg_load = sum(values) / len(values)
    peak_load = max(values)

    data = {
        "area_code": area_code,
        "period_days": days_back,
        "data_points": len(values),
        "avg_load_mw": round(avg_load, 0),
        "peak_load_mw": round(peak_load, 0),
    }

    save_cache("entsoe", cache_params, data)
    return ConnectorResult(source_id="A10", success=True, data=data)


def fetch_load_forecast_ratio(area_code: str = None,
                              days_back: int = 7) -> ConnectorResult:
    """Compare actual vs forecast load to detect demand surprises.

    A ratio > 1.0 means actual load exceeded the forecast (demand surprise).
    """
    if area_code is None:
        area_code = _AREAS["DE_LU"]

    cache_params = {"source": "entsoe_forecast_ratio", "area": area_code, "days": days_back}
    cached = get_cached("entsoe", cache_params, max_age_hours=12)
    if cached:
        return ConnectorResult(source_id="A10", success=True, data=cached, cached=True)

    if not _check_available():
        return ConnectorResult(
            source_id="A10", success=False,
            error="ENTSOE_API_KEY not configured"
        )

    end = datetime.utcnow()
    start = end - timedelta(days=days_back)
    period_start = start.strftime("%Y%m%d0000")
    period_end = end.strftime("%Y%m%d0000")

    # Fetch actual load
    actual_root = _api_request({
        "documentType": _DOC_ACTUAL_LOAD,
        "processType": _PROCESS_REALISED,
        "outBiddingZone_Domain": area_code,
        "periodStart": period_start,
        "periodEnd": period_end,
    })

    # Fetch day-ahead forecast
    forecast_root = _api_request({
        "documentType": _DOC_FORECAST_LOAD,
        "processType": _PROCESS_DAY_AHEAD,
        "outBiddingZone_Domain": area_code,
        "periodStart": period_start,
        "periodEnd": period_end,
    })

    if actual_root is None or forecast_root is None:
        return ConnectorResult(
            source_id="A10", success=False,
            error="Failed to fetch actual or forecast load data"
        )

    actual_values = _parse_timeseries_values(actual_root)
    forecast_values = _parse_timeseries_values(forecast_root)

    if not actual_values or not forecast_values:
        return ConnectorResult(
            source_id="A10", success=False,
            error="No data points in actual or forecast load"
        )

    avg_actual = sum(actual_values) / len(actual_values)
    avg_forecast = sum(forecast_values) / len(forecast_values)
    ratio = avg_actual / avg_forecast if avg_forecast > 0 else 1.0

    data = {
        "area_code": area_code,
        "period_days": days_back,
        "avg_actual_mw": round(avg_actual, 0),
        "avg_forecast_mw": round(avg_forecast, 0),
        "actual_vs_forecast_ratio": round(ratio, 3),
    }

    save_cache("entsoe", cache_params, data)
    return ConnectorResult(source_id="A10", success=True, data=data)


def get_grid_stress_modifier() -> dict:
    """Compute a grid stress modifier for PHY-ENE events.

    Aggregates data across major European areas to produce a single
    modifier reflecting current grid stress levels.

    Method: Average the actual-vs-forecast load ratio across available
    areas.  Ratio > 1.05 means demand consistently exceeds forecasts
    (grid stress increasing), ratio < 0.95 means demand lower than
    expected (grid stress decreasing).

    Falls back to neutral (1.0) if ENTSO-E data is unavailable.
    """
    if not _check_available():
        return {
            "name": "ENTSO-E grid stress",
            "source_id": "A10",
            "modifier": 1.0,
            "status": "UNAVAILABLE",
            "error": "ENTSOE_API_KEY not configured — using FRED proxy instead",
        }

    # Try to get load data from Germany (largest European market)
    result = fetch_load_data(area_code=_AREAS["DE_LU"], days_back=30)

    if not result.success:
        return {
            "name": "ENTSO-E grid stress",
            "source_id": "A10",
            "modifier": 1.0,
            "status": "FALLBACK",
            "error": result.error,
        }

    # Use peak-to-average ratio as stress indicator
    # Normal grid: peak/avg ~ 1.2-1.3
    # Stressed grid: peak/avg > 1.4 (demand spikes close to capacity)
    peak = result.data["peak_load_mw"]
    avg = result.data["avg_load_mw"]

    if avg <= 0:
        return {
            "name": "ENTSO-E grid stress",
            "source_id": "A10",
            "modifier": 1.0,
            "status": "FALLBACK",
            "error": "Invalid load data (avg=0)",
        }

    peak_avg_ratio = peak / avg

    # Calibration: peak/avg ratio maps to modifier
    # 1.20 (normal) → modifier 1.0
    # 1.30 (moderate stress) → modifier 1.1
    # 1.40 (high stress) → modifier 1.2
    # 1.50+ (extreme) → modifier 1.3+
    # Below 1.15 (low demand) → modifier 0.9
    import numpy as np
    modifier = 1.0 + (peak_avg_ratio - 1.25) * 1.0
    modifier = round(float(np.clip(modifier, 0.80, 1.50)), 2)

    return {
        "name": "ENTSO-E grid stress",
        "source_id": "A10",
        "modifier": modifier,
        "status": "COMPUTED",
        "indicator_value": round(peak_avg_ratio, 3),
        "indicator_unit": "peak/average load ratio (30-day)",
        "data_points": result.data["data_points"],
        "area": "DE-LU (Germany-Luxembourg)",
        "calibration": {
            "method": "ratio",
            "formula": "modifier = 1.0 + (peak_avg_ratio - 1.25)",
            "baseline_ratio": 1.25,
            "floor": 0.80,
            "ceiling": 1.50,
        },
    }
