"""
PRISM Engine — EIA connector.

Fetches energy data from the U.S. Energy Information Administration Open Data API v2.
Used for OPS-AIR-004 and OPS-RLD-005 fuel supply/disruption indicators.

Requires EIA_API_KEY environment variable.
"""

import logging
from datetime import datetime

from .base import ConnectorResult, fetch_with_retry, get_cached, save_cache
from ..config.credentials import get_credential

logger = logging.getLogger(__name__)

EIA_BASE = "https://api.eia.gov/v2"

# EIA series used by PRISM scoring functions
EIA_SERIES = {
    "WCRSTUS1": "Weekly U.S. Ending Stocks of Crude Oil (Thousand Barrels)",
    "WGTSTUS1": "Weekly U.S. Ending Stocks of Total Gasoline (Thousand Barrels)",
    "WDISTUS1": "Weekly U.S. Ending Stocks of Distillate Fuel Oil (Thousand Barrels)",
    "RWTC": "Cushing, OK WTI Spot Price FOB ($/Barrel)",
    "EMM_EPM0_PTE_NUS_DPG": "Weekly U.S. Regular Gasoline Price ($/gallon)",
}


def fetch_petroleum_stocks() -> ConnectorResult:
    """Fetch current U.S. petroleum stock levels.

    Returns days of supply equivalent (approximation from stock levels).
    """
    api_key = get_credential("eia")
    if not api_key:
        return ConnectorResult(
            source_id="EIA", success=False,
            error="EIA_API_KEY not configured"
        )

    cache_params = {"query": "petroleum_stocks"}
    cached = get_cached("eia", cache_params, max_age_hours=24)
    if cached:
        return ConnectorResult(source_id="EIA", success=True, data=cached, cached=True)

    # Fetch crude oil stocks from EIA v2 API (excluding SPR = commercial stocks)
    url = f"{EIA_BASE}/petroleum/stoc/wstk/data/"
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[product][]": "EPC0",     # Crude oil
        "facets[duoarea][]": "NUS",      # National US
        "facets[process][]": "SAX",      # Excluding SPR (commercial stocks)
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 52,  # Last year of weekly data
    }

    resp = fetch_with_retry(url, params=params, timeout=30)
    if not resp or resp.status_code != 200:
        logger.error("EIA petroleum stocks query failed")
        return ConnectorResult(source_id="EIA", success=False, error="HTTP failed for EIA stocks")

    try:
        body = resp.json()
        records = body.get("response", {}).get("data", [])
    except Exception as e:
        return ConnectorResult(source_id="EIA", success=False, error=f"Parse error: {e}")

    if not records:
        return ConnectorResult(source_id="EIA", success=False, error="No stock data returned")

    # Extract latest stock level and compute approximate days of supply
    # US consumption ~20M bbl/day; stocks in thousands of barrels
    latest_value = None
    for record in records:
        val = record.get("value")
        if val is not None:
            try:
                latest_value = float(val)
                break
            except (ValueError, TypeError):
                continue

    if latest_value is None:
        return ConnectorResult(source_id="EIA", success=False, error="No valid stock values")

    # Approximate days of supply: stocks (thousands bbl) / daily consumption (thousands bbl/day)
    daily_consumption_k = 20000  # ~20M bbl/day in thousands
    days_of_supply = round(latest_value / daily_consumption_k, 1)

    data = {
        "stocks_k_bbl": latest_value,
        "days_of_supply": days_of_supply,
        "period": records[0].get("period", ""),
        "records_count": len(records),
    }

    save_cache("eia", cache_params, data)
    return ConnectorResult(source_id="EIA", success=True, data=data)


def fetch_crude_price() -> ConnectorResult:
    """Fetch current WTI crude oil spot price and recent volatility.

    Returns latest price and year-over-year change percentage.
    """
    api_key = get_credential("eia")
    if not api_key:
        return ConnectorResult(
            source_id="EIA", success=False,
            error="EIA_API_KEY not configured"
        )

    cache_params = {"query": "crude_price"}
    cached = get_cached("eia", cache_params, max_age_hours=24)
    if cached:
        return ConnectorResult(source_id="EIA", success=True, data=cached, cached=True)

    url = f"{EIA_BASE}/petroleum/pri/spt/data/"
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[product][]": "EPCWTI",  # WTI Crude
        "facets[duoarea][]": "YCUOK",   # Cushing, OK
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 104,  # ~2 years of weekly data
    }

    resp = fetch_with_retry(url, params=params, timeout=30)
    if not resp or resp.status_code != 200:
        logger.error("EIA crude price query failed")
        return ConnectorResult(source_id="EIA", success=False, error="HTTP failed for EIA crude price")

    try:
        body = resp.json()
        records = body.get("response", {}).get("data", [])
    except Exception as e:
        return ConnectorResult(source_id="EIA", success=False, error=f"Parse error: {e}")

    # Extract prices (most recent first)
    prices = []
    for record in records:
        val = record.get("value")
        if val is not None:
            try:
                prices.append({"period": record.get("period", ""), "value": float(val)})
            except (ValueError, TypeError):
                continue

    if not prices:
        return ConnectorResult(source_id="EIA", success=False, error="No valid price data")

    latest_price = prices[0]["value"]

    # Year-over-year change (compare latest to ~52 weeks ago)
    yoy_change_pct = 0.0
    if len(prices) >= 52:
        year_ago_price = prices[51]["value"]
        if year_ago_price > 0:
            yoy_change_pct = round((latest_price - year_ago_price) / year_ago_price * 100, 1)

    # Price volatility: standard deviation of weekly % changes over last 26 weeks
    volatility_pct = 0.0
    if len(prices) >= 26:
        weekly_changes = []
        for i in range(min(26, len(prices) - 1)):
            if prices[i + 1]["value"] > 0:
                pct_change = (prices[i]["value"] - prices[i + 1]["value"]) / prices[i + 1]["value"] * 100
                weekly_changes.append(pct_change)
        if weekly_changes:
            mean_change = sum(weekly_changes) / len(weekly_changes)
            variance = sum((c - mean_change) ** 2 for c in weekly_changes) / len(weekly_changes)
            volatility_pct = round(variance ** 0.5, 1)

    data = {
        "latest_price": latest_price,
        "yoy_change_pct": yoy_change_pct,
        "volatility_pct": volatility_pct,
        "period": prices[0]["period"],
        "prices_count": len(prices),
    }

    save_cache("eia", cache_params, data)
    return ConnectorResult(source_id="EIA", success=True, data=data)


def fetch_refinery_outages() -> ConnectorResult:
    """Fetch refinery utilization rate (proxy for outages).

    Lower utilization = more outages/maintenance. EIA reports weekly
    operable capacity utilization percentage.
    """
    api_key = get_credential("eia")
    if not api_key:
        return ConnectorResult(
            source_id="EIA", success=False,
            error="EIA_API_KEY not configured"
        )

    cache_params = {"query": "refinery_utilization"}
    cached = get_cached("eia", cache_params, max_age_hours=24)
    if cached:
        return ConnectorResult(source_id="EIA", success=True, data=cached, cached=True)

    # EIA reports utilization by PADD region, not national total.
    # Fetch all PADD regions and compute weighted average per period.
    url = f"{EIA_BASE}/petroleum/pnp/wiup/data/"
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[process][]": "YUP",     # % Utilization Refinery Operable Capacity
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 260,  # ~52 weeks * 5 PADDs
    }

    resp = fetch_with_retry(url, params=params, timeout=30)
    if not resp or resp.status_code != 200:
        logger.error("EIA refinery utilization query failed")
        return ConnectorResult(source_id="EIA", success=False, error="HTTP failed for EIA refinery")

    try:
        body = resp.json()
        records = body.get("response", {}).get("data", [])
    except Exception as e:
        return ConnectorResult(source_id="EIA", success=False, error=f"Parse error: {e}")

    # Group by period and average across PADDs
    from collections import defaultdict
    period_values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        val = record.get("value")
        period = record.get("period", "")
        if val is not None and period:
            try:
                period_values[period].append(float(val))
            except (ValueError, TypeError):
                continue

    if not period_values:
        return ConnectorResult(source_id="EIA", success=False, error="No utilization data")

    # Compute average utilization per week, sorted by period descending
    weekly_avg = []
    for period in sorted(period_values.keys(), reverse=True):
        vals = period_values[period]
        weekly_avg.append({"period": period, "util": sum(vals) / len(vals)})

    latest_util = round(weekly_avg[0]["util"], 1)
    utilization_values = [w["util"] for w in weekly_avg]
    avg_util = sum(utilization_values) / len(utilization_values)

    # Count weeks with utilization below 85% as proxy for significant outage periods
    outage_weeks = sum(1 for v in utilization_values if v < 85.0)

    data = {
        "latest_utilization_pct": latest_util,
        "avg_utilization_pct": round(avg_util, 1),
        "outage_weeks": outage_weeks,
        "period": weekly_avg[0]["period"],
    }

    save_cache("eia", cache_params, data)
    return ConnectorResult(source_id="EIA", success=True, data=data)
