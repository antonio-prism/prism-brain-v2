"""
PRISM Engine — NOAA CPC connector (Source A13).

Fetches the NAO (North Atlantic Oscillation) monthly index for
atmospheric blocking pattern detection (proxy C04).

No API key required.
"""

import io
import logging

import pandas as pd

from .base import ConnectorResult, fetch_with_retry, get_cached, save_cache

logger = logging.getLogger(__name__)

NAO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table"


def fetch_nao_index() -> ConnectorResult:
    """
    Fetch and parse the NAO monthly index from NOAA CPC.

    Returns full time series, blocking month counts per year, and current modifier.
    """
    cache_params = {"source": "nao_monthly"}
    cached = get_cached("noaa_cpc", cache_params, max_age_hours=168)  # 7 days
    if cached:
        return ConnectorResult(source_id="A13", success=True, data=cached, cached=True)

    resp = fetch_with_retry(NAO_URL, timeout=30)
    if not resp or resp.status_code != 200:
        logger.error("Failed to fetch NOAA NAO index")
        return ConnectorResult(source_id="A13", success=False, error="HTTP request failed")

    try:
        lines = resp.text.strip().split("\n")
        records = []
        for line in lines:
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            for month_idx, val in enumerate(parts[1:13], 1):
                try:
                    records.append({"year": year, "month": month_idx, "nao": float(val)})
                except ValueError:
                    pass

        nao = pd.DataFrame(records)
    except Exception as e:
        return ConnectorResult(source_id="A13", success=False, error=f"Parse error: {e}")

    if nao.empty:
        return ConnectorResult(source_id="A13", success=False, error="No data parsed")

    # Blocking proxy: count months per year with NAO < -1.0
    nao["blocking"] = nao["nao"] < -1.0
    blocking_per_year = nao.groupby("year")["blocking"].sum()

    # Average blocking months per year (observation window)
    obs_window = blocking_per_year.loc[2000:2024] if 2024 in blocking_per_year.index else blocking_per_year.tail(25)
    avg_blocking_months = float(obs_window.mean()) if len(obs_window) > 0 else 1.0

    # Current year blocking months
    current_year = nao["year"].max()
    current_year_blocking = int(blocking_per_year.get(current_year, 0))

    # Modifier: current year vs average
    if avg_blocking_months > 0:
        ratio = current_year_blocking / avg_blocking_months
    else:
        ratio = 1.0
    modifier = round(max(0.50, min(2.00, ratio)), 2)

    # Latest NAO value
    latest_nao = float(nao["nao"].iloc[-1])

    data = {
        "n_years": len(blocking_per_year),
        "year_range": f"{int(nao['year'].min())}-{int(nao['year'].max())}",
        "avg_blocking_months_per_year": round(avg_blocking_months, 2),
        "current_year": int(current_year),
        "current_year_blocking_months": current_year_blocking,
        "latest_nao_value": round(latest_nao, 2),
        "modifier": modifier,
        "blocking_per_year": {int(k): int(v) for k, v in blocking_per_year.items()},
    }

    save_cache("noaa_cpc", cache_params, data)
    return ConnectorResult(source_id="A13", success=True, data=data)


def get_nao_blocking_modifier() -> dict:
    """Get the NAO-based blocking pattern modifier for cold wave events."""
    result = fetch_nao_index()
    if not result.success:
        return {"modifier": 1.0, "status": "FALLBACK", "error": result.error}

    return {
        "name": "NAO blocking pattern",
        "source_id": "A13",
        "modifier": result.data["modifier"],
        "status": "COMPUTED",
        "indicator_value": result.data["latest_nao_value"],
        "indicator_unit": "NAO index",
        "avg_blocking_months": result.data["avg_blocking_months_per_year"],
        "current_year_blocking": result.data["current_year_blocking_months"],
    }
