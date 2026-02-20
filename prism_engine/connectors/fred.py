"""
PRISM Engine — FRED connector (Source A03).

Fetches economic time series from the Federal Reserve Economic Data API.
Used for STR-ECO-001 (recession), STR-FIN-* (financial), and proxy C01/C05.

Requires FRED_API_KEY environment variable.
"""

import logging
from datetime import datetime

import pandas as pd

from .base import ConnectorResult, fetch_with_retry, get_cached, save_cache
from ..config.credentials import get_credential

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Key FRED series for PRISM
FRED_SERIES = {
    "T10Y2Y": "10Y-2Y Treasury spread (yield curve inversion → recession signal)",
    "UNRATE": "Unemployment rate",
    "BAMLH0A0HYM2": "High Yield OAS (credit stress signal)",
    "NAPMNOI": "ISM Manufacturing PMI New Orders (demand pressure)",
    "ACDGNO": "Durable goods new orders - computers (semi demand proxy)",
    "CPIAUCSL": "CPI All Items (inflation tracker)",
    "DTWEXBGS": "Trade-weighted USD index (currency strength)",
    "DCOILWTICO": "WTI Crude Oil Price (energy cost signal)",
    "PPIACO": "PPI All Commodities (industrial materials cost signal)",
    "CSUSHPISA": "S&P/Case-Shiller US Home Price Index",
}


def fetch_series(series_id: str, start: str = "2000-01-01") -> ConnectorResult:
    """
    Fetch a single FRED time series.

    Returns the full observation set as a list of {date, value} dicts.
    """
    api_key = get_credential("fred")
    if not api_key:
        return ConnectorResult(
            source_id="A03", success=False,
            error="FRED_API_KEY not configured"
        )

    cache_params = {"series": series_id, "start": start}
    cached = get_cached("fred", cache_params, max_age_hours=24)
    if cached:
        return ConnectorResult(source_id="A03", success=True, data=cached, cached=True)

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }

    resp = fetch_with_retry(FRED_BASE, params=params, timeout=30)
    if not resp or resp.status_code != 200:
        logger.error(f"FRED query failed for {series_id}")
        return ConnectorResult(source_id="A03", success=False, error=f"HTTP failed for {series_id}")

    try:
        observations = resp.json().get("observations", [])
    except Exception as e:
        return ConnectorResult(source_id="A03", success=False, error=f"Parse error: {e}")

    # Filter out missing values (FRED uses "." for missing)
    clean_obs = []
    for obs in observations:
        if obs.get("value") not in (".", "", None):
            try:
                clean_obs.append({
                    "date": obs["date"],
                    "value": float(obs["value"]),
                })
            except (ValueError, KeyError):
                pass

    data = {
        "series_id": series_id,
        "description": FRED_SERIES.get(series_id, ""),
        "n_observations": len(clean_obs),
        "date_range": f"{clean_obs[0]['date']} to {clean_obs[-1]['date']}" if clean_obs else "N/A",
        "latest_value": clean_obs[-1]["value"] if clean_obs else None,
        "latest_date": clean_obs[-1]["date"] if clean_obs else None,
        "observations": clean_obs,
    }

    save_cache("fred", cache_params, data)
    return ConnectorResult(source_id="A03", success=True, data=data)


def get_yield_curve_modifier() -> dict:
    """
    Compute recession modifier from 10Y-2Y Treasury spread.
    Inverted yield curve (negative spread) is a strong recession predictor.
    """
    result = fetch_series("T10Y2Y")
    if not result.success:
        return {"modifier": 1.0, "status": "FALLBACK", "error": result.error}

    obs = result.data.get("observations", [])
    if not obs:
        return {"modifier": 1.0, "status": "FALLBACK", "error": "No observations"}

    values = pd.Series([o["value"] for o in obs])
    latest = values.iloc[-1]

    # Standard ratio method: current vs rolling 60-month baseline
    rolling_mean = values.rolling(60, min_periods=36).mean()
    if rolling_mean.dropna().empty:
        return {"modifier": 1.0, "status": "FALLBACK", "error": "Not enough data for rolling mean"}

    # For yield curve: inversion (negative values) increases recession risk
    # Modifier logic: negative spread → higher modifier
    # We invert the ratio since lower spread = more risk
    baseline_mean = float(rolling_mean.dropna().iloc[-1])
    if baseline_mean != 0:
        # Invert: when spread drops below baseline, modifier increases
        ratio = 1.0 + (baseline_mean - latest) * 0.5  # 0.5 scaling factor
    else:
        ratio = 1.0

    modifier = round(max(0.50, min(2.50, ratio)), 2)

    return {
        "name": "Yield curve inversion",
        "source_id": "A03",
        "modifier": modifier,
        "status": "COMPUTED",
        "series_id": "T10Y2Y",
        "indicator_value": round(latest, 2),
        "indicator_unit": "percentage points (10Y-2Y spread)",
        "baseline_mean": round(baseline_mean, 2),
    }


def get_credit_spread_modifier() -> dict:
    """
    Compute financial stress modifier from High Yield OAS.
    Widening credit spreads signal market stress and recession risk.
    """
    result = fetch_series("BAMLH0A0HYM2")
    if not result.success:
        return {"modifier": 1.0, "status": "FALLBACK", "error": result.error}

    obs = result.data.get("observations", [])
    if not obs:
        return {"modifier": 1.0, "status": "FALLBACK", "error": "No observations"}

    values = pd.Series([o["value"] for o in obs])
    latest = values.iloc[-1]

    # Standard ratio method
    rolling_mean = values.rolling(60, min_periods=36).mean()
    if rolling_mean.dropna().empty:
        return {"modifier": 1.0, "status": "FALLBACK", "error": "Not enough data"}

    baseline = float(rolling_mean.dropna().iloc[-1])
    ratio = latest / baseline if baseline > 0 else 1.0

    p5 = float(values.quantile(0.05))
    p95 = float(values.quantile(0.95))

    modifier = round(max(0.50, min(2.50, ratio)), 2)

    return {
        "name": "Credit spread stress",
        "source_id": "A03",
        "modifier": modifier,
        "status": "COMPUTED",
        "series_id": "BAMLH0A0HYM2",
        "indicator_value": round(latest, 2),
        "indicator_unit": "basis points (HY OAS)",
        "baseline_mean": round(baseline, 2),
    }


def get_durable_goods_modifier() -> dict:
    """
    Compute semiconductor demand proxy (C01) from durable goods new orders.
    modifier = current_quarter / rolling_20q_mean
    """
    result = fetch_series("ACDGNO")
    if not result.success:
        return {"modifier": 1.0, "status": "FALLBACK", "error": result.error}

    obs = result.data.get("observations", [])
    if not obs:
        return {"modifier": 1.0, "status": "FALLBACK", "error": "No observations"}

    values = pd.Series([o["value"] for o in obs])

    # Rolling 20-quarter (= ~60 months) mean
    rolling = values.rolling(20, min_periods=12).mean()
    if rolling.dropna().empty:
        return {"modifier": 1.0, "status": "FALLBACK", "error": "Not enough data"}

    latest = float(values.iloc[-1])
    baseline = float(rolling.dropna().iloc[-1])
    ratio = latest / baseline if baseline > 0 else 1.0

    modifier = round(max(0.50, min(2.00, ratio)), 2)

    return {
        "name": "Durable goods demand pressure (semi proxy C01)",
        "source_id": "A03",
        "modifier": modifier,
        "status": "COMPUTED",
        "series_id": "ACDGNO",
        "indicator_value": round(latest, 0),
        "indicator_unit": "millions USD",
        "baseline_mean": round(baseline, 0),
        "proxy": "C01",
    }


def get_pmi_modifier() -> dict:
    """
    Compute PMI new orders demand modifier (C05).
    PMI 50 = neutral. modifier = NAPMNOI / 50.
    """
    result = fetch_series("NAPMNOI")
    if not result.success:
        return {"modifier": 1.0, "status": "FALLBACK", "error": result.error}

    obs = result.data.get("observations", [])
    if not obs:
        return {"modifier": 1.0, "status": "FALLBACK", "error": "No observations"}

    latest = obs[-1]["value"]
    modifier = round(max(0.50, min(2.00, latest / 50.0)), 2)

    return {
        "name": "ISM PMI new orders demand pressure (C05)",
        "source_id": "A03",
        "modifier": modifier,
        "status": "COMPUTED",
        "series_id": "NAPMNOI",
        "indicator_value": round(latest, 1),
        "indicator_unit": "PMI index points",
        "neutral_value": 50,
        "proxy": "C05",
    }


def count_threshold_years(series_id: str, threshold: float, comparison: str,
                          start_year: int = 2000, end_year: int = 2024,
                          label: str = "") -> ConnectorResult:
    """
    Generic Method A prior: count years where a FRED series crosses a threshold.

    comparison modes:
      "above"     — value > threshold (e.g. credit spread > 600bp)
      "below"     — value < threshold (e.g. yield curve < 0)
      "yoy_above" — year-over-year % change > threshold (e.g. CPI YoY > 10%)
      "yoy_below" — year-over-year % change < threshold (e.g. house price YoY < -10%)
    """
    result = fetch_series(series_id, start=f"{start_year - 1}-01-01")
    if not result.success:
        return result

    obs = result.data.get("observations", [])
    if not obs:
        return ConnectorResult(source_id="A03", success=False, error="No observations")

    # Group observations by year → compute annual average
    year_values: dict[int, list[float]] = {}
    for o in obs:
        year = int(o["date"][:4])
        if year not in year_values:
            year_values[year] = []
        year_values[year].append(o["value"])

    year_avgs = {y: sum(v) / len(v) for y, v in year_values.items()}

    qualifying_years = set()
    for year in range(start_year, end_year + 1):
        if year not in year_avgs:
            continue
        avg = year_avgs[year]

        if comparison == "above":
            if avg > threshold:
                qualifying_years.add(year)
        elif comparison == "below":
            if avg < threshold:
                qualifying_years.add(year)
        elif comparison == "yoy_above":
            prev = year_avgs.get(year - 1)
            if prev and prev != 0:
                yoy_pct = ((avg - prev) / abs(prev)) * 100
                if yoy_pct > threshold:
                    qualifying_years.add(year)
        elif comparison == "yoy_below":
            prev = year_avgs.get(year - 1)
            if prev and prev != 0:
                yoy_pct = ((avg - prev) / abs(prev)) * 100
                if yoy_pct < threshold:
                    qualifying_years.add(year)

    total_years = end_year - start_year + 1
    prior = round(len(qualifying_years) / total_years, 4) if total_years > 0 else 0

    return ConnectorResult(
        source_id="A03",
        success=True,
        data={
            "series_id": series_id,
            "threshold": threshold,
            "comparison": comparison,
            "label": label or f"{series_id} {comparison} {threshold}",
            "qualifying_years": sorted(qualifying_years),
            "event_years": len(qualifying_years),
            "total_years": total_years,
            "prior": prior,
            "formula": f"{len(qualifying_years)} qualifying years / {total_years} total years",
            "observation_window": f"{start_year}-{end_year}",
        },
    )


def count_recession_years(start_year: int = 2000, end_year: int = 2024) -> ConnectorResult:
    """
    Count years with yield curve inversion (T10Y2Y < 0) as recession signal.
    Alternative/complement to World Bank GDP data.
    """
    result = fetch_series("T10Y2Y", start=f"{start_year}-01-01")
    if not result.success:
        return result

    obs = result.data.get("observations", [])
    inverted_years = set()
    for o in obs:
        year = int(o["date"][:4])
        if start_year <= year <= end_year and o["value"] < 0:
            inverted_years.add(year)

    total_years = end_year - start_year + 1

    return ConnectorResult(
        source_id="A03",
        success=True,
        data={
            "inverted_years": sorted(inverted_years),
            "inversion_count": len(inverted_years),
            "total_years": total_years,
            "inversion_rate": round(len(inverted_years) / total_years, 4),
        },
    )
