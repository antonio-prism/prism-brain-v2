"""
PRISM Engine — GPR Index connector (Source A06).

Fetches the Geopolitical Risk Index from Matteo Iacoviello's website.
Used as a modifier for STR-GEO-* events and as a real-time tension indicator.

No API key required.
"""

import io
import logging

import pandas as pd

from .base import ConnectorResult, fetch_with_retry, get_cached, save_cache

logger = logging.getLogger(__name__)

GPR_URL_PRIMARY = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
GPR_URL_FALLBACK = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


def fetch_gpr_index() -> ConnectorResult:
    """
    Fetch and parse the monthly GPR Index.

    Returns the full time series, rolling baseline, current ratio, and calibrated modifier.
    """
    cache_params = {"source": "gpr_monthly"}
    cached = get_cached("gpr", cache_params, max_age_hours=168)  # 7 days
    if cached:
        return ConnectorResult(source_id="A06", success=True, data=cached, cached=True)

    # Try primary URL, then fallback
    resp = fetch_with_retry(GPR_URL_PRIMARY, timeout=30)
    if not resp or resp.status_code != 200:
        logger.warning("Primary GPR URL failed, trying fallback")
        resp = fetch_with_retry(GPR_URL_FALLBACK, timeout=30)

    if not resp or resp.status_code != 200:
        logger.error("Both GPR URLs failed")
        return ConnectorResult(source_id="A06", success=False, error="GPR download failed")

    try:
        gpr = pd.read_excel(io.BytesIO(resp.content), sheet_name=0)
    except Exception as e:
        return ConnectorResult(source_id="A06", success=False, error=f"Excel parse error: {e}")

    # Find the GPR column (might be 'GPR', 'GPRD', 'gpr', etc.)
    gpr_col = None
    for col in gpr.columns:
        if col.upper() in ("GPR", "GPRD"):
            gpr_col = col
            break
    if gpr_col is None:
        # Try the last numeric column as a heuristic
        numeric_cols = gpr.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            gpr_col = numeric_cols[-1]
        else:
            return ConnectorResult(source_id="A06", success=False, error="Cannot find GPR column")

    series = gpr[gpr_col].dropna()
    if len(series) < 60:
        return ConnectorResult(source_id="A06", success=False, error=f"Too few data points: {len(series)}")

    # Compute rolling 5-year (60-month) mean
    rolling_60m = series.rolling(60, min_periods=36).mean()
    ratio = series / rolling_60m
    ratio_clean = ratio.dropna()

    # Calibrate P5/P50/P95
    p5 = float(ratio_clean.quantile(0.05))
    p50 = float(ratio_clean.quantile(0.50))
    p95 = float(ratio_clean.quantile(0.95))

    floor = max(0.50, round(p5, 2))
    ceiling = min(3.00, round(p95, 2))

    current_ratio = float(ratio_clean.iloc[-1])
    current_modifier = round(max(floor, min(ceiling, current_ratio)), 2)

    data = {
        "n_observations": len(ratio_clean),
        "latest_gpr_value": round(float(series.iloc[-1]), 2),
        "rolling_60m_mean": round(float(rolling_60m.dropna().iloc[-1]), 2),
        "current_ratio": round(current_ratio, 2),
        "p5": round(p5, 2),
        "p50": round(p50, 2),
        "p95": round(p95, 2),
        "floor": floor,
        "ceiling": ceiling,
        "current_modifier": current_modifier,
    }

    save_cache("gpr", cache_params, data)
    return ConnectorResult(source_id="A06", success=True, data=data)


def get_gpr_modifier() -> dict:
    """
    Get the GPR-based geopolitical tension modifier.
    Used for STR-GEO-* events.
    """
    result = fetch_gpr_index()
    if not result.success:
        return {
            "modifier": 1.0,
            "status": "FALLBACK",
            "error": result.error,
        }

    return {
        "name": "GPR Index tension",
        "source_id": "A06",
        "modifier": result.data["current_modifier"],
        "status": "COMPUTED",
        "indicator_value": result.data["latest_gpr_value"],
        "indicator_unit": "GPR index points",
        "calibration": {
            "method": "ratio",
            "n_observations": result.data["n_observations"],
            "p5": result.data["p5"],
            "p50": result.data["p50"],
            "p95": result.data["p95"],
            "floor": result.data["floor"],
            "ceiling": result.data["ceiling"],
        },
    }
