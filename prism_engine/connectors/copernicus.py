"""
PRISM Engine — Copernicus / ERA5 temperature connector (Source A01).

Provides temperature anomaly modifier for PHY-CLI (heatwave) events.

Primary source: Published ERA5 European summer anomalies from the C3S
European State of the Climate (ESOTC) annual reports.  These are updated
once a year (April) and stored in era5_calibration.py — no CDS download
required, so the modifier is available instantly.

Optional: If CDS_API_KEY is configured and cdsapi is installed, the
connector can also download raw ERA5 monthly data for cross-validation
or to extend beyond the published anomaly table.
"""

import logging
import numpy as np

from .base import ConnectorResult
from ..config.credentials import get_credential

logger = logging.getLogger(__name__)

# Scaling constant derived via logistic regression (see era5_calibration.py)
_SCALING_CONSTANT = 0.21
_MODIFIER_FLOOR = 0.75
_MODIFIER_CEILING = 1.80


def _get_published_anomaly() -> dict | None:
    """Get the most recent published ERA5 anomaly from the calibration table.

    Returns dict with keys: year, anomaly_sigma, modifier — or None.
    """
    try:
        from ..computation.era5_calibration import get_anomaly_sigma_table
        sigma_table = get_anomaly_sigma_table()

        if not sigma_table:
            return None

        latest_year = max(sigma_table.keys())
        latest_sigma = sigma_table[latest_year]

        modifier = 1.0 + (latest_sigma * _SCALING_CONSTANT)
        modifier = round(float(np.clip(modifier, _MODIFIER_FLOOR, _MODIFIER_CEILING)), 2)

        return {
            "year": latest_year,
            "anomaly_sigma": latest_sigma,
            "modifier": modifier,
            "total_years": len(sigma_table),
        }
    except Exception as e:
        logger.error(f"Failed to load published ERA5 anomalies: {e}")
        return None


def get_temperature_modifier() -> dict:
    """Get the ERA5 temperature anomaly modifier for heatwave events.

    Uses the published C3S/ERA5 anomaly table (instant, no download).
    Falls back to neutral (1.0) only if the table is unavailable.
    """
    published = _get_published_anomaly()

    if published:
        return {
            "name": "ERA5 temperature anomaly",
            "source_id": "A01",
            "modifier": published["modifier"],
            "status": "COMPUTED",
            "indicator_value": published["anomaly_sigma"],
            "indicator_unit": "σ above 1991-2020 baseline",
            "data_year": published["year"],
            "observation_window": f"2000-{published['year']} ({published['total_years']}yr)",
            "calibration": {
                "method": "scaling",
                "scaling_formula": f"1.0 + (anomaly_sigma x {_SCALING_CONSTANT})",
                "scaling_constant": _SCALING_CONSTANT,
                "scaling_constant_status": "REGRESSION_DERIVED",
                "floor": _MODIFIER_FLOOR,
                "ceiling": _MODIFIER_CEILING,
            },
            "data_source": "C3S European State of the Climate (ERA5 reanalysis)",
        }

    # Fallback: neutral modifier
    return {
        "name": "ERA5 temperature anomaly",
        "source_id": "A01",
        "modifier": 1.0,
        "status": "FALLBACK",
        "error": "Published ERA5 anomaly table unavailable",
    }


def fetch_era5_temperature_cds(bbox: dict | None = None) -> ConnectorResult:
    """Download raw ERA5 data from CDS API (optional, for cross-validation).

    This is slow (30+ minutes) and only needed if you want to verify or
    extend beyond the published anomaly table.  Not used by the main
    engine modifier pipeline.
    """
    if bbox is None:
        bbox = {"north": 47, "south": 34, "west": -10, "east": 40}

    if not _check_cds_available():
        return ConnectorResult(
            source_id="A01", success=False,
            error="CDS API not available (missing cdsapi package or CDS_API_KEY)"
        )

    try:
        import cdsapi
        import xarray as xr
        from pathlib import Path
        from .base import get_cached, save_cache

        cache_params = {"source": "era5_t2m", "bbox": bbox}
        cached = get_cached("copernicus", cache_params, max_age_hours=720)
        if cached:
            return ConnectorResult(source_id="A01", success=True, data=cached, cached=True)

        data_dir = Path(__file__).parent.parent / "data" / "cache"
        api_key = get_credential("cds")
        client = cdsapi.Client(
            url="https://cds.climate.copernicus.eu/api",
            key=api_key,
        )
        output_file = str(data_dir / "era5_monthly_t2m_europe_summer.nc")

        client.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": "2m_temperature",
                "year": [str(y) for y in range(1991, 2026)],
                "month": ["06", "07", "08", "09"],
                "time": "00:00",
                "area": [bbox["north"], bbox["west"], bbox["south"], bbox["east"]],
                "data_format": "netcdf",
            },
            output_file,
        )

        ds = xr.open_dataset(output_file)
        t2m = ds["t2m"]
        weights = np.cos(np.deg2rad(t2m.latitude))
        monthly_mean = t2m.weighted(weights).mean(dim=["latitude", "longitude"])
        baseline = monthly_mean.sel(time=slice("1991", "2020")).groupby("time.month").mean()
        baseline_std = monthly_mean.sel(time=slice("1991", "2020")).groupby("time.month").std()
        anomaly = (monthly_mean.groupby("time.month") - baseline) / baseline_std
        latest_anomaly = float(anomaly.isel(time=-1).item())
        modifier = 1.0 + (latest_anomaly * _SCALING_CONSTANT)
        modifier = round(float(np.clip(modifier, _MODIFIER_FLOOR, _MODIFIER_CEILING)), 2)

        data = {
            "latest_anomaly_sigma": round(latest_anomaly, 2),
            "modifier": modifier,
            "scaling_constant": _SCALING_CONSTANT,
            "baseline_period": "1991-2020",
            "bbox": bbox,
        }

        ds.close()
        save_cache("copernicus", cache_params, data)
        return ConnectorResult(source_id="A01", success=True, data=data)

    except Exception as e:
        logger.error(f"CDS ERA5 fetch failed: {e}")
        return ConnectorResult(source_id="A01", success=False, error=str(e))


def _check_cds_available() -> bool:
    """Check if CDS API is configured and available."""
    try:
        import cdsapi  # noqa: F401
        key = get_credential("cds")
        return key is not None
    except ImportError:
        return False
