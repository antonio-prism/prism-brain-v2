"""
PRISM Engine — Copernicus CDS connector (Source A01).

Fetches ERA5 climate reanalysis data for temperature anomalies,
soil moisture, and other climate indicators.

Requires CDS_API_KEY environment variable and ~/.cdsapirc configuration.

Note: CDS requests can be slow (30+ minutes for large datasets).
For Phase 1, we use cached summary statistics and only request fresh
data when the cache is stale.
"""

import logging
from pathlib import Path

import numpy as np

from .base import ConnectorResult, get_cached, save_cache
from ..config.credentials import get_credential

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "cache"


def _check_cds_available() -> bool:
    """Check if CDS API is configured and available."""
    try:
        import cdsapi  # noqa: F401
        key = get_credential("cds")
        return key is not None
    except ImportError:
        logger.warning("cdsapi not installed — CDS connector unavailable")
        return False


def fetch_era5_temperature(bbox: dict | None = None) -> ConnectorResult:
    """
    Fetch ERA5 monthly mean 2m temperature for Europe (summer months).

    For Phase 1: check cache first. If no cached data, attempt CDS request.
    CDS requests are asynchronous and can take 30+ minutes.

    bbox: {"north": 47, "south": 34, "west": -10, "east": 40} for EU-South
    """
    if bbox is None:
        bbox = {"north": 47, "south": 34, "west": -10, "east": 40}

    cache_params = {"source": "era5_t2m", "bbox": bbox}
    cached = get_cached("copernicus", cache_params, max_age_hours=720)  # 30 days
    if cached:
        return ConnectorResult(source_id="A01", success=True, data=cached, cached=True)

    if not _check_cds_available():
        return ConnectorResult(
            source_id="A01", success=False,
            error="CDS API not available (missing cdsapi package or CDS_API_KEY)"
        )

    try:
        import cdsapi
        import xarray as xr

        api_key = get_credential("cds")
        client = cdsapi.Client(
            url="https://cds.climate.copernicus.eu/api",
            key=api_key,
        )
        output_file = str(DATA_DIR / "era5_monthly_t2m_europe_summer.nc")

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

        # Process the downloaded data
        ds = xr.open_dataset(output_file)
        t2m = ds["t2m"]

        # Area-weighted spatial mean
        weights = np.cos(np.deg2rad(t2m.latitude))
        monthly_mean = t2m.weighted(weights).mean(dim=["latitude", "longitude"])

        # Baseline: 1991-2020 mean and std per calendar month
        baseline = monthly_mean.sel(time=slice("1991", "2020")).groupby("time.month").mean()
        baseline_std = monthly_mean.sel(time=slice("1991", "2020")).groupby("time.month").std()

        # Anomaly per month (in standard deviations)
        anomaly = (monthly_mean.groupby("time.month") - baseline) / baseline_std

        # Heatwave year = any year with at least 1 month where anomaly > 2.0σ
        yearly_max_anomaly = anomaly.groupby("time.year").max()
        heatwave_years = int((yearly_max_anomaly > 2.0).sum().item())
        total_years = len(yearly_max_anomaly.year)
        prior = round(heatwave_years / total_years, 4)

        # Latest anomaly for modifier
        latest_anomaly = float(anomaly.isel(time=-1).item())
        # Modifier: 1σ above → 21% increase in heatwave risk
        # Derived via logistic regression on 25yr of ERA5 European summer
        # anomalies vs EM-DAT heatwave events (see era5_calibration.py).
        modifier = 1.0 + (latest_anomaly * 0.21)
        modifier = round(float(np.clip(modifier, 0.75, 1.80)), 2)

        data = {
            "heatwave_years": heatwave_years,
            "total_years": total_years,
            "prior": prior,
            "latest_anomaly_sigma": round(latest_anomaly, 2),
            "modifier": modifier,
            "scaling_constant": 0.21,
            "scaling_constant_status": "REGRESSION_DERIVED",
            "baseline_period": "1991-2020",
            "observation_window": f"2000-2024 ({total_years}yr)",
            "bbox": bbox,
        }

        ds.close()
        save_cache("copernicus", cache_params, data)
        return ConnectorResult(source_id="A01", success=True, data=data)

    except Exception as e:
        logger.error(f"CDS ERA5 fetch failed: {e}")
        return ConnectorResult(source_id="A01", success=False, error=str(e))


def get_temperature_modifier() -> dict:
    """
    Get the ERA5 temperature anomaly modifier for heatwave events.
    Falls back to neutral (1.0) if CDS data unavailable.
    """
    result = fetch_era5_temperature()
    if not result.success:
        return {
            "name": "ERA5 temperature anomaly",
            "source_id": "A01",
            "modifier": 1.0,
            "status": "FALLBACK",
            "error": result.error,
        }

    return {
        "name": "ERA5 temperature anomaly",
        "source_id": "A01",
        "modifier": result.data["modifier"],
        "status": "COMPUTED",
        "indicator_value": result.data["latest_anomaly_sigma"],
        "indicator_unit": "σ above 1991-2020 baseline",
        "calibration": {
            "method": "scaling",
            "scaling_formula": "1.0 + (anomaly_sigma x 0.21)",
            "scaling_constant": 0.21,
            "scaling_constant_status": result.data.get("scaling_constant_status",
                                                        "REGRESSION_DERIVED"),
            "floor": 0.75,
            "ceiling": 1.80,
        },
    }
