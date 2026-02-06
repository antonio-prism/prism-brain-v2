"""
PRISM Brain — Category-to-Indicator Mapping Configuration

Maps each event category prefix (e.g., "GEO", "CYBER") to the specific
data source indicators that should influence its probability calculation.

This fixes the critical accuracy bug where ALL events were using ALL indicators
regardless of relevance. Now each category only uses indicators that actually
relate to its risk domain.

Indicator names MUST match the exact keys returned by DataFetcher methods.
"""

# Beta parameters control how strongly an indicator influences probability.
# These must match the BETA_PARAMETERS dict in main.py:
#   direct_causal: 1.2  (strong direct relationship)
#   strong_correlation: 1.0  (clear statistical relationship)
#   moderate_correlation: 0.7  (some relationship)
#   weak_correlation: 0.4  (loose relationship)

CATEGORY_INDICATOR_MAP = {
    "GEO": {
        "description": "Natural catastrophes: earthquakes, tsunamis, volcanic eruptions, landslides",
        "default_baseline": 2,
        "indicators": [
            {"name": "usgs_earthquake_count",   "source": "USGS",       "weight": 0.25, "beta": "direct_causal",       "time_scale": "fast"},
            {"name": "usgs_significant_count",  "source": "USGS",       "weight": 0.20, "beta": "direct_causal",       "time_scale": "fast"},
            {"name": "usgs_max_magnitude",      "source": "USGS",       "weight": 0.20, "beta": "strong_correlation",  "time_scale": "fast"},
            {"name": "noaa_extreme_events",     "source": "NOAA",       "weight": 0.15, "beta": "strong_correlation",  "time_scale": "slow"},
            {"name": "noaa_climate_risk",       "source": "NOAA",       "weight": 0.10, "beta": "moderate_correlation","time_scale": "slow"},
            {"name": "gdelt_crisis_intensity",  "source": "GDELT",      "weight": 0.10, "beta": "weak_correlation",    "time_scale": "fast"},
        ]
    },
    "PHYS": {
        "description": "Physical infrastructure risks: bridges, dams, power grids, transportation",
        "default_baseline": 2,
        "indicators": [
            {"name": "usgs_earthquake_count",   "source": "USGS",       "weight": 0.20, "beta": "strong_correlation",  "time_scale": "fast"},
            {"name": "usgs_max_magnitude",      "source": "USGS",       "weight": 0.15, "beta": "strong_correlation",  "time_scale": "fast"},
            {"name": "noaa_extreme_events",     "source": "NOAA",       "weight": 0.20, "beta": "direct_causal",       "time_scale": "slow"},
            {"name": "noaa_climate_risk",       "source": "NOAA",       "weight": 0.15, "beta": "moderate_correlation","time_scale": "slow"},
            {"name": "eia_crude_oil_price",     "source": "EIA",        "weight": 0.10, "beta": "weak_correlation",    "time_scale": "medium"},
            {"name": "gdelt_crisis_intensity",  "source": "GDELT",      "weight": 0.10, "beta": "weak_correlation",    "time_scale": "fast"},
            {"name": "world_bank_gdp_growth",   "source": "WORLD_BANK", "weight": 0.10, "beta": "weak_correlation",    "time_scale": "slow"},
        ]
    },
    "ENRG": {
        "description": "Energy sector risks: oil spills, grid failures, price shocks, supply disruptions",
        "default_baseline": 3,
        "indicators": [
            {"name": "eia_crude_oil_price",     "source": "EIA",        "weight": 0.25, "beta": "direct_causal",       "time_scale": "fast"},
            {"name": "eia_natural_gas_price",   "source": "EIA",        "weight": 0.20, "beta": "direct_causal",       "time_scale": "fast"},
            {"name": "eia_energy_volatility",   "source": "EIA",        "weight": 0.15, "beta": "strong_correlation",  "time_scale": "fast"},
            {"name": "eia_strategic_reserve_level","source": "EIA",     "weight": 0.10, "beta": "moderate_correlation","time_scale": "medium"},
            {"name": "acled_conflict_events",   "source": "ACLED",      "weight": 0.10, "beta": "moderate_correlation","time_scale": "medium"},
            {"name": "gdelt_crisis_intensity",  "source": "GDELT",      "weight": 0.10, "beta": "weak_correlation",    "time_scale": "fast"},
            {"name": "fred_vix_index",          "source": "FRED",       "weight": 0.10, "beta": "weak_correlation",    "time_scale": "fast"},
        ]
    },
    "CYBER": {
        "description": "Cyber threats: ransomware, data breaches, critical infrastructure attacks",
        "default_baseline": 4,
        "indicators": [
            {"name": "otx_threat_pulse_count",  "source": "OTX",        "weight": 0.20, "beta": "direct_causal",       "time_scale": "fast"},
            {"name": "otx_malware_indicators",  "source": "OTX",        "weight": 0.15, "beta": "strong_correlation",  "time_scale": "fast"},
            {"name": "otx_ransomware_activity", "source": "OTX",        "weight": 0.15, "beta": "direct_causal",       "time_scale": "fast"},
            {"name": "cisa_recent_kev",         "source": "CISA",       "weight": 0.15, "beta": "direct_causal",       "time_scale": "fast"},
            {"name": "cisa_total_kev",          "source": "CISA",       "weight": 0.10, "beta": "strong_correlation",  "time_scale": "fast"},
            {"name": "nvd_critical_count",      "source": "NVD",        "weight": 0.15, "beta": "strong_correlation",  "time_scale": "fast"},
            {"name": "nvd_total_cves",          "source": "NVD",        "weight": 0.10, "beta": "moderate_correlation","time_scale": "fast"},
        ]
    },
    "TECH": {
        "description": "Technology risks: AI failures, semiconductor shortages, tech monopoly, digital divide",
        "default_baseline": 3,
        "indicators": [
            {"name": "nvd_total_cves",          "source": "NVD",        "weight": 0.15, "beta": "moderate_correlation","time_scale": "fast"},
            {"name": "nvd_critical_count",      "source": "NVD",        "weight": 0.15, "beta": "strong_correlation",  "time_scale": "fast"},
            {"name": "otx_threat_pulse_count",  "source": "OTX",        "weight": 0.15, "beta": "moderate_correlation","time_scale": "fast"},
            {"name": "cisa_recent_kev",         "source": "CISA",       "weight": 0.10, "beta": "moderate_correlation","time_scale": "fast"},
            {"name": "fred_unemployment_rate",  "source": "FRED",       "weight": 0.10, "beta": "weak_correlation",    "time_scale": "medium"},
            {"name": "world_bank_gdp_growth",   "source": "WORLD_BANK", "weight": 0.10, "beta": "weak_correlation",    "time_scale": "slow"},
            {"name": "gdelt_event_volume",      "source": "GDELT",      "weight": 0.10, "beta": "weak_correlation",    "time_scale": "fast"},
            {"name": "imf_world_gdp_growth",    "source": "IMF",        "weight": 0.15, "beta": "moderate_correlation","time_scale": "slow"},
        ]
    },
    "SUPL": {
        "description": "Supply chain risks: shipping disruptions, raw material shortages, trade wars",
        "default_baseline": 3,
        "indicators": [
            {"name": "eia_crude_oil_price",     "source": "EIA",        "weight": 0.15, "beta": "strong_correlation",  "time_scale": "fast"},
            {"name": "eia_energy_volatility",   "source": "EIA",        "weight": 0.10, "beta": "moderate_correlation","time_scale": "fast"},
            {"name": "fao_food_price_index",    "source": "FAO",        "weight": 0.15, "beta": "strong_correlation",  "time_scale": "medium"},
            {"name": "fao_supply_stress",       "source": "FAO",        "weight": 0.15, "beta": "direct_causal",       "time_scale": "medium"},
            {"name": "acled_conflict_events",   "source": "ACLED",      "weight": 0.10, "beta": "moderate_correlation","time_scale": "medium"},
            {"name": "gdelt_crisis_intensity",  "source": "GDELT",      "weight": 0.10, "beta": "moderate_correlation","time_scale": "fast"},
            {"name": "world_bank_gdp_growth",   "source": "WORLD_BANK", "weight": 0.10, "beta": "weak_correlation",    "time_scale": "slow"},
            {"name": "fred_vix_index",          "source": "FRED",       "weight": 0.15, "beta": "moderate_correlation","time_scale": "fast"},
        ]
    },
    "SYST": {
        "description": "Systemic risks: financial contagion, pandemic cascades, infrastructure interdependence",
        "default_baseline": 2,
        "indicators": [
            {"name": "fred_vix_index",          "source": "FRED",       "weight": 0.15, "beta": "direct_causal",       "time_scale": "fast"},
            {"name": "fred_unemployment_rate",  "source": "FRED",       "weight": 0.10, "beta": "strong_correlation",  "time_scale": "medium"},
            {"name": "fred_inflation_rate",     "source": "FRED",       "weight": 0.10, "beta": "moderate_correlation","time_scale": "medium"},
            {"name": "imf_world_gdp_growth",    "source": "IMF",        "weight": 0.10, "beta": "strong_correlation",  "time_scale": "slow"},
            {"name": "world_bank_gdp_growth",   "source": "WORLD_BANK", "weight": 0.10, "beta": "moderate_correlation","time_scale": "slow"},
            {"name": "acled_conflict_events",   "source": "ACLED",      "weight": 0.10, "beta": "moderate_correlation","time_scale": "medium"},
            {"name": "gdelt_crisis_intensity",  "source": "GDELT",      "weight": 0.10, "beta": "moderate_correlation","time_scale": "fast"},
            {"name": "eia_energy_volatility",   "source": "EIA",        "weight": 0.10, "beta": "weak_correlation",    "time_scale": "fast"},
            {"name": "otx_threat_pulse_count",  "source": "OTX",        "weight": 0.05, "beta": "weak_correlation",    "time_scale": "fast"},
            {"name": "fao_food_security_risk",  "source": "FAO",        "weight": 0.10, "beta": "moderate_correlation","time_scale": "slow"},
        ]
    },
    "CLIM": {
        "description": "Climate risks: sea level rise, droughts, wildfires, biodiversity loss",
        "default_baseline": 3,
        "indicators": [
            {"name": "noaa_temp_anomaly",       "source": "NOAA",       "weight": 0.25, "beta": "direct_causal",       "time_scale": "slow"},
            {"name": "noaa_extreme_events",     "source": "NOAA",       "weight": 0.20, "beta": "direct_causal",       "time_scale": "slow"},
            {"name": "noaa_climate_risk",       "source": "NOAA",       "weight": 0.15, "beta": "strong_correlation",  "time_scale": "slow"},
            {"name": "noaa_precipitation_index","source": "NOAA",       "weight": 0.10, "beta": "moderate_correlation","time_scale": "slow"},
            {"name": "fao_food_security_risk",  "source": "FAO",        "weight": 0.15, "beta": "strong_correlation",  "time_scale": "slow"},
            {"name": "fao_food_price_index",    "source": "FAO",        "weight": 0.10, "beta": "moderate_correlation","time_scale": "medium"},
            {"name": "eia_crude_oil_price",     "source": "EIA",        "weight": 0.05, "beta": "weak_correlation",    "time_scale": "fast"},
        ]
    },
    "CYB": {
        "description": "Legacy cyber event (maps to same indicators as CYBER)",
        "default_baseline": 4,
        "indicators": [
            {"name": "otx_threat_pulse_count",  "source": "OTX",        "weight": 0.20, "beta": "direct_causal",       "time_scale": "fast"},
            {"name": "otx_ransomware_activity", "source": "OTX",        "weight": 0.15, "beta": "direct_causal",       "time_scale": "fast"},
            {"name": "cisa_recent_kev",         "source": "CISA",       "weight": 0.20, "beta": "direct_causal",       "time_scale": "fast"},
            {"name": "nvd_critical_count",      "source": "NVD",        "weight": 0.20, "beta": "strong_correlation",  "time_scale": "fast"},
            {"name": "nvd_total_cves",          "source": "NVD",        "weight": 0.15, "beta": "moderate_correlation","time_scale": "fast"},
            {"name": "otx_malware_indicators",  "source": "OTX",        "weight": 0.10, "beta": "moderate_correlation","time_scale": "fast"},
        ]
    },
    "ECO": {
        "description": "Legacy economic event",
        "default_baseline": 3,
        "indicators": [
            {"name": "fred_unemployment_rate",  "source": "FRED",       "weight": 0.20, "beta": "direct_causal",       "time_scale": "medium"},
            {"name": "fred_inflation_rate",     "source": "FRED",       "weight": 0.15, "beta": "strong_correlation",  "time_scale": "medium"},
            {"name": "fred_vix_index",          "source": "FRED",       "weight": 0.15, "beta": "strong_correlation",  "time_scale": "fast"},
            {"name": "world_bank_gdp_growth",   "source": "WORLD_BANK", "weight": 0.20, "beta": "direct_causal",       "time_scale": "slow"},
            {"name": "imf_world_gdp_growth",    "source": "IMF",        "weight": 0.15, "beta": "strong_correlation",  "time_scale": "slow"},
            {"name": "fred_fed_funds_rate",     "source": "FRED",       "weight": 0.15, "beta": "moderate_correlation","time_scale": "medium"},
        ]
    },
    "POL": {
        "description": "Legacy geopolitical event",
        "default_baseline": 3,
        "indicators": [
            {"name": "acled_conflict_events",   "source": "ACLED",      "weight": 0.25, "beta": "direct_causal",       "time_scale": "medium"},
            {"name": "acled_fatalities",        "source": "ACLED",      "weight": 0.15, "beta": "strong_correlation",  "time_scale": "medium"},
            {"name": "gdelt_crisis_intensity",  "source": "GDELT",      "weight": 0.20, "beta": "strong_correlation",  "time_scale": "fast"},
            {"name": "gdelt_event_volume",      "source": "GDELT",      "weight": 0.15, "beta": "moderate_correlation","time_scale": "fast"},
            {"name": "acled_instability_index", "source": "ACLED",      "weight": 0.15, "beta": "strong_correlation",  "time_scale": "medium"},
            {"name": "world_bank_gdp_growth",   "source": "WORLD_BANK", "weight": 0.10, "beta": "weak_correlation",    "time_scale": "slow"},
        ]
    },
    "CLI": {
        "description": "Legacy climate event (maps to same indicators as CLIM)",
        "default_baseline": 3,
        "indicators": [
            {"name": "noaa_temp_anomaly",       "source": "NOAA",       "weight": 0.25, "beta": "direct_causal",       "time_scale": "slow"},
            {"name": "noaa_extreme_events",     "source": "NOAA",       "weight": 0.20, "beta": "direct_causal",       "time_scale": "slow"},
            {"name": "noaa_climate_risk",       "source": "NOAA",       "weight": 0.20, "beta": "strong_correlation",  "time_scale": "slow"},
            {"name": "fao_food_security_risk",  "source": "FAO",        "weight": 0.20, "beta": "strong_correlation",  "time_scale": "slow"},
            {"name": "fao_food_price_index",    "source": "FAO",        "weight": 0.15, "beta": "moderate_correlation","time_scale": "medium"},
        ]
    },
}


def get_category_prefix(event_id: str) -> str:
    for sep in ('_', '-'):
        if sep in event_id:
            prefix = event_id.split(sep)[0]
            return prefix
    return event_id


def get_indicators_for_event(event_id: str) -> list:
    prefix = get_category_prefix(event_id)
    category = CATEGORY_INDICATOR_MAP.get(prefix, {})
    return category.get("indicators", [])


def get_all_categories() -> list:
    return list(CATEGORY_INDICATOR_MAP.keys())


def get_default_baseline(event_id: str) -> int:
    """Get the default baseline scale (1-5) for an event based on its category."""
    prefix = get_category_prefix(event_id)
    category = CATEGORY_INDICATOR_MAP.get(prefix, {})
    return category.get("default_baseline", 3)


def validate_weights():
    issues = {}
    for cat, config in CATEGORY_INDICATOR_MAP.items():
        total = sum(ind["weight"] for ind in config["indicators"])
        if abs(total - 1.0) > 0.01:
            issues[cat] = total
    return issues
