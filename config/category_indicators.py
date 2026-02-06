"""
PRISM Brain — Category-to-Indicator Mapping Configuration

Maps each event category prefix (e.g., "GEO", "CYBER") to the specific
data source indicators that should influence its probability calculation.

This fixes the critical accuracy bug where ALL events were using ALL indicators
regardless of relevance. Now each category only uses indicators that actually
relate to its risk domain.

Indicator names MUST match the exact keys returned by DataFetcher methods.
"""
import re
import hashlib
import math


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


# Approximate category sizes for severity scaling
CATEGORY_SIZES = {
    "GEO": 75, "PHYS": 60, "ENRG": 80, "CYBER": 65, "TECH": 85,
    "SUPL": 90, "SYST": 50, "CLIM": 100, "CYB": 100, "ECO": 50,
    "POL": 50, "CLI": 100
}


def get_event_sensitivity(event_id: str) -> dict:
    """
    Returns event-specific sensitivity profile for probability differentiation.
    Creates unique adjustments per event based on severity position within category.
    
    Returns dict with:
    - baseline_offset: adjustment to category baseline (float, -0.8 to +0.8)
    - weight_multipliers: dict of indicator_name -> multiplier
    - severity_factor: overall signal scaling (0.6 to 1.4)
    """
    prefix = get_category_prefix(event_id)
    category = CATEGORY_INDICATOR_MAP.get(prefix, {})
    indicators = category.get("indicators", [])
    
    # Extract event number from ID (e.g., GEO-001 -> 1, CYB_050 -> 50)
    num_match = re.search(r"(\d+)$", event_id)
    event_num = int(num_match.group(1)) if num_match else 1
    
    # Get category size
    cat_size = CATEGORY_SIZES.get(prefix, 75)
    
    # Severity position: 0.0 (most severe) to 1.0 (least severe)
    severity_pos = min(1.0, (event_num - 1) / max(cat_size - 1, 1))
    
    # Baseline offset: severe events get higher baseline, mild get lower
    baseline_offset = 0.8 * (1.0 - 2.0 * severity_pos)
    
    # Severity factor: amplifies signals for severe events
    severity_factor = 1.4 - 0.8 * severity_pos
    
    # Weight multipliers per indicator based on beta type and severity
    weight_multipliers = {}
    for ind in indicators:
        name = ind["name"]
        beta = ind["beta"]
        if beta == "direct_causal":
            mult = 1.3 - 0.6 * severity_pos
        elif beta == "strong_correlation":
            mult = 1.15 - 0.3 * severity_pos
        elif beta == "moderate_correlation":
            mult = 0.9 + 0.3 * severity_pos
        else:
            mult = 0.8 + 0.5 * severity_pos
        weight_multipliers[name] = mult
    
    # Add deterministic variation from event_id hash
    hash_val = int(hashlib.md5(event_id.encode()).hexdigest()[:8], 16)
    hash_noise = ((hash_val % 1000) / 1000.0 - 0.5) * 0.2
    baseline_offset += hash_noise
    
    # Clamp baseline_offset so final baseline stays in valid range
    baseline_offset = max(-1.5, min(1.5, baseline_offset))
    
    return {
        "baseline_offset": round(baseline_offset, 4),
        "weight_multipliers": weight_multipliers,
        "severity_factor": round(severity_factor, 4),
        "severity_position": round(severity_pos, 4)
    }

def validate_weights():
    issues = {}
    for cat, config in CATEGORY_INDICATOR_MAP.items():
        total = sum(ind["weight"] for ind in config["indicators"])
        if abs(total - 1.0) > 0.01:
            issues[cat] = total
    return issues
