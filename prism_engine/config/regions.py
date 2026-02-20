"""
PRISM Engine — Region definitions.

All country lists (ISO 3166-1 alpha-2) and geographic bounding boxes
used for filtering events by observation region.
"""

REGIONS = {
    "EU27": [
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
        "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
        "PL", "PT", "RO", "SK", "SI", "ES", "SE"
    ],
    "EEA_EXTENDED": [
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
        "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
        "PL", "PT", "RO", "SK", "SI", "ES", "SE", "GB", "CH", "NO", "IS", "LI"
    ],
    "OECD_TRADING": [
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
        "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
        "PL", "PT", "RO", "SK", "SI", "ES", "SE",
        "US", "GB", "JP", "KR", "CA", "AU", "CH", "MX"
    ],
    "TOP20_SUPPLIERS": [
        "CN", "US", "GB", "CH", "RU", "NO", "JP", "KR", "TR", "IN",
        "BR", "VN", "TW", "TH", "SA", "ID", "MY", "MX", "UA", "ZA"
    ],
}

BOUNDING_BOXES = {
    "europe": {"north": 72, "south": 34, "west": -25, "east": 40},
    "europe_south": {"north": 47, "south": 34, "west": -10, "east": 40},
    "europe_north": {"north": 72, "south": 55, "west": -25, "east": 40},
}

SEISMIC_ZONES = {
    "mediterranean": {"north": 45, "south": 34, "west": -5, "east": 40},
    "japan": {"north": 46, "south": 30, "west": 128, "east": 146},
    "us_west_coast": {"north": 50, "south": 32, "west": -130, "east": -115},
    "mexico_central_am": {"north": 20, "south": 14, "west": -105, "east": -85},
    "indonesia": {"north": 6, "south": -11, "west": 95, "east": 141},
    "chile_peru": {"north": -15, "south": -45, "west": -80, "east": -68},
}

CHOKEPOINTS = {
    "suez": {"lat": 30.5, "lon": 32.3, "radius_km": 500},
    "panama": {"lat": 9.1, "lon": -79.7, "radius_km": 200},
    "malacca": {"lat": 2.5, "lon": 101.5, "radius_km": 300},
    "hormuz": {"lat": 26.5, "lon": 56.3, "radius_km": 200},
    "bab_al_mandab": {"lat": 12.6, "lon": 43.3, "radius_km": 300},
    "turkish_straits": {"lat": 41.0, "lon": 29.0, "radius_km": 100},
}

# Region assignment per event domain prefix
DOMAIN_REGION_MAP = {
    "PHY-CLI": "europe",
    "PHY-ENE": "EEA_EXTENDED",
    "PHY-MAT": "GLOBAL",
    "PHY-WAT": "europe",
    "PHY-GEO": "ALL_SEISMIC",
    "PHY-POL": "EEA_EXTENDED",
    "PHY-BIO": "GLOBAL",
    "STR-GEO": "TOP20_SUPPLIERS",
    "STR-TRD": "OECD_TRADING",
    "STR-REG": "EU27_US_GB",
    "STR-ECO": "OECD_TRADING",
    "STR-ENP": "EU27_US_GB",
    "STR-TEC": "OECD_TRADING",
    "STR-FIN": "GLOBAL",
    "DIG": "GLOBAL",
    "OPS": "GLOBAL",
}


def get_countries_for_region(region_name: str) -> list[str]:
    """Get ISO country codes for a named region."""
    if region_name == "GLOBAL":
        return []  # No country filter = worldwide
    if region_name == "EU27_US_GB":
        return REGIONS["EU27"] + ["US", "GB"]
    if region_name == "ALL_SEISMIC":
        return []  # Use bounding boxes instead
    return REGIONS.get(region_name, [])


def get_bbox_for_event(event_id: str) -> dict | None:
    """Get the geographic bounding box for an event's region."""
    prefix = "-".join(event_id.split("-")[:2])
    region = DOMAIN_REGION_MAP.get(prefix)
    if region and region in BOUNDING_BOXES:
        return BOUNDING_BOXES[region]
    return None
