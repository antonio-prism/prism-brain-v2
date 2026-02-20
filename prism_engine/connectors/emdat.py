"""
PRISM Engine — EM-DAT connector (Source A08).

Reads the EM-DAT disaster database from a local CSV/Excel file.
Handles column name inconsistencies between portal and HDX downloads.

Requires manual download: the user must place the file at
prism_engine/data/emdat_public.csv (or .xlsx).
"""

import logging
from pathlib import Path

import pandas as pd

from .base import ConnectorResult
from ..config.regions import REGIONS

logger = logging.getLogger(__name__)

# ISO Alpha-2 → Alpha-3 mapping for EM-DAT country filtering.
# EM-DAT uses 3-letter codes; our REGIONS config uses 2-letter codes.
ISO2_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CH": "CHE", "CY": "CYP",
    "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST", "ES": "ESP",
    "FI": "FIN", "FR": "FRA", "GB": "GBR", "GR": "GRC", "HR": "HRV",
    "HU": "HUN", "IE": "IRL", "IS": "ISL", "IT": "ITA", "LI": "LIE",
    "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT", "NL": "NLD",
    "NO": "NOR", "PL": "POL", "PT": "PRT", "RO": "ROU", "SE": "SWE",
    "SI": "SVN", "SK": "SVK", "TR": "TUR", "RS": "SRB", "ME": "MNE",
    "MK": "MKD", "AL": "ALB", "BA": "BIH", "XK": "XKX",
    # G7 extras
    "US": "USA", "JP": "JPN", "CA": "CAN",
    # Other major
    "AU": "AUS", "NZ": "NZL", "KR": "KOR", "IL": "ISR", "RU": "RUS",
    "CN": "CHN", "IN": "IND", "BR": "BRA", "MX": "MEX", "ZA": "ZAF",
    "SA": "SAU", "AE": "ARE", "SG": "SGP", "TW": "TWN", "TH": "THA",
    "MY": "MYS", "ID": "IDN", "PH": "PHL", "VN": "VNM", "PK": "PAK",
    "EG": "EGY", "NG": "NGA", "KE": "KEN", "CL": "CHL", "CO": "COL",
    "PE": "PER", "AR": "ARG",
}

EMDAT_DIR = Path(__file__).parent.parent / "data"
EMDAT_PATHS = [
    EMDAT_DIR / "emdat_public.csv",
    EMDAT_DIR / "emdat_public.xlsx",
    EMDAT_DIR / "emdat.csv",
    EMDAT_DIR / "emdat.xlsx",
]

# Column alias mapping — handles both portal and HDX formats
EMDAT_COLUMN_ALIASES = {
    "year": ["Year", "year", "Start Year", "start_year"],
    "iso": ["ISO", "iso3", "Country ISO", "ISO3", "iso"],
    "country": ["Country", "country", "Country Name"],
    "disaster_type": ["Disaster Type", "disaster_type", "Type"],
    "disaster_subtype": ["Disaster Subtype", "disaster_subtype", "Sub-Type", "Subtype"],
    "deaths": ["Total Deaths", "total_deaths", "Deaths"],
    "affected": ["No. Affected", "total_affected", "Total Affected"],
    "damage_usd": ["Total Damage (USD)", "total_damage_adj", "Total Damage",
                    "Total Damage, Adjusted ('000 US$)", "Reconstruction Costs ('000 US$)"],
    "event_name": ["Event Name", "event_name", "Disaster Name"],
    "dis_no": ["Dis No", "dis_no", "DisNo."],
}

# EM-DAT disaster subtype strings for PRISM events
EMDAT_EVENT_MAPPINGS = {
    # Climate Extremes & Weather Events
    "PHY-CLI-001": {"type_pattern": "Riverine flood", "countries": "EEA_EXTENDED"},
    "PHY-CLI-002": {"type_pattern": "Coastal flood|Storm surge", "countries": "EEA_EXTENDED"},
    "PHY-CLI-003": {"type_pattern": "Heat wave", "countries": "EEA_EXTENDED"},
    "PHY-CLI-004": {"type_pattern": "Drought", "countries": "EEA_EXTENDED"},
    "PHY-CLI-005": {"type_pattern": "Forest fire|Wildfire", "countries": "EEA_EXTENDED"},
    "PHY-CLI-006": {"type_pattern": "Cold wave|Extreme winter", "countries": "EEA_EXTENDED"},
    # Geophysical Disasters
    "PHY-GEO-001": {"type_pattern": "Earthquake", "countries": "ALL_SEISMIC"},
    "PHY-GEO-002": {"type_pattern": "Volcanic", "countries": "EEA_EXTENDED"},
    "PHY-GEO-003": {"type_pattern": "Landslide|Mudslide", "countries": "EEA_EXTENDED"},
    "PHY-GEO-004": {"type_pattern": "Tsunami", "countries": "ALL_SEISMIC"},
    "PHY-GEO-005": {"type_pattern": "Sinkhole|Ground collapse|Subsidence", "countries": "EEA_EXTENDED"},
    # Contamination & Pollution
    "PHY-POL-001": {"type_pattern": "Chemical spill|Industrial accident", "countries": "EEA_EXTENDED"},
    "PHY-POL-002": {"type_pattern": "Fog|Sand|Ash|Air pollution", "countries": "EEA_EXTENDED"},
    "PHY-POL-004": {"type_pattern": "Oil spill", "countries": "ALL_SEISMIC"},
    # Water Resources
    "PHY-WAT-001": {"type_pattern": "Drought", "countries": "EEA_EXTENDED"},
    "PHY-WAT-006": {"type_pattern": "Flood", "countries": "EEA_EXTENDED"},
    # Biological & Pandemic Risks
    "PHY-BIO-001": {"type_pattern": "Epidemic|Pandemic", "countries": "ALL_SEISMIC"},
    "PHY-BIO-002": {"type_pattern": "Epidemic", "countries": "EEA_EXTENDED"},
    "PHY-BIO-004": {"type_pattern": "Poisoning", "countries": "EEA_EXTENDED"},
    # Energy Supply (storms causing grid outages)
    "PHY-ENE-001": {"type_pattern": "Storm|Convective|Extra-tropical", "countries": "EEA_EXTENDED"},
    # Operational — events with physical disaster component
    "OPS-MAR-006": {"type_pattern": "Transport|Maritime", "countries": "ALL_SEISMIC"},
    "OPS-SUP-002": {"type_pattern": "Fire|Explosion|Industrial", "countries": "EEA_EXTENDED"},
    "OPS-MFG-002": {"type_pattern": "Fire|Explosion|Industrial", "countries": "EEA_EXTENDED"},
    "OPS-MFG-005": {"type_pattern": "Industrial accident", "countries": "EEA_EXTENDED"},
    "OPS-WHS-001": {"type_pattern": "Fire", "countries": "EEA_EXTENDED"},
}


def _find_emdat_file() -> Path | None:
    """Find the EM-DAT data file."""
    for path in EMDAT_PATHS:
        if path.exists():
            return path
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names regardless of source format."""
    rename_map = {}
    for canonical, aliases in EMDAT_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    return df.rename(columns=rename_map)


def _get_country_list(region_key: str) -> list[str]:
    """Get country ISO-3 codes for EM-DAT filtering.

    REGIONS uses ISO-2 codes; EM-DAT uses ISO-3.  This function converts.
    """
    if region_key == "ALL_SEISMIC":
        return []  # No country filter for seismic (uses all countries)
    iso2_list = REGIONS.get(region_key, [])
    # Convert ISO-2 → ISO-3 for EM-DAT matching
    return [ISO2_TO_ISO3.get(c, c) for c in iso2_list]


def load_emdat() -> pd.DataFrame | None:
    """
    Load and normalize the EM-DAT dataset.

    Returns normalized DataFrame or None if file not found.
    """
    path = _find_emdat_file()
    if path is None:
        logger.warning(
            "EM-DAT file not found. Download from public.emdat.be and place at "
            f"{EMDAT_DIR / 'emdat_public.csv'}"
        )
        return None

    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        else:
            df = pd.read_excel(path)
    except Exception as e:
        logger.error(f"Error reading EM-DAT file: {e}")
        return None

    df = _normalize_columns(df)
    logger.info(f"Loaded EM-DAT: {len(df)} records, columns: {list(df.columns)}")
    return df


def count_event_years(event_id: str, start_year: int = 2000,
                      end_year: int = 2024) -> ConnectorResult:
    """
    Count distinct years with at least 1 qualifying disaster event for a PRISM event.

    Uses EM-DAT disaster subtype pattern matching and region filtering.
    """
    mapping = EMDAT_EVENT_MAPPINGS.get(event_id)
    if mapping is None:
        return ConnectorResult(
            source_id="A08", success=False,
            error=f"No EM-DAT mapping for event {event_id}"
        )

    df = load_emdat()
    if df is None:
        return ConnectorResult(
            source_id="A08", success=False,
            error="EM-DAT file not available"
        )

    type_pattern = mapping["type_pattern"]
    countries = _get_country_list(mapping["countries"])

    # Filter by disaster type/subtype (check both columns if available)
    type_col = "disaster_subtype" if "disaster_subtype" in df.columns else "disaster_type"
    mask = df[type_col].str.contains(type_pattern, case=False, na=False)

    # Filter by year
    if "year" in df.columns:
        mask &= (df["year"] >= start_year) & (df["year"] <= end_year)

    # Filter by country (if applicable)
    if countries and "iso" in df.columns:
        mask &= df["iso"].isin(countries)

    filtered = df[mask]

    if "year" in filtered.columns:
        event_years = int(filtered["year"].nunique())
    else:
        event_years = 0

    total_years = end_year - start_year + 1
    prior = round(event_years / total_years, 4) if total_years > 0 else 0

    return ConnectorResult(
        source_id="A08",
        success=True,
        data={
            "event_id": event_id,
            "type_pattern": type_pattern,
            "region": mapping["countries"],
            "n_countries_filtered": len(countries) if countries else "ALL",
            "total_matching_records": len(filtered),
            "event_years": event_years,
            "total_years": total_years,
            "prior": prior,
            "formula": f"{event_years} event-years / {total_years} total years",
            "observation_window": f"{start_year}-{end_year}",
        },
    )
