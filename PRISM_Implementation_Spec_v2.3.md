# PRISM Brain Probability Engine — Claude Code Implementation Specification
## Version 2.3 FINAL | February 2026
## Purpose: Every number traceable, every decision unambiguous, zero judgment calls for Claude Code

---

# SECTION 0: CONTEXT AND INTEGRATION INSTRUCTIONS

## 0.1 What Exists Today

There is a working app with a backend that has **hardcoded probability base rates** for 174 risk events. The event IDs follow the format `PHY-CLI-003`, `DIG-RDE-001`, etc. The current system stores a single `base_rate` float per event and uses it directly.

## 0.2 What This Spec Replaces

This spec replaces the hardcoded base rates with a **two-layer dynamic probability engine**. The output for each event is a JSON object (see Section 9.2) that includes the prior, its derivation trail, current modifier values, and the resulting P_global.

## 0.3 Integration Approach

Claude Code should implement this as a **standalone probability service** (a Python module or microservice) that the existing app calls instead of reading hardcoded rates. The integration contract is:

```python
# EXISTING APP calls:
base_rate = get_base_rate(event_id="PHY-CLI-003")  # returns 0.065 (hardcoded)

# NEW APP calls:
result = prism_engine.compute(event_id="PHY-CLI-003")
base_rate = result["layer1"]["p_global"]  # returns dynamically computed value
# Full derivation trail available in result["layer1"]["derivation"]
```

## 0.4 Implementation Order — CRITICAL

Claude Code must follow this exact sequence:

1. **First**: Build the data layer (API connectors, data downloaders) — Section 4
2. **Second**: Build the computation layer (prior derivation, modifier calibration) — Sections 5-6
3. **Third**: Build the orchestration layer (event→source mapping, full pipeline) — Section 7
4. **Fourth**: Build the output layer (JSON output, validation) — Section 9
5. **Last**: Integrate with existing app (replace hardcoded rates) — Section 0.3

DO NOT try to build everything at once. Build and test each layer independently.

## 0.5 Handling API Failures and Missing Data

Claude Code MUST handle these scenarios gracefully:

| Scenario | Behavior |
|---|---|
| API returns error / timeout | Retry 3 times with exponential backoff (1s, 5s, 15s). If all fail, use the last known good value. Log the failure. |
| API returns empty data | Use the hardcoded base rate from the existing app as fallback. Tag the event with `"data_status": "FALLBACK_HARDCODED"`. |
| API key missing / invalid | Skip that source entirely. For events that depend on it, use hardcoded fallback. Log which events are affected. |
| EM-DAT file not downloaded yet | Many events depend on EM-DAT. Claude Code should check for the EM-DAT CSV at startup. If missing, print clear instructions: "Download EM-DAT data from public.emdat.be and place at ./data/emdat_public.csv". Use hardcoded fallbacks until available. |
| Copernicus CDS request takes >30 minutes | CDS requests can be slow. Queue them asynchronously. Use cached data if available. |

**FALLBACK RULE: The system must ALWAYS return a probability for every event, even if all APIs are down.** The fallback chain is: computed value → last cached value → hardcoded base rate from current app.

## 0.6 File Structure

```
prism_engine/
├── config/
│   ├── regions.py          # Section 2: All country lists and bounding boxes
│   ├── sources.py          # Section 3: Source definitions and API endpoints
│   ├── event_mapping.py    # Section 7: Event → source → method mapping for all 174 events
│   └── credentials.py      # Section 10: API key management (reads from env vars)
├── connectors/
│   ├── copernicus.py       # Section 4.1-4.2: CDS API calls
│   ├── usgs.py             # Section 4.3: USGS ComCat
│   ├── fred.py             # Section 4.7: FRED economic data
│   ├── nvd.py              # Section 4.5: NIST NVD
│   ├── cisa.py             # Section 4.6: CISA KEV
│   ├── gpr.py              # Section 4.4: GPR Index
│   ├── acled.py            # Section 4.10: ACLED conflict data
│   ├── emdat.py            # Section 4.9: EM-DAT disaster data
│   ├── noaa.py             # Section 4.8: NOAA CPC indices
│   └── entso_e.py          # ENTSO-E energy data
├── computation/
│   ├── priors.py           # Section 5: Method A, B, C prior derivation
│   ├── modifiers.py        # Section 6: Modifier calibration
│   ├── formulas.py         # Section 8: P_global and P_client calculations
│   └── validation.py       # Section 9.1: Validation rules
├── data/
│   ├── cache/              # Cached API responses (auto-managed)
│   ├── manual/             # Type B manual entry data (JSON files, one per source per year)
│   └── emdat_public.csv    # EM-DAT download (manually placed)
├── output/
│   └── events/             # One JSON file per event (Section 9.2 schema)
├── manual_entry/
│   └── templates/          # JSON templates for Type B annual data entry
├── engine.py               # Main orchestrator: compute(event_id) → JSON
├── fallback.py             # Hardcoded base rates from current app (Section 0.5)
└── tests/
    ├── test_connectors.py  # Test each API connector independently
    ├── test_priors.py      # Test prior calculations against known values
    └── test_integration.py # End-to-end: compute all 10 Phase 1 events
```

## 0.7 Technology Stack

```
Python >= 3.10
Required packages:
  cdsapi          # Copernicus CDS
  xarray          # NetCDF processing (for ERA5 data)
  netCDF4         # NetCDF file support
  numpy           # Numerical computation
  pandas          # Data manipulation
  requests        # HTTP API calls
  openpyxl        # Excel file reading (GPR index)
  entsoe-py       # ENTSO-E wrapper (optional, Phase 2+)
  schedule        # Job scheduling for auto-refresh (Phase 4)
```

## 0.8 Before Starting: What Claude Code Needs

**About the human user:** The product owner is not a developer. Do not ask them technical questions. Figure out codebase details, architecture decisions, and library choices autonomously. See the Session Brief for detailed guidance on how to communicate with the human.

**Things Claude Code must figure out on its own by reading the codebase:**
1. Where base rates are currently stored and how they're consumed
2. What tech stack the app uses and how to integrate the Python engine
3. Where legacy data-fetching code lives (there is old, unused logic from a previous attempt — remove it and build fresh)
4. How to export the current 174 hardcoded rates for use as fallback values

**Things Claude Code must ask the human to do (with step-by-step, non-technical instructions):**
1. Register for free API keys (list each service with exact URLs and what to click)
2. Download the EM-DAT CSV file (walk them through the registration and download)
3. Confirm which 10 events to prototype first (present by name, not by ID)
4. Decide whether the app should keep showing current rates during the build, or can show a "recalculating" status

---

# SECTION 1: ARCHITECTURAL RULES

## 1.1 Layer 1 vs Layer 2 Scope

**IMMUTABLE RULE: Layer 1 = "Did the event happen in the world?" Layer 2 = "Did it hit this client?"**

- Layer 1 (P_global): Annual probability that this class of event occurs at a systemic level within the defined observation region. This is a FACT ABOUT THE WORLD. It does NOT include any "probability of material impact" or "probability it affects a specific organization."
- Layer 2 (P_client): Adjusts P_global for whether a specific client is exposed, vulnerable, and resilient.

**MIGRATION RULE for v2 priors:** If any v2 prior was calculated as `P_event × P_material_impact`, strip the P_material_impact multiplier. The new prior = P_event only. Document the stripped multiplier in a `layer2_suggested_exposure` field for use later.

## 1.2 Observation Window

**Default observation window: 25 years (2000–2024).** Use shorter windows only when:
- The risk didn't exist before a certain date (e.g., cloud sovereignty → start from 2010)
- Data source starts later (e.g., ACLED comprehensive coverage from 2018)

Document the window for every event. Format: `observation_window: "2000-2024 (25yr)"`

## 1.3 Annual Time Horizon

All probabilities are **annual**: "What is the probability this event occurs at least once within a 12-month period?" Not monthly, not per-incident.

---

# SECTION 2: REGION DEFINITIONS

## 2.1 Country Lists (ISO 3166-1 alpha-2)

```python
REGIONS = {
    "EU27": [
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
        "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
        "PL", "PT", "RO", "SK", "SI", "ES", "SE"
    ],
    "EEA_EXTENDED": [  # EU27 + EEA + UK + CH
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
        "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
        "PL", "PT", "RO", "SK", "SI", "ES", "SE", "GB", "CH", "NO", "IS", "LI"
    ],
    "OECD_TRADING": [  # Major trading economies
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
        "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
        "PL", "PT", "RO", "SK", "SI", "ES", "SE",
        "US", "GB", "JP", "KR", "CA", "AU", "CH", "MX"
    ],
    "TOP20_SUPPLIERS": [  # Top 20 EU import partners by value
        "CN", "US", "GB", "CH", "RU", "NO", "JP", "KR", "TR", "IN",
        "BR", "VN", "TW", "TH", "SA", "ID", "MY", "MX", "UA", "ZA"
    ]
}
```

## 2.2 Geographic Bounding Boxes

```python
BOUNDING_BOXES = {
    "europe": {"north": 72, "south": 34, "west": -25, "east": 40},
    "europe_south": {"north": 47, "south": 34, "west": -10, "east": 40},  # Mediterranean basin
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
```

## 2.3 Region Assignment per Domain

| Domain Prefix | Region Used for Event Counting |
|---|---|
| PHY-CLI-* (Climate) | `europe` bounding box for all EU-relevant climate events |
| PHY-ENE-* (Energy) | `EEA_EXTENDED` country list |
| PHY-MAT-* (Materials) | `GLOBAL` — any disruption anywhere |
| PHY-WAT-* (Water) | `europe` bounding box |
| PHY-GEO-* (Geophysical) | `SEISMIC_ZONES` — union of all zones |
| PHY-POL-* (Pollution) | `EEA_EXTENDED` country list |
| PHY-BIO-* (Bio) | `GLOBAL` with WHO DON scope |
| STR-GEO-* (Geopolitical) | `TOP20_SUPPLIERS` for conflict, `GLOBAL` for sanctions |
| STR-TRD-* (Trade) | `OECD_TRADING` country list |
| STR-REG-* (Regulatory) | `EU27` + `US` + `GB` |
| STR-ECO-* (Economic) | `OECD_TRADING` |
| STR-ENP-* (Energy Policy) | `EU27` + `US` + `GB` |
| STR-TEC-* (Tech Policy) | `OECD_TRADING` |
| STR-FIN-* (Financial) | `GLOBAL` |
| DIG-* (All Digital) | `GLOBAL` — cyber is borderless |
| OPS-* (All Operational) | `GLOBAL` for supply chain; `EEA_EXTENDED` for manufacturing/warehouse |

---

# SECTION 3: DATA SOURCE MASTER TABLE

## 3.1 Source Classification

Every source is Type A (API), Type B (Annual Manual), or Type C (Needs Proxy).

### Type A — Automated API Sources

| ID | Source Name | Endpoint | Auth | Format | Install | Events Served |
|---|---|---|---|---|---|---|
| A01 | Copernicus CDS ERA5 | `cds.climate.copernicus.eu/api` | Free token | NetCDF | `pip install cdsapi` | PHY-CLI-*, PHY-WAT-* |
| A02 | USGS ComCat | `earthquake.usgs.gov/fdsnws/event/1/query` | None | GeoJSON | HTTP GET | PHY-GEO-001, PHY-GEO-004 |
| A03 | FRED | `api.stlouisfed.org/fred/series/observations` | Free key | JSON | HTTP GET | STR-ECO-*, STR-FIN-*, OPS-CMP-001 (proxy) |
| A04 | NIST NVD | `services.nvd.nist.gov/rest/json/cves/2.0` | Free key | JSON | HTTP GET | DIG-CIC-*, DIG-SCC-*, DIG-HWS-004 |
| A05 | CISA KEV | `cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json` | None | JSON | HTTP GET | DIG-CIC-*, DIG-SCC-* |
| A06 | GPR Index | `matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls` | None | Excel | HTTP GET | STR-GEO-* |
| A07 | ACLED | `api.acleddata.com/acled/read` | Free key | JSON | HTTP GET | STR-GEO-001, STR-GEO-004, STR-GEO-005 |
| A08 | EM-DAT (via HDX) | `data.humdata.org/dataset/emdat-country-profiles` | Free reg | CSV | HTTP GET | PHY-CLI-*, PHY-GEO-*, PHY-POL-* |
| A09 | World Bank API | `api.worldbank.org/v2/` | None | JSON | HTTP GET | STR-ECO-*, PHY-MAT-* |
| A10 | ENTSO-E | `transparency.entsoe.eu/api` | Free token | XML | `pip install entsoe-py` | PHY-ENE-* |
| A11 | WHO DON | `who.int/emergencies/disease-outbreak-news` | None | HTML/RSS | Web scrape | PHY-BIO-* |
| A12 | UCDP API | `ucdpapi.pcr.uu.se/api/gedevents/` | None | JSON | HTTP GET | STR-GEO-001, STR-GEO-005, STR-GEO-007 |
| A13 | NOAA CPC Indices | `cpc.ncep.noaa.gov/data/indices/` | None | Text | HTTP GET | PHY-CLI-006 (NAO, AO indices) |

### Type B — Annual Manual Entry Sources

Claude Code must create a JSON template for each Type B source. When the annual report is published, a human fills in the values. Claude Code reads these files to compute priors.

**Template location:** `./data/manual/{source_id}_{year}.json`

**Example template for B01 (DBIR):**
```json
{
    "source_id": "B01",
    "source_name": "Verizon DBIR",
    "report_year": 2025,
    "entry_date": null,
    "entered_by": null,
    "data": {
        "ransomware_pct": null,
        "third_party_pct": null,
        "social_engineering_pct": null,
        "credential_theft_pct": null,
        "web_app_exploit_pct": null,
        "insider_misuse_pct": null,
        "total_incidents": null,
        "total_breaches": null,
        "contributing_orgs": null,
        "double_extortion_pct": null
    },
    "validation": {
        "all_pct_fields_between_0_and_1": true,
        "total_breaches_less_than_total_incidents": true
    }
}
```

**When a manual file is missing or has null values:** Use the most recent available year's data. If no manual data exists for a source, use the hardcoded fallback rate. Log: `"manual_data_status": "USING_PRIOR_YEAR_2024"`.

| ID | Source Name | Report / URL | Publication Month | Events Served | Data Points to Extract |
|---|---|---|---|---|---|
| B01 | Verizon DBIR | `verizon.com/dbir` | May | DIG-RDE-*, DIG-FSD-* | `ransomware_pct`, `third_party_pct`, `social_engineering_pct`, `total_incidents`, `total_breaches` |
| B02 | FBI IC3 | `ic3.gov/AnnualReport` | April | DIG-FSD-001, DIG-FSD-002 | `bec_complaints`, `bec_losses_usd`, `ransomware_complaints`, `total_complaints`, `total_losses_usd` |
| B03 | ENISA ETL | `enisa.europa.eu/publications/enisa-threat-landscape-*` | October | DIG-SCC-*, DIG-CIC-* | `top10_threats`, `supply_chain_pct`, `total_incidents_analyzed` |
| B04 | Dragos YiR | `dragos.com/ot-cybersecurity-year-in-review` | February | DIG-CIC-002, DIG-HWS-* | `ransomware_groups_count`, `ransomware_incidents`, `threat_groups_active`, `ot_shutdown_pct` |
| B05 | IRU | IRU Driver Shortage Report | November | OPS-RLD-001 | `driver_vacancy_pct_eu`, `driver_vacancy_pct_us`, `total_shortage_estimate` |
| B06 | Allianz Trade | Global Insolvency Report | January | OPS-SUP-001, OPS-SUP-006 | `global_insolvency_index`, `eu_insolvency_rate`, `yoy_change_pct` |
| B07 | Flexera | State of Cloud Report | February | DIG-CLS-* | `multicloud_pct`, `cloud_cost_overspend_pct`, `vendor_lockin_concern_pct` |
| B08 | WTO TMR | G20 Trade Measures Report | July + December | STR-TRD-* | `new_restrictive_measures_count`, `trade_coverage_usd_b`, `cumulative_stockpile_usd_b` |
| B09 | Munich Re NatCat | Natural Catastrophe Review | January | PHY-CLI-*, PHY-GEO-* | `total_events`, `insured_losses_usd_b`, `economic_losses_usd_b`, `deadliest_events` |
| B10 | IATA WATS | World Air Transport Statistics | June | OPS-AIR-* | `cargo_load_factor_pct`, `ctk_growth_pct`, `capacity_shortfall_months` |
| B11 | Marsh/WTW | Political Risk Map | January | STR-GEO-* | `countries_elevated_risk`, `key_risk_themes` |

### Type C — Proxy Required (No Direct Access)

| ID | Original Source | Proxy Indicator | Proxy Source (Type A ID) | Proxy Calculation |
|---|---|---|---|---|
| C01 | Susquehanna semi lead times | Durable goods new orders (computers) | A03 (FRED: `ACDGNO`) | `modifier = current_quarter / rolling_20q_mean`. High orders → demand surge → lead time pressure |
| C02 | Dragos OT connectivity index | ICS-tagged CVE annual count | A04 (NIST NVD) | Query: `keywordSearch=ICS+SCADA+PLC+HMI+OT`. YoY growth rate as attack surface expansion proxy |
| C03 | Panama Canal water levels | Soil moisture Panama watershed | A01 (CDS: `volumetric_soil_water_layer_1`) | Area: lat 8-10, lon -80 to -79. Deficit below 1991-2020 mean = drought risk |
| C04 | NOAA blocking pattern frequency | NAO Index (monthly) | A13 (CPC: `/data/indices/nao`) | Strongly negative NAO (< -1.0) correlates with European blocking. Count months < -1.0 per year |
| C05 | Semiconductor lead times | FRED ISM Manufacturing PMI new orders | A03 (FRED: `NAPMNOI`) | PMI new orders > 55 for >3 months = demand pressure. Modifier = NAPMNOI / 50 |

---

# SECTION 4: EXACT API CALL SPECIFICATIONS

## 4.1 Copernicus CDS — ERA5 Temperature (Source A01)

```python
import cdsapi

client = cdsapi.Client()  # Requires ~/.cdsapirc with token

# Monthly mean 2m temperature for Europe (summer = June-September)
client.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': '2m_temperature',
        'year': [str(y) for y in range(1991, 2026)],
        'month': ['06', '07', '08', '09'],
        'time': '00:00',
        'area': [72, -25, 34, 40],  # North, West, South, East
        'data_format': 'netcdf',
    },
    'era5_monthly_t2m_europe_summer.nc'
)
```

**Processing for heatwave prior (PHY-CLI-003):**
```python
import xarray as xr
import numpy as np

ds = xr.open_dataset('era5_monthly_t2m_europe_summer.nc')
# Compute area-weighted spatial mean for EU-South (34-47N, -10 to 40E)
eu_south = ds['t2m'].sel(latitude=slice(47, 34), longitude=slice(-10, 40))
weights = np.cos(np.deg2rad(eu_south.latitude))
monthly_mean = eu_south.weighted(weights).mean(dim=['latitude', 'longitude'])

# Baseline: 1991-2020 mean per calendar month
baseline = monthly_mean.sel(time=slice('1991', '2020')).groupby('time.month').mean()
baseline_std = monthly_mean.sel(time=slice('1991', '2020')).groupby('time.month').std()

# Anomaly per month
anomaly = (monthly_mean.groupby('time.month') - baseline) / baseline_std

# Heatwave year = any year with at least 1 month where anomaly > 2.0 (i.e., >2σ)
yearly_max_anomaly = anomaly.groupby('time.year').max()
heatwave_years = (yearly_max_anomaly > 2.0).sum().item()
total_years = len(yearly_max_anomaly.year)
prior = heatwave_years / total_years
```

**Processing for temperature modifier:**
```python
# Current value
latest_anomaly = anomaly.isel(time=-1).item()
# Convert to modifier: 1σ above → 15% increase
modifier = 1.0 + (latest_anomaly * 0.15)
modifier = np.clip(modifier, 0.75, 1.80)
```

**JUSTIFICATION for 0.15 scaling factor:** The 0.15 means "each standard deviation of temperature anomaly changes the heatwave probability by 15%." This is derived from ERA5 data: in months with anomaly ~+2σ, the probability of an actual heatwave event (as logged in EM-DAT) is ~30% higher than baseline. So 2σ × 0.15 = 0.30, which matches. Claude Code should recalculate this from the actual ERA5-vs-EM-DAT correlation after downloading both datasets — the 0.15 is an initial estimate.

## 4.2 Copernicus CDS — Soil Moisture (Source A01)

```python
client.retrieve(
    'reanalysis-era5-land-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'volumetric_soil_water_layer_1',
        'year': [str(y) for y in range(1991, 2026)],
        'month': [f'{m:02d}' for m in range(1, 13)],
        'time': '00:00',
        'area': [72, -25, 34, 40],
        'data_format': 'netcdf',
    },
    'era5_land_monthly_swvl1_europe.nc'
)
```

## 4.3 USGS ComCat — Earthquakes (Source A02)

```python
import requests

def count_earthquakes(zone_name, zone_bbox, start="2000-01-01", end="2024-12-31", min_mag=6.0):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/count"
    params = {
        "format": "text",
        "starttime": start,
        "endtime": end,
        "minmagnitude": min_mag,
        "minlatitude": zone_bbox["south"],
        "maxlatitude": zone_bbox["north"],
        "minlongitude": zone_bbox["west"],
        "maxlongitude": zone_bbox["east"],
    }
    r = requests.get(url, params=params)
    count = int(r.text.strip())
    years = 25
    return {"zone": zone_name, "count": count, "years": years, "prior": count / years}

# Run for all zones
for name, bbox in SEISMIC_ZONES.items():
    result = count_earthquakes(name, bbox)
    print(f"{name}: {result['count']} events in {result['years']} years = prior {result['prior']:.3f}")

# PHY-GEO-001 prior = "at least 1 M6.0+ event in ANY zone" 
# Use inclusion-exclusion or simply: P(at least 1) = 1 - ∏(1 - P_zone_i)
```

## 4.4 GPR Index (Source A06)

```python
import pandas as pd

# Download monthly GPR
# NOTE: URL may change. Fall back to: matteoiacoviello.com/gpr.htm and find current link
url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
gpr = pd.read_excel(url, sheet_name='export')

# Key columns: 'month' (YYYYMM), 'GPRH' (historical), 'GPR' (recent)
# Combine into single series
# Compute ratio to rolling 5-year mean
gpr['rolling_60m'] = gpr['GPR'].rolling(60, min_periods=36).mean()
gpr['ratio'] = gpr['GPR'] / gpr['rolling_60m']

# Calibrate P5/P95
p5 = gpr['ratio'].dropna().quantile(0.05)
p95 = gpr['ratio'].dropna().quantile(0.95)
median = gpr['ratio'].dropna().quantile(0.50)

# Current modifier = latest ratio, clipped to [max(0.50, p5), min(3.00, p95)]
current_modifier = gpr['ratio'].iloc[-1]
current_modifier = max(max(0.50, p5), min(min(3.00, p95), current_modifier))
```

## 4.5 NIST NVD — ICS Vulnerability Count (Source A04)

```python
import requests
import time

def count_ics_cves(year):
    url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    params = {
        "keywordSearch": "ICS SCADA PLC HMI industrial control",
        "pubStartDate": f"{year}-01-01T00:00:00.000",
        "pubEndDate": f"{year}-12-31T23:59:59.999",
        "resultsPerPage": 1  # We only need totalResults count
    }
    # NVD API rate limit: 5 requests per 30 seconds without key, 50 with key
    r = requests.get(url, params=params, headers={"apiKey": NVD_API_KEY})
    data = r.json()
    return data.get("totalResults", 0)

# Build annual time series
ics_cve_counts = {}
for year in range(2015, 2026):
    ics_cve_counts[year] = count_ics_cves(year)
    time.sleep(2)  # Rate limiting
```

## 4.6 CISA KEV Catalog (Source A05)

```python
import requests
from collections import Counter
from datetime import datetime

url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
kev = requests.get(url).json()

# Count additions per year
yearly_counts = Counter()
for v in kev['vulnerabilities']:
    year = datetime.strptime(v['dateAdded'], '%Y-%m-%d').year
    yearly_counts[year] += 1

# ICS-relevant: filter by vendorProject containing industrial keywords
ics_keywords = ['siemens', 'schneider', 'rockwell', 'honeywell', 'abb', 
                'emerson', 'yokogawa', 'ge', 'mitsubishi', 'omron']
ics_kev = [v for v in kev['vulnerabilities'] 
           if any(kw in v.get('vendorProject', '').lower() for kw in ics_keywords)]
```

## 4.7 FRED Economic Series (Source A03)

```python
import requests

FRED_API_KEY = "YOUR_KEY"

def get_fred_series(series_id, start="2000-01-01"):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
    }
    r = requests.get(url, params=params)
    return r.json()["observations"]

# Key series for PRISM:
FRED_SERIES = {
    "T10Y2Y": "10Y-2Y Treasury spread (yield curve inversion → recession signal)",
    "UNRATE": "Unemployment rate",
    "BAMLH0A0HYM2": "High Yield OAS (credit stress signal)",
    "NAPMNOI": "ISM Manufacturing PMI New Orders (demand pressure)",
    "ACDGNO": "Durable goods new orders - computers (semi demand proxy)",
    "CPIAUCSL": "CPI All Items (inflation tracker)",
    "DTWEXBGS": "Trade-weighted USD index (currency strength)",
}
```

## 4.8 NOAA CPC NAO Index (Source A13)

```python
import requests
import io
import pandas as pd

# NAO Monthly Index — direct download as fixed-width text
url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table"
response = requests.get(url)
# Parse: columns are Year, Jan, Feb, ..., Dec
lines = response.text.strip().split('\n')
records = []
for line in lines:
    parts = line.split()
    year = int(parts[0])
    for month_idx, val in enumerate(parts[1:13], 1):
        records.append({"year": year, "month": month_idx, "nao": float(val)})
nao = pd.DataFrame(records)

# Blocking proxy: count months per year with NAO < -1.0
# Strongly negative NAO → atmospheric blocking over Europe
nao['blocking'] = nao['nao'] < -1.0
blocking_months_per_year = nao.groupby('year')['blocking'].sum()
```

## 4.9 EM-DAT via HDX (Source A08)

```python
import pandas as pd

# HDX aggregated country profiles — download CSV
# URL pattern: https://data.humdata.org/dataset/emdat-country-profiles
# Direct CSV links per country available at HDX
# Alternative: Register at public.emdat.be, download full Excel

# IMPORTANT: EM-DAT column names vary between the portal download and HDX download.
# The portal download (recommended) typically uses these columns:
#   'Year', 'ISO', 'Country', 'Disaster Group', 'Disaster Subgroup', 
#   'Disaster Type', 'Disaster Subtype', 'Event Name',
#   'Total Deaths', 'No. Affected', 'Total Damage (USD)',
#   'Dis No' (disaster number, unique ID)
#
# The HDX country profiles use aggregated columns:
#   'year', 'country', 'iso3', 'disaster_subtype', 'events_count',
#   'total_affected', 'total_deaths', 'total_damage_adj'
#
# RULE: At startup, Claude Code must read the header row and auto-detect column names.
# Map to canonical names using this lookup:
EMDAT_COLUMN_ALIASES = {
    "year": ["Year", "year", "Start Year"],
    "iso": ["ISO", "iso3", "Country ISO", "ISO3"],
    "disaster_subtype": ["Disaster Subtype", "disaster_subtype", "Sub-Type"],
    "deaths": ["Total Deaths", "total_deaths", "Deaths"],
    "affected": ["No. Affected", "total_affected", "Total Affected"],
    "damage_usd": ["Total Damage (USD)", "total_damage_adj", "Total Damage"],
}

def normalize_emdat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names regardless of source format."""
    rename_map = {}
    for canonical, aliases in EMDAT_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    return df.rename(columns=rename_map)

# Once loaded and normalized, standard filtering:
def count_event_years(emdat_df, disaster_type, countries, start_year=2000, end_year=2024):
    """Count distinct years with at least 1 qualifying event."""
    filtered = emdat_df[
        (emdat_df['Disaster Subtype'].str.contains(disaster_type, case=False, na=False)) &
        (emdat_df['ISO'].isin(countries)) &
        (emdat_df['Start Year'] >= start_year) &
        (emdat_df['Start Year'] <= end_year)
    ]
    event_years = filtered['Start Year'].nunique()
    total_years = end_year - start_year + 1
    return event_years, total_years, event_years / total_years

# EM-DAT disaster type strings for PRISM events:
EMDAT_MAPPINGS = {
    "PHY-CLI-001": {"type": "Riverine flood", "countries": "EEA_EXTENDED"},
    "PHY-CLI-002": {"type": "Coastal flood|Storm surge", "countries": "EEA_EXTENDED"},
    "PHY-CLI-003": {"type": "Heat wave", "countries": "EEA_EXTENDED"},
    "PHY-CLI-004": {"type": "Drought", "countries": "EEA_EXTENDED"},
    "PHY-CLI-005": {"type": "Forest fire|Wildfire", "countries": "EEA_EXTENDED"},
    "PHY-CLI-006": {"type": "Cold wave|Extreme winter", "countries": "EEA_EXTENDED"},
    "PHY-GEO-001": {"type": "Earthquake", "countries": "ALL_SEISMIC"},
    "PHY-GEO-002": {"type": "Volcanic", "countries": "EEA_EXTENDED"},
    "PHY-GEO-003": {"type": "Landslide|Mudslide", "countries": "EEA_EXTENDED"},
    "PHY-GEO-004": {"type": "Tsunami", "countries": "ALL_SEISMIC"},
    "PHY-POL-001": {"type": "Chemical spill|Industrial accident", "countries": "EEA_EXTENDED"},
}
```

## 4.10 ACLED Conflict Events (Source A07)

```python
import requests

ACLED_KEY = "YOUR_KEY"
ACLED_EMAIL = "your@email.com"

def count_conflict_years(countries, start_year=2000, end_year=2024):
    """Count years with at least 1 battle event in any of the listed countries."""
    years_with_conflict = set()
    for year in range(start_year, end_year + 1):
        url = "https://api.acleddata.com/acled/read"
        params = {
            "key": ACLED_KEY,
            "email": ACLED_EMAIL,
            "event_type": "Battles",
            "year": year,
            "iso": "|".join(countries),  # ACLED uses ISO numeric or country names
            "limit": 1,
        }
        r = requests.get(url, params=params)
        data = r.json()
        if data.get("count", 0) > 0:
            years_with_conflict.add(year)
    return len(years_with_conflict), end_year - start_year + 1
```

---

# SECTION 5: PRIOR DERIVATION METHODS

## 5.1 Method A: Frequency Count

**When:** Discrete events logged in structured databases (EM-DAT, USGS, WHO DON, ACLED).

```python
def method_a_prior(event_years: int, total_years: int) -> dict:
    prior = event_years / total_years
    confidence = "High" if total_years >= 20 and event_years >= 5 else "Medium"
    return {
        "prior": round(prior, 4),
        "method": "A",
        "formula": f"{event_years} event-years / {total_years} total years",
        "confidence": confidence,
    }
```

## 5.2 Method B: Incidence Rate

**When:** Per-organization risk measured by surveys (DBIR, IC3, insurance data).

```python
def method_b_prior(incident_rate: float, dark_figure: float) -> dict:
    """
    incident_rate: reported annual probability per organization (e.g., 0.12 from DBIR)
    dark_figure: underreporting multiplier (see table below)
    """
    prior = min(0.95, incident_rate * dark_figure)
    return {
        "prior": round(prior, 4),
        "method": "B",
        "formula": f"{incident_rate} × {dark_figure} dark figure",
        "confidence": "Medium",
    }
```

**CRITICAL: How to derive incident_rate from DBIR data for specific events:**

The DBIR reports overall breach statistics, not per-event-type annual probabilities. Claude Code must decompose as follows:

```python
# Step 1: Get DBIR's overall breach probability per organization
# DBIR 2025: 12,195 confirmed breaches from ~28,000 contributing organizations
# This gives an APPROXIMATE annual breach probability of: 12195/28000 ≈ 0.435
# BUT this is biased — DBIR contributors are organizations that HAVE incidents.
# Use the DBIR's own "% of organizations experiencing a breach" when published,
# or use insurance data as a more representative denominator.
# Conservative estimate: ~15-20% of organizations experience any breach per year.
DBIR_BASE_BREACH_RATE = 0.18  # 18% annual probability of any breach type

# Step 2: Multiply by the attack-type percentage from DBIR
DBIR_ATTACK_SHARES = {
    # From DBIR 2025 (update annually from Type B manual entry)
    "ransomware": 0.44,           # 44% of breaches involved ransomware
    "social_engineering": 0.25,   # 25% involved social engineering 
    "credential_theft": 0.38,     # 38% involved stolen credentials
    "third_party": 0.30,          # 30% involved third-party supplier
    "web_app_exploit": 0.26,      # 26% involved web application attacks
    "insider_misuse": 0.08,       # 8% involved insider misuse
}

# Step 3: Map PRISM events to DBIR attack shares
DBIR_EVENT_MAPPING = {
    "DIG-RDE-001": {"share": "ransomware", "subsplit": 0.50},    # 50% of ransomware hits ERP/business systems
    "DIG-RDE-002": {"share": "ransomware", "subsplit": 0.30},    # 30% involves database encryption
    "DIG-RDE-003": {"share": "ransomware", "subsplit": 0.62},    # 62% involves double extortion (DBIR 2025)
    "DIG-RDE-004": {"share": "credential_theft", "subsplit": 0.40}, # 40% of cred theft → personal data breach
    "DIG-RDE-005": {"share": "insider_misuse", "subsplit": 0.35}, # 35% of insider misuse → IP theft
    "DIG-RDE-006": {"share": "web_app_exploit", "subsplit": 0.25}, # 25% of web attacks → financial data
    "DIG-RDE-007": {"share": "credential_theft", "subsplit": 1.0}, # All credential theft counts
    "DIG-RDE-008": {"share": "third_party", "subsplit": 0.30},   # 30% of third-party → data exposure
    "DIG-FSD-001": {"share": "social_engineering", "subsplit": 0.40}, # 40% of social eng → BEC
    "DIG-FSD-002": {"share": "social_engineering", "subsplit": 0.25}, # 25% → wire transfer fraud
    "DIG-FSD-003": {"share": "credential_theft", "subsplit": 0.60}, # 60% of cred theft via phishing
    "DIG-FSD-005": {"share": None, "fixed_rate": 0.12},          # DDoS: use Netscout data directly
}

# Step 4: Calculate per-event incident rate
def dbir_incident_rate(event_id: str) -> float:
    mapping = DBIR_EVENT_MAPPING[event_id]
    if mapping.get("fixed_rate"):
        return mapping["fixed_rate"]
    share = DBIR_ATTACK_SHARES[mapping["share"]]
    subsplit = mapping["subsplit"]
    return DBIR_BASE_BREACH_RATE * share * subsplit

# Example: DIG-RDE-001 (ERP ransomware)
# = 0.18 × 0.44 × 0.50 = 0.0396 ≈ 4.0% annual probability
```

**IMPORTANT: The `subsplit` values above are estimates based on DBIR breakdown tables. In Phase 1, Claude Code should:**
1. Log the subsplit values used
2. Tag each with `"subsplit_source": "DBIR_2025_figure_XX"` or `"subsplit_source": "ESTIMATE"`
3. When DBIR 2026 is released (May 2026), update DBIR_ATTACK_SHARES and any available subsplits

**For DIG-CIC events (ICS/OT), use Dragos data instead of DBIR:**
```python
DRAGOS_EVENT_MAPPING = {
    "DIG-CIC-001": {"incidents_2025": 0, "note": "No healthcare-specific OT attacks in Dragos sample — use DBIR healthcare ransomware rate"},
    "DIG-CIC-002": {"incidents_2025": 3300, "manufacturing_pct": 0.67, "total_mfg_orgs": 300000, 
                     "formula": "3300 × 0.67 / 300000 = 0.0074 (0.74% per mfg org) × dark_figure 3.0 = 2.2%"},
    "DIG-CIC-003": {"incidents_2025": 3300, "energy_pct": 0.08, "note": "Use Dragos energy sector share"},
    "DIG-CIC-004": {"note": "Use EPA/CISA water sector data — estimate from CISA advisories"},
    "DIG-CIC-005": {"note": "Use Dragos transport sector share"},
    "DIG-CIC-006": {"note": "Extremely rare — 1-2 known incidents (Stuxnet, Triton). Prior = 0.005"},
}
```

**Dark Figure Multiplier Table (with sources):**

| Threat Type | Multiplier | Source & Rationale |
|---|---|---|
| Ransomware (enterprise) | 1.0 | DBIR methodology already includes forensic firm data + insurance claims. Verizon DBIR 2025 analyzed 22,052 incidents from 139 contributors including forensic firms. No adjustment needed. |
| BEC / Wire fraud | 1.5 | FBI IC3 2024: 859,532 complaints logged vs FTC estimate ~5% reporting rate for consumer fraud. For enterprise BEC specifically, reporting rate is higher (~67%) because financial losses trigger investigation. Multiplier = 1/0.67 ≈ 1.5 |
| ICS/OT compromise | 3.0 | Dragos 2026 YiR: "Many incidents are mislabeled as IT incidents when Windows servers hosting SCADA software are compromised." Dragos directly states misclassification represents substantial undercount. Conservative 3× based on their "persistent mischaracterization" language. |
| Supply chain software attacks | 2.0 | ENISA ETL 2024: analyzed 4,875 incidents but notes "data leak sites have started being considered unreliable" and "many duplicates." Supply chain attacks specifically difficult to attribute. 2× conservative. |
| General data breaches | 1.0 | DBIR + IC3 + HIBP provide comprehensive coverage for enterprise-scale breaches. No multiplier. |
| DDoS attacks | 1.0 | Netscout/Akamai telemetry provides near-complete visibility. No multiplier. |

## 5.3 Method C: Structural Calibration

**When:** Policy/regulatory/market outcomes without long frequency histories.

```python
def method_c_prior(
    p_preconditions: float,  # Are structural conditions in place?
    p_trigger: float,        # Will a trigger event occur?
    p_implementation: float, # Will it be implemented within 12 months?
    evidence: dict           # Must document evidence for each
) -> dict:
    prior = p_preconditions * p_trigger * p_implementation
    return {
        "prior": round(prior, 4),
        "method": "C",
        "formula": f"{p_preconditions} × {p_trigger} × {p_implementation}",
        "confidence": "Low",
        "sub_probabilities": evidence,
    }
```

**Assigning Method C sub-probabilities — mandatory rules:**

Each sub-probability MUST have one of these evidence types:

| Evidence Type | How to Assign | Example |
|---|---|---|
| Legislative pipeline | Count bills introduced / bills passed in observation window | EU AI Act: 3 major bills introduced 2019-2024, 1 passed → P_implementation ≈ 0.33 per bill |
| Election cycle | Binary: is there an election in the next 12 months in a relevant jurisdiction? | US election year → P_trigger for protectionist tariffs = 0.70 (based on WTO data: 70% of election years 2016-2024 had new measures) |
| Historical policy frequency | Count analogous policy changes in observation window | REACH-style restrictions: 8 major amendments in 17 years → P ≈ 0.47/year |
| Expert survey | Reference specific survey (e.g., WEF Global Risks Report ranking) | If risk ranked in top 5 by >50% of respondents → P_preconditions = 0.80 |

**If Claude Code cannot find evidence for a sub-probability, it must set it to 0.50 (maximum ignorance) and tag it as `evidence: "DEFAULT_0.50_NO_DATA"`.** This makes the uncertainty visible.

---

# SECTION 6: MODIFIER CALIBRATION PROCEDURE

## 6.1 Standard Ratio Method (for continuous indicators)

```python
import pandas as pd
import numpy as np

def calibrate_modifier(time_series: pd.Series, window: int = 60) -> dict:
    """
    Standard modifier calibration from a monthly time series.
    
    Args:
        time_series: Monthly values of the indicator
        window: Rolling window for baseline (months). Default 60 = 5 years.
    
    Returns:
        Dict with p5, p50, p95, floor, ceiling
    """
    baseline = time_series.rolling(window, min_periods=window // 2).mean()
    ratio = time_series / baseline
    ratio = ratio.dropna()
    
    p5 = ratio.quantile(0.05)
    p50 = ratio.quantile(0.50)
    p95 = ratio.quantile(0.95)
    
    floor = max(0.50, round(p5, 2))
    ceiling = min(3.00, round(p95, 2))
    
    return {
        "n_observations": len(ratio),
        "p5": round(p5, 2),
        "p50": round(p50, 2),
        "p95": round(p95, 2),
        "floor": floor,
        "ceiling": ceiling,
        "current_value": round(ratio.iloc[-1], 2),
        "current_modifier": round(max(floor, min(ceiling, ratio.iloc[-1])), 2),
    }
```

## 6.2 Categorical Modifier Method (for binary/categorical indicators)

```python
def categorical_modifier(condition: bool, if_true: float, justification: str) -> dict:
    """
    For indicators that are on/off (e.g., election year, sanctions active).
    
    Args:
        condition: Is the condition currently true?
        if_true: Modifier value when condition is true (must be justified)
        justification: Empirical basis for the value
    """
    return {
        "type": "categorical",
        "condition_met": condition,
        "modifier": if_true if condition else 1.00,
        "justification": justification,
    }
```

**Pre-defined categorical modifiers:**

| Condition | if_true | Justification |
|---|---|---|
| US presidential election year | 1.25 | WTO data: Trade restrictive measures increase ~25% in US election years (2016: +28%, 2020: +22%, 2024: +31% vs non-election year baseline) |
| Active military conflict involving OECD member | 1.40 | GPR Index averaged 40% above 5yr mean during 2022-2024 (Ukraine conflict period) |
| Active PHEIC declared by WHO | 1.50 | During COVID PHEIC (2020-2023), supply chain disruption frequency was ~50% above historical average per Munich Re |
| ENSO El Niño active (ONI > 0.5) | 1.20 | EM-DAT data: 20% more climate-related disasters globally during El Niño years |
| ECB in rate-hiking cycle | 1.15 | During rate-hiking periods, Allianz insolvency index rises ~15% (observed 2022-2023) |

## 6.3 Scaling Constant Justification

Some modifiers convert a raw indicator (like temperature anomaly in °C) to a dimensionless multiplier. Each scaling constant must be justified:

| Indicator | Raw Unit | Scaling Formula | Constant | Justification |
|---|---|---|---|---|
| ERA5 temperature anomaly | σ (standard deviations) | `1.0 + (σ × 0.15)` | 0.15 | **Initial estimate.** Claude Code must verify by regressing EM-DAT heatwave event count against ERA5 anomaly for 2000-2024. The 0.15 implies 2σ → +30% probability, consistent with published climate-impact literature. After regression, replace 0.15 with actual β coefficient. |
| Soil moisture deficit | fraction (0 to 1) | `1.0 + (deficit × 2.0)` | 2.0 | **Initial estimate.** Severe drought (50% deficit) → 2× probability. Claude Code must verify by correlating Copernicus soil moisture with EM-DAT drought events. Replace with actual regression coefficient. |
| GPR ratio | dimensionless (already a ratio) | Direct use | N/A | No conversion needed — the ratio IS the modifier. |
| FRED PMI new orders | index points | `NAPMNOI / 50` | 50 | PMI 50 = neutral. Values above 50 indicate expansion (demand pressure). The modifier is literally "how much above/below neutral." |

**RULE: Any scaling constant tagged as "Initial estimate" must be replaced with a regression-derived coefficient in Phase 1.**

---

# SECTION 7: COMPLETE EVENT → SOURCE → METHOD MAPPING

## 7.1 The 10 Phase 1 Prototype Events

These were selected to cover: all 4 domains, all 3 methods, all 3 source types, and a range of confidence levels.

| # | Event ID | Event Name | Method | Primary Source | Indicator for Modifier | Modifier Source |
|---|---|---|---|---|---|---|
| 1 | PHY-CLI-003 | Heat wave affecting production | A | A01 (ERA5) + A08 (EM-DAT) | ERA5 T2m anomaly | A01 |
| 2 | PHY-GEO-001 | Major earthquake | A | A02 (USGS) | Recent seismicity rate / long-term rate | A02 |
| 3 | STR-GEO-001 | Armed conflict in supplier country | A | A07 (ACLED) + A12 (UCDP) | GPR Index ratio | A06 |
| 4 | STR-TRD-001 | Major tariff increases >25% | C | B08 (WTO) | Election year (cat.) + GPR trade tension | A06 + categorical |
| 5 | DIG-RDE-001 | ERP system ransomware | B | B01 (DBIR) | CISA KEV growth rate | A05 |
| 6 | DIG-CIC-002 | SCADA/ICS compromise | B | B04 (Dragos) | NVD ICS CVE count (proxy C02) | A04 |
| 7 | OPS-MAR-002 | Canal/strait closure | A | News/UNCTAD (manual list) | Chokepoint security (GPR sub-index) | A06 + A07 |
| 8 | OPS-CMP-001 | Semiconductor chip shortage | C | B04 proxy + news | FRED durable goods orders (proxy C01) | A03 |
| 9 | STR-ECO-001 | Recession in major market | A | A03 (FRED) + A09 (World Bank) | Yield curve + credit spread | A03 |
| 10 | PHY-BIO-001 | Zoonotic disease outbreak | A | A11 (WHO DON) | WHO alert status (categorical) | Categorical |

## 7.2 Full 174-Event Mapping Rules

For the remaining 164 events, Claude Code should assign method and source using these rules:

**Auto-assignment algorithm:**

```python
def assign_method_and_source(event_id: str, domain: str, family: str) -> dict:
    prefix = event_id.split("-")[0]  # PHY, STR, DIG, OPS
    
    # PHYSICAL domain → Method A (frequency count)
    if prefix == "PHY":
        if "CLI" in event_id:
            return {"method": "A", "primary": "A01+A08", "modifier": "A01"}
        elif "GEO" in event_id:
            return {"method": "A", "primary": "A02+A08", "modifier": "A02"}
        elif "ENE" in event_id:
            return {"method": "A", "primary": "A10+A08", "modifier": "A10"}
        elif "WAT" in event_id:
            return {"method": "A", "primary": "A01+A08", "modifier": "A01"}
        elif "MAT" in event_id:
            return {"method": "A", "primary": "A09+A08", "modifier": "A09"}
        elif "POL" in event_id:
            return {"method": "A", "primary": "A08", "modifier": "A01"}
        elif "BIO" in event_id:
            return {"method": "A", "primary": "A11", "modifier": "categorical"}
    
    # STRUCTURAL domain → Method A for geopolitical/economic, Method C for regulatory/policy
    elif prefix == "STR":
        if "GEO" in event_id:
            return {"method": "A", "primary": "A07+A12", "modifier": "A06"}
        elif "TRD" in event_id:
            return {"method": "C", "primary": "B08", "modifier": "A06+categorical"}
        elif "REG" in event_id:
            return {"method": "C", "primary": "legislative_scan", "modifier": "categorical"}
        elif "ECO" in event_id:
            return {"method": "A", "primary": "A03+A09", "modifier": "A03"}
        elif "ENP" in event_id:
            return {"method": "C", "primary": "legislative_scan", "modifier": "categorical"}
        elif "TEC" in event_id:
            return {"method": "C", "primary": "legislative_scan", "modifier": "categorical"}
        elif "FIN" in event_id:
            return {"method": "A", "primary": "A03", "modifier": "A03"}
    
    # DIGITAL domain → Method B (incidence rate from DBIR/IC3/Dragos)
    elif prefix == "DIG":
        if "CIC" in event_id:
            return {"method": "B", "primary": "B04+B01", "modifier": "A04+A05"}
        elif "RDE" in event_id:
            return {"method": "B", "primary": "B01", "modifier": "A05"}
        elif "SCC" in event_id:
            return {"method": "B", "primary": "B01+B03", "modifier": "A04"}
        elif "FSD" in event_id:
            return {"method": "B", "primary": "B01+B02", "modifier": "A05"}
        elif "CLS" in event_id:
            return {"method": "C", "primary": "B07", "modifier": "categorical"}
        elif "HWS" in event_id:
            return {"method": "C", "primary": "B04+news", "modifier": "C01"}
        elif "SWS" in event_id:
            return {"method": "C", "primary": "B03+B07", "modifier": "categorical"}
    
    # OPERATIONAL domain → Method A for historical events, Method C for structural
    elif prefix == "OPS":
        if "MAR" in event_id or "AIR" in event_id:
            return {"method": "A", "primary": "A08+news", "modifier": "A06+C03"}
        elif "RLD" in event_id:
            return {"method": "A", "primary": "B05+A08", "modifier": "A03"}
        elif "CMP" in event_id:
            return {"method": "C", "primary": "news+B04", "modifier": "C01+C05"}
        elif "SUP" in event_id:
            return {"method": "A", "primary": "B06+A08", "modifier": "A03"}
        elif "MFG" in event_id:
            return {"method": "A", "primary": "A08+insurance_data", "modifier": "categorical"}
        elif "WHS" in event_id:
            return {"method": "A", "primary": "A08+insurance_data", "modifier": "categorical"}
    
    # Fallback
    return {"method": "C", "primary": "manual_research", "modifier": "categorical"}
```

**Handling undefined source references:**

The mapping algorithm above references some sources that are not automated APIs. Here's what Claude Code should do for each:

| Source Reference | What It Means | Claude Code Action |
|---|---|---|
| `"legislative_scan"` | Requires scanning EU/US legislative databases for active bills | **Phase 1:** Use Method C with `p_preconditions = 0.50` (default ignorance). Tag with `"needs_legislative_scan": true`. **Phase 2:** Build a scraper for EUR-Lex (EU) and Congress.gov (US) to count active bills per topic. |
| `"news"` | Requires structured event list from news sources | **Phase 1:** Manually compile an event list as a JSON file at `./data/manual/news_events.json` with format `[{"event_id": "OPS-MAR-002", "year": 2021, "description": "Ever Given Suez blockage"}, ...]`. Use this for Method A counting. |
| `"insurance_data"` | Requires Marsh/Swiss Re/Allianz loss data | **Phase 1:** Use EM-DAT damage figures as a proxy. If EM-DAT damage > $100M for an event type in a year, count it. **Phase 2:** Add Marsh B11 manual entry as a Type B source. |
| `"manual_research"` | Fallback — no automated source exists | Use hardcoded fallback rate. Tag with `"data_status": "NO_AUTOMATED_SOURCE"`. |

---

# SECTION 8: P_GLOBAL AND P_CLIENT FORMULAS

## 8.1 P_global

```python
def calculate_p_global(prior: float, modifiers: list[float]) -> float:
    raw = prior
    for m in modifiers:
        raw *= m
    floor = max(0.001, 0.1 * prior)
    ceiling = 0.95
    return round(max(floor, min(ceiling, raw)), 4)
```

## 8.2 P_client

```python
def calculate_p_client(
    p_global: float,
    geographic_exposure: float,   # 0.0 to 3.0
    industry_exposure: float,     # 0.0 to 3.0
    scale_factor: float,          # 0.5 to 2.0
    vulnerability_score: float,   # 0 to 100
    resilience_score: float       # 0 to 100
) -> float:
    exposure = geographic_exposure * industry_exposure * scale_factor
    net_vulnerability = (vulnerability_score / 100) * (1 - resilience_score / 100)
    p_client = p_global * exposure * net_vulnerability
    return round(max(0.001, min(0.95, p_client)), 4)
```

## 8.3 Layer 2 Factor Lookup Tables

Claude Code should populate these from client intake questionnaire. Default values for "generic European manufacturer":

```python
DEFAULT_CLIENT_FACTORS = {
    # Geographic exposure: how much does the client's footprint overlap with the risk region?
    "geographic_exposure": {
        "EU_only_operations": 1.0,       # Baseline
        "Global_supply_chain": 1.5,      # More exposed to global disruptions
        "Single_country": 0.7,           # Less exposed
        "High_risk_region_dependent": 2.0, # Heavily dependent on unstable regions
    },
    
    # Industry exposure: how susceptible is this industry to this risk type?
    "industry_exposure": {
        # By event domain:
        "PHY-CLI": {"manufacturing": 1.2, "services": 0.6, "agriculture": 2.0, "tech": 0.8},
        "DIG-RDE": {"healthcare": 1.8, "financial": 1.5, "manufacturing": 1.0, "retail": 1.3},
        "STR-TRD": {"automotive": 1.8, "pharma": 1.2, "tech": 1.5, "services": 0.5},
        "OPS-CMP": {"automotive": 2.0, "electronics": 2.5, "chemicals": 1.0, "services": 0.3},
    },
    
    # Scale factor: larger organizations = bigger target but more resources
    "scale_factor": {
        "SME_under_250": 0.8,      # Smaller target, but less resilience
        "MidCap_250_5000": 1.0,    # Baseline
        "LargeCap_over_5000": 1.2, # Bigger target, more complex supply chain
    },
}
```

---

# SECTION 9: VALIDATION AND OUTPUT SCHEMA

## 9.1 Validation Rules

```python
VALIDATION_RULES = {
    "prior": (0.001, 0.95),
    "modifier": (0.50, 3.00),
    "p_global": (0.001, 0.95),
    "p_client": (0.001, 0.95),
    "geographic_exposure": (0.0, 3.0),
    "industry_exposure": (0.0, 3.0),
    "scale_factor": (0.5, 2.0),
    "vulnerability_score": (0, 100),
    "resilience_score": (0, 100),
}

def validate(field: str, value: float) -> bool:
    low, high = VALIDATION_RULES[field]
    return low <= value <= high
```

## 9.2 Output Schema per Event

Every computed event must produce a JSON object with this exact structure:

```json
{
    "event_id": "PHY-CLI-003",
    "event_name": "Extreme heat wave affecting production/logistics",
    "domain": "Physical",
    "family": "Climate Extremes & Weather Events",
    
    "layer1": {
        "prior": 0.72,
        "method": "A",
        "derivation": {
            "formula": "18 event-years / 25 total years",
            "data_source": "EM-DAT + ERA5 cross-validation",
            "source_id": "A08+A01",
            "observation_window": "2000-2024 (25yr)",
            "n_observations": 25,
            "calculation_steps": "Counted years where ERA5 EU-South T2m summer anomaly >2σ AND EM-DAT logged heatwave event in any EU27 country. Result: 18 of 25 years qualify.",
            "confidence": "High"
        },
        "modifiers": [
            {
                "name": "Temperature anomaly",
                "source_id": "A01",
                "indicator_value": 2.3,
                "indicator_unit": "σ above 1991-2020 baseline",
                "modifier_value": 1.35,
                "calibration": {
                    "method": "ratio",
                    "n_observations": 300,
                    "p5": 0.75,
                    "p50": 1.02,
                    "p95": 1.80,
                    "floor": 0.75,
                    "ceiling": 1.80,
                    "scaling_formula": "1.0 + (anomaly_sigma × 0.15)",
                    "scaling_constant_status": "INITIAL_ESTIMATE_NEEDS_REGRESSION"
                }
            },
            {
                "name": "El Niño status",
                "source_id": "categorical",
                "indicator_value": true,
                "modifier_value": 1.20,
                "calibration": {
                    "method": "categorical",
                    "justification": "EM-DAT data: 20% more climate disasters during El Niño years"
                }
            }
        ],
        "p_global": 0.95,
        "p_global_raw": 1.166,
        "p_global_capped_at": 0.95
    },
    
    "layer2": {
        "geographic_exposure": 1.0,
        "industry_exposure": 1.2,
        "scale_factor": 1.0,
        "vulnerability_score": 65,
        "resilience_score": 30,
        "p_client": 0.52
    },
    
    "metadata": {
        "last_computed": "2026-02-19T12:00:00Z",
        "spec_version": "2.2",
        "data_freshness": {
            "A01": "2026-01-31",
            "A08": "2025-12-31"
        }
    }
}
```

---

# SECTION 10: CREDENTIALS AND SETUP

| Service | Registration URL | Environment Variable | Rate Limit |
|---|---|---|---|
| Copernicus CDS | `cds.climate.copernicus.eu/user/register` | `CDS_API_KEY` | ~100 requests/day |
| FRED | `fred.stlouisfed.org/docs/api/api_key.html` | `FRED_API_KEY` | 120 requests/minute |
| NIST NVD | `nvd.nist.gov/developers/request-an-api-key` | `NVD_API_KEY` | 50 requests/30 seconds |
| ACLED | `acleddata.com/register` | `ACLED_KEY` + `ACLED_EMAIL` | 500 requests/day |
| EM-DAT | `public.emdat.be/register` | Manual download | N/A |
| ENTSO-E | `transparency.entsoe.eu/register` | `ENTSOE_API_KEY` | 400 requests/minute |

**No key needed:** USGS (A02), CISA KEV (A05), GPR Index (A06), World Bank (A09), NOAA CPC (A13).

---

# SECTION 11: IMPLEMENTATION PHASES

## Phase 1 (Weeks 1-4): 10-Event Prototype
1. Register for all API keys
2. Implement data connectors for: A01, A02, A03, A04, A05, A06, A07, A08, A13
3. Download time series for all 10 prototype events
4. Compute priors using Methods A, B, C
5. Run modifier calibration (Section 6)
6. **Validate scaling constants:** Regress ERA5 anomaly against EM-DAT event counts. Replace 0.15 with actual β.
7. Compare computed priors against v2 workbook estimates. Log discrepancies.
8. **Acceptance test:** ≥8 of 10 priors within 50% of v2 estimates, OR documented explanation for divergence.

## Phase 2 (Weeks 5-12): Scale to 174 Events
1. Run assignment algorithm (Section 7.2) for all 174 events
2. Group events by primary source — batch API calls
3. Compute all priors and modifiers
4. Manual research for Method C events (identify legislative pipeline, election cycle, etc.)
5. Build manual entry interface for Type B sources
6. Output: Complete JSON for all 174 events

## Phase 3 (Weeks 13-16): Client Integration
1. Build client intake questionnaire (geographic exposure, industry, scale, vulnerability, resilience)
2. Populate Layer 2 lookup tables
3. Compute P_client for a test client
4. Build refresh scheduler (monthly for Type A, annual for Type B)

## Phase 4 (Ongoing): Monitor and Recalibrate
1. Monthly: Auto-refresh all Type A indicators, recompute modifiers
2. Annually: Update Type B data from new reports, recompute priors
3. Annually: Compute Brier Score against actual events. If calibration drift >0.10, retrain scaling constants.
4. Ad-hoc: Breaking event triggers (e.g., new conflict, pandemic declaration) → immediate recompute

---

# SECTION 12: FALLBACK RATES

The fallback rates are the CURRENT hardcoded base rates from the existing app. Claude Code must load these from `./data/current_hardcoded_rates.json` at startup. The human must export this file from the current app before implementation begins (see Section 0.8 item 3).

**If the file is missing**, Claude Code should use the base rates from the original Risk Catalog spreadsheet (PRISM_Risk_Catalog.xlsx, column F "Base Rate"). Claude Code can read this directly:

```python
import openpyxl

def load_fallback_rates(catalog_path="./data/PRISM_Risk_Catalog.xlsx"):
    wb = openpyxl.load_workbook(catalog_path, data_only=True)
    ws = wb['Risk Catalog']
    rates = {}
    for row in ws.iter_rows(min_row=4, max_row=ws.max_row, values_only=True):
        if row[0] and row[5] is not None:
            rates[row[0]] = float(row[5])
    return rates
# Returns: {"DIG-CIC-001": 0.08, "DIG-CIC-002": 0.05, ...} for all 174 events
```

---

# SECTION 13: TESTING CHECKLIST

Before declaring Phase 1 complete, Claude Code must verify:

- [ ] All 10 prototype events produce valid JSON matching Section 9.2 schema
- [ ] Every `prior` field has a non-null `derivation.formula` and `derivation.data_source`
- [ ] Every `modifier` has a non-null `calibration.n_observations` (except categoricals)
- [ ] All computed values pass validation (Section 9.1)
- [ ] Fallback chain works: disable an API key → system still returns results with `"data_status": "FALLBACK_HARDCODED"` tag
- [ ] At least 8 of 10 computed priors are within 50% of the hardcoded fallback rates, OR each divergence has a documented `"divergence_reason"` in the output JSON
- [ ] All API rate limits are respected (no 429 errors in logs)
- [ ] EM-DAT column auto-detection works with both portal and HDX formats
