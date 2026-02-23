# PRISM Brain V2 — Functional Specification

**Version:** 2.3
**Date:** February 23, 2026
**Purpose:** Complete end-to-end specification enabling full reconstruction of the PRISM Brain risk intelligence platform.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Risk Taxonomy](#3-risk-taxonomy)
4. [Probability Engine](#4-probability-engine)
5. [Data Connectors](#5-data-connectors)
6. [Dynamic Scoring System](#6-dynamic-scoring-system)
7. [Modifier System](#7-modifier-system)
8. [Fallback Chain](#8-fallback-chain)
9. [Client Risk Assessment](#9-client-risk-assessment)
10. [Database Schema](#10-database-schema)
11. [API Reference](#11-api-reference)
12. [Frontend Application](#12-frontend-application)
13. [Data Files & Configuration](#13-data-files--configuration)
14. [Deployment](#14-deployment)

---

## 1. System Overview

PRISM Brain is a **Risk Intelligence Engine** that calculates probabilities for 174 global risk events across 4 domains, then applies those probabilities to industrial clients' business processes to quantify risk exposure in monetary terms.

### What it does

1. **Computes probabilities** for 174 risk events using three methods (frequency analysis, incidence rates, structural calibration), pulling live data from 12+ external APIs
2. **Applies modifiers** from real-time data sources (energy markets, seismicity, geopolitical tension, cyber threat growth) to adjust base probabilities
3. **Maps risks to client business processes** using a 222-process hierarchical framework
4. **Calculates financial exposure** using the formula: `Exposure = Criticality × Vulnerability × (1−Resilience) × Downtime × Probability`
5. **Presents results** through an interactive dashboard with heatmaps, charts, rankings, and Excel exports
6. **Archives every computation** for historical tracking, comparison, and audit

### Core Formula

```
P_global = Prior × ∏(Modifiers)    clamped to [floor, 0.95]
    where floor = max(0.001, 0.1 × Prior)

Exposure = Criticality_per_day × Vulnerability × (1 − Resilience) × Downtime_days × P_global
```

---

## 2. Architecture

### Two-Process Design

| Component | Technology | Port | Purpose |
|-----------|-----------|------|---------|
| Backend | FastAPI + SQLAlchemy | 8000 | REST API, probability engine, data connectors, database |
| Frontend | Streamlit | 8501 | Interactive dashboard, 8 pages, Excel import/export |

### Technology Stack

**Backend:**
- Python 3.11, FastAPI, Uvicorn (ASGI)
- SQLAlchemy 2.0 ORM, PostgreSQL (primary), SQLite (frontend fallback)
- scikit-learn (logistic regression for ERA5 calibration)
- aiohttp (async HTTP), requests (sync HTTP)
- python-dotenv (environment config)

**Frontend:**
- Streamlit 1.28+
- Plotly (interactive charts), Altair (declarative viz)
- Pandas, openpyxl/xlsxwriter (Excel I/O)

### File Structure

```
prism-brain-v2/
├── main.py                          # App entry point, route registration, migrations
├── client_routes.py                 # Client CRUD endpoints (/api/v1/clients/*)
├── v2_routes.py                     # V2 taxonomy/event endpoints (/api/v2/*)
├── routes/
│   ├── calculations.py              # Signal extraction, Bayesian calculation engine
│   ├── data_sources.py              # 28-source data fetcher, indicator storage
│   └── events.py                    # Basic event listing
├── config/
│   └── settings.py                  # Environment-based settings (DB URL, API keys)
├── database/
│   ├── connection.py                # PostgreSQL connection pool (10+20 overflow)
│   └── models.py                    # 17 SQLAlchemy ORM models
├── prism_engine/                    # Probability computation engine
│   ├── engine.py                    # Orchestrator: compute_one(), compute_all()
│   ├── api_routes.py                # Engine API endpoints (/api/v2/engine/*)
│   ├── history.py                   # Archive compute runs to database
│   ├── fallback.py                  # Fallback rate loader (seeds → catalog → cache)
│   ├── method_c_loader.py           # Method C research data loader
│   ├── indicator_store.py           # Indicator value store (get/set/save)
│   ├── indicator_fetch.py           # Tier 1 auto-fetch orchestrator
│   ├── annual_data.py               # Annual report data management (DBIR, Dragos)
│   ├── computation/
│   │   ├── priors.py                # Prior computation (Methods A, B, C)
│   │   ├── formulas.py              # Mathematical formulas per method
│   │   ├── modifiers.py             # Ratio + categorical modifier system
│   │   ├── scoring.py               # Dynamic scoring (normalize → weight → sigmoid)
│   │   ├── validation.py            # Input/output validation and clipping
│   │   └── era5_calibration.py      # ERA5 temperature logistic regression
│   ├── config/
│   │   ├── event_mapping.py         # Event → method/source dispatch
│   │   ├── sources.py               # Data source definitions
│   │   ├── indicator_mapping.py     # Tier 1 auto-fetch indicator mappings
│   │   ├── credentials.py           # API key management
│   │   └── regions.py               # Geographic regions, seismic zones, chokepoints
│   ├── connectors/                  # External API connectors
│   │   ├── base.py                  # ConnectorResult, retry logic, file cache
│   │   ├── eia.py                   # EIA energy data (stocks, prices, refinery)
│   │   ├── fred.py                  # FRED economic series (yield curve, credit)
│   │   ├── cisa.py                  # CISA Known Exploited Vulnerabilities
│   │   ├── usgs.py                  # USGS earthquake data
│   │   ├── copernicus.py            # ERA5 temperature anomaly
│   │   ├── noaa.py                  # NOAA NAO index (cold wave blocking)
│   │   ├── world_bank.py            # World Bank GDP growth
│   │   ├── ucdp.py                  # UCDP armed conflict data
│   │   ├── nvd.py                   # NVD ICS vulnerability trends
│   │   ├── entsoe.py                # ENTSO-E European grid stress
│   │   ├── gpr.py                   # Geopolitical Risk Index
│   │   └── emdat.py                 # EM-DAT disaster database (local file)
│   └── data/
│       ├── fallback_rates.json      # Cached base rates for all 174 events
│       ├── method_c_overrides.json  # Method C research priors (115 events)
│       ├── method_c_full_research.json  # Full scoring functions + indicators
│       ├── indicator_store_global.json  # Live indicator values (Tier 1+2)
│       ├── annual_updates.json      # Annual report data (DBIR, Dragos)
│       └── manual/news_events.json  # Historical event validation data
├── frontend/
│   ├── Welcome.py                   # Landing page
│   ├── pages/
│   │   ├── 1_Client_Setup.py        # Client profile management
│   │   ├── 2_Process_Criticality.py # Business process selection + criticality
│   │   ├── 3_Risk_Selection.py      # Risk event selection by domain/family
│   │   ├── 4_Risk_Assessment.py     # Vulnerability/resilience/downtime scoring
│   │   ├── 5_Results_Dashboard.py   # Exposure analysis, charts, Excel export
│   │   ├── 6_Annual_Data_Update.py  # Update DBIR/Dragos/dark figures annually
│   │   ├── 7_Data_Sources.py        # Engine status, compute, history archive
│   │   └── 8_Indicator_Data_Entry.py # Tier 2/3 indicator data entry
│   ├── modules/
│   │   ├── api_client.py            # HTTP client for all backend API calls
│   │   └── database.py              # Hybrid backend+SQLite data layer
│   ├── utils/
│   │   ├── constants.py             # App constants, colors, domain definitions
│   │   ├── helpers.py               # Data loaders, formatters, risk level logic
│   │   └── theme.py                 # PRISM CSS theme injection
│   └── data/
│       ├── risk_database.json       # 174 risk events with metadata
│       ├── process_framework.json   # 222 business processes (hierarchy)
│       ├── data_summary.json        # Summary statistics
│       └── seeds/                   # Domain seed files (4 files)
└── method_c_v3_complete.json        # Complete Method C research (115 events)
```

---

## 3. Risk Taxonomy

### 4 Domains, 28 Families, 174 Events

| Domain | Events | Description |
|--------|--------|-------------|
| **PHYSICAL** | 44 | Climate extremes, geophysical, biological, energy, materials, water, pollution |
| **STRUCTURAL** | 42 | Geopolitical, economic, financial, trade, regulatory, reputational |
| **DIGITAL** | 47 | Ransomware, data theft, cloud, supply chain, ICS/OT, hardware, AI risks |
| **OPERATIONAL** | 41 | Supply chain, maritime, air freight, manufacturing, labor, compliance |

### Event ID Format

```
{DOMAIN}-{FAMILY}-{NUMBER}
```

Examples:
- `PHY-CLI-003` — Physical domain, Climate family, event 3 (extreme heat wave)
- `DIG-RDE-001` — Digital domain, Ransomware & Data Extortion family, event 1
- `STR-GEO-001` — Structural domain, Geopolitical family, event 1

### Family Codes

Each family maps to a numeric code (Layer_2_Secondary) for process framework linkage:

| Code | Family Name | Domain |
|------|------------|--------|
| 1.1 | Climate Extremes & Weather Events | Physical |
| 1.2 | Geophysical Events | Physical |
| 1.3 | Biological Hazards | Physical |
| 1.4 | Energy Supply & Grid Events | Physical |
| 1.5 | Material & Resource Supply | Physical |
| 1.6 | Water Stress & Contamination | Physical |
| 1.7 | Pollution & Environmental Contamination | Physical |
| 2.1 | Geopolitical & Conflict Risk | Structural |
| 2.2 | Economic & Recession Risk | Structural |
| 2.3 | Financial System Risk | Structural |
| 2.4 | Trade & Sanctions Risk | Structural |
| 2.5 | Regulatory & Compliance Risk | Structural |
| 2.6 | Reputational & Social Risk | Structural |
| 2.7 | Legal & Liability Risk | Structural |
| 3.1 | Ransomware & Data Extortion | Digital |
| 3.2 | Fraud, Scams & Data Theft | Digital |
| 3.3 | Cloud Services & Infrastructure | Digital |
| 3.4 | Software Supply Chain | Digital |
| 3.5 | Critical Infrastructure Cyber | Digital |
| 3.6 | Hardware & Semiconductor | Digital |
| 3.7 | AI & Emerging Technology Risk | Digital |
| 4.1 | Supply Chain & Logistics | Operational |
| 4.2 | Maritime & Shipping | Operational |
| 4.3 | Air Freight & Aviation | Operational |
| 4.4 | Road & Land Transport | Operational |
| 4.5 | Manufacturing & Production | Operational |
| 4.6 | Labor & Workforce | Operational |
| 4.7 | Business Continuity & Compliance | Operational |

### Risk Event Data Structure

Each event in `risk_database.json`:

```json
{
  "Event_ID": "PHY-CLI-003",
  "Event_Name": "Extreme heat wave affecting production/logistics",
  "Event_Description": "Extended period of extreme heat...",
  "Layer_1_Primary": "PHYSICAL",
  "Layer_2_Primary": "Climate Extremes & Weather Events",
  "Layer_2_Secondary": "1.1",
  "Super_Risk": "NO",
  "base_probability": 0.065,
  "base_rate_pct": 6.5,
  "confidence_level": "HIGH",
  "Geographic_Scope": "EU",
  "Time_Horizon": "annual",
  "data_sources": [
    {"tier": "PRIMARY", "name": "EM-DAT", "url": "https://emdat.be", "cost": "free"},
    {"tier": "SECONDARY", "name": "ERA5", "url": "https://cds.climate.copernicus.eu", "cost": "free"}
  ]
}
```

---

## 4. Probability Engine

### Three Computation Methods

#### Method A — Frequency Analysis (67 events)

Counts years with qualifying events in a historical database, then divides by total observation years.

```
Prior = Event_years / Total_years
```

**Data sources by event type:**

| Source | Events | Method |
|--------|--------|--------|
| EM-DAT | PHY-CLI, PHY-GEO, PHY-BIO, PHY-ENE, PHY-POL, PHY-WAT, some OPS | Count disaster years in EEA region |
| USGS | PHY-GEO-001 (earthquake) | Poisson model: P = 1 − e^(−λ) across seismic zones |
| UCDP | STR-GEO-001 (armed conflict) | Count war-years (1000+ deaths) in top-20 supplier countries |
| World Bank | STR-ECO-001 (recession) | Count years with any G7 country GDP < 0 |
| FRED threshold | STR-ECO-002, STR-ECO-005, PHY-ENE-003, etc. | Count years crossing threshold (e.g., HY spread > 600bp) |
| Manual events | OPS-MAR-002, etc. | Count from curated historical event list |

**Example — PHY-CLI-003 (heat wave):**
```
Source: EM-DAT (2000–2024)
Filter: disaster_type matches "Heat wave" in EEA-Extended region
Result: 15 qualifying years / 25 total years = 0.60
Confidence: High
```

**Example — PHY-GEO-001 (earthquake):**
```
Source: USGS ComCat
Method: Poisson across 7 seismic zones
Per zone: λ = count / years → P(zone) = 1 − e^(−λ)
Combined: P(≥1 anywhere) = 1 − ∏(1 − P_i)
Result: ~0.95 (M6.0+ earthquake occurs somewhere nearly every year)
```

#### Method B — Incidence Rate (48 events)

Uses survey-based rates from annual industry reports (DBIR for cyber, Dragos for OT/ICS), adjusted by attack-type share, event subsplit, and dark figure multiplier.

**DBIR-based events (30 events):**
```
Prior = Base_breach_rate × Attack_share × Subsplit × Dark_figure
```

Where:
- `Base_breach_rate` = 0.18 (DBIR 2025: 18% of organizations breached annually)
- `Attack_share` = fraction attributed to this attack type (e.g., ransomware = 0.44)
- `Subsplit` = fraction of that attack type specific to this event variant
- `Dark_figure` = underreporting multiplier (1.0–3.0)

**Attack shares (from DBIR):**
| Category | Share | Events |
|----------|-------|--------|
| Ransomware | 0.44 | DIG-RDE-* |
| Social engineering | 0.25 | DIG-FSD-* |
| Credential theft | 0.38 | DIG-FSD-* |
| Third-party breach | 0.30 | DIG-SCC-* |
| Web app exploit | 0.26 | DIG-FSD-*, DIG-SCC-* |
| Insider misuse | 0.08 | DIG-FSD-* |

**Dragos-based events (18 events):**
```
Prior = (Incidents × Sector_pct × Dark_figure) / Total_orgs
```

Where:
- `Incidents` = 3,300 (Dragos 2025)
- `Sector_pct` = 0.67 (manufacturing share)
- `Dark_figure` = 3.0 (ICS incidents heavily underreported)
- `Total_orgs` = 300,000

```
Result: (3300 × 0.67 × 3.0) / 300000 = 0.0221 → then × subsplit per event
```

#### Method C — Structural Calibration (59 events)

Decomposes probability into three independent sub-probabilities:

```
Prior = P_preconditions × P_trigger × P_implementation
```

Where:
- **P_preconditions**: Structural conditions that make the event possible (e.g., market concentration, infrastructure age)
- **P_trigger**: Likelihood that a triggering event occurs (e.g., policy change, extreme weather)
- **P_implementation**: Probability the trigger actually materializes into the full event

**Source priority chain for Method C:**

1. **Phase 1 hand-crafted** (2 events: STR-TRD-001, OPS-CMP-001) — custom logic
2. **Dynamic scoring** — compute from live indicator data (see Section 6)
3. **Event-specific research** — from `method_c_overrides.json` (expert-calibrated values)
4. **Family-level defaults** — generic values per family (22 families defined)
5. **Generic fallback** — P_pre=0.50, P_trig=0.50, P_impl=0.50 → Prior=0.125

**Family default examples:**

| Family | P_pre | P_trig | P_impl | Rationale |
|--------|-------|--------|--------|-----------|
| STR-GEO | 0.55 | 0.45 | 0.55 | Conflict conditions elevated post-2022 |
| DIG-RDE | 0.65 | 0.50 | 0.55 | Ransomware infrastructure mature |
| PHY-CLI | 0.60 | 0.55 | 0.50 | Climate trends well-established |
| OPS-SUP | 0.50 | 0.40 | 0.55 | Single-source dependencies common |

### Compute Flow

```
compute_all():
  1. fetch_tier1_indicators()          # Auto-fetch from EIA, FRED, NVD
  2. for each of 174 events:
     a. get_event_config(event_id)     # Determine method (A/B/C) and sources
     b. compute_prior(config)          # Method-specific prior calculation
     c. get_modifiers(config)          # Fetch real-time modifier values
     d. p_global = prior × ∏(mods)    # Apply modifiers with floor/ceiling
     e. validate_output(result)        # Clip, check divergence
  3. archive_compute_run(results)      # Background: save to PostgreSQL
  4. return all 174 results
```

### Output Schema (per event)

```json
{
  "event_id": "PHY-CLI-003",
  "event_name": "Extreme heat wave affecting production/logistics",
  "domain": "Physical",
  "family": "Climate Extremes & Weather Events",
  "layer1": {
    "prior": 0.60,
    "method": "A",
    "derivation": {
      "formula": "15 event-years / 25 total years",
      "data_source": "EM-DAT",
      "source_id": "A08",
      "observation_window": "2000-2024",
      "n_observations": 25,
      "confidence": "High",
      "sub_probabilities": null
    },
    "modifiers": [
      {
        "name": "ERA5 temperature anomaly",
        "source_id": "A01",
        "indicator_value": 1.4,
        "indicator_unit": "σ",
        "modifier_value": 1.21,
        "status": "COMPUTED"
      }
    ],
    "p_global": 0.726
  },
  "metadata": {
    "spec_version": "2.3",
    "engine_version": "2.0.0",
    "fallback_rate": 0.065,
    "divergence_from_fallback": 8.23
  }
}
```

---

## 5. Data Connectors

### Connector Infrastructure

All connectors use a common base (`connectors/base.py`):

- **ConnectorResult**: Standard return type with `source_id`, `success`, `data`, `error`, `cached`, `timestamp`
- **File cache**: JSON files in `prism_engine/data/cache/` with MD5-hashed keys
- **Retry logic**: 3 attempts with exponential backoff [1s, 5s, 15s]; retry on 429/5xx, fail on 4xx
- **Timeout**: 30 seconds per request (configurable)

### Connector Reference

| ID | Connector | API | Key Required | Cache TTL | Events |
|----|-----------|-----|-------------|-----------|--------|
| A01 | Copernicus ERA5 | Published anomaly table | No* | 30 days | PHY-CLI-* (climate) |
| A02 | USGS | earthquake.usgs.gov/fdsnws | No | 7 days | PHY-GEO-001 (earthquake) |
| A03 | FRED | api.stlouisfed.org | Yes | 24h | STR-ECO, STR-FIN, OPS-CMP |
| A04 | NVD | services.nvd.nist.gov | Yes | 7 days | DIG-CIC (ICS cyber) |
| A05 | CISA KEV | cisa.gov/feeds | No | 24h | DIG-* (all digital) |
| A06 | GPR Index | matteoiacoviello.com | No | 7 days | STR-GEO, STR-TRD |
| A07 | UCDP | ucdpapi.pcr.uu.se | No | 7 days | STR-GEO-001 (conflict) |
| A08 | EM-DAT | Local file (emdat.be) | No | N/A | PHY-CLI, PHY-GEO, PHY-BIO |
| A09 | World Bank | api.worldbank.org | No | 7 days | STR-ECO-001 (recession) |
| A10 | ENTSO-E | web-api.tp.entsoe.eu | Yes | 12-24h | PHY-ENE (grid stress) |
| A13 | NOAA CPC | cpc.ncep.noaa.gov | No | 7 days | PHY-CLI-006 (cold wave) |
| — | EIA | api.eia.gov/v2 | Yes | 24h | OPS-AIR, OPS-RLD (fuel) |

*\*CDS_API_KEY optional for raw ERA5 download; published anomaly table used by default*

### Connector Details

**EIA (Energy Information Administration)**
- `fetch_petroleum_stocks()`: U.S. crude oil commercial stocks → days of supply
  - Endpoint: `/petroleum/stoc/wstk/data/`, facets: product=EPC0, area=NUS, process=SAX
  - Calculation: `days_of_supply = stocks_thousand_bbl / 20000`
- `fetch_crude_price()`: WTI spot price + YoY change + 26-week volatility
  - Endpoint: `/petroleum/pri/spt/data/`, facets: product=EPCWTI, area=YCUOK
- `fetch_refinery_outages()`: Refinery utilization averaged across 5 PADD regions
  - Endpoint: `/petroleum/pnp/wiup/data/`, facets: process=YUP
  - Groups by period, averages across regions

**FRED (Federal Reserve Economic Data)**
- `fetch_series(series_id)`: Generic series fetcher, returns observations list
- `get_yield_curve_modifier()`: T10Y2Y series → `modifier = 1.0 + (baseline − latest) × 0.5`, bounds [0.50, 2.50]
- `get_credit_spread_modifier()`: BAMLH0A0HYM2 → `ratio = latest / 60-month mean`, bounds [0.50, 2.50]
- `get_durable_goods_modifier()`: ACDGNO → `ratio = quarter / 20-quarter mean`, bounds [0.50, 2.00]
- `count_threshold_years()`: Generic prior calc — counts years crossing threshold

**CISA KEV**
- `fetch_kev_catalog()`: Downloads full catalog, counts by year, identifies ICS vendors
  - ICS vendors tracked: Siemens, Schneider, Rockwell, Honeywell, ABB, Emerson, Yokogawa, GE, Mitsubishi, Omron, AVEVA, CodeSys, Moxa, Phoenix Contact, WAGO
- `get_kev_modifier()`: YoY growth rate → modifier, bounds [0.50, 2.00]

**USGS**
- `count_earthquakes(zone, bbox, start, end, min_mag)`: ComCat count query
- `compute_earthquake_prior()`: Poisson model across 7 seismic zones → combined P(≥1)
- `get_recent_seismicity_modifier()`: 90-day rate vs 25-year average, bounds [0.50, 3.00]

**Copernicus ERA5**
- `get_temperature_modifier()`: Published anomaly → `modifier = 1.0 + (σ × 0.21)`, bounds [0.75, 1.80]
  - Coefficient 0.21 derived via logistic regression on 25 years of heatwave vs anomaly data

**NVD**
- `fetch_ics_cve_timeseries()`: Searches keywords ["SCADA", "ICS industrial control", "PLC HMI"]
- `get_ics_cve_modifier()`: YoY ICS CVE growth → modifier, bounds [0.50, 2.00]

**GPR Index (Geopolitical Risk)**
- `fetch_gpr_index()`: Downloads Excel from Iacoviello, extracts GPR series
- `get_gpr_modifier()`: `current / 60-month mean`, calibrated to [p5, p95] percentiles

**UCDP (Uppsala Conflict Data)**
- `count_conflict_years()`: UCDP/PRIO dataset v25.1, counts years with war (1000+ deaths)
- Uses Gleditsch-Ward country codes; handles interstate conflicts with comma-separated codes

**NOAA CPC**
- `fetch_nao_index()`: North Atlantic Oscillation monthly values
- `get_nao_blocking_modifier()`: Counts blocking months (NAO < −1.0), bounds [0.50, 2.00]

**World Bank**
- `fetch_gdp_growth()`: G7 GDP growth → counts recession years (any country GDP < 0)

**ENTSO-E**
- `fetch_load_data()`: Actual electricity load by bidding zone
- `get_grid_stress_modifier()`: `1.0 + (peak_avg_ratio − 1.25)`, bounds [0.80, 1.50]

**EM-DAT**
- Local file only (user downloads from emdat.be)
- `count_event_years()`: Regex matching on disaster types, region filtering by ISO codes
- Handles both portal and HDX format variations

---

## 6. Dynamic Scoring System

For 115 Method C events, indicators can drive sub-probabilities dynamically instead of using static research values.

### Three-Tier Indicator System

| Tier | Source | Coverage | TTL | Entry Method |
|------|--------|----------|-----|--------------|
| **Tier 1** | Public APIs (FRED, EIA, NVD) | ~5% of indicators | 24h | Auto-fetch |
| **Tier 2** | Analyst reports (Gartner, IATA, Drewry) | ~25% | 1 year | Manual entry |
| **Tier 3** | Client operational data | ~25% | 6 months | Manual/Excel |

### Indicator Store

Key format: `"{event_id}/{sub_probability}/{indicator_id}"`

```json
{
  "OPS-AIR-004/p_preconditions/stocks": {
    "value": 21.0,
    "unit": "days",
    "tier": 1,
    "source": "eia.fetch_petroleum_stocks",
    "updated_at": "2026-02-23T08:45:32Z",
    "ttl_hours": 24
  }
}
```

### Scoring Pipeline (per sub-probability)

```
1. NORMALIZE each indicator to [0, 1]:
   - linear_scale:         (value − min) / (max − min)
   - inverse_linear_scale: 1 − linear_scale
   - log_scale:            log(value − min + 1) / log(max − min + 1)
   - discrete_map:         lookup table {value → score}
   - threshold:            1 if value ≥ threshold, else 0

2. WEIGHT: Re-normalize weights across available indicators
   If coverage < 30%: use static research value instead
   renorm_weight = w_i / Σ(w_available)

3. WEIGHTED SCORE:
   ws = Σ(normalized_i × renorm_weight_i)

4. SIGMOID:
   P_sub = 1 / (1 + exp(−steepness × (ws − midpoint)))
   Default: midpoint=0.5, steepness=6.0

5. CLAMP to [0.05, 0.95]

6. COMPOSITE:
   Prior = P_pre × P_trig × P_impl
```

### Indicator Mapping Example

For event `OPS-AIR-004` (Jet fuel shortage), sub-probability `p_preconditions`:

| Indicator | Source | Normalization | Weight | Params |
|-----------|--------|---------------|--------|--------|
| stocks | EIA | inverse_linear_scale | 0.40 | min=15, max=35 days |
| vol | EIA | linear_scale | 0.30 | min=1, max=15% |
| outages | EIA | linear_scale | 0.30 | min=0, max=12 weeks |

### Tier 1 Auto-Fetch

On every `compute_all()`, before computing events:

```
fetch_tier1_indicators():
  for each mapped indicator:
    call connector (EIA, FRED, NVD)      # Cached per-run
    extract value from result
    set_indicator_value(event_id, sub_prob, indicator_id, value)
  save_global_store()                     # Persist to JSON
```

Currently mapped auto-fetch indicators:
- `stocks` → `eia.fetch_petroleum_stocks` → days_of_supply
- `vol` → `eia.fetch_crude_price` → volatility_pct
- `spike` → `eia.fetch_crude_price` → yoy_change_pct
- `outages` → `eia.fetch_refinery_outages` → outage_weeks
- `ig_spread` → `fred.fetch_series(BAMLC0A4CBBB)` → latest value

---

## 7. Modifier System

After computing the prior, modifiers adjust it based on real-time conditions.

### Application Formula

```
p_global = prior × ∏(modifier_values)
floor = max(0.001, 0.1 × prior)
ceiling = 0.95
p_global = clamp(p_global, floor, ceiling)
```

### Ratio Modifiers

Continuous values computed as ratio of current to baseline:

```
baseline = rolling N-month mean of indicator
modifier = clamp(current / baseline, [0.50, 3.00])
```

| Source | Indicator | Events | Typical Range |
|--------|-----------|--------|---------------|
| ERA5 (A01) | Temperature anomaly σ | PHY-CLI-* | 0.75–1.80 |
| USGS (A02) | 90-day seismicity rate | PHY-GEO-001 | 0.50–3.00 |
| FRED (A03) | Yield curve spread | STR-ECO, STR-FIN | 0.50–2.50 |
| FRED (A03) | Durable goods orders | OPS-CMP, DIG-HWS | 0.50–2.00 |
| NVD (A04) | ICS CVE YoY growth | DIG-CIC-* | 0.50–2.00 |
| CISA (A05) | KEV catalog YoY growth | DIG-* | 0.50–2.00 |
| GPR (A06) | Geopolitical risk ratio | STR-GEO, STR-TRD | 0.50–3.00 |
| ENTSO-E (A10) | Grid stress ratio | PHY-ENE-* | 0.80–1.50 |

### Categorical Modifiers

Binary conditions that apply preset multipliers:

| Condition | Modifier | Evidence |
|-----------|----------|----------|
| US election year | 1.25 | WTO: +28-31% trade measures in election years |
| Active OECD military conflict | 1.40 | GPR Index +40% during 2022-2024 |
| WHO PHEIC active | 1.50 | COVID PHEIC: +50% supply chain disruptions |
| El Niño active | 1.20 | EM-DAT: +20% climate disasters in El Niño years |
| ECB rate hiking cycle | 1.15 | Allianz: +15% insolvencies during 2022-2023 |

---

## 8. Fallback Chain

When a computation method fails, the system falls back gracefully:

### Prior Fallback

```
1. COMPUTED: Live computation from data source
2. CACHED: Previous computation result (24h TTL)
3. HARDCODED: From seed JSON files or catalog
4. DEFAULT: 0.05 (5%) for completely unknown events
```

### Fallback Rate Loading

```
_load_fallback_rates():
  1. Load from 4 seed files (physical, structural, digital, operational)
  2. Load from PRISM_Risk_Catalog.xlsx (if available)
  3. Merge (seeds take priority over catalog)
  4. Cache to fallback_rates.json for fast reload
```

### Modifier Fallback

If any modifier source fails, the modifier returns 1.0 (neutral) with status `FALLBACK`. The computation continues with remaining modifiers.

### Divergence Tracking

```
divergence = |computed_prior − fallback_rate| / fallback_rate
if divergence > 0.50:
  log reason if available in DIVERGENCE_REASONS
```

This documents cases where the computed prior differs significantly from the seed value (usually because the seed was a Layer 2 estimate while the engine computes Layer 1).

---

## 9. Client Risk Assessment

### Workflow (5 Steps)

```
Step 1: Client Setup       → Company profile (name, industry, revenue, employees)
Step 2: Process Criticality → Select business processes, assign criticality (€/day)
Step 3: Risk Selection      → Choose which of 174 risks apply to this client
Step 4: Risk Assessment     → Score vulnerability, resilience, downtime per risk×process
Step 5: Results Dashboard   → View exposure analysis, charts, rankings, export
```

### Process Framework

222 processes organized in 2 levels across 4 scopes:

| Scope | Name | Processes | Examples |
|-------|------|-----------|---------|
| A | Physical Assets & Infrastructure | 1–6 | Buildings, storage, production, equipment |
| B | Business Operations | 7–18 | Supply chain, customers, transport, utilities |
| C | Strategic & Commercial | 19–25 | IT, cybersecurity, production ops, quality |
| D | Digital, Production & Workforce | 26–32 | HR, product development, maintenance |

Each scope has 5-8 macro-processes, each with 4-8 sub-processes. Clients select the sub-processes relevant to their operations.

### Criticality Calculation

```
Default criticality per process = Annual_revenue / 250_working_days / Number_of_processes
```

Users can override per process based on actual impact assessment.

### Exposure Formula

```
Exposure = Criticality_per_day × Vulnerability × (1 − Resilience) × Expected_downtime_days × Probability
```

| Parameter | Range | Meaning |
|-----------|-------|---------|
| Criticality | EUR/day | Revenue lost per day of disruption |
| Vulnerability | 0.0–1.0 | How exposed to this risk (0=immune, 1=fully exposed) |
| Resilience | 0.0–1.0 | Ability to withstand (0=no resilience, 1=fully resilient) |
| Downtime | Days | Expected duration of disruption |
| Probability | 0.0–1.0 | Annual probability of the event (from engine) |

### Exposure Summary (aggregated)

```json
{
  "total_exposure": 1500000,
  "revenue_at_risk_pct": 3.0,
  "by_domain": {"DIGITAL": 500000, "PHYSICAL": 1000000, ...},
  "by_process": {"ERP Operations": 900000, "Production Floor": 600000},
  "by_risk": {"Ransomware": 450000, "Heat wave": 200000, ...},
  "assessments": [{process_name, risk_name, domain, criticality, vulnerability, resilience, downtime, probability, exposure}]
}
```

---

## 10. Database Schema

### Core Tables

**RiskEvent** — Master list of 174 risk events
| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | Auto-increment |
| event_id | String(50) unique | e.g., "PHY-CLI-003" |
| event_name | String(500) | Display name |
| layer1_primary | String(100) | Domain |
| layer2_primary | String(100) | Family name |
| layer2_secondary | String(100) | Family code |
| baseline_probability | Float | Default probability |
| super_risk | Boolean | Super-risk flag |
| geographic_scope | String(100) | EU, Global, etc. |
| description | Text | Full description |

**Client** — Company profiles
| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | |
| name | String(500) | Company name |
| industry | String(200) | Sector |
| revenue | Float | Annual EUR |
| employees | Integer | Headcount |
| currency | String(10) | Default EUR |
| export_percentage | Float | % exports |
| primary_markets | Text | CSV |

**ClientProcess** — Selected business processes
| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | |
| client_id | Integer FK → Client | CASCADE delete |
| process_id | String(50) | e.g., "1.3" |
| process_name | String(500) | From framework |
| criticality_per_day | Float | EUR/day |

**ClientRisk** — Selected risks per client
| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | |
| client_id | Integer FK → Client | CASCADE delete |
| risk_id | String(50) | Event ID |
| risk_name | String(500) | Event name |
| domain | String(100) | |
| probability | Float | Latest probability |
| is_prioritized | Boolean | Flagged for focus |

**ClientRiskAssessment** — Vulnerability/resilience/downtime scores
| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | |
| client_id | Integer FK → Client | CASCADE delete |
| process_id | Integer FK → ClientProcess | CASCADE delete |
| risk_id | Integer FK → ClientRisk | CASCADE delete |
| vulnerability | Float | 0–1 |
| resilience | Float | 0–1 |
| expected_downtime | Integer | Days |
| Unique constraint | | (client_id, process_id, risk_id) |

### History Tables

**CalculationLog** — Run-level metadata
| Column | Type | Notes |
|--------|------|-------|
| calculation_id | String(50) unique | e.g., "abc12345-20260223-143210" |
| start_time, end_time | DateTime | |
| events_processed | Integer | Usually 174 |
| events_succeeded | Integer | |
| events_failed | Integer | |
| duration_seconds | Float | |
| status | String(50) | COMPLETED or COMPLETED_WITH_ERRORS |
| trigger | String(50) | manual or scheduled |
| method | String(50) | engine_v2 |

**ProbabilitySnapshot** — Per-event per-run snapshots
| Column | Type | Notes |
|--------|------|-------|
| event_id | String(50) indexed | |
| probability_pct | Float | p_global × 100 |
| confidence_score | Float | 0–1 |
| calculation_id | String(50) indexed | FK to CalculationLog |
| snapshot_date | DateTime | |
| prior | Float | Raw prior |
| method | String(10) | A, B, C, or FALLBACK |
| data_source | String(200) | e.g., "EM-DAT" |
| is_dynamic | Boolean | Dynamic scoring used |
| modifier_count | Integer | |
| domain | String(100) | |
| family | String(100) | |
| event_name | String(500) | |

### Signal & ML Tables (Phase 4)

**RiskProbability** — Full calculation results with signal extraction
- Contains 40+ columns for Bayesian calculation, signal extraction, ML ensemble, explainability, dependency adjustments
- Key fields: probability_pct, confidence_score, signal, momentum, trend, is_anomaly, ensemble_method, attribution (JSON), explanation (Text)

**IndicatorWeight** — Indicator-to-event configuration
- Maps which indicators (from 28 sources) apply to which events
- Fields: event_id, indicator_name, weight, normalized_weight, data_source, beta_type

**IndicatorValue** — Time-series indicator storage (appended, never overwritten)
- Fields: event_id, indicator_name, value, raw_value, z_score, data_source, timestamp, signal, momentum, trend, is_anomaly

**DataSourceHealth** — External source health tracking
- Fields: source_name, status (OPERATIONAL/DEGRADED/ERROR), response_time_ms, success_rate_24h

### Additional Tables

- **ProbabilityAlert** — Threshold-based alert rules
- **AlertEvent** — Alert trigger log
- **IndustryProfile** — Pre-configured risk profiles by industry
- **ProfileRiskEvent** — Profile-to-risk mappings
- **ReportSchedule** — Scheduled report configuration
- **ReportHistory** — Generated report log

---

## 11. API Reference

### Health

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Backend health check |
| GET | `/api/v2/health` | V2 API health |

### Client Management (V1)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/clients` | Create client |
| GET | `/api/v1/clients` | List all clients |
| GET | `/api/v1/clients/{id}` | Get client detail |
| PUT | `/api/v1/clients/{id}` | Update client |
| DELETE | `/api/v1/clients/{id}` | Delete client (cascades) |
| POST | `/api/v1/clients/{id}/processes` | Add process |
| GET | `/api/v1/clients/{id}/processes` | List processes |
| PUT | `/api/v1/clients/{id}/processes/{pid}` | Update process |
| DELETE | `/api/v1/clients/{id}/processes/{pid}` | Delete process |
| POST | `/api/v1/clients/{id}/risks` | Add risk |
| GET | `/api/v1/clients/{id}/risks` | List risks |
| PUT | `/api/v1/clients/{id}/risks/{rid}` | Update risk |
| POST | `/api/v1/clients/{id}/assessments` | Save assessment |
| GET | `/api/v1/clients/{id}/assessments` | List assessments |
| GET | `/api/v1/clients/{id}/exposure-summary` | Exposure analysis |

### Events & Taxonomy (V2)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v2/taxonomy` | Full domain→family→event tree |
| GET | `/api/v2/events` | List events (filters: domain, family_code, search) |
| GET | `/api/v2/events/{event_id}` | Single event detail |
| GET | `/api/v2/domains/{domain}` | Domain with families & events |
| GET | `/api/v2/families/{family_code}` | Family with events |
| GET | `/api/v2/probabilities` | Latest probabilities |

### Engine Computation

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v2/engine/status` | Engine version, event counts, credentials |
| GET | `/api/v2/engine/compute/{event_id}` | Compute single event |
| GET | `/api/v2/engine/compute-all` | Compute all 174 events (auto-archives) |
| GET | `/api/v2/engine/compute-phase1` | Compute 10 Phase 1 events only |
| GET | `/api/v2/engine/fallback-rates` | All 174 fallback rates |

### Engine History

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v2/engine/history/runs` | List historical runs (newest first) |
| GET | `/api/v2/engine/history/runs/{calc_id}` | All snapshots for a run |
| GET | `/api/v2/engine/history/events/{event_id}` | Probability history for one event |
| GET | `/api/v2/engine/history/compare?run_a=X&run_b=Y` | Side-by-side run comparison |

### Engine Indicators

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v2/engine/indicators/{event_id}` | Get indicator values + coverage |
| PUT | `/api/v2/engine/indicators` | Save indicator values (batch) |
| GET | `/api/v2/engine/indicator-coverage` | Coverage summary all events |
| GET | `/api/v2/engine/indicator-sources` | List unique data sources |
| POST | `/api/v2/engine/indicator-fetch` | Trigger Tier 1 auto-fetch |

### Annual Data & Method C

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v2/engine/annual-data` | Get DBIR/Dragos/dark figure config |
| PUT | `/api/v2/engine/annual-data` | Save annual data updates |
| GET | `/api/v2/engine/method-c-status` | Method C integration status |
| POST | `/api/v2/engine/method-c-integrate` | Integrate research from file |

### Data Sources (V1)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/data-sources/health` | Source health status |
| POST | `/api/v1/data/refresh` | Fetch from 28 sources + recalculate |
| GET | `/api/v1/stats` | System statistics |

---

## 12. Frontend Application

### Page Flow

```
Welcome → Client Setup → Process Criticality → Risk Selection → Risk Assessment → Results Dashboard
                                                                                        ↓
                                                           Annual Data Update / Data Sources / Indicator Entry
```

### Page 1: Welcome

- PRISM branding and overview
- Backend connectivity status
- "Get Started" navigation to Client Setup

### Page 2: Client Setup

**Tab 1 — Company Profile:**
- Form: Name, Location, Industry (dropdown with templates), Revenue, Employees, Currency, Export %, Markets, Sectors
- Industry templates auto-select relevant processes on creation

**Tab 2 — Client List:**
- Table of all clients with Edit/Delete actions
- "Delete All" with confirmation

### Page 3: Process Criticality

**Tab 1 — Select Processes:**
- 4 scope sections (A–D), each with collapsible macro-process expanders
- Each expander shows sub-processes with checkboxes
- Bulk select/deselect per macro-process
- Sorted by process ID (1.1, 1.2, ... 32.6)

**Tab 2 — Criticality:**
- Table of selected processes with editable criticality (€/day)
- Default calculated from revenue: `Revenue / 250 / num_processes`
- Shows Daily Downtime Revenue Impact per process

**Tab 3 — Import/Export:**
- Download: Professional PRISM Questionnaire Excel template (multi-sheet with dropdowns)
- Upload: Parse completed questionnaire, preview, apply

### Page 4: Risk Selection

**Tab 1 — Select Risks:**
- 4 domain sections, each with collapsible family expanders
- Each family shows events with checkboxes and probability badges (colored by risk level)
- Sorted by family code (1.1, 1.2, ... 4.7)
- Probabilities fetched from engine (if backend online) or fallback rates

**Tab 2 — Summary:**
- Selected risk count by domain
- Coverage analysis

### Page 5: Risk Assessment

**Tab 1 — Assess:**
- For each risk × process combination:
  - Vulnerability slider (0–100%)
  - Resilience slider (0–100%)
  - Downtime (days, number input)
  - Notes (text)
- Quick-fill templates by domain
- Live exposure calculation as values change

**Tab 2 — Review:**
- Table of all assessments with computed exposure
- Sort and filter options

**Tab 3 — Import/Export:**
- Download: Excel with all assessment combinations
- Upload: Parse and apply completed assessments

### Page 6: Results Dashboard

**Tab 1 — Executive Summary:**
- Total exposure (large metric card)
- Revenue at risk percentage
- Domain breakdown (4-column with pie chart)
- Top 10 risks by exposure (bar chart)
- Top 10 processes by exposure (bar chart)

**Tab 2 — Detailed Analysis:**
- Full sortable table of all assessments
- Filter by domain, process, risk
- Each row shows: process, risk, domain, criticality, vulnerability, resilience, downtime, probability, exposure

**Tab 3 — Visualizations:**
- Process × Domain heatmap (always shows all 4 domains on x-axis)
- Domain exposure pie chart
- Risk ranking horizontal bar chart
- Process ranking horizontal bar chart

**Tab 4 — Export:**
- Download comprehensive Excel report (3 sheets: Executive Summary, Detailed Assessments, By Domain)
- Styled with PRISM colors, auto-width columns, formatted currency

### Page 7: Data Sources

**Tab 1 — Engine Status:**
- Engine version, total events, fallback rates loaded
- API credential status table (configured vs missing)

**Tab 2 — Compute Probabilities:**
- "Compute All 174 Events" button (120s timeout)
- Progress spinner during computation
- Results: method distribution, probability histogram

**Tab 3 — History Archive:**
- Run list table (date, ID, events, duration, status)
- Drill-down: select run → see all 174 event probabilities with domain filter
- Excel download of any run (3-tab styled workbook)
- Compare two runs: side-by-side delta analysis

### Page 8: Annual Data Update

**Tab 1 — DBIR Data:**
- Base breach rate, year
- Attack category shares (ransomware, social engineering, etc.)

**Tab 2 — Dragos/OT Data:**
- ICS incident count, manufacturing share, dark figure

**Tab 3 — Dark Figures:**
- Underreporting multipliers by event category

**Tab 4 — Event Subsplits:**
- Per-event subsplit overrides for DBIR-based events
- Shows computed probability live

### Page 9: Indicator Data Entry

**Tab 1 — Tier 2 (Research Reports):**
- Source selector (Gartner, IATA, Drewry, etc.)
- Indicator entry form with normalization ranges
- Shows affected events per indicator

**Tab 2 — Tier 3 (Client Data):**
- Client selector
- Grouped by family, shows client's selected risks
- Entry form with current values and status badges
- Excel questionnaire download/upload

### Design System

**Colors:**
- Primary: #1B3A4B (dark teal)
- Accent: #E8862A (orange)
- Background: #F5F7FA
- Domain colors: Physical=#E8862A, Structural=#1B3A4B, Digital=#7030A0, Operational=#22876C

**Typography:** Inter font family, non-blocking load

**Components:** Orange primary buttons, orange-bordered metric cards, orange-underlined active tabs, striped tables, collapsible expanders

### Hybrid Data Layer

The frontend uses a hybrid approach:
1. Try backend API first (FastAPI on port 8000)
2. On failure, fall back to local SQLite (`frontend/data/prism_brain.db`)
3. Read operations cached for 30 seconds (survives Streamlit reruns)
4. Backend URL: configurable via Streamlit secrets, env var, or default `http://127.0.0.1:8000`

---

## 13. Data Files & Configuration

### Risk Data

| File | Content | Records |
|------|---------|---------|
| `frontend/data/risk_database.json` | All 174 risk events with metadata | 174 |
| `frontend/data/process_framework.json` | Business process hierarchy | 222 (32 L1 + 190 L2) |
| `frontend/data/seeds/*.json` | Domain-specific event seeds (4 files) | 174 total |
| `prism_engine/data/fallback_rates.json` | Cached base rates | 174 |
| `method_c_v3_complete.json` | Full Method C research + scoring functions | 115 events, 1072 indicators |
| `prism_engine/data/method_c_overrides.json` | Expert-calibrated priors | 115 events |
| `prism_engine/data/method_c_full_research.json` | Scoring function definitions | 115 events |

### Runtime Data

| File | Content | Update Frequency |
|------|---------|-----------------|
| `prism_engine/data/indicator_store_global.json` | Live indicator values | Per computation |
| `prism_engine/data/annual_updates.json` | DBIR/Dragos annual config | Annually |
| `prism_engine/data/cache/*.json` | Connector response cache | Per TTL |
| `prism_engine/data/manual/news_events.json` | Historical event references | As needed |

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `FRED_API_KEY` | Recommended | FRED economic data |
| `EIA_API_KEY` | Recommended | EIA energy data |
| `NVD_API_KEY` | Recommended | NVD vulnerability data |
| `ENTSOE_API_KEY` | Optional | ENTSO-E grid data |
| `CDS_API_KEY` | Optional | Copernicus raw data (not needed for published table) |

---

## 14. Deployment

### Local Development

```bash
# Backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend
streamlit run Welcome.py --server.port 8501
```

### Docker

```dockerfile
FROM python-3.11-slim
# Installs both backend and frontend requirements
# Exposes 8000 + 8501
# Runs start_app.sh
```

### Railway (Production)

```
Procfile: web: bash start_app.sh
runtime.txt: python-3.11.7
```

### Database

- **Production:** PostgreSQL (Railway or local)
- **Frontend fallback:** SQLite at `frontend/data/prism_brain.db`
- **Connection pool:** 10 persistent + 20 overflow, 30s timeout, 30min recycle, pre-ping enabled
- **Schema migration:** `ensure_schema_updates()` in `main.py` runs ALTER TABLE IF NOT EXISTS on startup

---

## Appendix A: Complete Event Method Assignment

| Method | Count | Event Patterns |
|--------|-------|---------------|
| A (Frequency) | 67 | PHY-CLI-*, PHY-GEO-*, PHY-BIO-*, PHY-ENE-*, PHY-POL-*, PHY-WAT-*, some STR-*, some OPS-* |
| B (Incidence) | 48 | DIG-RDE-*, DIG-FSD-*, DIG-SCC-*, DIG-CIC-*, some DIG-CLS-* |
| C (Structural) | 59 | Remaining events without frequency or survey data |

## Appendix B: Geographic Regions

| Region | Countries | Used By |
|--------|-----------|---------|
| EU27 | 27 EU member states | General EU scope |
| EEA_EXTENDED | EU27 + GB, CH, NO, IS, LI | PHY-CLI, PHY-ENE |
| TOP20_SUPPLIERS | CN, US, GB, CH, RU, NO, JP, KR, TR, IN, BR, VN, TW, TH, SA, ID, MY, MX, UA, ZA | STR-GEO |
| OECD_TRADING | EU27 + US, GB, JP, KR, CA, AU, CH, MX | STR-TRD |
| Seismic zones | Mediterranean, Japan, US West Coast, Mexico, Central America, Indonesia, Chile/Peru | PHY-GEO |
| Maritime chokepoints | Suez, Panama, Malacca, Hormuz, Bab al-Mandab, Turkish Straits | OPS-MAR |

## Appendix C: Validation Bounds

| Field | Min | Max |
|-------|-----|-----|
| prior | 0.001 | 0.95 |
| modifier | 0.50 | 3.00 |
| p_global | 0.001 | 0.95 |
| vulnerability | 0.0 | 1.0 |
| resilience | 0.0 | 1.0 |
| geographic_exposure | 0.0 | 3.0 |
| industry_exposure | 0.0 | 3.0 |
