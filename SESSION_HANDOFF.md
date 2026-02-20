# PRISM Brain V2 — Session Handoff Document

**Created:** February 19, 2026
**Purpose:** Provides full context for continuing work in a new Claude Cowork session.
**User:** Antonio (non-developer — always guide step by step with rationale)

---

## 1. WHAT IS THIS PROJECT?

PRISM Brain V2 is a **Risk Intelligence Engine** that calculates probabilities for 174 global risk events across 4 domains. It pulls data from 28+ external sources (USGS, CISA, FRED, NOAA, etc.), applies Bayesian probability calculations with signal extraction and ML enhancement, and presents results through a Streamlit dashboard.

The app has two parts:
- **Backend (FastAPI):** REST API that manages risk events, calculates probabilities, fetches external data. Runs on port 8000.
- **Frontend (Streamlit):** Visual dashboard with 7 pages for client management, risk selection, assessment, and data exploration. Runs on port 8501.

---

## 2. CURRENT STATE (as of Feb 20, 2026 — Session 7)

### What works:
- Backend starts successfully: `python -m uvicorn main:app --host 0.0.0.0 --port 8000`
- Database initialized (PostgreSQL local)
- All syntax checks pass
- Major cleanup completed (20,954 lines removed)
- **Performance fixes applied:** non-blocking Google Fonts, cached init_database, lazy external tables, HTTP connection pooling, localhost→127.0.0.1, 30s cache TTL
- **Process Criticality redesigned:** "Criticality" tab shows Daily Downtime Revenue Impact; inline process IDs
- **Import/Export redesigned:** uses professional PRISM Questionnaire template (multi-sheet Excel with dropdowns)
- **Risk Selection sorted** by domain → family → event ID
- **Results Dashboard fixed:** workflow progress indicator, heatmap, plotly charts all working
- **V2 Event Explorer page removed** (no longer needed)
- **Probability Engine (prism_engine) built and integrated:**
  - **ALL 174 events computing successfully (Phase 2 complete)**
  - Engine version 2.0.0
  - 10 data connectors (USGS, CISA, GPR, NOAA, World Bank, FRED, NVD, UCDP, EM-DAT, Copernicus)
  - 3 computation methods: A (frequency, 33 events), B (incidence rate, 26 events), C (structural calibration, 115 events)
  - Modifier system (ratio + categorical)
  - Fallback chain: computed → cached → hardcoded base rate
  - API routes registered at `/api/v2/engine/*`
  - Full derivation trail in JSON output (data source, formula, observation window, confidence)
  - Config-driven dispatch (no hardcoded if/elif chains)
  - Method C family-level calibration for 23 event families

### What still needs work:
- ~~**ACLED API access tier**~~ — **RESOLVED.** Replaced ACLED with UCDP (Uppsala Conflict Data Program). Free, no API key, 25-year window. STR-GEO-001 now computes 48% prior (12 war-years in 25).
- ~~**ERA5 scaling constant 0.15**~~ — **RESOLVED.** Derived coefficient 0.21 via logistic regression on 25yr of ERA5 European summer anomalies vs EM-DAT heatwave events. See `prism_engine/computation/era5_calibration.py`.
- **Copernicus ERA5 downloads are slow:** cdsapi + xarray + netcdf4 installed and configured. CDS API key works. But ERA5 data requests take 30+ minutes — first download not yet completed. Temperature modifier falls back to 1.0 until first cache.
- **Method C research in progress:** A parallel Claude session is researching evidence-based sub-probabilities for ~115 Method C events using the PRD at `docs/PRD_Method_C_Research.md`. Output expected as `method_c_research_output.json`.
- ~~**Legacy V1 routes still active**~~ — **RESOLVED.** V1 data/calculation routes retired from `main.py`. Data Sources page rewired to V2 engine endpoints. Client CRUD routes (`/api/v1/clients/*`) kept (still active).

### Git status:
- Latest pushed commit: `b798cfc` — Phase 2 engine scaling (all 174 events)
- Unpushed changes: Connector fixes (FRED AMTMNO, Copernicus cdsapi, ACLED OAuth rewrite)
- Remote: `https://github.com/antonio-prism/prism-brain-v2` (main branch)

### Rollback if needed:
```bash
git checkout 3ddd23e -- .
```

---

## 3. PROJECT ARCHITECTURE

### File Structure:
```
prism-brain-v2/
├── main.py                    # App entry point (v3.1.0), health check, registers active routes
├── routes/                    # RETIRED — V1 data/calculation routes (no longer registered)
│   ├── __init__.py
│   ├── events.py              # RETIRED — GET /api/v1/events
│   ├── calculations.py        # RETIRED — Old probability engine (replaced by prism_engine)
│   └── data_sources.py        # RETIRED — Old DataFetcher (replaced by prism_engine)
├── prism_engine/              # Probability engine (v2.0 — all 174 events)
│   ├── __init__.py
│   ├── api_routes.py          # FastAPI routes at /api/v2/engine/*
│   ├── annual_data.py         # Annual data update persistence (DBIR, Dragos overrides)
│   ├── engine.py              # Main orchestrator: compute(), compute_all()
│   ├── fallback.py            # Loads 174 fallback rates from seed files + Risk Catalog
│   ├── config/
│   │   ├── credentials.py     # API key management from env vars
│   │   ├── event_mapping.py   # All 174 event configs (auto-loaded from seed files)
│   │   ├── regions.py         # Geographic definitions (EU27, OECD, bounding boxes, etc.)
│   │   └── sources.py         # Type A/B/C data source registry
│   ├── connectors/
│   │   ├── base.py            # HTTP infrastructure, caching, retry logic
│   │   ├── usgs.py            # USGS earthquakes (no key)
│   │   ├── cisa.py            # CISA KEV catalog (no key)
│   │   ├── gpr.py             # GPR geopolitical risk index (no key)
│   │   ├── noaa.py            # NOAA NAO index (no key)
│   │   ├── world_bank.py      # World Bank GDP growth (no key)
│   │   ├── fred.py            # FRED economic indicators (needs key)
│   │   ├── nvd.py             # NIST NVD vulnerabilities (needs key)
│   │   ├── acled.py           # ACLED conflict data (DEPRECATED — replaced by UCDP)
│   │   ├── ucdp.py            # UCDP/PRIO conflict data (free, no key)
│   │   ├── emdat.py           # EM-DAT disasters (local file)
│   │   └── copernicus.py      # Copernicus ERA5 climate (needs key + cdsapi)
│   ├── computation/
│   │   ├── priors.py          # Methods A, B, C for prior calculation
│   │   ├── modifiers.py       # Ratio and categorical modifier calibration
│   │   ├── formulas.py        # P_global = prior x modifiers with floor/ceiling
│   │   ├── era5_calibration.py # ERA5 temperature scaling regression (derived 0.21 coefficient)
│   │   └── validation.py      # Value validation and clipping rules
│   ├── data/
│   │   ├── manual/news_events.json  # Manual event lists (canal closures, disease outbreaks)
│   │   ├── annual_updates.json      # Annual data overrides (written by manual entry page)
│   │   ├── fallback_rates.json      # Cached fallback rates (auto-generated)
│   │   └── cache/                   # Connector response cache (auto-managed)
│   ├── manual_entry/templates/      # Templates for manual data entry (DBIR, Dragos)
│   ├── output/events/               # Computed event JSON files (Section 9.2 format)
│   └── tests/
│       └── test_integration.py      # Phase 1 integration tests
├── client_routes.py           # Client CRUD, processes, risks, assessments
├── v2_routes.py               # V2 taxonomy API (domains/families)
├── config/
│   ├── settings.py            # App settings from environment variables
│   └── category_indicators.py # LEGACY — Old indicator mapping
├── database/
│   ├── connection.py          # SQLAlchemy engine, init_db(), session context
│   └── models.py              # All SQLAlchemy table models
├── frontend/
│   ├── Welcome.py             # Streamlit entry point
│   ├── modules/
│   │   ├── api_client.py      # Frontend → Backend API communication
│   │   ├── database.py        # SQLite client data + backend API bridge
│   │   ├── external_data.py   # External data fetching helpers
│   │   └── probability_engine.py # Client-side probability calculations
│   ├── pages/
│   │   ├── 1_Client_Setup.py
│   │   ├── 2_Process_Criticality.py
│   │   ├── 3_Risk_Selection.py
│   │   ├── 4_Risk_Assessment.py
│   │   ├── 5_Results_Dashboard.py
│   │   ├── 6_Annual_Data_Update.py  # Type B manual entry (DBIR, Dragos, dark figures)
│   │   └── 7_Data_Sources.py        # Engine status & compute-all (rewired from V1 to V2)
│   ├── data/
│   │   ├── seeds/             # 4 JSON files with 174 risk event definitions
│   │   │   ├── physical_domain_seed.json    (44 events)
│   │   │   ├── structural_domain_seed.json  (43 events)
│   │   │   ├── digital_domain_seed.json     (47 events)
│   │   │   └── operational_domain_seed.json (41 events)
│   │   ├── process_framework.json  # 222 business processes
│   │   ├── risk_database.json      # OLD 900-event version (not used by current app)
│   │   ├── data_summary.json       # Summary stats for Welcome page
│   │   └── prism_brain.db          # SQLite for client-side data
│   ├── utils/
│   │   ├── constants.py
│   │   ├── helpers.py
│   │   └── theme.py
│   └── .streamlit/
│       ├── config.toml
│       ├── pages.toml
│       └── secrets.toml
├── load_events.py             # Script to load seed JSON → PostgreSQL
├── parse_events.py            # Script that parsed taxonomy MD files → JSON
├── migrate.py                 # Database migration (for existing tables only)
├── regenerate_weights.py      # Regenerate indicator weights
├── sync_from_railway.py       # Sync from Railway DB (password removed for security)
├── start_backend.bat          # Windows: double-click to start backend
├── start_frontend.bat         # Windows: double-click to start frontend
├── load_data.bat              # Windows: double-click to load events into DB
├── cleanup_files.bat          # Windows: removes leftover files from cleanup
├── CLEANUP_LOG.md             # Documents what was changed + rollback instructions
├── requirements.txt           # Python backend dependencies
├── runtime.txt                # Python version spec
├── Procfile                   # Railway deployment config
├── .env                       # Local environment variables (NOT in git)
├── .gitignore                 # venv, __pycache__, .env, docs/
└── docs/
    ├── source_data/           # Original taxonomy MD files, Excel
    │   ├── Risk_Family_Taxonomy_REVISED_v2.1.md
    │   ├── DOMAIN_2_STRUCTURAL_COMPLETE v2.md
    │   ├── DOMAIN_3_DIGITAL_RESILIENCE_COMPLETE.md
    │   ├── DOMAIN_4_OPERATIONAL_MASTER (2).md
    │   ├── Family_1.1 through 1.7 (7 MD files)
    │   └── 20260217-Expanded Process Masterist-NEW-v02.xlsx
    ├── setup_guides/          # Old setup guide Word docs (v1-v4)
    ├── AUDIT_REPORT.md
    └── SETUP_TODO.md
```

### Active API Endpoints (v3.1.0):
| Endpoint | File | Purpose |
|----------|------|---------|
| GET /health | main.py | Health check |
| All /api/v1/clients/* | client_routes.py | Client CRUD, processes, risks, assessments |
| All /api/v2/* | v2_routes.py | V2 taxonomy (domains, families, events) |
| GET /api/v2/engine/compute/{event_id} | prism_engine/api_routes.py | Compute single event probability |
| GET /api/v2/engine/compute-all | prism_engine/api_routes.py | Compute all 174 events (?domain= filter) |
| GET /api/v2/engine/compute-phase1 | prism_engine/api_routes.py | Compute original 10 Phase 1 events only |
| GET /api/v2/engine/status | prism_engine/api_routes.py | Engine health & credentials |
| GET /api/v2/engine/fallback-rates | prism_engine/api_routes.py | All 174 fallback rates |
| GET /api/v2/engine/annual-data | prism_engine/api_routes.py | Get annual update data (DBIR, Dragos, dark figures) |
| PUT /api/v2/engine/annual-data | prism_engine/api_routes.py | Save annual update data from manual entry page |
| GET /api/v2/engine/era5-calibration | prism_engine/api_routes.py | Run ERA5 temperature scaling regression |
| POST /api/v2/engine/load-method-c-research | prism_engine/api_routes.py | Load Method C research output JSON |

**Retired V1 routes** (no longer registered in main.py):
`/api/v1/events`, `/api/v1/probabilities`, `/api/v1/data-sources/health`, `/api/v1/data/refresh`, `/api/v1/calculations/trigger-full`, `/api/v1/stats`

---

## 4. DATABASE SETUP

### Two databases, same code:
- **Local (PC testing):** PostgreSQL at `127.0.0.1:5432`, database `prism_brain_v2`, user `prism_user`, password `prism2026`
- **Railway (production):** Automatically configured via Railway's environment variables

The `.env` file controls which database the local version uses. Railway ignores `.env` and injects its own variables.

### Current .env:
```
DATABASE_URL=postgresql://prism_user:prism2026@127.0.0.1:5432/prism_brain_v2
DEBUG=true
FRED_API_KEY=<configured>
NOAA_API_KEY=your_key_here
NVD_API_KEY=<configured>
EIA_API_KEY=your_key_here
CDS_API_KEY=<configured>
ACLED_EMAIL=<configured>
ACLED_PASSWORD=<configured>
```
**Note:** ACLED credentials configured and OAuth works, but account needs Research-tier access for API data queries (currently returns 403).

### Railway database (for reference):
- Public URL: `postgresql://postgres:GuDIefSxVNllIGCEHlyYIgwGQfYPHRVp@switchyard.proxy.rlwy.net:16186/railway`
- Internal URL: `postgresql://postgres:GuDIefSxVNllIGCEHlyYIgwGQfYPHRVp@postgres.railway.internal:5432/railway`
- NOTE: The Railway PostgreSQL database had 0 risk events. The live app loads data from seed JSON files and SQLite, not PostgreSQL.

### PostgreSQL tables (16 total):
risk_events, risk_probabilities, indicator_weights, indicator_values, data_source_health, calculation_logs, clients, client_processes, client_risks, client_risk_assessments, probability_snapshots, probability_alerts, alert_events, industry_profiles, profile_risk_events, report_schedules

### Frontend also uses:
- SQLite (`frontend/data/prism_brain.db`) — for client-side data
- JSON files (`frontend/data/seeds/*.json`) — for risk event definitions
- JSON (`frontend/data/process_framework.json`) — for 222 business processes

---

## 5. HOW TO START THE APP

### Prerequisites (already installed on Antonio's PC):
- Python 3.11.9 (also has 3.14, but venv uses 3.11)
- PostgreSQL 18
- Git Bash
- Node.js (for document generation scripts)

### Start the backend:
```bash
cd ~/Documents/prism-brain-v2
source venv/Scripts/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Start the frontend (in a second Git Bash window):
```bash
cd ~/Documents/prism-brain-v2
source venv/Scripts/activate
cd frontend
streamlit run Welcome.py
```

### Or use the .bat files:
- Double-click `start_backend.bat`
- Double-click `start_frontend.bat`

### Note on performance:
If running without the backend, the frontend will use local SQLite (fast fallback).
The backend is needed for PostgreSQL data, external data refresh, and Railway deployment.

---

## 6. THE 174 RISK EVENTS

### 4 Domains:
| Code | Domain | Events | Families |
|------|--------|--------|----------|
| PHY | Physical & Environmental | 44 | 7 (Climate, Energy, Resources, Water, Geophysical, Contamination, Biological) |
| STR | Structural & Systemic | 43 | 7 (Financial, Trade, Governance, Infrastructure, Social, Geopolitical, Legal) |
| DIG | Digital & Technological | 47 | 7 (Cyber, Data, AI/Automation, Network, Digitaltic, Software, Emerging Tech) |
| OPS | Operational & Business | 41 | 7 (Supply Chain, Workforce, Compliance, Reputation, Process, Financial Ops, Strategic) |

### Source of truth:
The 4 seed files in `frontend/data/seeds/` contain the complete definitions with:
- event_id, event_name, domain, family_code, family_name
- description, base_rate_pct, confidence_level
- affected_industries, geographic_scope, data_sources
- formulas and validation rules

### Important:
- `risk_database.json` (900 events) is the OLD version — do NOT use it
- `data_summary.json` says 174 events but references the old risk_database.json format
- The taxonomy document (`Risk_Family_Taxonomy_REVISED_v2.1.md`) is in `docs/source_data/`

---

## 7. DEPLOYMENT (RAILWAY)

### Live URL:
`https://prism-brain-production.up.railway.app/`

### How it deploys:
- Push to GitHub `main` branch → Railway auto-deploys
- Uses `Dockerfile` and `railway_start.sh` (these files are in the frontend folder, NOT in the repo root — they were part of the Mac files)
- Railway runs both FastAPI and Streamlit in one container

### Important deployment note:
The Dockerfile and railway deployment scripts (`railway.toml`, `railway_start.sh`, `Dockerfile`, `start_app.sh`) exist in the `frontend/` folder from the Mac copy. They may need updating after the cleanup since `dashboard_routes.py` was removed and `main.py` was restructured.

---

## 8. KNOWN ISSUES / THINGS TO FIX

1. ~~**Risk events not in local PostgreSQL**~~ — **DONE.** `load_events.py` rewritten to read from 4 seed files. Run via `load_data.bat`.

2. ~~**Frontend may have broken imports**~~ — **DONE.** All 17 frontend files pass syntax checks. No broken references.

3. ~~**risk_database.json (900 events)**~~ — **DONE.** Replaced with consolidated 174-event version from seed files.

4. ~~**data_summary.json outdated**~~ — **DONE.** Regenerated with correct domain counts.

5. ~~**Railway deployment**~~ — **DONE.** New Dockerfile, start_app.sh, updated Procfile.

6. ~~**Physical files not all deleted**~~ — **DONE.** Verified all cleanup files removed.

7. ~~**App extremely slow**~~ — **FIXED.** Three root causes found and fixed:
   - **Google Fonts `@import` in CSS** — Made a blocking network request on EVERY page load (500ms-2s per click). Replaced with non-blocking `<link>` tag.
   - **`init_database()` running at module import** — Re-checked/created 5 SQLite tables on every Streamlit rerun. Moved to `@st.cache_resource` (runs only once per session).
   - **`init_external_data_tables()` running at module import** — Created 5 more tables on every import of external_data.py. Changed to lazy initialization (only runs when Data Sources page is visited).

8. **API keys not configured** — The `.env` file has placeholder values for FRED, NOAA, NVD, EIA. These are needed for the data refresh feature but NOT for basic app operation.

7. **Git lock files** — The VM creates `.git/index.lock` and `.git/HEAD.lock` files that can't be auto-cleaned. If git commands fail, the user needs to manually run `rm -f .git/index.lock .git/HEAD.lock`.

---

## 9. WHAT WAS DONE IN THIS SESSION

### Phase 1: Migration Setup
- Explored workspace and GitHub repo
- Compared local Mac files with GitHub code
- Decided: clean start from GitHub + keep Streamlit frontend from Mac
- Created setup guide (Word doc, 4 versions)

### Phase 2: Local Setup
- User installed Python 3.11.9, PostgreSQL 18, Git
- Created venv with `py -3.11 -m venv venv`
- Installed dependencies via `pip install -r requirements.txt`
- Created PostgreSQL database and user
- Got backend running successfully

### Phase 3: Data Investigation
- Found Railway PostgreSQL had 0 risk events
- Found risk_database.json had 900 events (old version)
- User uploaded 4 seed JSON files with correct 174 events
- Created V2 API routes (`v2_routes.py`) for taxonomy-based access
- Created `load_events.py` to load seed data into PostgreSQL

### Phase 4: Performance Fixes
- Reduced API timeout from 30s to 5s
- Reduced health check timeout from 10s to 3s
- Removed 90-second timeout on probabilities endpoint
- Added caching to JSON file loading and API calls

### Phase 5: Bug Fixes
- Fixed domain name: "STRATEGIC" → "STRUCTURAL" in v2_routes.py
- Fixed base_rate_pct conversion for events with rates < 1.0
- Fixed super_risk always showing False in api_client.py

### Phase 6: Major Cleanup
- Created backup commit (3ddd23e)
- Split main.py: 3,700 lines → 160 lines + 3 route modules
- Removed 18 unused V1 API endpoints
- Removed Phase 3 code (dashboard_routes.py + 4 disabled pages)
- Removed 6 dead frontend files
- Fixed security: removed Railway password from sync_from_railway.py
- Organized docs into docs/ folder
- Added .gitignore
- Committed cleanup (de212d0)

### Phase 7: Performance Fixes (Session 2)
- Fixed Google Fonts blocking @import → non-blocking `<link>` tag
- Wrapped `init_database()` in `@st.cache_resource` (runs once per session)
- Changed `init_external_data_tables()` to lazy initialization
- Committed: ce089fe

### Phase 8: Process Criticality Redesign (Session 2)
- Renamed "Criticality" tab concept to "Daily Downtime Revenue Impact"
- Criticality tab now shows 3 columns: Process, Criticality (always "High"), Daily Downtime Revenue Impact (€/day)
- Process ID now shown inline with process name (e.g. "4.4. Electronic Production Equipment")
- Selected processes = High criticality processes (rationale: only high-criticality processes are selected)
- Replaced old simple import/export with PRISM Process Criticality Questionnaire template
  - Template has: Frontpage, Instructions, Company Profile, SCOPE A-D, Summary
  - Each scope sheet has: Process ID, Process Title, Applicable?, Criticality Level (dropdown), Daily Revenue Impact (€), Rationale
  - Upload logic: reads all 4 SCOPE sheets, imports processes with Criticality = "Critical" or "High"
  - Daily Revenue Impact values from the spreadsheet are imported into the Criticality tab
  - Import REPLACES all previous selections (not additive) — the imported file becomes the new truth
  - After import, Process Selection checkboxes are synced to match the imported state
  - Template stored at: `frontend/data/PRISM_Process_Criticality_Template.xlsx`

### Phase 9: Additional Performance Fixes (Session 2)
- Changed localhost → 127.0.0.1 in api_client.py (avoids slow Windows DNS IPv6→IPv4 fallback)
- Added HTTP connection pooling via requests.Session() (reuses TCP connections)
- Increased database cache TTL from 5s to 30s
- Sorted risks in Risk Selection page by domain → family → event ID

### Phase 10: Probability Engine — prism_engine (Session 4)
Built the entire `prism_engine/` module from scratch per Implementation Spec v2.3.

**What was built:**
- **10 data connectors** for external APIs: USGS, CISA KEV, GPR Index, NOAA NAO, World Bank GDP, FRED (4 series), NVD, ACLED, EM-DAT, Copernicus ERA5
- **3 computation methods:**
  - Method A: Frequency count (event-years / total-years over 25yr window)
  - Method B: Incidence rate × dark figure (DBIR decomposition for cyber events, Dragos for ICS/OT)
  - Method C: Structural calibration (preconditions × trigger × implementation)
- **Modifier system:** Ratio method (compare current indicator to historical baseline) + categorical method (binary conditions like election year, active PHEIC)
- **Fallback chain:** Engine-computed value → last cached value → hardcoded base rate from seed files
- **Full derivation trail:** Every computed probability includes formula, data source, observation window, confidence level, and calculation steps
- **Output format:** Section 9.2 JSON schema with layer1, layer2, and metadata
- **API routes:** 4 endpoints at `/api/v2/engine/*` registered in `main.py`
- **Integration tests:** `prism_engine/tests/test_integration.py`

**Phase 1 prototype results (10 events):**
| Event ID | Event Name | P_global | Fallback | Status |
|----------|-----------|---------|----------|--------|
| PHY-CLI-003 | Extreme heat event / heatwave | 6.5% | 6.5% | MATCH |
| PHY-GEO-001 | Major earthquake near industrial zone | 86.45% | 3.5% | DOCUMENTED (Layer 1 vs Layer 2) |
| STR-GEO-001 | Armed conflict in key supplier country | 12% | 12% | MATCH (ACLED key needed) |
| STR-TRD-001 | Sudden import tariff increase | 33.6% | 22% | DOCUMENTED (Method C current evidence) |
| DIG-RDE-001 | Ransomware attack on ERP systems | 5.23% | 12% | DOCUMENTED (DBIR subsplit precision) |
| DIG-CIC-002 | SCADA/ICS protocol exploitation | 2.92% | 5% | DOCUMENTED (Dragos methodology) |
| OPS-MAR-002 | Critical maritime chokepoint closure | 12% | 8% | WITHIN 50% |
| OPS-CMP-001 | Critical semiconductor chip shortage | 16.8% | 20% | WITHIN 50% |
| STR-ECO-001 | Recession in major trading partner | 44% | 15% | DOCUMENTED (G7 scope vs single market) |
| PHY-BIO-001 | Major zoonotic disease outbreak | 36% | 35% | MATCH |

**Key fixes during development:**
- USGS earthquake: Raw counts (247 events) used as probabilities → Fixed with Poisson conversion: P = 1 - e^(-λ)
- Recession prior 68%: All OECD countries (35) → Restricted to G7 only → 44%
- CISA KEV YoY: Partial 2025 vs full 2024 bias → Use two most recent COMPLETE years
- NVD 404 loops: 60+ second delays → Early return when no API key configured
- eval() security risk in formulas.py → Replaced with functools.reduce(mul, ...)

**What's NOT done yet:**
- Legacy V1 routes kept for frontend compatibility
- Phase 2+ (remaining 164 events) not started

### Phase 11: Frontend Wired to Engine (Session 4)
Connected the Streamlit frontend to the prism_engine so dashboard shows engine-computed probabilities.

**Files modified:**
- `frontend/modules/api_client.py` — Added engine API functions:
  - `api_engine_compute_all()` — calls `/api/v2/engine/compute-all`, 60s timeout, cached 5min
  - `api_engine_get_fallback_rates()` — calls `/api/v2/engine/fallback-rates`
  - `get_engine_probability(event_id)` — get P_global for a single event
  - `get_best_probability(event_id, fallback)` — engine probability > fallback > default
- `frontend/pages/3_Risk_Selection.py` — All 4 tabs updated:
  - **Tab 1 (Select Risks):** Column renamed "Base Rate" → "Probability", shows engine P_global for Phase 1 events with indicator dot
  - **Tab 2 (Probabilities):** Now shows engine results with Method (A/B/C), Confidence, Source columns instead of old V2 probabilities
  - **Tab 3 (Save):** Uses `get_best_probability()` — engine values for Phase 1, base rates for others. Shows count of engine-computed risks in success message
  - **Tab 4 (Import/Export):** Export includes engine probability and Source column

**How it works:**
- When backend is running: engine computes real probabilities from external data (USGS, World Bank, etc.)
- When backend is offline: gracefully falls back to base rates (no errors)
- Engine results cached 5 min in memory, connector responses cached 24h on disk
- Results Dashboard automatically shows engine values because it reads from `client_risks.probability` (set during save)

### Phase 12: API Keys, EM-DAT, Connector Fixes (Session 4)
Configured API keys and fixed data connector issues.

**API keys configured in `.env`:** FRED, NVD, CDS (ACLED pending)
**EM-DAT loaded:** 27,505 records. Fixed ISO2→ISO3 code mismatch in emdat.py.
**NVD connector rewritten:** Date-filtered queries returned 404 (API date params unreliable). Rewrote to fetch all ICS CVEs without date filter, count by year client-side.
**PHY-CLI-003 divergence documented:** EM-DAT shows 60% heatwave frequency (15/25 years). Old rate (6.5%) was Layer 2.

### Phase 13: Engine Scaled to 174 Events (Session 5)
Replaced hardcoded 10-event dispatch with config-driven routing so all 174 events compute through the engine.

**Architecture change — config-driven dispatch:**
- Each event now has a `prior_source` field (e.g., "emdat", "usgs", "fred_threshold", "dbir", "family_defaults")
- `engine.py` routes to the correct connector/method based on this field — no more `if event_id == "..."` chains
- Engine version bumped from 1.0.0 to 2.0.0

**Files modified (8 files):**

1. **`prism_engine/config/event_mapping.py`** — Core rewrite:
   - Added `_load_all_events()`: auto-loads all 174 events from seed JSON files at import time
   - Added `EMDAT_EVENTS` set (25 event IDs that use EM-DAT data)
   - Added `DBIR_EVENTS` set (25 event IDs that use DBIR decomposition)
   - Added `FRED_THRESHOLD_EVENTS` dict (5 events: STR-ECO-002, STR-ECO-005, STR-ECO-006, PHY-ENE-003, PHY-MAT-003)
   - Added `METHOD_C_FAMILY_DEFAULTS` dict (23 family prefixes with calibrated sub-probabilities)
   - Added `DEFAULT_MODIFIER_SOURCES` dict (23 family prefixes with modifier routing)
   - Method auto-assignment: EM-DAT → Method A, DBIR → Method B, FRED threshold → Method A, rest → Method C

2. **`prism_engine/connectors/emdat.py`** — Expanded from 11 to 25 EM-DAT mappings:
   - New: PHY-GEO-005 (Sinkhole), PHY-WAT-001 (Drought), PHY-WAT-006 (Flood), PHY-POL-002 (Air pollution), PHY-POL-004 (Oil spill), PHY-BIO-001/002/004, PHY-ENE-001 (Storm), OPS-MAR-006, OPS-SUP-002, OPS-MFG-002/005, OPS-WHS-001

3. **`prism_engine/computation/priors.py`** — Expanded from 12 to 27 DBIR event mappings:
   - New: DIG-FSD-004 (deepfake), DIG-FSD-006 (app-layer DDoS), DIG-FSD-007 (DNS/BGP), DIG-SCC-001-005 (supply chain), DIG-CIC-001/003-006 (critical infrastructure OT)

4. **`prism_engine/connectors/fred.py`** — Added `count_threshold_years()` for Method A priors:
   - Generic function: groups FRED observations by year, compares annual average to threshold
   - 4 comparison modes: "above", "below", "yoy_above", "yoy_below"
   - Added 3 new FRED series: DCOILWTICO (oil), PPIACO (PPI), CSUSHPISA (house prices)

5. **`prism_engine/engine.py`** — Complete dispatch rewrite:
   - `_get_method_a_prior()`: Routes by `prior_source` field → emdat, usgs, acled, world_bank, fred_threshold, manual_events
   - `_get_method_b_prior()`: Routes by DBIR_EVENT_MAPPING membership or dragos special path
   - `_get_method_c_prior()`: Family-level calibration from METHOD_C_FAMILY_DEFAULTS, then generic 0.50 fallback
   - `compute_all()`: Iterates all 174 events. `compute_all_phase1()` kept as backward-compat alias

6. **`prism_engine/api_routes.py`** — Updated for 174 events:
   - `/compute-all` calls `compute_all()`, optional `?domain=` filter, method distribution in response
   - Added `/compute-phase1` endpoint for backward compatibility

7. **`frontend/modules/api_client.py`** — Timeout increased 60s → 120s for 174 events

8. **`frontend/pages/3_Risk_Selection.py`** — Removed "Phase 1 covers 10 prototype events" caption

**Test results (all 174 events):**
| Method | Events | P_global Range | Mean |
|--------|--------|---------------|------|
| A (frequency) | 33 | 0.001 – 0.95 | 0.3774 |
| B (incidence) | 26 | 0.0014 – 0.1584 | 0.0331 |
| C (calibrated) | 115 | 0.0612 – 0.3360 | 0.1457 |
| **Total** | **174** | — | **0 fallbacks** |

**Bugs fixed during Phase 13:**
- ~~FRED NAPMNOI returns HTTP 400~~ → **Fixed in Phase 14:** Replaced with AMTMNO, PHY-ENE modifiers re-enabled
- "Unknown modifier source 'categorical' for PHY-BIO-*" → Changed PHY-BIO default modifiers to `[]`
- FastAPI not installed → Ran `pip install -r requirements.txt`

### Phase 14: Connector Fixes — Real Data for Engine (Session 6)
Fixed 3 practical gaps preventing the engine from using real external data.

**Fix 1: FRED NAPMNOI → AMTMNO (FIXED)**
- Problem: ISM removed all data from FRED in 2016. NAPMNOI series doesn't exist → HTTP 400.
- Solution: Replaced with AMTMNO (Manufacturers' New Orders: Total Manufacturing) which has data through Nov 2025.
- New modifier: ratio-to-rolling-60-month-mean (instead of dividing by 50). Tested: modifier = 1.08.
- Files: `fred.py` (new `get_manufacturing_orders_modifier()`), `engine.py`, `event_mapping.py`, `sources.py`

**Fix 2: Copernicus ERA5 (FIXED — packages installed, config updated)**
- Problem: cdsapi package not installed, API key not being passed to client.
- Solution: Installed cdsapi 0.7.7, xarray 2026.2.0, netcdf4 1.7.4. Updated `copernicus.py` to pass API key from env var to `cdsapi.Client()` constructor.
- Note: ERA5 data requests take 30+ minutes. First download not yet triggered. Falls back to modifier=1.0 until cache populated.
- Files: `copernicus.py`

**Fix 3: ACLED → UCDP Replacement (FIXED)**
- Problem: ACLED switched from API keys to OAuth in 2025. Even with OAuth working, data queries return HTTP 403 because the account needs paid Research-tier access.
- Solution: Replaced ACLED entirely with **UCDP** (Uppsala Conflict Data Program) — the academic gold standard for armed conflict data, used by UN and World Bank. Completely free, no API key required.
- Improvements over ACLED:
  - 25-year observation window (2000-2024) vs ACLED's 7 years (2018-2024) — much more statistically robust
  - Pre-aggregated country-year records (no need to count individual events)
  - Filters for war-level conflicts (1000+ battle deaths/yr) to capture supply-chain-disrupting events
  - Handles interstate conflicts with comma-separated country codes (e.g. "365, 369" for Russia-Ukraine)
- Result: STR-GEO-001 prior = 48% (12 war-years in 25). War years: 2000-2005, 2014-2016, 2022-2024.
- Files: `ucdp.py` (new connector), `engine.py`, `event_mapping.py`, `sources.py`
- Old `acled.py` kept but no longer used by the engine

### Phase 15: Type B Manual Entry Page + ERA5 Regression (Session 7)
Built the annual data update infrastructure and calibrated the ERA5 temperature scaling.

**Step A: Type B Manual Entry Page (COMPLETED)**
Created a Streamlit page for updating DBIR breach rates, Dragos ICS statistics, and dark figure multipliers when new annual reports are published.

Architecture:
- `prism_engine/annual_data.py` — Persistence layer. Loads/saves annual data overrides to `prism_engine/data/annual_updates.json`. Falls back to hardcoded defaults if no override file exists.
- `prism_engine/computation/priors.py` — Modified `method_b_prior()` to call `get_dbir_base_rate()`, `get_dbir_attack_shares()`, and `get_subsplit_override()` from the annual data module instead of using hardcoded constants directly.
- `prism_engine/api_routes.py` — Added GET/PUT endpoints at `/api/v2/engine/annual-data`.
- `frontend/modules/api_client.py` — Added `api_engine_get_annual_data()` and `api_engine_save_annual_data()`.
- `frontend/pages/6_Annual_Data_Update.py` — 4-tab page:
  - Tab 1: DBIR Breach Rates (base rate + 6 attack type shares, with live probability preview)
  - Tab 2: Dragos ICS/OT (incidents, manufacturing %, total orgs, dark figure, with live DIG-CIC-002 preview)
  - Tab 3: Dark Figure Multipliers (6 underreporting correction factors)
  - Tab 4: Event Subsplits (Advanced) — override individual event subsplit factors
- `frontend/.streamlit/pages.toml` — Added "Annual Data Update" navigation entry

**Step D: ERA5 Temperature Scaling Regression (COMPLETED)**
Replaced the hardcoded 0.15 scaling constant with a regression-derived 0.21 coefficient.

Methodology:
- X = European summer (JJA) temperature anomaly in standard deviations (from C3S/ERA5 published data, 2000-2024)
- Y = Binary heatwave indicator (1 if EM-DAT reports any heatwave in EEA-Extended that year)
- Method: Logistic regression via Newton-Raphson IRLS, then marginal effect → relative risk conversion
- Result: Optimal coefficient = **0.21** (up from 0.15 initial estimate)
  - At 1.5σ: modifier = 1.32 (logistic predicts 1.34 — excellent fit)
  - At 2.0σ: modifier = 1.42 (logistic predicts 1.48 — good fit)
  - At 2.5σ: modifier = 1.53 (logistic predicts 1.57 — good fit)
  - Pseudo-R² = 0.16, accuracy = 64% (16/25)
  - Key finding: ALL years with anomaly > 1.5σ had EM-DAT heatwave events (6/6 = 100%)

Files:
- `prism_engine/computation/era5_calibration.py` — New module with published anomaly table, logistic regression, scaling derivation
- `prism_engine/connectors/copernicus.py` — Updated: 0.15 → 0.21, status "INITIAL_ESTIMATE_NEEDS_REGRESSION" → "REGRESSION_DERIVED"
- `prism_engine/api_routes.py` — Added GET `/api/v2/engine/era5-calibration` endpoint

**Step E: ENTSO-E Connector (DEFERRED)**
Marked as optional in the original plan. Can be added later if needed for PHY-ENE events.

**Method C Research (DELEGATED)**
A parallel Claude session is researching evidence-based sub-probabilities for 115 Method C events using the PRD at `docs/PRD_Method_C_Research.md`. When the research output (`method_c_research_output.json`) is received, it can be loaded via the POST `/api/v2/engine/load-method-c-research` endpoint or placed at `method_c_research_output.json` in the project root.

### Phase 16: Method C Loader + V1 Route Cleanup (Session 8)
Finalized the engine integration and removed legacy V1 routes.

**Method C Research Output Loader (COMPLETED)**
Built the integration pipeline for consuming Method C research output from the parallel Claude session.

Files:
- `prism_engine/method_c_loader.py` — New module:
  - `load_research_output(path)` — Validates JSON schema: requires event_id, p_pre, p_trig, p_impl (each 0.01-0.99), plus evidence dict
  - `integrate_research(data)` — Writes event-level overrides to `prism_engine/data/method_c_overrides.json`
  - `get_method_c_override(event_id)` — Returns calibrated sub-probabilities for a specific event, or None
- `prism_engine/engine.py` — Updated `_get_method_c_prior()` priority chain:
  1. Phase 1 hand-crafted configs (highest priority)
  2. Event-specific research overrides (from method_c_overrides.json)
  3. Family-level calibrated defaults (from METHOD_C_FAMILY_DEFAULTS)
  4. Generic 0.50 defaults (lowest priority)
- `prism_engine/api_routes.py` — Added POST `/api/v2/engine/load-method-c-research`

**Legacy V1 Route Cleanup (COMPLETED)**
Retired all V1 data/calculation routes. The probability engine now handles everything.

Changes:
- `main.py` — Removed imports and registration of `routes/events.py`, `routes/calculations.py`, `routes/data_sources.py`. Version bumped to 3.1.0. Health endpoint features updated.
- `frontend/modules/api_client.py` — Removed 6 unused V1 functions: `fetch_events`, `fetch_probabilities`, `fetch_data_sources`, `trigger_data_refresh`, `trigger_recalculation`, `get_event_probability`. Added `api_engine_status()`.
- `frontend/pages/7_Data_Sources.py` — Complete rewrite. Replaced V1 refresh/recalculate buttons with V2 engine compute-all. Shows engine status, API credentials, method distribution. Fixed broken navigation link (6_Results → 5_Results).
- Route files in `routes/` kept on disk for reference but no longer registered in the FastAPI app.

---

## 10. WINDOWS/VM LIMITATIONS

When working through Claude Cowork on this PC:
- **Cannot delete files** — Windows mount permissions prevent `rm` from the VM. Use `.bat` scripts or ask user to delete manually.
- **Git lock files** — Every git operation creates lock files that can't be auto-removed. User must run `rm -f .git/index.lock .git/HEAD.lock` before git commands.
- **Git add -A times out** — The venv folder has thousands of files. Always add specific files by name, never `git add -A`.
- **No pip packages in VM** — Can't test Python imports directly. Use `python -m py_compile` for syntax checking.
- **User can't always see Claude's output** — There was a UI bug where responses after tool calls were invisible. Always provide clear text summaries the user can read.
