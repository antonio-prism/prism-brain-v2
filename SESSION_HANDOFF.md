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

## 2. CURRENT STATE (as of Feb 19, 2026)

### What works:
- Backend starts successfully: `python -m uvicorn main:app --host 0.0.0.0 --port 8000`
- Database initialized (PostgreSQL local)
- All syntax checks pass
- Major cleanup completed (20,954 lines removed)

### What still needs work:
- **Risk events not loaded into local PostgreSQL.** The 174 events exist in 4 seed JSON files (`frontend/data/seeds/`) but haven't been loaded into the database yet. The script `load_events.py` was created for this but hasn't been run successfully.
- **Frontend not fully tested** after the cleanup. Needs verification that all 7 pages work.
- **API keys not configured.** The `.env` file has placeholder values (`your_key_here`) for FRED, NOAA, NVD, EIA. These are needed for the data refresh feature but NOT for basic app operation.
- **Changes not pushed to GitHub** yet. Two local commits exist that haven't been pushed.

### Git status:
- Latest commit: `de212d0` — "Major cleanup: split main.py, remove Phase 3 and dead code"
- Backup commit: `3ddd23e` — "Backup before major cleanup" (rollback point)
- Remote: `https://github.com/antonio-prism/prism-brain-v2` (main branch)
- **Not pushed yet** — user should push when ready

### Rollback if needed:
```bash
git checkout 3ddd23e -- .
```

---

## 3. PROJECT ARCHITECTURE

### File Structure:
```
prism-brain-v2/
├── main.py                    # 160 lines — App entry point, health check, startup
├── routes/
│   ├── __init__.py
│   ├── events.py              # 60 lines — GET /api/v1/events
│   ├── calculations.py        # 1,254 lines — Probability engine + trigger-full
│   └── data_sources.py        # 1,602 lines — DataFetcher (28 sources) + health/stats
├── client_routes.py           # 509 lines — Client CRUD, processes, risks, assessments
├── v2_routes.py               # 494 lines — V2 taxonomy API (domains/families)
├── config/
│   ├── settings.py            # App settings from environment variables
│   └── category_indicators.py # Maps event categories to indicators
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
│   │   ├── 6_V2_Event_Explorer.py
│   │   └── 7_Data_Sources.py
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

### Active API Endpoints:
| Endpoint | File | Purpose |
|----------|------|---------|
| GET /health | main.py | Health check |
| GET /api/v1/events | routes/events.py | List all risk events |
| GET /api/v1/probabilities | routes/calculations.py | List latest probabilities |
| POST /api/v1/calculations/trigger-full | routes/calculations.py | Run full probability calculation |
| GET /api/v1/data-sources/health | routes/data_sources.py | Data source status |
| GET /api/v1/stats | routes/data_sources.py | System statistics |
| POST /api/v1/data/refresh | routes/data_sources.py | Refresh data from 28 sources |
| GET /api/v1/data/sources | routes/data_sources.py | List configured sources |
| All /api/v1/clients/* | client_routes.py | Client CRUD, processes, risks, assessments |
| All /api/v2/* | v2_routes.py | V2 taxonomy (domains, families, events) |

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
FRED_API_KEY=your_key_here
NOAA_API_KEY=your_key_here
NVD_API_KEY=your_key_here
EIA_API_KEY=your_key_here
```

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

1. **Risk events not in local PostgreSQL** — Need to run `load_events.py` successfully. The script reads from the seed JSON files and inserts into PostgreSQL. This hasn't been verified yet.

2. **Frontend may have broken imports** — After removing `demo_data.py` and `smart_prioritization.py`, any page that imported them will break. Need to check all 7 pages.

3. **risk_database.json (900 events) still exists** — This old file is still in `frontend/data/`. The frontend's `helpers.py` reads from it. This could cause confusion since it has 900 events, not 174. May need to update `helpers.py` to read from the seed files instead, or replace `risk_database.json` with the 174-event version.

4. **data_summary.json is outdated** — Shows 174 events but its internal structure references old data format.

5. **Railway deployment** — The cleanup changed the backend structure significantly. Railway deployment may need updating (Dockerfile, start scripts).

6. **Physical files not all deleted** — The `cleanup_files.bat` was created to handle leftover files that couldn't be deleted from the VM. Antonio ran it but we didn't verify all files were removed.

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

---

## 10. WINDOWS/VM LIMITATIONS

When working through Claude Cowork on this PC:
- **Cannot delete files** — Windows mount permissions prevent `rm` from the VM. Use `.bat` scripts or ask user to delete manually.
- **Git lock files** — Every git operation creates lock files that can't be auto-removed. User must run `rm -f .git/index.lock .git/HEAD.lock` before git commands.
- **Git add -A times out** — The venv folder has thousands of files. Always add specific files by name, never `git add -A`.
- **No pip packages in VM** — Can't test Python imports directly. Use `python -m py_compile` for syntax checking.
- **User can't always see Claude's output** — There was a UI bug where responses after tool calls were invisible. Always provide clear text summaries the user can read.
