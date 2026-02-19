# PRISM Brain V2 - Cleanup Log

## Date: February 19, 2026

## How to Roll Back
If anything breaks after this cleanup, run:
```bash
git log --oneline -5
# Find the commit that says "Backup before major cleanup"
git checkout <that-commit-hash> -- .
```
This restores ALL files to exactly how they were before the cleanup.

The backup commit hash is: `3ddd23e`

So the full rollback command is:
```bash
git checkout 3ddd23e -- .
```

## What Was Changed

### 1. main.py Split (3,700 lines -> 160 lines)
- **Before:** One giant file with everything
- **After:** Slim entry point that delegates to route modules
- **New files created:**
  - `routes/__init__.py`
  - `routes/events.py` (60 lines) - GET /api/v1/events
  - `routes/calculations.py` (1,254 lines) - Probability engine + trigger-full
  - `routes/data_sources.py` (1,602 lines) - Data fetching + health + stats

### 2. Removed V1 Endpoints (18 endpoints removed)
These were not called by the frontend:
- Single event detail, bulk events, bulk weights, bulk values
- Indicator weights/values listing
- Probability history, attribution, explanation, dependencies
- Dashboard summary, calculations listing, legacy trigger
- HTML dashboard page, redundant data sources listing

### 3. Removed Phase 3 Code
- `dashboard_routes.py` - Trends, Alerts, Profiles, Reports backend
- `frontend/pages_disabled/` - 4 disabled Phase 3 pages
- Phase 3 functions in `frontend/modules/api_client.py` (~120 lines)

### 4. Removed Dead Frontend Code
- `frontend/api_client.py` - Old duplicate (replaced by modules/api_client.py)
- `frontend/modules/demo_data.py` - Never imported
- `frontend/modules/smart_prioritization.py` - Never imported
- `frontend/data/risk_events_v2.json` - Never referenced by code
- `frontend/BUGFIX_NOTES.md` - Old notes
- `frontend/README.md` - Old readme

### 5. Security Fix
- `sync_from_railway.py` - Removed hardcoded Railway database password
  (now reads from RAILWAY_DATABASE_URL environment variable)

### 6. Documentation Organized
- Moved all .md/.docx/.xlsx reference files to `docs/source_data/` and `docs/setup_guides/`
- Added `docs/` to .gitignore so reference files don't clutter the repo

### 7. .gitignore Added
- `venv/`, `__pycache__/`, `*.pyc`, `.env`, `*.lock`
- `docs/source_data/`, `docs/setup_guides/`

## Endpoints Still Active
| Endpoint | Source File |
|----------|------------|
| GET /health | main.py |
| GET /api/v1/events | routes/events.py |
| GET /api/v1/probabilities | routes/calculations.py |
| POST /api/v1/calculations/trigger-full | routes/calculations.py |
| GET /api/v1/data-sources/health | routes/data_sources.py |
| GET /api/v1/stats | routes/data_sources.py |
| POST /api/v1/data/refresh | routes/data_sources.py |
| GET /api/v1/data/sources | routes/data_sources.py |
| All /api/v1/clients/* | client_routes.py |
| All /api/v2/* | v2_routes.py |
