# PRISM Brain v2 — Full Code Audit Report

**Date:** February 19, 2026
**Audited by:** Claude (Cowork)

---

## Executive Summary

The app works, but it has accumulated a lot of baggage from iterative development. The main issues are: duplicate API endpoints (V1 still exists but nothing uses it), three different data sources doing the same job (PostgreSQL + SQLite + JSON files), disabled features with fully-coded backends, dead code, and 15+ documentation/reference files cluttering the project root.

Below is a prioritized list of what can be cleaned up, organized from "safe and easy" to "bigger refactor."

---

## 1. Files You Can Safely Delete (No Impact on Running App)

These files are NOT used by the application. Removing them won't break anything.

### Documentation clutter in project root (740 KB total)

These are reference documents and old setup guides that were used during development but aren't part of the app:

| File | Size | Why it's safe to delete |
|------|------|------------------------|
| `PRISM_V2_PC_Setup_Guide.docx` | 17 KB | Superseded by v4 |
| `PRISM_V2_PC_Setup_Guide_v2.docx` | 18 KB | Superseded by v4 |
| `PRISM_V2_PC_Setup_Guide_v3.docx` | 19 KB | Superseded by v4 |
| `PRISM_V2_PC_Setup_Guide_v4.docx` | 19 KB | Superseded by SETUP_TODO.md |
| `DOMAIN_2_STRUCTURAL_COMPLETE.md` | — | Source doc, data already in seed JSONs |
| `DOMAIN_3_DIGITAL_RESILIENCE_COMPLETE.md` | 107 KB | Source doc, data already in seed JSONs |
| `DOMAIN_4_OPERATIONAL...v2.md` | 162 KB | Source doc, data already in seed JSONs |
| `Family_1.1_Climate_Extremes_COMPLETE.md` | 29 KB | Source doc, data already in seed JSONs |
| `Family_1.2_Energy_Supply_COMPLETE.md` | 43 KB | Source doc, data already in seed JSONs |
| `Family_1.3_Natural_Resources_COMPLETE.md` | 64 KB | Source doc, data already in seed JSONs |
| `Family_1.4_Water_Resources_COMPLETE.md` | 18 KB | Source doc, data already in seed JSONs |
| `Family_1.5_Geophysical_COMPLETE.md` | 15 KB | Source doc, data already in seed JSONs |
| `Family_1.6_Contamination_Pollution_COMPLETE.md` | 20 KB | Source doc, data already in seed JSONs |
| `Family_1.7_Biological_Pandemic_COMPLETE.md` | 16 KB | Source doc, data already in seed JSONs |
| `Risk_Family_Taxonomy_REVISED_v2.1.md` | 32 KB | Source of truth doc, but data is in seed JSONs |
| `20260217-Expanded Process Masterist-NEW-v02.xlsx` | 18 KB | Source spreadsheet, data is in process_framework.json |

**Recommendation:** Move these to a `docs/` folder if you want to keep them for reference, or just delete them.

### Dead code file

| File | Size | Why it's safe to delete |
|------|------|------------------------|
| `frontend/modules/demo_data.py` | 191 lines | Contains `seed_demo_clients()` but it's never imported or called by any page |

### One-time utility scripts (optional)

These were only needed during the migration. You can keep them if you want, but they're not part of the app:

| File | Purpose | Status |
|------|---------|--------|
| `sync_from_railway.py` | Sync production DB to local | **Contains hardcoded Railway password!** Move to env var or delete |
| `parse_events.py` | Parse markdown files into events | Superseded by seed JSON files |

### Seed JSON files (optional)

The 4 seed files in `frontend/data/seeds/` (368 KB total) were used once to create `risk_events_v2.json`. They could be archived since the merged file is the one actually used.

---

## 2. Security Issue

**File:** `sync_from_railway.py` line 16 contains your production Railway database password in plain text:

```
RAILWAY_URL = "postgresql://postgres:GuDIef...@switchyard.proxy.rlwy.net:16186/railway"
```

**Fix:** Either delete the file (it's a one-time utility) or move the URL to `.env` and read from there. If this file has been pushed to GitHub, you should rotate the Railway database password.

---

## 3. V1 API Endpoints — Can Be Removed

The backend has **22 V1 endpoints** in `main.py` that the frontend no longer uses. The frontend exclusively uses the **8 V2 endpoints** in `v2_routes.py`.

**V1 endpoints still in main.py (not used by frontend):**

- `/api/v1/events` (GET, POST)
- `/api/v1/events/{event_id}` (GET, PUT)
- `/api/v1/indicators/weights` (GET, POST, PUT)
- `/api/v1/indicators/values` (GET, POST)
- `/api/v1/probabilities` (GET)
- `/api/v1/probabilities/{event_id}/history` (GET)
- `/api/v1/probabilities/{event_id}/attribution` (GET)
- `/api/v1/probabilities/{event_id}/explanation` (GET)
- `/api/v1/dependencies/{event_id}` (GET)
- `/api/v1/dashboard/summary` (GET)
- `/api/v1/calculations/trigger` (POST)
- `/api/v1/data-sources/health` (GET)
- `/api/v1/data-sources/refresh` (POST)

**Removing these would cut main.py from ~3,700 lines to ~2,500 lines**, making it much easier to maintain.

**However:** If other tools or external systems call V1 endpoints, keep them. The Railway production app might depend on V1 if the frontend there hasn't been updated to V2 yet.

---

## 4. Disabled Phase 3 Features — Decide Their Fate

Four frontend pages are disabled (in `pages_disabled/`), but their backend code is fully implemented:

| Disabled Page | Backend File | Lines of Backend Code |
|---|---|---|
| `8_Probability_Trends.py` | `dashboard_routes.py` | ~250 lines |
| `9_Alert_Management.py` | `dashboard_routes.py` | ~250 lines |
| `10_Industry_Profiles.py` | `dashboard_routes.py` | ~250 lines |
| `11_Automated_Reports.py` | `dashboard_routes.py` | ~295 lines |

Plus 7 database tables that only these features use (probability_snapshots, probability_alerts, alert_events, industry_profiles, profile_risk_events, report_schedules, report_history).

**Options:**
- **Re-enable them** — Move pages back from `pages_disabled/` to `pages/`
- **Remove them entirely** — Delete the 4 pages, `dashboard_routes.py` (1,045 lines), and the 7 table definitions in models.py
- **Keep as-is** — Leave them disabled for future use (current state)

---

## 5. Three Data Sources — The Biggest Architecture Issue

The frontend loads the same data from three different places:

| Source | What | Where Used |
|---|---|---|
| **PostgreSQL** (via API) | Risk events, probabilities | Pages 3, 6, 7 via `api_client.py` |
| **SQLite** (local file) | Clients, processes, risks | Pages 1-5 via `database.py` |
| **JSON files** | Risk events, processes | `smart_prioritization.py`, `helpers.py` |

This means:
- Risk event data exists in PostgreSQL AND `risk_events_v2.json` AND `risk_database.json`
- Process data exists in the API AND `process_framework.json`
- If you update data in one place, the others don't automatically update

**Ideal cleanup (bigger refactor):** Make PostgreSQL the single source of truth. Have the frontend read everything from the API. Remove the JSON file dependency and SQLite fallback, or at least make them auto-sync.

**Quick win:** At minimum, `smart_prioritization.py` could read from the API instead of JSON files.

---

## 6. main.py Size — 3,700 Lines

`main.py` is doing too many things in one file. It contains:

- FastAPI app setup and middleware (~100 lines)
- 10 Pydantic models (~200 lines)
- Probability calculation engine (~400 lines)
- V1 API endpoint handlers (~800 lines)
- Client route registration (~50 lines)
- Dashboard route registration (~50 lines)
- 25 data source fetch functions (~1,500 lines)
- DataSourceCollector orchestrator (~200 lines)
- Startup/shutdown events (~100 lines)
- Error handling (~50 lines)

**Possible refactor:** Split into separate files:
- `main.py` — Just app setup, startup, middleware (~200 lines)
- `v1_routes.py` — V1 endpoints (or delete them)
- `v2_routes.py` — Already extracted
- `probability_engine.py` — Calculation logic
- `data_sources/` — One file per data source or grouped by domain

This would make each file manageable and easier to understand.

---

## 7. Summary of Recommended Actions

### Quick wins (do now, no risk):
1. Delete the 4 old setup guide .docx files
2. Move domain/family .md files and the Excel to a `docs/` folder
3. Delete `frontend/modules/demo_data.py`
4. Delete or secure `sync_from_railway.py` (has hardcoded credentials)
5. Delete `parse_events.py` (superseded by seed files)

### Medium effort (do when convenient):
6. Remove V1 API endpoints from main.py (if Railway production uses V2)
7. Decide on Phase 3 features: re-enable or remove
8. Archive seed JSON files to `docs/seeds/`

### Bigger refactor (plan for later):
9. Split main.py into smaller modules
10. Unify data sources (PostgreSQL as single source of truth)
11. Remove SQLite fallback or make it auto-sync from API
