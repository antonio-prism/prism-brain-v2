# PRISM Brain v2 — Setup TODO (Windows PC)

Everything below assumes you've already done the initial Git clone and PostgreSQL setup from the previous session.

---

## What I've Done (Autonomous Work)

1. **Fixed v2_routes.py** — Changed "STRATEGIC" to "STRUCTURAL" in the domain map, and updated all family name fallbacks to use the full names from your seed data (e.g., "Climate Extremes & Weather Events" instead of just "Climate").

2. **Fixed load_events.py** — The base rate conversion had a bug where events with base_rate_pct ≤ 1.0 (like 0.12%) would incorrectly get a 50% probability. Now all 174 events convert correctly.

3. **Fixed api_client.py** — The super_risk field was hardcoded to False. Now it reads the actual value from the API.

4. **Verified process_framework.json** — Confirmed it has the correct 222 entries (32 macro + 190 sub-processes), rebuilt from your Excel.

5. **Verified risk_events_v2.json** — 174 events, domains match (PHYSICAL: 44, STRUCTURAL: 42, DIGITAL: 47, OPERATIONAL: 41), all event IDs consistent with risk_database.json.

6. **Created startup scripts** — Three `.bat` files for easy launching on Windows.

---

## Steps You Need to Do

### Step 1: Activate Your Virtual Environment

Open a terminal in your `prism-brain-v2` folder.

```
venv\Scripts\activate
```

If you haven't created one yet:

```
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Make Sure PostgreSQL Is Running

Open pgAdmin or check that the PostgreSQL service is running in Windows Services.

### Step 3: Create Database Tables

Start the backend once — it auto-creates all tables:

```
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Wait until you see "Application startup complete", then press Ctrl+C to stop it.

### Step 4: Load the Data

```
python load_events.py
```

You should see: "Inserted: 174 new events"

Then:

```
python regenerate_weights.py
```

### Step 5: Start the App

**Option A — Use the batch files (easiest):**
- Double-click `start_backend.bat`
- Double-click `start_frontend.bat` (in a second window)

**Option B — Manual:**

Terminal 1 (backend):
```
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Terminal 2 (frontend):
```
cd frontend
streamlit run Welcome.py
```

### Step 6: Verify It Works

- Backend API: http://localhost:8000/api/v2/health → should show 174 events
- Frontend dashboard: http://localhost:8501 → should open the PRISM dashboard

---

## Optional Later Steps

- **Add API keys** in `.env` for real data sources (FRED, NOAA, NVD, EIA)
- **Set Super Risk flags** — The seed data doesn't include super_risk markers. You can set these via the dashboard or directly in the database
- **Deploy to Railway** — Push changes to GitHub and Railway will auto-deploy. The Railway instance uses its own DATABASE_URL so your local changes won't affect production data

---

## File Summary

| File | What It Does |
|------|-------------|
| `load_events.py` | Loads 174 risk events from JSON into PostgreSQL |
| `v2_routes.py` | NEW: V2 API endpoints the frontend needs |
| `start_backend.bat` | One-click backend launcher |
| `start_frontend.bat` | One-click frontend launcher |
| `load_data.bat` | One-click data loader (runs Steps 5) |
| `frontend/data/risk_events_v2.json` | 174 events in seed format |
| `frontend/data/risk_database.json` | Same events in legacy format for helpers.py |
| `frontend/data/process_framework.json` | 222 business processes |
| `frontend/data/seeds/` | Your original 4 seed JSON files (preserved) |
