# PRISM Brain - Risk Intelligence System

A web-based application for comprehensive enterprise risk assessment.

## Features

- **174 Curated Risk Events** across 4 domains and 28 families (Physical, Structural, Digital, Operational)
- **222 Business Processes** across 4 scopes and 32 macro-processes
- **Client Management** - Create and manage multiple client profiles
- **Smart Risk Selection** - Risks auto-scored by relevance to client
- **Guided Assessment** - Step-by-step vulnerability/resilience input
- **Interactive Dashboard** - Visualizations and exports

## Quick Start

### 1. Install Dependencies

```bash
cd prism_brain_app
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. For Web Deployment

To deploy to Streamlit Cloud or other hosting:

```bash
# Streamlit Cloud
# 1. Push this folder to GitHub
# 2. Connect to share.streamlit.io
# 3. Deploy from your repository
```

## Application Workflow

### Step 1: Client Setup 🏢
- Create a new client with company information
- Select relevant business processes from the APQC framework
- Set criticality values (€/day revenue impact)

### Step 2: Risk Selection ⚡
- Browse 174 risk events across 4 domains and 28 families
- Each event has a research-backed base probability and confidence level
- Select risks to include in assessment

### Step 3: Risk Assessment 🎯
- For each process-risk combination, input:
  - **Vulnerability** (0-100%): How likely is this process affected?
  - **Resilience** (0-100%): How quickly can you recover?
  - **Expected Downtime** (days): Duration until normal operations

### Step 4: Results Dashboard 💰
- View total annual risk exposure
- Analyze by domain, process, or risk event
- Export to Excel or CSV

## PRISM Formula

```
Risk Exposure (€/year) = Criticality × Vulnerability × (1-Resilience) × Downtime × Probability
```

Where:
- **Criticality**: Revenue impact per day of disruption (€/day)
- **Vulnerability**: Likelihood process is affected (0-1)
- **Resilience**: Recovery capability (0-1)
- **Downtime**: Expected disruption duration (days)
- **Probability**: Annual probability of risk event (0-1)

## Project Structure

```
prism_brain_app/
├── app.py                          # Main entry point
├── pages/
│   ├── 1_🏢_Client_Setup.py        # Client & process management
│   ├── 2_⚡_Risk_Selection.py       # Risk selection & prioritization
│   ├── 3_🎯_Risk_Assessment.py     # Vulnerability input
│   └── 4_💰_Results_Dashboard.py   # Visualizations & export
├── data/
│   ├── risk_database.json          # Legacy V1 risk events (V2 uses backend database)
│   ├── process_framework.json      # APQC processes
│   └── prism_brain.db              # SQLite database (auto-created)
├── modules/
│   └── database.py                 # Database operations
├── utils/
│   ├── constants.py                # Configuration
│   └── helpers.py                  # Utility functions
└── requirements.txt
```

## Configuration

Edit `utils/constants.py` to customize:
- Default currency
- Risk level thresholds
- Probability factor weights
- Industry templates

## Data Persistence

All data is stored in `data/prism_brain.db` (SQLite database).
To backup, simply copy this file.

## Adding New Features

The application is designed for easy modification:

1. **New page**: Add a file to `pages/` with naming pattern `N_emoji_Name.py`
2. **New database tables**: Add to `modules/database.py`
3. **New risk data**: Update `data/risk_database.json`
4. **New processes**: Update `data/process_framework.json`

## Support

For questions or issues, please contact the development team.

---

**Version:** 2.0.0
**Last Updated:** February 2026
