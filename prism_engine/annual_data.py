"""
PRISM Engine — Annual data update persistence.

Manages the JSON file that stores manually-entered annual report data
(DBIR breach rates, Dragos ICS stats, dark figures, subsplit overrides).
The Streamlit UI writes to this file via API; priors.py reads from it.

If no overrides file exists, all functions return the hardcoded defaults
from priors.py — zero-risk migration.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
ANNUAL_FILE = DATA_DIR / "annual_updates.json"

# ── Hardcoded defaults (mirrors priors.py at time of writing) ────────────
DEFAULTS = {
    "dbir_year": 2025,
    "dbir_base_breach_rate": 0.18,
    "dbir_attack_shares": {
        "ransomware": 0.44,
        "social_engineering": 0.25,
        "credential_theft": 0.38,
        "third_party": 0.30,
        "web_app_exploit": 0.26,
        "insider_misuse": 0.08,
    },
    "dragos_year": 2025,
    "dragos_incidents": 3300,
    "dragos_manufacturing_pct": 0.67,
    "dragos_total_mfg_orgs": 300000,
    "dragos_dark_figure": 3.0,
    "dark_figures": {
        "ransomware_enterprise": 1.0,
        "bec_wire_fraud": 1.5,
        "ics_ot_compromise": 3.0,
        "supply_chain_software": 2.0,
        "general_data_breaches": 1.0,
        "ddos": 1.0,
    },
    "subsplit_overrides": {},
}


def load_annual_data() -> dict:
    """Load annual update data, falling back to hardcoded defaults."""
    if not ANNUAL_FILE.exists():
        return {**DEFAULTS, "source": "defaults"}

    try:
        with open(ANNUAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Merge with defaults so missing keys get filled in
        merged = {**DEFAULTS}
        for key, val in data.items():
            if isinstance(val, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **val}
            else:
                merged[key] = val
        merged["source"] = "file"
        return merged
    except Exception as e:
        logger.error(f"Error loading annual data from {ANNUAL_FILE}: {e}")
        return {**DEFAULTS, "source": "defaults_error"}


def save_annual_data(data: dict) -> bool:
    """Save annual update data to the JSON file."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(ANNUAL_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Annual data saved to {ANNUAL_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving annual data to {ANNUAL_FILE}: {e}")
        return False


def get_dbir_base_rate() -> float:
    """Get the current DBIR base breach rate (overrides or default)."""
    return load_annual_data().get("dbir_base_breach_rate", DEFAULTS["dbir_base_breach_rate"])


def get_dbir_attack_shares() -> dict[str, float]:
    """Get the current DBIR attack share percentages."""
    return load_annual_data().get("dbir_attack_shares", DEFAULTS["dbir_attack_shares"])


def get_dragos_config() -> dict:
    """Get the current Dragos ICS/OT configuration."""
    data = load_annual_data()
    return {
        "incidents": data.get("dragos_incidents", DEFAULTS["dragos_incidents"]),
        "manufacturing_pct": data.get("dragos_manufacturing_pct", DEFAULTS["dragos_manufacturing_pct"]),
        "total_mfg_orgs": data.get("dragos_total_mfg_orgs", DEFAULTS["dragos_total_mfg_orgs"]),
        "dark_figure": data.get("dragos_dark_figure", DEFAULTS["dragos_dark_figure"]),
    }


def get_dark_figures() -> dict[str, float]:
    """Get the current dark figure multipliers."""
    return load_annual_data().get("dark_figures", DEFAULTS["dark_figures"])


def get_subsplit_override(event_id: str) -> float | None:
    """Get a subsplit override for a specific event, or None if not overridden."""
    overrides = load_annual_data().get("subsplit_overrides", {})
    entry = overrides.get(event_id)
    if isinstance(entry, dict):
        return entry.get("subsplit")
    return None
