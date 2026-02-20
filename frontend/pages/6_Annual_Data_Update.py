"""
PRISM Brain - Annual Data Update (Type B Manual Entry)
======================================================
Update DBIR breach rates, Dragos ICS statistics, and dark figure
multipliers when new annual reports are published. These values
feed into Method B probability calculations for 27 cyber events.

Update schedule:
  - DBIR data: Every February (when Verizon publishes the DBIR)
  - Dragos data: Every January (when Dragos publishes Year-in-Review)
  - Dark figures: Review every 2-3 years
"""

import streamlit as st
import sys
from pathlib import Path

from utils.theme import inject_prism_theme

APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))

from modules.api_client import (
    api_engine_get_annual_data,
    api_engine_save_annual_data,
    clear_cache,
)
from modules.database import is_backend_online

st.set_page_config(
    page_title="Annual Data Update | PRISM Brain",
    page_icon=":material/edit_calendar:",
    layout="wide",
)

inject_prism_theme()


# ── Fallback: direct file access when backend is offline ──────────────────

def _load_annual_data_local():
    """Load annual data directly from JSON file (when backend is offline)."""
    try:
        engine_dir = APP_DIR.parent / "prism_engine"
        sys.path.insert(0, str(engine_dir.parent))
        from prism_engine.annual_data import load_annual_data
        return load_annual_data()
    except Exception as e:
        st.error(f"Cannot load annual data: {e}")
        return None


def _save_annual_data_local(data):
    """Save annual data directly to JSON file (when backend is offline)."""
    try:
        engine_dir = APP_DIR.parent / "prism_engine"
        sys.path.insert(0, str(engine_dir.parent))
        from prism_engine.annual_data import save_annual_data
        return save_annual_data(data)
    except Exception as e:
        st.error(f"Cannot save annual data: {e}")
        return False


def load_data():
    """Load annual data via API or direct file access."""
    backend = is_backend_online()
    if backend:
        data = api_engine_get_annual_data()
        if data:
            return data
    return _load_annual_data_local()


def save_data(data):
    """Save annual data via API or direct file access."""
    backend = is_backend_online()
    if backend:
        result = api_engine_save_annual_data(data)
        if result and result.get("status") == "saved":
            return True
    return _save_annual_data_local(data)


# ── DBIR event descriptions (for the subsplit table) ─────────────────────

DBIR_EVENT_NAMES = {
    "DIG-RDE-001": "ERP system encrypted by ransomware",
    "DIG-RDE-002": "Cloud backup destroyed in ransomware attack",
    "DIG-RDE-003": "Double extortion: data exfil + encryption",
    "DIG-RDE-004": "Credential theft leading to data breach",
    "DIG-RDE-005": "Insider data exfiltration",
    "DIG-RDE-006": "Web application exploit leading to breach",
    "DIG-RDE-007": "Credential stuffing / brute-force breach",
    "DIG-RDE-008": "Third-party data breach (vendor compromise)",
    "DIG-FSD-001": "Business email compromise (BEC)",
    "DIG-FSD-002": "Invoice fraud via social engineering",
    "DIG-FSD-003": "Account takeover via credential theft",
    "DIG-FSD-004": "Deepfake-enabled social engineering",
    "DIG-FSD-005": "Volumetric DDoS attack",
    "DIG-FSD-006": "Application-layer DDoS attack",
    "DIG-FSD-007": "DNS/BGP hijacking attack",
    "DIG-SCC-001": "Software supply chain compromise",
    "DIG-SCC-002": "MSP/MSSP compromise (managed service)",
    "DIG-SCC-003": "Open-source dependency compromise",
    "DIG-SCC-004": "Cloud provider breach affecting tenants",
    "DIG-SCC-005": "SaaS platform compromise",
    "DIG-CIC-001": "Healthcare ICS/OT compromise",
    "DIG-CIC-003": "Energy grid OT compromise",
    "DIG-CIC-004": "Water/wastewater OT compromise",
    "DIG-CIC-005": "Transportation OT compromise",
    "DIG-CIC-006": "Nuclear/chemical OT compromise",
}


# ── Main page ─────────────────────────────────────────────────────────────

st.title("Annual Data Update")
st.markdown(
    "Update the annual report data that feeds into **Method B** probability "
    "calculations. These values affect **27 digital risk events** computed "
    "from the Verizon DBIR and Dragos Year-in-Review reports."
)

data = load_data()
if data is None:
    st.error("Could not load annual data. Make sure the backend is running or the engine is accessible.")
    st.stop()

# Show data source
source = data.get("source", "unknown")
last_updated = data.get("last_updated", "Never")
if source == "file":
    st.info(f"Loaded from saved file. Last updated: **{last_updated}**")
elif source == "defaults":
    st.warning("No saved overrides found. Showing hardcoded defaults. Save to create your first override file.")

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_dbir, tab_dragos, tab_dark, tab_subsplits = st.tabs([
    "DBIR Breach Rates",
    "Dragos ICS/OT",
    "Dark Figure Multipliers",
    "Event Subsplits (Advanced)",
])

# ── Tab 1: DBIR Data ─────────────────────────────────────────────────────
with tab_dbir:
    st.subheader("Verizon DBIR Annual Data")
    st.markdown(
        "These values come from the annual **Verizon Data Breach Investigations "
        "Report (DBIR)**. Update them each February when the new edition is published."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        dbir_year = st.number_input(
            "Report year",
            min_value=2020, max_value=2030, step=1,
            value=int(data.get("dbir_year", 2025)),
            key="dbir_year",
        )
        base_rate = st.number_input(
            "Base breach rate (0-1)",
            min_value=0.01, max_value=0.99, step=0.01,
            value=float(data.get("dbir_base_breach_rate", 0.18)),
            format="%.2f",
            key="base_rate",
            help="Annual probability of any breach type affecting an organization. "
                 "DBIR 2025 reports ~18%.",
        )

    with col2:
        st.markdown("**What this means:**")
        st.markdown(
            f"- Base rate = **{base_rate:.0%}** chance of any breach per organization per year\n"
            f"- This base rate is then split by attack type (shares below) and "
            f"event-specific subsplits to derive per-event probabilities"
        )

    st.divider()
    st.markdown("**Attack type shares** (must sum to more than 100% because breaches can involve multiple attack types)")

    shares = data.get("dbir_attack_shares", {})
    share_cols = st.columns(3)
    share_labels = {
        "ransomware": ("Ransomware", "% of breaches involving ransomware (DBIR 2025: 44%)"),
        "social_engineering": ("Social Engineering", "% involving phishing, BEC, pretexting (DBIR 2025: 25%)"),
        "credential_theft": ("Credential Theft", "% involving stolen credentials (DBIR 2025: 38%)"),
        "third_party": ("Third Party", "% involving supply chain / vendor (DBIR 2025: 30%)"),
        "web_app_exploit": ("Web App Exploit", "% involving web application attacks (DBIR 2025: 26%)"),
        "insider_misuse": ("Insider Misuse", "% involving privilege misuse (DBIR 2025: 8%)"),
    }

    share_values = {}
    for i, (key, (label, help_text)) in enumerate(share_labels.items()):
        with share_cols[i % 3]:
            share_values[key] = st.number_input(
                label,
                min_value=0.01, max_value=0.99, step=0.01,
                value=float(shares.get(key, 0.10)),
                format="%.2f",
                key=f"share_{key}",
                help=help_text,
            )

    # Preview: show what the top events would compute to
    st.divider()
    st.markdown("**Preview: Computed probabilities for key events**")
    preview_events = [
        ("DIG-RDE-001", "ransomware", 0.50, "ERP ransomware"),
        ("DIG-RDE-003", "ransomware", 0.62, "Double extortion"),
        ("DIG-FSD-001", "social_engineering", 0.40, "BEC fraud"),
        ("DIG-SCC-001", "third_party", 0.35, "Supply chain"),
    ]
    prev_cols = st.columns(len(preview_events))
    for col, (eid, share_key, subsplit, label) in zip(prev_cols, preview_events):
        computed = base_rate * share_values.get(share_key, 0) * subsplit
        with col:
            st.metric(label, f"{computed:.1%}")


# ── Tab 2: Dragos Data ───────────────────────────────────────────────────
with tab_dragos:
    st.subheader("Dragos ICS/OT Annual Data")
    st.markdown(
        "These values come from the annual **Dragos Year-in-Review** report. "
        "Update them each January when the new edition is published. "
        "They feed into the **DIG-CIC-002** (SCADA/ICS compromise) probability."
    )

    col1, col2 = st.columns(2)
    with col1:
        dragos_year = st.number_input(
            "Report year",
            min_value=2020, max_value=2030, step=1,
            value=int(data.get("dragos_year", 2025)),
            key="dragos_year",
        )
        dragos_incidents = st.number_input(
            "Total ICS/OT incidents reported",
            min_value=100, max_value=50000, step=100,
            value=int(data.get("dragos_incidents", 3300)),
            key="dragos_incidents",
            help="Total number of ICS/OT security incidents tracked by Dragos. "
                 "Dragos 2025 YiR: ~3,300.",
        )
        dragos_mfg_pct = st.number_input(
            "Manufacturing sector percentage (0-1)",
            min_value=0.01, max_value=0.99, step=0.01,
            value=float(data.get("dragos_manufacturing_pct", 0.67)),
            format="%.2f",
            key="dragos_mfg_pct",
            help="Percentage of incidents in manufacturing sector. Dragos 2025: 67%.",
        )

    with col2:
        dragos_total_orgs = st.number_input(
            "Estimated total manufacturing organizations",
            min_value=10000, max_value=1000000, step=10000,
            value=int(data.get("dragos_total_mfg_orgs", 300000)),
            key="dragos_total_orgs",
            help="Estimated total number of manufacturing organizations globally "
                 "with ICS/OT systems. Census/industry estimates: ~300,000.",
        )
        dragos_dark = st.number_input(
            "Dark figure multiplier",
            min_value=1.0, max_value=10.0, step=0.5,
            value=float(data.get("dragos_dark_figure", 3.0)),
            format="%.1f",
            key="dragos_dark",
            help="Underreporting multiplier for ICS/OT incidents. "
                 "Dragos notes 'persistent mischaracterization' of OT incidents as IT. "
                 "Conservative estimate: 3x.",
        )

    # Live preview
    rate = (dragos_incidents * dragos_mfg_pct) / dragos_total_orgs if dragos_total_orgs > 0 else 0
    prior = min(0.95, rate * dragos_dark)
    st.divider()
    st.markdown("**Preview: DIG-CIC-002 (SCADA/ICS compromise)**")
    st.markdown(
        f"Formula: ({dragos_incidents:,} incidents x {dragos_mfg_pct:.0%} mfg "
        f"/ {dragos_total_orgs:,} orgs) x {dragos_dark:.1f} dark figure = **{prior:.2%}**"
    )


# ── Tab 3: Dark Figure Multipliers ───────────────────────────────────────
with tab_dark:
    st.subheader("Dark Figure Multipliers")
    st.markdown(
        "These multipliers correct for underreporting bias in different attack "
        "categories. A multiplier of 1.0 means no correction needed (good coverage). "
        "Higher values indicate more underreporting."
    )

    dark_figs = data.get("dark_figures", {})
    dark_labels = {
        "ransomware_enterprise": (
            "Ransomware (Enterprise)",
            "DBIR includes forensic firm + insurance data. No adjustment needed.",
        ),
        "bec_wire_fraud": (
            "BEC / Wire Fraud",
            "FBI IC3: enterprise BEC reporting rate ~67%. Multiplier = 1/0.67.",
        ),
        "ics_ot_compromise": (
            "ICS/OT Compromise",
            "Dragos: 'persistent mischaracterization' of OT incidents as IT. Conservative 3x.",
        ),
        "supply_chain_software": (
            "Supply Chain (Software)",
            "ENISA ETL 2024: supply chain attacks difficult to attribute. Conservative 2x.",
        ),
        "general_data_breaches": (
            "General Data Breaches",
            "DBIR + IC3 + HIBP comprehensive coverage. No multiplier.",
        ),
        "ddos": (
            "DDoS Attacks",
            "Netscout/Akamai telemetry near-complete. No multiplier.",
        ),
    }

    dark_values = {}
    dark_cols = st.columns(3)
    for i, (key, (label, help_text)) in enumerate(dark_labels.items()):
        with dark_cols[i % 3]:
            dark_values[key] = st.number_input(
                label,
                min_value=1.0, max_value=10.0, step=0.1,
                value=float(dark_figs.get(key, 1.0)),
                format="%.1f",
                key=f"dark_{key}",
                help=help_text,
            )


# ── Tab 4: Event Subsplits (Advanced) ────────────────────────────────────
with tab_subsplits:
    st.subheader("Event-Level Subsplit Overrides")
    st.markdown(
        "Each cyber event has a **subsplit factor** that determines what fraction "
        "of its attack category applies to that specific event. For example, "
        "DIG-RDE-001 (ERP ransomware) uses 50% of the ransomware share because "
        "not all ransomware targets ERP systems specifically.\n\n"
        "Override subsplits here only if you have evidence from the latest DBIR "
        "or other sources. Leave blank to keep the default."
    )

    # Load current DBIR mapping from priors.py defaults
    try:
        from prism_engine.computation.priors import DBIR_EVENT_MAPPING
        mapping = DBIR_EVENT_MAPPING
    except ImportError:
        mapping = {}

    overrides = data.get("subsplit_overrides", {})

    if mapping:
        # Group by family prefix
        families = {}
        for eid, conf in sorted(mapping.items()):
            prefix = "-".join(eid.split("-")[:2])
            families.setdefault(prefix, []).append((eid, conf))

        for family, events in families.items():
            with st.expander(f"{family} ({len(events)} events)", expanded=False):
                for eid, conf in events:
                    name = DBIR_EVENT_NAMES.get(eid, eid)
                    share_key = conf.get("share", "N/A")
                    default_subsplit = conf.get("subsplit", 0)
                    fixed = conf.get("fixed_rate")

                    if fixed is not None:
                        st.markdown(
                            f"**{eid}** - {name}  \n"
                            f"Fixed rate: `{fixed}` (from {conf.get('subsplit_source', 'unknown')}). "
                            f"Not overridable via subsplit."
                        )
                        continue

                    col_a, col_b, col_c = st.columns([2, 1, 1])
                    with col_a:
                        st.markdown(f"**{eid}** - {name}")
                        st.caption(f"Share: {share_key} | Default subsplit: {default_subsplit} | Source: {conf.get('subsplit_source', 'unknown')}")
                    with col_b:
                        current_override = overrides.get(eid, {})
                        has_override = isinstance(current_override, dict) and "subsplit" in current_override
                        override_val = st.number_input(
                            "Override",
                            min_value=0.01, max_value=1.0, step=0.01,
                            value=float(current_override.get("subsplit", default_subsplit)) if has_override else float(default_subsplit),
                            format="%.2f",
                            key=f"subsplit_{eid}",
                            label_visibility="collapsed",
                        )
                    with col_c:
                        # Show computed probability with this subsplit
                        computed = base_rate * share_values.get(share_key, 0) * override_val
                        st.markdown(f"P = **{computed:.1%}**")

                        # Track if user changed from default
                        if abs(override_val - default_subsplit) > 0.001:
                            if eid not in overrides:
                                overrides[eid] = {}
                            overrides[eid] = {"subsplit": override_val}
                        elif eid in overrides:
                            del overrides[eid]
    else:
        st.info("DBIR event mapping not available. Run the app with the engine to see event subsplits.")


# ── Save button (always visible) ─────────────────────────────────────────
st.divider()
col_save, col_reset = st.columns([1, 3])
with col_save:
    if st.button("Save All Changes", type="primary", use_container_width=True):
        updated = {
            "dbir_year": dbir_year,
            "dbir_base_breach_rate": base_rate,
            "dbir_attack_shares": share_values,
            "dragos_year": dragos_year,
            "dragos_incidents": dragos_incidents,
            "dragos_manufacturing_pct": dragos_mfg_pct,
            "dragos_total_mfg_orgs": dragos_total_orgs,
            "dragos_dark_figure": dragos_dark,
            "dark_figures": dark_values,
            "subsplit_overrides": overrides,
        }
        success = save_data(updated)
        if success:
            clear_cache()
            st.success("Annual data saved. Engine probabilities will use these values on next computation.")
        else:
            st.error("Failed to save annual data. Check the console for errors.")

with col_reset:
    st.caption(
        "Saving writes to `prism_engine/data/annual_updates.json`. "
        "Delete that file to revert to hardcoded defaults."
    )
