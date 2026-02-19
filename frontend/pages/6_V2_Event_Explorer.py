"""
PRISM Brain V2 - Event Explorer
================================
Browse the V2 risk taxonomy: Domains → Families → Events.
174 events across 4 domains (Physical, Structural, Digital, Operational) and 28 families.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))

from modules.api_client import (
    api_v2_get_taxonomy,
    api_v2_get_events,
    api_v2_get_event,
    api_v2_get_domain,
    api_v2_get_family,
    api_v2_get_probabilities,
    api_v2_get_stats,
    api_v2_health,
    API_BASE_URL,
)
from modules.database import is_backend_online
from utils.theme import (
    inject_prism_theme, page_header, page_footer,
    domain_card_html, DOMAIN_COLORS, DOMAIN_ICONS,
)

st.set_page_config(
    page_title="V2 Event Explorer | PRISM Brain",
    page_icon="\U0001f30d",
    layout="wide"
)

# PRISM Design System CSS
inject_prism_theme()

# ---------- Domain colors & icons ----------
DOMAIN_STYLE = {
    "PHYSICAL": {"color": "#FFC000", "icon": "\U0001f30d", "label": "Physical"},
    "STRUCTURAL": {"color": "#5B9BD5", "icon": "\U0001f3db\ufe0f", "label": "Structural"},
    "OPERATIONAL": {"color": "#70AD47", "icon": "\u2699\ufe0f", "label": "Operational"},
    "DIGITAL": {"color": "#7030A0", "icon": "\U0001f4bb", "label": "Digital"},
}

CONFIDENCE_COLORS = {
    "HIGH": "#2ecc71",
    "MEDIUM-HIGH": "#27ae60",
    "MEDIUM": "#f39c12",
    "MEDIUM-LOW": "#e67e22",
    "LOW": "#e74c3c",
}


def risk_color(pct):
    if pct >= 65:
        return "#FF6B6B"
    elif pct >= 40:
        return "#FFE066"
    return "#69DB7C"


# ======================================================================
# Sidebar — V2 health check
# ======================================================================
with st.sidebar:
    st.markdown("---")
    st.caption("**V2 Backend**")
    backend_ok = is_backend_online()
    if backend_ok:
        v2h = api_v2_health()
        if v2h and v2h.get("status") == "healthy":
            st.markdown(f"\U0001f7e2 **Connected** — {v2h.get('v2_events', 0)} events")
        elif v2h and v2h.get("status") == "empty":
            st.markdown("\U0001f7e1 **Connected** — No events loaded. Run migrate_v2.py first.")
        else:
            st.markdown("\U0001f534 **V2 tables not found.** Run migrate_v2.py.")
    else:
        st.markdown("\U0001f534 **Backend offline**")
    st.markdown("---")


# ======================================================================
# Page header
# ======================================================================
page_header("Event Explorer", "Browse the risk taxonomy: Domains \u2192 Families \u2192 Events", icon="\U0001f30d")

if not backend_ok:
    st.error("Backend is offline. Please start the backend with `./start_app.sh` and refresh.")
    st.stop()


# ======================================================================
# Taxonomy overview
# ======================================================================
taxonomy = api_v2_get_taxonomy()

if not taxonomy or taxonomy.get("total_events", 0) == 0:
    st.warning("No V2 events loaded yet. You need to run the migration script first.\n\n"
               "Open a Terminal and run:\n\n"
               "```\ncd backend && python migrate_v2.py\n```")
    st.stop()

# Overview metrics
st.subheader("Taxonomy Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("\U0001f4ca Total Events", taxonomy["total_events"])
with col2:
    st.metric("\U0001f310 Domains", len(taxonomy["domains"]))
with col3:
    total_families = sum(d["family_count"] for d in taxonomy["domains"])
    st.metric("\U0001f4c1 Families", total_families)
with col4:
    stats = api_v2_get_stats()
    st.metric("\U0001f4c8 Calculations", stats.get("total_calculations", 0) if stats else 0)

st.divider()

# ======================================================================
# Domain cards
# ======================================================================
st.subheader("Risk Domains")

domain_cols = st.columns(len(taxonomy["domains"]))
for i, dom in enumerate(taxonomy["domains"]):
    with domain_cols[i]:
        st.markdown(
            domain_card_html(dom["domain"], dom["event_count"], dom["family_count"]),
            unsafe_allow_html=True,
        )

st.divider()

# ======================================================================
# Navigation: pick a domain then a family (or view all)
# ======================================================================
st.subheader("Explore Events")

domain_options = ["All Domains"] + [d["domain"] for d in taxonomy["domains"]]
selected_domain = st.selectbox("Filter by Domain", domain_options)

# Get families based on domain
if selected_domain == "All Domains":
    all_families = []
    for d in taxonomy["domains"]:
        for f in d.get("families", []):
            all_families.append(f)
    family_options = ["All Families"] + [f"{f['family_code']} — {f['family_name']}" for f in all_families]
else:
    domain_data = api_v2_get_domain(selected_domain)
    all_families = domain_data.get("families", []) if domain_data else []
    family_options = ["All Families"] + [f"{f['family_code']} — {f['family_name']}" for f in all_families]

selected_family_str = st.selectbox("Filter by Family", family_options)

# Parse selection
filter_domain = None if selected_domain == "All Domains" else selected_domain
filter_family = None
if selected_family_str != "All Families":
    filter_family = selected_family_str.split(" — ")[0].strip()

# Search
search_term = st.text_input("\U0001f50d Search events by name", "")

# Fetch events
events = api_v2_get_events(
    domain=filter_domain,
    family_code=filter_family,
    search=search_term if search_term else None
)

if events is None:
    st.error("Could not fetch events from the backend.")
    st.stop()

st.caption(f"Showing {len(events)} events")

# ======================================================================
# Events table
# ======================================================================
if events:
    # Build a DataFrame for nice display
    df = pd.DataFrame(events)
    df = df.rename(columns={
        "event_id": "ID",
        "event_name": "Event Name",
        "domain": "Domain",
        "family_code": "Family",
        "family_name": "Family Name",
        "base_rate_pct": "Base Rate %",
        "confidence_level": "Confidence",
    })
    display_cols = ["ID", "Event Name", "Domain", "Family", "Family Name", "Base Rate %", "Confidence"]
    df_display = df[display_cols]

    st.dataframe(
        df_display,
        width="stretch",
        hide_index=True,
        column_config={
            "Base Rate %": st.column_config.ProgressColumn(
                "Base Rate %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        },
    )

    # ======================================================================
    # Event detail expander
    # ======================================================================
    st.divider()
    st.subheader("Event Detail")

    event_ids = [e["event_id"] for e in events]
    selected_event_id = st.selectbox("Select an event to view details", event_ids,
                                      format_func=lambda eid: f"{eid} — {next((e['event_name'] for e in events if e['event_id'] == eid), eid)}")

    if selected_event_id:
        detail = api_v2_get_event(selected_event_id)
        if detail:
            col_l, col_r = st.columns([2, 1])

            with col_l:
                st.markdown(f"### {detail['event_name']}")
                st.markdown(f"**ID:** `{detail['event_id']}` &nbsp;&nbsp; "
                            f"**Domain:** {detail['domain']} &nbsp;&nbsp; "
                            f"**Family:** {detail['family_code']} — {detail['family_name']}")

                if detail.get("description"):
                    st.markdown(f"**Description:** {detail['description']}")

                st.markdown(f"**Base Rate:** {detail['base_rate_pct']}% "
                            f"({detail.get('base_rate_frequency', 'annual')}) &nbsp;&nbsp; "
                            f"**Confidence:** {detail.get('confidence_level', 'MEDIUM')} &nbsp;&nbsp; "
                            f"**Update:** {detail.get('update_frequency', 'quarterly')} &nbsp;&nbsp; "
                            f"**Scope:** {detail.get('geographic_scope', 'EU')}")

            with col_r:
                # Probability card
                prob_pct = detail.get("current_probability_pct") or detail["base_rate_pct"]
                color = risk_color(prob_pct)
                st.markdown(f"""
                <div style="background:{color}20; border:2px solid {color};
                            padding:20px; border-radius:12px; text-align:center;">
                    <span style="font-size:2.5rem; font-weight:bold; color:{color};">{prob_pct:.1f}%</span><br>
                    <span style="font-size:0.9rem;">Current Probability</span><br>
                    <span style="font-size:0.8rem; color:#888;">
                        {detail.get('change_direction') or 'STABLE'}
                        &nbsp;&middot;&nbsp; Confidence: {detail.get('probability_confidence') or 'N/A'}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # Data sources
            if detail.get("data_sources"):
                with st.expander("Data Sources", expanded=False):
                    for src in detail["data_sources"]:
                        tier_emoji = {"PRIMARY": "\U0001f7e2", "SECONDARY": "\U0001f7e1", "TERTIARY": "\U0001f7e0"}.get(src.get("tier", ""), "\u26aa")
                        st.markdown(f"{tier_emoji} **{src.get('name', 'Unknown')}** ({src.get('tier', 'N/A')})")
                        if src.get("url"):
                            st.caption(f"URL: {src['url']}")
                        if src.get("method"):
                            st.caption(f"Method: {src['method']} &nbsp;&middot;&nbsp; Update: {src.get('update_freq', 'N/A')}")

            # Validation rules
            if detail.get("validation_rules"):
                with st.expander("Validation Rules", expanded=False):
                    vr = detail["validation_rules"]
                    st.json(vr)

else:
    st.info("No events match your filters.")

# ======================================================================
# Footer
# ======================================================================
page_footer()
