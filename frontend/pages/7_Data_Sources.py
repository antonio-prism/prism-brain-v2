"""PRISM Brain — Data Sources & Engine Status
==============================================
Monitor the probability engine, check data source credentials,
and trigger recomputation of all 174 risk events.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from utils.theme import inject_prism_theme

import sys
APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))

from modules.database import is_backend_online
from modules.api_client import (
    api_engine_status,
    api_engine_compute_all,
    clear_cache,
)

st.set_page_config(
    page_title="Data Sources | PRISM Brain",
    page_icon="📡",
    layout="wide",
)

inject_prism_theme()


def show_engine_status():
    """Display engine health, version, and credential status."""
    st.subheader("Engine Status")

    # Backend connectivity
    backend_online = is_backend_online()
    if backend_online:
        st.success("Backend connected and operational")
    else:
        st.error("Backend offline — start the server with `python main.py`")
        return

    # Fetch engine status
    status = api_engine_status(use_cache=False)
    if not status:
        st.warning("Could not reach the engine status endpoint.")
        return

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    with col1:
        st.metric("Engine Version", status.get("engine_version", "—"))
    with col2:
        st.metric("Spec Version", status.get("spec_version", "—"))
    with col3:
        st.metric("Total Events", status.get("total_events", "—"))
    with col4:
        st.metric("Fallback Rates Loaded", status.get("fallback_rates_loaded", "—"))

    # API credentials
    credentials = status.get("api_credentials", {})
    no_key_sources = status.get("no_key_sources", [])

    if credentials:
        with st.expander("API Credentials", expanded=False):
            cred_rows = []
            for source, available in credentials.items():
                cred_rows.append({
                    "Source": source,
                    "Status": "Configured" if available else "Missing",
                    "Icon": "✅" if available else "❌",
                })
            df = pd.DataFrame(cred_rows)
            st.dataframe(df, width=600, hide_index=True)

            if no_key_sources:
                st.caption(f"Sources that require no API key: {', '.join(no_key_sources)}")


def show_compute_section():
    """Buttons to trigger engine computation for all 174 events."""
    st.divider()
    st.subheader("Compute Probabilities")

    st.info(
        "Click below to compute dynamic probabilities for all 174 risk events "
        "using the PRISM probability engine. The engine fetches live data from "
        "external APIs (EM-DAT, FRED, World Bank, etc.) and applies the three-method "
        "framework (A / B / C) to produce calibrated probabilities."
    )

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        compute_btn = st.button(
            "Compute All 174 Events",
            key="compute_all_btn",
            type="primary",
            use_container_width=True,
        )

    with col2:
        if st.button("Clear Cache", key="clear_cache_btn", use_container_width=True):
            clear_cache()
            st.success("Cache cleared. Next computation will fetch fresh data.")

    if compute_btn:
        progress = st.empty()
        result_area = st.empty()

        with progress.container():
            st.info("Computing probabilities for all 174 events — this may take up to 2 minutes on first run...")

        result = api_engine_compute_all(use_cache=False)

        progress.empty()

        if result is None:
            result_area.error("Engine computation failed. Check that the backend is running.")
            return

        # Show results
        with result_area.container():
            st.success(f"Computed {len(result)} events successfully!")

            # Method breakdown
            methods = {"A": 0, "B": 0, "C": 0, "FALLBACK": 0}
            probs = []
            for eid, r in result.items():
                if isinstance(r, dict):
                    m = r.get("layer1", {}).get("method", "FALLBACK")
                    methods[m] = methods.get(m, 0) + 1
                    p = r.get("layer1", {}).get("p_global")
                    if p is not None:
                        probs.append(p)

            met_cols = st.columns(4, gap="medium")
            labels = {
                "A": "Method A (Frequency)",
                "B": "Method B (Survey)",
                "C": "Method C (Structural)",
                "FALLBACK": "Fallback Only",
            }
            for i, (key, label) in enumerate(labels.items()):
                with met_cols[i]:
                    st.metric(label, methods.get(key, 0))

            # Probability distribution
            if probs:
                bins = {"0–25%": 0, "25–50%": 0, "50–75%": 0, "75–100%": 0}
                for p in probs:
                    pct = p * 100 if p <= 1 else p
                    if pct < 25:
                        bins["0–25%"] += 1
                    elif pct < 50:
                        bins["25–50%"] += 1
                    elif pct < 75:
                        bins["50–75%"] += 1
                    else:
                        bins["75–100%"] += 1
                st.write("**Probability Distribution:**")
                st.bar_chart(pd.Series(bins, name="Events"))

            st.caption(f"Computed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    st.title("📡 Data Sources & Engine")
    st.markdown(
        "Monitor engine health, check API credentials, and compute probabilities "
        "for all 174 risk events."
    )

    show_engine_status()
    show_compute_section()

    st.divider()

    # Navigation
    nav_col1, _, nav_col3 = st.columns([1, 2, 1], gap="large")
    with nav_col1:
        if st.button("← Results Dashboard", use_container_width=True, key="nav_results"):
            st.switch_page("pages/5_Results_Dashboard.py")
    with nav_col3:
        if st.button("Risk Selection →", use_container_width=True, key="nav_risk"):
            st.switch_page("pages/3_Risk_Selection.py")


if __name__ == "__main__":
    main()
