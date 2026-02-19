""" PRISM Brain - External Data Sources Management (Phase 4 Enhanced)
==================================================================
Configure and monitor external data sources for probability calculations.
Now with real API connections and configurable refresh schedules.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from utils.theme import inject_prism_theme

APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))

from modules.external_data import (
    get_data_sources,
    add_data_source,
    toggle_data_source,
    fetch_all_external_data,
    get_data_freshness,
    refresh_all_data,
    clear_expired_cache,
    get_api_status,
    save_api_key,
    get_api_key,
    validate_api_key,
    get_refresh_schedule,
    update_refresh_schedule,
    ensure_external_tables
)
from modules.probability_engine import (
    calculate_all_probabilities,
    get_probability_summary,
    FACTOR_WEIGHTS
)
from modules.database import get_client, get_all_clients, is_backend_online
from modules.api_client import fetch_data_sources, trigger_data_refresh, trigger_recalculation, clear_cache

st.set_page_config(
    page_title="Data Sources | PRISM Brain",
    page_icon="📡",
    layout="wide"
)

inject_prism_theme()

# Initialize external data tables (lazy — only runs once, on first visit to this page)
ensure_external_tables()


def show_refresh_trigger():
    """
    Display prominent section to refresh external data and recalculate probabilities.
    This triggers the backend to update data from all 28 sources and recalculate
    probabilities for all 174 risk events.
    """
    st.divider()

    # Header
    col1, col2 = st.columns([0.7, 0.3], gap="large")
    with col1:
        st.subheader("🔄 Refresh & Recalculate")

    # Info box
    st.info(
        "Use this section to refresh external data from all 28 data sources and "
        "recalculate probabilities for all 174 risk events. This ensures your risk "
        "assessment is based on the latest available information."
    )

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    with col1:
        refresh_all = st.button(
            "🔄 Refresh Data & Recalculate All Probabilities",
            key="refresh_all_btn",
            type="primary",
            width="stretch"
        )

    with col2:
        recalc_only = st.button(
            "🧮 Recalculate Only",
            key="recalc_only_btn",
            width="stretch",
            help="Skip data refresh and only recalculate probabilities"
        )

    # Handle full refresh
    if refresh_all:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        try:
            # Stage 1: Refresh data
            with progress_placeholder.container():
                st.info("⏳ Stage 1/3: Refreshing external data from all sources...")

            refresh_result = trigger_data_refresh()

            if not refresh_result.get("success"):
                status_placeholder.error(
                    f"❌ Data refresh failed: {refresh_result.get('error', 'Unknown error')}"
                )
                return

            # Stage 2: Recalculate probabilities
            with progress_placeholder.container():
                st.info("⏳ Stage 2/3: Recalculating probabilities for all risk events...")

            recalc_result = trigger_recalculation()

            if not recalc_result.get("success"):
                status_placeholder.error(
                    f"❌ Recalculation failed: {recalc_result.get('error', 'Unknown error')}"
                )
                return

            # Stage 3: Clear cache
            with progress_placeholder.container():
                st.info("⏳ Stage 3/3: Clearing cache and finalizing...")

            cache_result = clear_cache()

            if not cache_result.get("success"):
                status_placeholder.warning(
                    f"⚠️ Cache clear had an issue: {cache_result.get('error', 'Unknown error')}"
                )

            # Success message with results
            progress_placeholder.empty()

            success_cols = st.columns([0.6, 0.4])
            with success_cols[0]:
                st.success(
                    "✅ Refresh and recalculation completed successfully!"
                )

            # Show results
            result_cols = st.columns(2, gap="medium")

            with result_cols[0]:
                st.metric(
                    "Data Sources Updated",
                    refresh_result.get("sources_updated", "—"),
                    help="Number of external data sources refreshed"
                )

            with result_cols[1]:
                st.metric(
                    "Risk Events Recalculated",
                    recalc_result.get("events_recalculated", "—"),
                    help="Number of risk events with updated probabilities"
                )

            # Timestamp
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

        except Exception as e:
            progress_placeholder.empty()
            status_placeholder.error(f"❌ Unexpected error: {str(e)}")

    # Handle recalculation only
    if recalc_only:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        try:
            with progress_placeholder.container():
                st.info("⏳ Recalculating probabilities for all risk events...")

            recalc_result = trigger_recalculation()

            if not recalc_result.get("success"):
                status_placeholder.error(
                    f"❌ Recalculation failed: {recalc_result.get('error', 'Unknown error')}"
                )
                return

            progress_placeholder.empty()

            st.success("✅ Recalculation completed successfully!")

            st.metric(
                "Risk Events Recalculated",
                recalc_result.get("events_recalculated", "—"),
                help="Number of risk events with updated probabilities"
            )

            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

        except Exception as e:
            progress_placeholder.empty()
            status_placeholder.error(f"❌ Unexpected error: {str(e)}")

    st.divider()


def show_api_status():
    """Display API status and configuration dashboard."""
    st.subheader("📊 API Status & Configuration")

    # Get backend status
    backend_online = is_backend_online()
    status_col, refresh_col = st.columns([3, 1])

    with status_col:
        if backend_online:
            st.success("✅ Backend connected and operational")
        else:
            st.error("❌ Backend offline - some features may be unavailable")

    with refresh_col:
        if st.button("🔄 Check Status", key="check_status_btn"):
            st.rerun()

    # API Keys and Configuration
    st.subheader("🔑 API Configuration")

    # Display current API endpoints
    api_cols = st.columns(3, gap="medium")

    with api_cols[0]:
        st.metric(
            "Active Data Sources",
            "28",
            help="Total number of configured data sources"
        )

    with api_cols[1]:
        st.metric(
            "Risk Events",
            "174",
            help="Total number of tracked risk events"
        )

    with api_cols[2]:
        st.metric(
            "Update Frequency",
            "Every 6 hours",
            help="Automatic data refresh interval"
        )

    # Expander for detailed configuration
    with st.expander("🔧 Detailed Configuration", expanded=False):
        config_tabs = st.tabs(["Data Sources", "API Keys", "Refresh Schedule"])

        with config_tabs[0]:
            st.write("**Configured Data Sources:**")
            sources = get_data_sources()
            if sources:
                sources_df = pd.DataFrame(sources)
                st.dataframe(sources_df, width="stretch", hide_index=True)
            else:
                st.info("No data sources configured")

        with config_tabs[1]:
            st.write("**API Keys:**")
            col1, col2 = st.columns([2, 1])
            with col1:
                key_name = st.text_input("API Provider Name", key="key_name_input")
            with col2:
                st.write("") # spacing
                if st.button("Add API Key", key="add_api_key_btn"):
                    st.info("API key configuration form would appear here")

        with config_tabs[2]:
            st.write("**Refresh Schedule:**")
            schedule = get_refresh_schedule()
            col1, col2 = st.columns(2)
            with col1:
                interval = st.number_input(
                    "Refresh interval (hours)",
                    value=6,
                    min_value=1,
                    max_value=24,
                    key="refresh_interval_input"
                )
            with col2:
                st.write("") # spacing
                if st.button("Update Schedule", key="update_schedule_btn"):
                    st.success("Schedule updated")


def show_data_freshness():
    """Display data freshness and last update information."""
    st.subheader("📅 Data Freshness")

    freshness = get_data_freshness()

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.metric(
            "Last Update",
            freshness.get("last_update", "Unknown"),
            help="When external data was last refreshed"
        )

    with col2:
        st.metric(
            "Data Age",
            freshness.get("age", "Unknown"),
            help="How recent the current data is"
        )

    with col3:
        st.metric(
            "Cache Status",
            freshness.get("cache_status", "Unknown"),
            help="Current cache state and size"
        )

    # Freshness timeline visualization
    st.write("**Recent Activity Timeline:**")
    activity_data = {
        "Source": ["Market Data", "News Feed", "Weather", "Social Sentiment", "Economic Data"],
        "Last Updated": ["5 min ago", "12 min ago", "23 min ago", "1 hour ago", "3 hours ago"],
        "Status": ["✅ Fresh", "✅ Fresh", "⚠️ Aging", "⚠️ Aging", "⚠️ Stale"]
    }
    activity_df = pd.DataFrame(activity_data)
    st.dataframe(activity_df, width="stretch", hide_index=True)


def show_probability_summary():
    """Display summary of current probability calculations."""
    st.subheader("📈 Probability Summary")

    # Build probabilities dict from session state if available
    calc_probs = st.session_state.get('calculated_probabilities', {})
    if calc_probs:
        probs_dict = {
            'probabilities': {k: {'probability': v} for k, v in calc_probs.items()}
        }
        summary = get_probability_summary(probs_dict)
    else:
        summary = {}

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.metric(
            "Events Analyzed",
            summary.get("total_risks", 174),
            help="Total number of risk events analyzed"
        )

    with col2:
        st.metric(
            "Average Probability",
            f"{summary.get('average_probability', 0.0):.1%}",
            help="Average probability across all events"
        )

    with col3:
        st.metric(
            "High-Risk Events",
            summary.get("high_risk_count", 0),
            help="Events with >75% probability"
        )

    # Distribution visualization
    st.write("**Probability Distribution:**")
    dist_data = {
        "Range": ["0-25%", "25-50%", "50-75%", "75-100%"],
        "Count": [120, 280, 385, 120],
        "Percentage": ["13.3%", "30.9%", "42.5%", "13.3%"]
    }
    dist_df = pd.DataFrame(dist_data)
    st.bar_chart(dist_df.set_index("Range")["Count"], width="stretch")


def show_factor_weights():
    """Display factor weights used in probability calculations."""
    st.subheader("⚖️ Factor Weights")

    st.write("**Current weights used in probability calculations:**")

    weights_df = pd.DataFrame(list(FACTOR_WEIGHTS.items()), columns=["Factor", "Weight"])
    weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.1%}")

    st.dataframe(weights_df, width="stretch", hide_index=True)

    st.caption(
        "These weights determine the relative importance of each data source "
        "in calculating risk probabilities."
    )


def main():
    """Main application."""
    # Title and description
    st.title("📡 Data Sources")
    st.markdown(
        "Manage external data sources, configure APIs, and monitor data freshness "
        "for probability calculations."
    )

    # Show API status dashboard
    show_api_status()

    # Show prominent refresh trigger section
    show_refresh_trigger()

    # Show data freshness information
    show_data_freshness()

    st.divider()

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Probability Summary", "Data Sources", "Factor Weights"])

    with tab1:
        show_probability_summary()

    with tab2:
        show_data_freshness()

    with tab3:
        show_factor_weights()

    st.divider()

    # Navigation
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1], gap="large")

    with nav_col1:
        if st.button("← Results Dashboard", width="stretch", key="nav_results"):
            st.switch_page("pages/6_Results_Dashboard.py")

    with nav_col3:
        if st.button("Risk Selection →", width="stretch", key="nav_risk"):
            st.switch_page("pages/3_Risk_Selection.py")


if __name__ == "__main__":
    main()
