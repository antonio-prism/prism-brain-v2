"""
PRISM Brain - Risk Intelligence System
======================================
Main application entry point.

Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add app directory to path for imports
APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))

from utils.constants import APP_NAME, APP_VERSION, APP_SUBTITLE, RISK_DOMAINS
from utils.helpers import load_data_summary
from utils.theme import inject_prism_theme, page_header, page_footer, domain_card_html, DOMAIN_COLORS
from modules.database import init_database, is_backend_online, get_data_source, refresh_backend_status
# Page configuration
st.set_page_config(
    page_title=f"{APP_NAME} - Welcome",
    page_icon="\U0001f3af",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_database()

# --- Backend Status Indicator (sidebar) ---
with st.sidebar:
    st.markdown("---")
    st.caption("**Backend Connection**")
    backend_online = is_backend_online()
    if backend_online:
        st.markdown("\U0001f7e2 **Connected** (PostgreSQL)")
    else:
        st.markdown("\U0001f534 **Offline** (using local SQLite)")
    st.caption(f"Data: {get_data_source()}")
    if st.button("\U0001f504 Refresh", key="refresh_backend"):
        refresh_backend_status()
        st.rerun()
    st.markdown("---")

# PRISM Design System CSS
inject_prism_theme()


def main():
    """Main application page - Welcome."""

    # Header
    page_header(APP_NAME, f"{APP_SUBTITLE} v{APP_VERSION}", icon="\U0001f3af")

    st.divider()

    # Load summary data
    try:
        summary = load_data_summary()
    except Exception as e:
        summary = {"risks": {"total": 174, "families": 28}, "processes": {"total": 222}}
        st.sidebar.caption(f"\u26a0\ufe0f Data summary fallback: {type(e).__name__}")

    # Overview metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("\U0001f4ca Risk Events", f"{summary['risks']['total']:,}")

    with col2:
        st.metric("\U0001f3c6 Risk Families", f"{summary['risks'].get('families', 28)}")

    with col3:
        st.metric("\U0001f4cb Processes", f"{summary['processes']['total']:,}")

    st.divider()

    # Main content
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("\U0001f4cd Quick Start Guide")

        st.markdown("""
        Welcome to **PRISM Brain V2** — your comprehensive risk intelligence system,
        with 174 curated risk events across 4 domains and 28 families.

        ### How to Use This Application

        **Step 1: Client Setup** \U0001f3e2
        - Create a new client profile with company information
        - Select the industry and geographic location

        **Step 2: Process Criticality** \u2699\ufe0f
        - Select relevant business processes from the process framework
        - Set criticality values (revenue impact per day of disruption)

        **Step 3: Risk Selection** \u26a1
        - Browse 174 risk events across 4 domains: Physical, Structural, Digital, and Operational
        - Each event has a research-backed base probability and confidence level
        - Select the risks relevant to your client

        **Step 4: Risk Assessment** \U0001f4dd
        - For each prioritized process-risk combination:
          - Set Vulnerability (0-100%)
          - Set Resilience (0-100%)
          - Set Expected Downtime (days)

        **Step 5: Results Dashboard** \U0001f4b0
        - View total risk exposure in your chosen currency
        - Analyze risk by domain, process, and risk event
        - Export results to Excel for reporting

        You can also use the **Event Explorer** to browse and search the full risk taxonomy,
        or the **Data Sources** page to manage external data feeds.
        """)

        st.info("\U0001f448 Use the sidebar to navigate between modules.")

    with col_right:
        st.subheader("\U0001f4ca Risk Domains")

        for domain, info in RISK_DOMAINS.items():
            domain_risks = summary['risks'].get('by_domain', {}).get(domain, 0)
            st.markdown(
                domain_card_html(domain, domain_risks, 0, extra=info['description']),
                unsafe_allow_html=True,
            )

    # Footer
    page_footer(APP_VERSION)


if __name__ == "__main__":
    main()
