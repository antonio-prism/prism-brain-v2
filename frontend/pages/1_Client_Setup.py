"""
PRISM Brain - Client Setup Module
==================================
Create and manage client profiles.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from utils.theme import inject_prism_theme

# Add app directory to path
APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))

from utils.constants import (
    CURRENCY_SYMBOLS,
    INDUSTRY_TEMPLATES,
)
from utils.helpers import (
    format_currency,
)
from modules.database import (
    create_client,
    get_all_clients,
    get_client,
    update_client,
    delete_client,
    delete_all_clients,
    add_client_process,
    get_client_processes,
    update_client_process,
    delete_client_process
)

st.set_page_config(
    page_title="Client Setup | PRISM Brain",
    page_icon="🏢",
    layout="wide"
)

inject_prism_theme()

# Initialize session state
if 'current_client_id' not in st.session_state:
    st.session_state.current_client_id = None

if 'selected_processes' not in st.session_state:
    st.session_state.selected_processes = set()


def client_selector():
    """Sidebar client selector."""
    st.sidebar.header("🏢 Client Selection")
    clients = get_all_clients()

    # New client button
    if st.sidebar.button("➕ New Client", width="stretch"):
        st.session_state.current_client_id = None
        st.session_state.selected_processes = set()
        st.rerun()

    st.sidebar.divider()

    # Existing clients
    if clients:
        st.sidebar.subheader("Existing Clients")
        for client in clients:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(
                    f"📁 {client['name']}",
                    key=f"select_{client['id']}",
                    width="stretch"
                ):
                    st.session_state.current_client_id = client['id']
                    # Load existing processes
                    processes = get_client_processes(client['id'])
                    st.session_state.selected_processes = set(
                        p['process_id'] for p in processes
                    )
                    st.rerun()

            with col2:
                if st.button("🗑️", key=f"delete_{client['id']}"):
                    delete_client(client['id'])
                    if st.session_state.current_client_id == client['id']:
                        st.session_state.current_client_id = None
                    st.rerun()

        # Delete All button at the bottom
        st.sidebar.divider()
        if st.sidebar.button("🗑️ Delete All Clients", type="secondary"):
            st.session_state.confirm_delete_all = True
        if st.session_state.get("confirm_delete_all"):
            st.sidebar.warning("This will permanently delete all clients and their data.")
            col_yes, col_no = st.sidebar.columns(2)
            with col_yes:
                if st.button("Yes, delete all", type="primary"):
                    delete_all_clients()
                    st.session_state.current_client_id = None
                    st.session_state.selected_processes = set()
                    st.session_state.confirm_delete_all = False
                    st.rerun()
            with col_no:
                if st.button("Cancel"):
                    st.session_state.confirm_delete_all = False
                    st.rerun()


def company_profile_form():
    """Company information form."""
    st.subheader("📋 Company Profile")
    client = None
    if st.session_state.current_client_id:
        client = get_client(st.session_state.current_client_id)

    with st.form("company_profile"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input(
                "Company Name *",
                value=client['name'] if client else "",
                placeholder="Enter company name"
            )

            location = st.text_input(
                "Location",
                value=client['location'] if client else "",
                placeholder="City, Country"
            )

            industry = st.selectbox(
                "Industry",
                options=[""] + list(INDUSTRY_TEMPLATES.keys()),
                index=list(INDUSTRY_TEMPLATES.keys()).index(client['industry']) + 1
                if client and client.get('industry') in INDUSTRY_TEMPLATES
                else 0
            )

            revenue = st.number_input(
                "Annual Revenue",
                min_value=0.0,
                value=float(client['revenue']) if client else 0.0,
                step=100000.0,
                format="%.0f"
            )

        with col2:
            currency = st.selectbox(
                "Currency",
                options=list(CURRENCY_SYMBOLS.keys()),
                index=list(CURRENCY_SYMBOLS.keys()).index(client['currency'])
                if client and client.get('currency')
                else 0
            )

            employees = st.number_input(
                "Number of Employees",
                min_value=0,
                value=int(client['employees']) if client else 0,
                step=1
            )

            export_percentage = st.slider(
                "Export Percentage",
                min_value=0,
                max_value=100,
                value=int(client['export_percentage']) if client else 0,
                help="Percentage of revenue from exports"
            )

            primary_markets = st.text_input(
                "Primary Markets",
                value=client['primary_markets'] if client else "",
                placeholder="e.g., Norway, Germany, USA"
            )

        sectors = st.text_input(
            "Key Sectors",
            value=client['sectors'] if client else "",
            placeholder="e.g., Defense, Marine, Renewable Energy"
        )

        notes = st.text_area(
            "Notes",
            value=client['notes'] if client else "",
            placeholder="Additional information about the client..."
        )

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            submitted = st.form_submit_button(
                "💾 Save Client" if client else "➕ Create Client",
                type="primary",
                width="stretch"
            )

        if submitted:
            if not name:
                st.error("Company name is required")
            else:
                if client:
                    # Update existing client
                    update_client(
                        client['id'],
                        name=name,
                        location=location,
                        industry=industry,
                        revenue=revenue,
                        currency=currency,
                        employees=employees,
                        export_percentage=export_percentage,
                        primary_markets=primary_markets,
                        sectors=sectors,
                        notes=notes
                    )
                    st.success("✅ Client updated successfully!")
                else:
                    # Create new client
                    client_id = create_client(
                        name=name,
                        location=location,
                        industry=industry,
                        revenue=revenue,
                        currency=currency,
                        employees=employees,
                        export_percentage=export_percentage,
                        primary_markets=primary_markets,
                        sectors=sectors,
                        notes=notes
                    )
                    st.session_state.current_client_id = client_id
                    st.success("✅ Client created successfully!")
                    st.rerun()


def main():
    """Main page function."""
    st.title("🏢 Client Setup")
    st.markdown("Create and configure client profiles for risk assessment.")

    # Sidebar client selector
    client_selector()

    # Show current client name if selected
    if st.session_state.current_client_id:
        client = get_client(st.session_state.current_client_id)
        if client:
            st.success(f"📁 Working with: **{client['name']}**")

    # Company Profile (only tab now)
    company_profile_form()

    # Navigation
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("← Back to Home"):
            st.switch_page("Welcome.py")

    with col3:
        if st.session_state.current_client_id:
            if st.button("Next: Process Criticality →", type="primary"):
                st.switch_page("pages/2_Process_Criticality.py")


if __name__ == "__main__":
    main()
