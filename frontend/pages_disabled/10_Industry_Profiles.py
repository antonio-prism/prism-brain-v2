import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from utils.theme import inject_prism_theme

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.api_client import (
    check_backend_health,
    api_get_profiles,
    api_get_profile,
    api_get_profile_by_industry,
    api_create_profile,
    api_apply_profile,
    fetch_events,
    api_get_all_clients
)

# Page configuration
st.set_page_config(
    page_title="Industry Risk Profiles",
    page_icon="ð¢",
    layout="wide"
)

inject_prism_theme()

# Initialize session state
if "form_reset" not in st.session_state:
    st.session_state.form_reset = False
if "create_profile_success" not in st.session_state:
    st.session_state.create_profile_success = False
if "apply_profile_success" not in st.session_state:
    st.session_state.apply_profile_success = False

# Industry options
INDUSTRIES = [
    "Technology",
    "Finance",
    "Healthcare",
    "Manufacturing",
    "Energy",
    "Agriculture",
    "Transportation",
    "Retail",
    "Construction",
    "Government",
    "Education",
    "Telecommunications"
]

# Helper functions
def get_backend_status():
    """Check backend health and return status indicator."""
    health = check_backend_health()
    return health.get("status") == "healthy"

def format_timestamp(ts):
    """Format timestamp string for display."""
    if not ts:
        return "Unknown"
    try:
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        return str(ts)
    except:
        return str(ts)

def get_profiles_data():
    """Fetch all profiles from backend."""
    profiles = api_get_profiles()
    return profiles if profiles else []

def get_events_data():
    """Fetch all events from backend."""
    events = fetch_events(limit=500, use_cache=True)
    return events if events else []

def get_clients_data():
    """Fetch all clients from backend."""
    clients = api_get_all_clients()
    return clients if clients else []

# Header
st.title("ð¢ Industry Risk Profiles")
st.markdown("""
Industry Risk Profiles help you quickly apply pre-configured, industry-specific risk assessments
to new clients. Each profile contains a curated set of relevant risk events and their importance
scores based on sector best practices.
""")

# Sidebar - Backend Status and Stats
st.sidebar.markdown("### Backend Status")
backend_healthy = get_backend_status()
status_color = "ð¢" if backend_healthy else "ð´"
st.sidebar.markdown(f"{status_color} **Status:** {'Healthy' if backend_healthy else 'Unhealthy'}")

if backend_healthy:
    try:
        profiles = get_profiles_data()
        events = get_events_data()
        profile_count = len(profiles) if profiles else 0
        industry_count = len(set([p.get("industry", "Unknown") for p in profiles])) if profiles else 0
        event_count = len(events) if events else 0
        st.sidebar.metric("Total Profiles", profile_count)
        st.sidebar.metric("Industries Covered", industry_count)
        st.sidebar.metric("Risk Events Available", event_count)
    except Exception as e:
        st.sidebar.warning(f"Could not load stats: {str(e)}")
else:
    st.sidebar.error("Backend unavailable.")

if not backend_healthy:
    st.error("Backend is unavailable.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Browse Profiles", "Create Profile", "Apply Profile"])

with tab1:
    st.header("Browse Industry Profiles")
    try:
        profiles = get_profiles_data()
        if not profiles:
            st.info("No profiles yet. Create one in the 'Create Profile' tab.")
        else:
            industries = sorted(set([p.get("industry", "Unknown") for p in profiles]))
            filter_ind = st.selectbox("Filter by Industry", ["All"] + industries)
            filtered = profiles if filter_ind == "All" else [p for p in profiles if p.get("industry") == filter_ind]
            st.markdown(f"**Found {len(filtered)} profile(s)**")
            for p in filtered:
                with st.expander(f"{p.get('profile_name', 'Unnamed')} - {p.get('industry', 'Unknown')}"):
                    c1, c2 = st.columns(2)
                    c1.text(f"Industry: {p.get('industry')}")
                    c1.text(f"Template: {'lYes' if p.get('is_template') else 'No'}")
                    c2.text(f"ID: {p.get('id')}")
                    evts = p.get("events", [])
                    c2.metric("Events", len(evts) if evts else 0)
                    st.text(p.get("description", "No description"))
                    if evts:
                        st.dataframe(pd.DataFrame([{"ID": e.get("event_id"), "Name": e.get("event_name"), "Category": e.get("category")} for e in evts]), hide_index=True)
    except Exception as e:
        st.error(f"Error: {e}")

with tab2:
    st.header("Create Industry Profile")
    try:
        events = get_events_data()
        if not events:
            st.warning("No risk events available.")
        else:
            with st.form("create_profile_form"):
                c1, c2 = st.columns(2)
                industry = c1.selectbox("Industry *", INDUSTRIES)
                profile_name = c1.text_input("Profile Name *")
                is_template = c2.checkbox("Mark as Template", value=True)
                description = st.text_area("Description", height=100)
                event_opts = {e.get("event_id"): f"{e.get('event_name')} ({e.get('category', 'General')})" for e in events}
                selected_evts = st.multiselect("Risk Events *", options=list(event_opts.keys()), format_func=lambda x: event_opts.get(x, x))
                if selected_evts:
                    st.info(f"{len(selected_evts)} event(s) selected")
                if st.form_submit_button("Create Profile", type="primary"):
                    if not profile_name or not selected_evts:
                        st.error("Please fill name and select events.")
                    else:
                        events_data = [{"event_id": eid, "event_name": next((e.get("event_name") for e in events if e.get("event_id") == eid), ""), "relevance_score": 1.0} for eid in selected_evts]
                        result = api_create_profile(industry, profile_name, description, is_template, events_data)
                        if result:
                            st.success(f"Profile created! (ID: {result})")
                        else:
                            st.error("Failed to create profile.")
    except Exception as e:
        st.error(f"Error: {e}")

with tab3:
    st.header("Apply Profile to Client")
    try:
        profiles = get_profiles_data()
        clients = get_clients_data()
        if not profiles:
            st.warning("No profiles available. Create one first.")
        elif not clients:
            st.warning("No clients available. Create one first.")
        else:
            with st.form("apply_profile_form"):
                prof_opts = {p.get("id"): f"{p.get('profile_name')} ({p.get('industry')})" for p in profiles}
                client_opts = {c.get("id"): c.get("name", "Unknown") for c in clients}
                c1, c2 = st.columns(2)
                pid = c1.selectbox("Profile", list(prof_opts.keys()), format_func=lambda x: prof_opts.get(x, str(x)))
                cid = c2.selectbox("Client", list(client_opts.keys()), format_func=lambda x: client_opts.get(x, str(x)))
                st.warning("Applying will add all profile events to the client.")
                if st.form_submit_button("Apply Profile", type="primary"):
                    result = api_apply_profile(pid, cid)
                    if result:
                        st.success("Profile applied successfully!")
                    else:
                        st.error("Failed to apply profile.")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.markdown("**PRISM Brain â Industry Risk Profiles**")
