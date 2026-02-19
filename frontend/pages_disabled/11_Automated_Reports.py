import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys, os
from utils.theme import inject_prism_theme

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.api_client import (
    check_backend_health, api_generate_report, api_create_report_schedule,
    api_get_report_schedules, api_get_report_schedule,
    api_update_report_schedule, api_delete_report_schedule,
    fetch_events, api_get_all_clients
)

st.set_page_config(page_title="Automated Reports", page_icon="ð", layout="wide")


inject_prism_theme()
for key, default in [("generate_report_success", False), ("create_schedule_success", False),
                      ("schedule_action_success", False), ("report_history", []),
                      ("last_generated_report", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

REPORT_TYPES = ["Risk Summary", "Probability Analysis", "Trend Report",
                "Client Risk Assessment", "Full Executive Report"]
REPORT_TYPE_MAP = {"Risk Summary": "risk_summary", "Probability Analysis": "probability_analysis",
                   "Trend Report": "trend_report", "Client Risk Assessment": "client_risk_assessment",
                   "Full Executive Report": "full_executive_report"}
FREQUENCY_OPTIONS = ["Daily", "Weekly", "Monthly"]
FREQUENCY_MAP = {"Daily": "daily", "Weekly": "weekly", "Monthly": "monthly"}

def get_backend_status():
    health = check_backend_health()
    return health.get("status") == "healthy"

def format_timestamp(ts):
    if not ts:
        return "Unknown"
    try:
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        return str(ts)
    except:
        return str(ts)

def format_report_type(api_type):
    for display, api in REPORT_TYPE_MAP.items():
        if api == api_type:
            return display
    return api_type

def format_frequency(api_freq):
    for display, api in FREQUENCY_MAP.items():
        if api == api_freq:
            return display
    return api_freq

def add_to_history(report_type, status, size=None):
    entry = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "type": report_type, "status": status, "size": size or "N/A"}
    st.session_state.report_history.insert(0, entry)
    st.session_state.report_history = st.session_state.report_history[:20]

# Header
st.title("ð Automated Reports")
st.markdown("""
Automated Reports provide comprehensive risk analysis, trend insights, and executive summaries.
Generate on-demand reports or schedule recurring reports to track risk metrics over time.
""")

# Sidebar
st.sidebar.markdown("### Backend Status")
backend_healthy = get_backend_status()
st.sidebar.markdown(f"{'ð¢' if backend_healthy else 'ð´'} **Status:** {'Healthy' if backend_healthy else 'Unhealthy'}")

if backend_healthy:
    try:
        schedules = api_get_report_schedules() or []
        st.sidebar.metric("Total Schedules", len(schedules))
        st.sidebar.metric("Active Schedules", len([s for s in schedules if s.get("is_active", True)]))
        st.sidebar.metric("Report Types", len(REPORT_TYPES))
    except Exception as e:
        st.sidebar.warning(f"Could not load stats: {str(e)}")
else:
    st.sidebar.error("Backend is currently unavailable.")

if not backend_healthy:
    st.error("Backend is unavailable. Please check your connection and try again later.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["â¡ Generate Report", "ð Scheduled Reports", "ð Report History"])

# ============================================================================
# TAB 1: Generate Report
# ============================================================================
with tab1:
    st.header("Generate Report On-Demand")
    st.markdown("Create and download a custom report immediately.")
    try:
        with st.form("generate_report_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                report_type_display = st.selectbox("Report Type *", REPORT_TYPES)
            with col2:
                clients = api_get_all_clients() or []
                client_options = ["All Clients"] + [c.get("name", f"Client {c.get('id')}") for c in clients]
                selected_client = st.selectbox("Filter by Client", client_options)

            st.markdown("**Date Range Filter (Optional)**")
            dc1, dc2 = st.columns(2)
            with dc1:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
            with dc2:
                end_date = st.date_input("End Date", value=datetime.now())

            events = fetch_events(limit=500, use_cache=True) or []
            categories = sorted(set([e.get("category", "Unknown") for e in events])) if events else []
            selected_categories = st.multiselect("Risk Categories (Optional)", categories)

            gc, cc = st.columns(2)
            with gc:
                generate_btn = st.form_submit_button("ð Generate Report", width="stretch", type="primary")
            with cc:
                cancel_btn = st.form_submit_button("Cancel", width="stretch")

            if generate_btn:
                try:
                    filters = {}
                    if selected_client != "All Clients":
                        cid = next((c.get("id") for c in clients if c.get("name", f"Client {c.get('id')}") == selected_client), None)
                        if cid:
                            filters["client_id"] = cid
                    if start_date and end_date:
                        filters["date_range"] = {"start": start_date.isoformat(), "end": end_date.isoformat()}
                    if selected_categories:
                        filters["categories"] = selected_categories

                    report_type_api = REPORT_TYPE_MAP.get(report_type_display, "risk_summary")
                    result = api_generate_report(report_type=report_type_api, filters=filters if filters else None)

                    if result:
                        st.session_state.last_generated_report = result
                        add_to_history(report_type_display, "Generated", result.get("size", "Unknown"))
                        st.success("Report generated successfully!")

                        d1, d2, d3 = st.columns(3)
                        with d1:
                            st.metric("Report Type", report_type_display)
                        with d2:
                            st.metric("Generated", datetime.now().strftime("%Y-%m-%d %H:%M"))
                        with d3:
                            st.metric("Size", result.get("size", "N/A"))

                        if result.get("content"):
                            st.markdown("### Report Content")
                            st.write(result.get("content"))
                        if result.get("summary"):
                            st.info(result.get("summary"))
                        if result.get("metrics"):
                            m = result["metrics"]
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            with mc1:
                                st.metric("Total Risks", m.get("total_risks", 0))
                            with mc2:
                                st.metric("High Risk", m.get("high_risk_count", 0))
                            with mc3:
                                st.metric("Medium Risk", m.get("medium_risk_count", 0))
                            with mc4:
                                st.metric("Low Risk", m.get("low_risk_count", 0))
                        if result.get("data"):
                            try:
                                st.dataframe(pd.DataFrame(result("data")), width="stretch", hide_index=True)
                            except:
                                pass
                    else:
                        st.error("Failed to generate report. Please try again.")
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    add_to_history(report_type_display, "Failed")
            if cancel_btn:
                st.info("Report generation cancelled.")
    except Exception as e:
        st.error(f"Error loading form: {str(e)}")

# ============================================================================
# TAB 2: Scheduled Reports
# ============================================================================
with tab2:
    st.header("Manage Scheduled Reports")
    col_schedules, col_create = st.columns([1.5, 1], gap="large")

    with col_schedules:
        st.subheader("Current Schedules")
        try:
            schedules = api_get_report_schedules() or []
            if not schedules:
                st.info("No scheduled reports yet. Create one using the form on the right.")
            else:
                st.markdown(f"**Found {len(schedules)} schedule(s)**")
                for idx, schedule in enumerate(schedules):
                    sid = schedule.get("id")
                    sname = schedule.get("name", "Unnamed")
                    rtype = format_report_type(schedule.get("report_type", "risk_summary"))
                    freq = format_frequency(schedule.get("schedule_type", "weekly"))
                    active = schedule.get("is_active", True)
                    badge = "ð¢ Active" if active else "ð´ Inactive"

                    with st.expander(f"ð {sname} â {freq.capitalize()} ({badge})", expanded=False):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.text(f"Type: {rtype}")
                            st.text(f"Frequency: {freq.capitalize()}")
                            st.code(str(sid))
                        with c2:
                            st.text(f"Status: {'Active' if active else 'Inactive'}")
                            st.text(f"Last Run: {format_timestamp(schedule.get('last_run'))}")
                            st.text(f"Next Run: {format_timestamp(schedule.get('next_run'))}")

                        a1, a2, a3 = st.columns(3)
                        with a1:
                            if st.button("Edit", key=f"edit_{sid}", width="stretch"):
                                st.info(f"Edit schedule {sid}")
                        with a2:
                            lbl = "Disable" if active else "Enable"
                            if st.button(lbl, key=f"toggle_{sid}", width="stretch"):
                                try:
                                    api_update_report_schedule(sid, {"is_active": not active})
                                    st.success(f"Schedule {'disabled' if active else 'enabled'}!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        with a3:
                            if st.button("Delete", key=f"del_{sid}", width="stretch"):
                                try:
                                    api_delete_report_schedule(sid)
                                    st.success("Schedule deleted!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"Error loading schedules: {str(e)}")

    with col_create:
        st.subheader("Create Schedule")
        try:
            clients = api_get_all_clients() or []
            with st.form("create_schedule_form", clear_on_submit=False):
                schedule_name = st.text_input("Schedule Name *", placeholder="e.g., Weekly Risk Summary")
                report_type_d = st.selectbox("Report Type *", REPORT_TYPES, key="sched_rtype")
                frequency_d = st.selectbox("Frequency *", FREQUENCY_OPTIONS)
                cl_options = ["All Clients"] + [c.get("name", f"Client {c.get('id')}") for c in clients]
                sel_client = st.selectbox("Client (Optional)", cl_options, key="sched_client")

                submit_btn = st.form_submit_button("Create Schedule", width="stretch", type="primary")
                if submit_btn:
                    if not schedule_name:
                        st.error("Please fill in all required fields.")
                    else:
                        try:
                            filters = {}
                            if sel_client != "All Clients":
                                cid = next((c.get("id") for c in clients if c.get("name") == sel_client), None)
                                if cid:
                                    filters["client_id"] = cid
                            result = api_create_report_schedule(
                                name=schedule_name,
                                report_type=REPORT_TYPE_MAP.get(report_type_d, "risk_summary"),
                                schedule_type=FREQUENCY_MAP.get(frequency_d, "weekly"),
                                filters=filters if filters else None
                            )
                            if result:
                                st.success(f"Schedule created! (ID: {result.get('id', 'Unknown')})")
                                st.rerun()
                            else:
                                st.error("Failed to create schedule.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"Error loading form: {str(e)}")

# ============================================================================
# TAB 3: Report History
# ============================================================================
with tab3:
    st.header("Report History")
    try:
        if not st.session_state.report_history:
            st.info("No reports generated yet. Visit the 'Generate Report' tab to create one.")
        else:
            st.markdown(f"**{len(st.session_state.report_history)} report(s) in history**")
            st.dataframe(
                pd.DataFrame(st.session_state.report_history),
                width="stretch", hide_index=True,
                column_config={
                    "date": st.column_config.TextColumn("Generated"),
                    "type": st.column_config.TextColumn("Report Type"),
                    "status": st.column_config.TextColumn("Status"),
                    "size": st.column_config.TextColumn("Size")
                }
            )
            if st.session_state.last_generated_report:
                with st.expander("View Last Generated Report", expanded=False):
                    st.json(st.session_state.last_generated_report)
            if st.button("Clear History", width="stretch"):
                st.session_state.report_history = []
                st.session_state.last_generated_report = None
                st.success("History cleared.")
                st.rerun()
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**PRISM Brain â Automated Reports** | Generate risk analysis reports on-demand or on a recurring schedule.")
