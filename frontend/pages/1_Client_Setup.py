"""
PRISM Brain - Client Setup Module
==================================
Create and manage client profiles.
Includes AI-powered prefill for automatic process and risk selection.
"""

import streamlit as st
import pandas as pd
import json
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
    delete_client_process,
    add_client_risk,
    get_client_risks,
    is_backend_online,
)
from modules.api_client import (
    check_backend_health,
    api_engine_ai_prefill,
    api_engine_status,
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

if 'ai_prefill_result' not in st.session_state:
    st.session_state.ai_prefill_result = None

if 'ai_prefill_status' not in st.session_state:
    st.session_state.ai_prefill_status = None


def client_selector():
    """Sidebar client selector."""
    st.sidebar.header("🏢 Client Selection")
    clients = get_all_clients()

    # New client button
    if st.sidebar.button("➕ New Client", width="stretch"):
        st.session_state.current_client_id = None
        st.session_state.selected_processes = set()
        st.session_state.ai_prefill_result = None
        st.session_state.ai_prefill_status = None
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
                    # Clear AI state when switching clients
                    st.session_state.ai_prefill_result = None
                    st.session_state.ai_prefill_status = None
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


# =====================================================================
# AI-Powered Prefill Section
# =====================================================================

def _check_ai_available():
    """Check if AI prefill is available (backend online + API key configured)."""
    if not is_backend_online():
        return False, "Backend connection required"
    status = api_engine_status(use_cache=True)
    if not status:
        return False, "Cannot check engine status"
    creds = status.get("api_credentials", {})
    if not creds.get("anthropic", False):
        return False, "Configure ANTHROPIC_API_KEY in .env to enable AI features"
    return True, "Ready"


def ai_prefill_section():
    """AI-Powered prefill section — analyze company and suggest processes/risks."""
    if not st.session_state.current_client_id:
        return

    client = get_client(st.session_state.current_client_id)
    if not client:
        return

    st.divider()
    st.subheader("AI-Powered Setup")
    st.markdown(
        "Analyze this company using AI to automatically suggest relevant "
        "business processes and risks. The AI will search the web for company "
        "information and review any uploaded documents."
    )

    # Check if AI is available
    ai_available, ai_message = _check_ai_available()

    # Show existing AI result if available
    if st.session_state.ai_prefill_status == "done" and st.session_state.ai_prefill_result:
        _render_review_screen(client)
        return

    # Show error if last run failed
    if st.session_state.ai_prefill_status == "error":
        st.error(st.session_state.ai_prefill_result or "AI analysis failed")
        if st.button("Try Again", key="ai_retry"):
            st.session_state.ai_prefill_status = None
            st.session_state.ai_prefill_result = None
            st.rerun()
        return

    # Document uploader
    uploaded_files = st.file_uploader(
        "Upload company documents (optional)",
        type=["pdf", "docx", "xlsx", "txt", "csv"],
        accept_multiple_files=True,
        help="Upload annual reports, risk registers, audit reports, or other "
             "company documents for a more detailed analysis. Max 5 files, 20MB total.",
        key="ai_doc_upload",
    )

    # Validate uploads
    if uploaded_files:
        if len(uploaded_files) > 5:
            st.warning("Maximum 5 files allowed. Only the first 5 will be used.")
            uploaded_files = uploaded_files[:5]
        total_size = sum(f.size for f in uploaded_files)
        if total_size > 20 * 1024 * 1024:
            st.error(f"Total file size ({total_size / 1024 / 1024:.1f} MB) exceeds 20MB limit.")
            uploaded_files = None

    # Run button
    if not ai_available:
        st.button("Run AI Analysis", disabled=True, help=ai_message, key="ai_run_disabled")
        st.caption(f"*{ai_message}*")
    else:
        doc_count = len(uploaded_files) if uploaded_files else 0
        btn_label = f"Run AI Analysis" + (f" ({doc_count} document{'s' if doc_count != 1 else ''})" if doc_count else "")

        if st.button(btn_label, type="primary", key="ai_run"):
            _run_ai_analysis(client, uploaded_files)


def _run_ai_analysis(client, uploaded_files):
    """Execute the AI analysis with a spinner."""
    client_name = client.get("name", "the company")
    doc_count = len(uploaded_files) if uploaded_files else 0

    if doc_count:
        spinner_msg = (
            f"Researching **{client_name}**... Analyzing {doc_count} uploaded "
            f"document{'s' if doc_count != 1 else ''}, searching the web, and mapping "
            f"to risk framework. This takes 30-60 seconds."
        )
    else:
        spinner_msg = (
            f"Researching **{client_name}**... Searching the web, analyzing "
            f"industry profile, and mapping to risk framework. This takes 15-30 seconds."
        )

    with st.spinner(spinner_msg):
        result = api_engine_ai_prefill(
            client_id=st.session_state.current_client_id,
            files=uploaded_files,
        )

    if result and result.get("status") == "success":
        st.session_state.ai_prefill_result = result
        st.session_state.ai_prefill_status = "done"
        st.rerun()
    else:
        error_msg = result.get("message", "Unknown error") if result else "No response from backend"
        st.session_state.ai_prefill_result = error_msg
        st.session_state.ai_prefill_status = "error"
        st.rerun()


def _render_review_screen(client):
    """Render the AI analysis review screen with checkboxes."""
    result = st.session_state.ai_prefill_result

    st.success("**AI Analysis Complete**")

    # Show documents analyzed
    docs = result.get("documents_processed", [])
    if docs:
        doc_names = ", ".join(
            f"{d['filename']} ({d.get('pages', d.get('sheets', '?'))} "
            f"{'pages' if d['type'] == 'pdf' else 'sheets' if d['type'] == 'xlsx' else 'items'})"
            for d in docs
        )
        st.caption(f"Based on: web research + {len(docs)} document(s): {doc_names}")
    else:
        st.caption("Based on: web research")

    # Company analysis
    analysis = result.get("company_analysis", "")
    if analysis:
        st.markdown(f"> {analysis}")

    # Warnings
    for w in result.get("warnings", []):
        st.warning(w)

    # Validation stats
    validation = result.get("validation", {})
    if validation.get("processes_dropped", 0) > 0 or validation.get("risks_dropped", 0) > 0:
        st.info(
            f"Validation: {validation.get('processes_dropped', 0)} process suggestions and "
            f"{validation.get('risks_dropped', 0)} risk suggestions were filtered out "
            f"(IDs not in PRISM catalog)."
        )

    # ── Suggested Processes ──
    processes = result.get("processes", [])
    st.markdown(f"### Suggested Processes ({len(processes)})")

    if processes:
        # Initialize checkbox states for processes
        for i, p in enumerate(processes):
            key = f"ai_proc_{i}"
            if key not in st.session_state:
                st.session_state[key] = True

        # Render as a compact table with checkboxes
        for i, p in enumerate(processes):
            col_check, col_id, col_name, col_scope, col_rationale = st.columns([0.5, 1, 3, 1.5, 4])
            with col_check:
                st.checkbox("", value=True, key=f"ai_proc_{i}", label_visibility="collapsed")
            with col_id:
                st.caption(p.get("process_id", ""))
            with col_name:
                st.markdown(f"**{p.get('process_name', p.get('process_id', ''))}**")
            with col_scope:
                st.caption(p.get("scope", ""))
            with col_rationale:
                st.caption(p.get("rationale", ""))
    else:
        st.warning("No processes suggested.")

    # ── Suggested Risks ──
    risks = result.get("risks", [])
    st.markdown(f"### Suggested Risks ({len(risks)})")

    if risks:
        # Initialize checkbox states for risks
        for i, r in enumerate(risks):
            key = f"ai_risk_{i}"
            if key not in st.session_state:
                st.session_state[key] = True

        # Render as a compact table with checkboxes
        for i, r in enumerate(risks):
            col_check, col_id, col_name, col_domain, col_vr, col_rationale = st.columns([0.5, 1.2, 3, 1.5, 1.5, 3.5])
            with col_check:
                st.checkbox("", value=True, key=f"ai_risk_{i}", label_visibility="collapsed")
            with col_id:
                st.caption(r.get("event_id", ""))
            with col_name:
                st.markdown(f"**{r.get('event_name', r.get('event_id', ''))}**")
            with col_domain:
                st.caption(r.get("domain", ""))
            with col_vr:
                v = r.get("vulnerability", 0)
                res = r.get("resilience", 0)
                st.caption(f"V:{v:.0%} R:{res:.0%}")
            with col_rationale:
                st.caption(r.get("rationale", ""))
    else:
        st.warning("No risks suggested.")

    # ── Action Buttons ──
    st.divider()
    col_apply, col_discard, col_spacer = st.columns([1, 1, 3])

    with col_apply:
        if st.button("Apply Selections", type="primary", key="ai_apply"):
            _apply_ai_selections(result)

    with col_discard:
        if st.button("Discard", key="ai_discard"):
            st.session_state.ai_prefill_result = None
            st.session_state.ai_prefill_status = None
            # Clean up checkbox states
            for key in list(st.session_state.keys()):
                if key.startswith("ai_proc_") or key.startswith("ai_risk_"):
                    del st.session_state[key]
            st.rerun()


def _apply_ai_selections(result):
    """Apply the checked AI suggestions to the client's data."""
    client_id = st.session_state.current_client_id
    processes = result.get("processes", [])
    risks = result.get("risks", [])

    # Collect checked items
    selected_processes = []
    for i, p in enumerate(processes):
        if st.session_state.get(f"ai_proc_{i}", True):
            selected_processes.append(p)

    selected_risks = []
    for i, r in enumerate(risks):
        if st.session_state.get(f"ai_risk_{i}", True):
            selected_risks.append(r)

    # Save processes
    proc_saved = 0
    for p in selected_processes:
        try:
            add_client_process(
                client_id=client_id,
                process_id=p["process_id"],
                process_name=p.get("process_name", p["process_id"]),
                category=p.get("scope", ""),
                criticality_per_day=0,
                notes=f"AI: {p.get('rationale', '')}",
            )
            proc_saved += 1
        except Exception as e:
            st.warning(f"Could not save process {p['process_id']}: {e}")

    # Save risks
    risk_saved = 0
    for r in selected_risks:
        try:
            base_rate = r.get("base_rate_pct", 0)
            probability = base_rate / 100.0 if base_rate > 1 else base_rate
            add_client_risk(
                client_id=client_id,
                risk_id=r["event_id"],
                risk_name=r.get("event_name", r["event_id"]),
                domain=r.get("domain", ""),
                category=r.get("family_name", ""),
                probability=probability,
                notes=f"AI: {r.get('rationale', '')}",
            )
            risk_saved += 1
        except Exception as e:
            st.warning(f"Could not save risk {r['event_id']}: {e}")

    # Store V/R suggestions for the Risk Assessment page
    vr_suggestions = {}
    for r in selected_risks:
        vr_suggestions[r["event_id"]] = {
            "vulnerability": r.get("vulnerability", 0.5),
            "resilience": r.get("resilience", 0.3),
        }
    st.session_state.ai_vr_suggestions = vr_suggestions

    # Update session state sets
    st.session_state.selected_processes = set(p["process_id"] for p in selected_processes)
    st.session_state.selected_risks = set(r["event_id"] for r in selected_risks)

    # Force re-sync on Process Criticality and Risk Selection pages
    st.session_state._processes_synced_for = None
    st.session_state._risks_synced_for = None

    # Clear AI state and checkbox keys
    st.session_state.ai_prefill_result = None
    st.session_state.ai_prefill_status = "applied"
    for key in list(st.session_state.keys()):
        if key.startswith("ai_proc_") or key.startswith("ai_risk_"):
            del st.session_state[key]

    st.success(f"Applied **{proc_saved}** processes and **{risk_saved}** risks. "
               f"Next: set revenue impact per process on the Process Criticality page.")
    st.balloons()


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

    # Company Profile
    company_profile_form()

    # AI-Powered Prefill (only shown when a client is saved)
    if st.session_state.current_client_id:
        ai_prefill_section()

        # Show success message if just applied
        if st.session_state.ai_prefill_status == "applied":
            st.session_state.ai_prefill_status = None  # Clear after showing

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
