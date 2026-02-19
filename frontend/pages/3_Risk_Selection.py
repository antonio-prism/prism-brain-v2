"""
PRISM Brain V2 - Risk Selection Module
=======================================
Select and prioritize risks relevant to the client.
Uses V2 taxonomy: Domains -> Families -> Events with base-rate probabilities.
"""

import streamlit as st
import pandas as pd
import sys
import io
from pathlib import Path
from utils.theme import inject_prism_theme

APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))

from utils.constants import RISK_DOMAINS
from utils.helpers import (
    get_domain_color,
    get_domain_icon,
    format_percentage
)
from modules.database import (
    get_client,
    get_client_processes,
    get_all_clients,
    add_client_risk,
    get_client_risks,
    update_client_risk,
    is_backend_online
)
from modules.api_client import (
    fetch_v2_events_normalized,
    api_v2_get_taxonomy,
    api_v2_get_probabilities,
    api_v2_health,
)

st.set_page_config(
    page_title="Risk Selection | PRISM Brain",
    page_icon="\u26A1",
    layout="wide"
)

inject_prism_theme()


# ====================== Session State ======================
if 'current_client_id' not in st.session_state:
    st.session_state.current_client_id = None

if 'selected_risks' not in st.session_state:
    st.session_state.selected_risks = set()


# ====================== Sidebar ======================

def client_selector_sidebar():
    """Sidebar for client selection and V2 backend status."""
    st.sidebar.header("\U0001F3E2 Current Client")
    clients = get_all_clients()

    if not clients:
        st.sidebar.warning("No clients created yet")
        if st.sidebar.button("Create Client"):
            st.switch_page("pages/1_Client_Setup.py")
        return

    client_names = {c['id']: c['name'] for c in clients}
    selected_id = st.sidebar.selectbox(
        "Select Client",
        options=list(client_names.keys()),
        format_func=lambda x: client_names[x],
        index=list(client_names.keys()).index(st.session_state.current_client_id)
        if st.session_state.current_client_id in client_names
        else 0
    )

    if selected_id != st.session_state.current_client_id:
        st.session_state.current_client_id = selected_id
        existing_risks = get_client_risks(selected_id, prioritized_only=True)
        st.session_state.selected_risks = set(r['risk_id'] for r in existing_risks)
        st.rerun()

    if st.session_state.current_client_id:
        client = get_client(st.session_state.current_client_id)
        st.sidebar.divider()
        st.sidebar.markdown(f"**\U0001F4CD {client.get('location', 'N/A')}**")
        st.sidebar.markdown(f"\U0001F3ED {client.get('industry', 'N/A')}")
        st.sidebar.markdown(f"\U0001F4CA {client.get('sectors', 'N/A')}")

    # V2 backend status
    st.sidebar.divider()
    st.sidebar.caption("**V2 Backend**")
    if is_backend_online():
        v2h = api_v2_health()
        if v2h and v2h.get("status") == "healthy":
            st.sidebar.markdown(f"\U0001F7E2 **Connected** \u2014 {v2h.get('v2_events', 0)} events")
        else:
            st.sidebar.markdown("\U0001F7E1 **Connected** \u2014 No V2 events loaded")
    else:
        st.sidebar.markdown("\U0001F534 **Backend offline**")


# ====================== Load V2 Events ======================

def load_risks():
    """Load V2 events from the backend API, normalized for frontend use.

    Returns list of dicts with keys: id, name, domain, family_code,
    family_name, default_probability (0-1), base_rate_pct (0-100), etc.
    """
    return fetch_v2_events_normalized()


# ====================== Risk Selection Tab ======================

def risk_selection_interface():
    """Main risk selection interface using V2 taxonomy."""
    client = get_client(st.session_state.current_client_id)
    risks = load_risks()

    if not risks:
        st.error("Could not load V2 events from the backend. "
                 "Make sure the backend is running and migrate_v2.py has been executed.")
        return

    st.markdown(f"## Select Risks for {client['name']}")
    st.markdown("Browse risks by domain and family, then check the ones relevant to this client.")

    # ---------- Filters ----------
    # Get taxonomy for family dropdown
    taxonomy = api_v2_get_taxonomy()
    all_families = []
    if taxonomy:
        for dom in taxonomy.get("domains", []):
            for fam in dom.get("families", []):
                all_families.append(fam)

    col1, col2, col3 = st.columns(3)
    with col1:
        domain_options = ["All Domains"] + list(RISK_DOMAINS.keys())
        selected_domain = st.selectbox("Filter by Domain", domain_options, key="domain_filter")

    with col2:
        if selected_domain == "All Domains":
            family_list = all_families
        else:
            family_list = [f for f in all_families
                           if any(r['domain'] == selected_domain and r['family_code'] == f['family_code']
                                  for r in risks)]
        family_options = ["All Families"] + [
            f"{f['family_code']} \u2014 {f['family_name']}" for f in family_list
        ]
        selected_family_str = st.selectbox("Filter by Family", family_options, key="family_filter")

    with col3:
        search_term = st.text_input("\U0001F50D Search", key="risk_search")

    # Parse family selection
    filter_domain = None if selected_domain == "All Domains" else selected_domain
    filter_family = None
    if selected_family_str != "All Families":
        filter_family = selected_family_str.split(" \u2014 ")[0].strip()

    # Fetch filtered events from API
    filtered_risks = fetch_v2_events_normalized(
        domain=filter_domain,
        family_code=filter_family,
        search=search_term if search_term else None
    )

    # ---------- Bulk actions ----------
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("\u2713 Select All Visible"):
            for r in filtered_risks:
                st.session_state.selected_risks.add(r['id'])
            st.rerun()
    with col2:
        if st.button("\u2717 Clear Selection"):
            st.session_state.selected_risks.clear()
            st.rerun()
    with col3:
        st.write(f"**{len(st.session_state.selected_risks)}** risks selected")

    # ---------- Risk table ----------
    st.subheader(f"Available Risks ({len(filtered_risks)})")

    header_cols = st.columns([1, 4, 2, 2, 2])
    with header_cols[0]:
        st.write("**Select**")
    with header_cols[1]:
        st.write("**Risk Name**")
    with header_cols[2]:
        st.write("**Family**")
    with header_cols[3]:
        st.write("**Domain**")
    with header_cols[4]:
        st.write("**Base Rate**")

    st.divider()

    for risk in filtered_risks:
        risk_id = risk['id']
        cols = st.columns([1, 4, 2, 2, 2])

        with cols[0]:
            is_selected = risk_id in st.session_state.selected_risks
            new_value = st.checkbox(
                "Select",
                value=is_selected,
                key=f"risk_{risk_id}",
                label_visibility="collapsed"
            )
            if new_value != is_selected:
                if new_value:
                    st.session_state.selected_risks.add(risk_id)
                else:
                    st.session_state.selected_risks.discard(risk_id)

        with cols[1]:
            domain_icon = get_domain_icon(risk['domain'])
            st.markdown(f"{domain_icon} **{risk['name']}**")

        with cols[2]:
            st.write(f"{risk.get('family_code', '')} {risk.get('family_name', '')}")

        with cols[3]:
            st.write(risk['domain'])

        with cols[4]:
            prob = risk.get('default_probability', 0)
            st.write(format_percentage(prob))

    # ---------- Save inline ----------
    st.divider()
    if st.button("\U0001F4BE Save Risk Selection", key="save_risks", type="primary"):
        _save_selected_risks(risks)


# ====================== Probabilities Tab ======================

def probability_overview_interface():
    """Show current V2 probabilities for selected risks."""
    st.subheader("\U0001F4CA Risk Probabilities")

    risks = load_risks()
    selected_risks = [r for r in risks if r['id'] in st.session_state.selected_risks]

    if not selected_risks:
        st.info("Select risks in the 'Select Risks' tab first.")
        return

    # Fetch latest probabilities from backend
    v2_probs = api_v2_get_probabilities()
    prob_map = {}
    if v2_probs:
        for p in v2_probs:
            prob_map[p['event_id']] = p

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Selected Risks", len(selected_risks))
    with col2:
        backend_status = "\U0001F7E2 Online" if is_backend_online() else "\U0001F534 Offline"
        st.write(f"**Backend:** {backend_status}")

    st.divider()

    # Results table
    results_data = []
    for risk in selected_risks:
        p = prob_map.get(risk['id'])
        if p:
            prob_pct = p.get('probability_pct', risk.get('base_rate_pct', 0))
            method = p.get('calculation_method', 'BASE_RATE')
            direction = p.get('change_direction', 'STABLE')
        else:
            prob_pct = risk.get('base_rate_pct', 0)
            method = 'BASE_RATE'
            direction = 'STABLE'

        results_data.append({
            'Event ID': risk['id'],
            'Risk Name': risk['name'],
            'Family': f"{risk.get('family_code', '')} {risk.get('family_name', '')}",
            'Probability %': prob_pct,
            'Method': method,
            'Trend': direction,
        })

    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('Probability %', ascending=False)

    st.dataframe(
        results_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Probability %": st.column_config.ProgressColumn(
                "Probability %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        },
    )

    st.caption("Probabilities are Phase 1 base rates (environmental exposure). "
               "Phase 2 will add industry/geography multipliers.")


# ====================== Save Tab ======================

def save_risks_interface():
    """Review and save selected risks to the client's portfolio."""
    st.subheader("\U0001F4BE Save Risk Selections")

    if not st.session_state.selected_risks:
        st.warning("No risks selected yet. Use the 'Select Risks' tab first.")
        return

    client = get_client(st.session_state.current_client_id)
    risks = load_risks()
    selected_risks = [r for r in risks if r['id'] in st.session_state.selected_risks]

    st.write(f"**Client:** {client['name']}  \u2014  **Selected Risks:** {len(selected_risks)}")

    # Summary table
    selected_df = pd.DataFrame([
        {
            'Event ID': r['id'],
            'Risk Name': r['name'],
            'Domain': r['domain'],
            'Family': r.get('family_name', ''),
            'Probability (%)': f"{r.get('default_probability', 0) * 100:.1f}%"
        }
        for r in selected_risks
    ])
    st.dataframe(selected_df, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("\u2713 Confirm & Save Risks", type="primary"):
            _save_selected_risks(risks)

    with col2:
        if st.button("\u2190 Back to Risk Selection"):
            st.rerun()


def _save_selected_risks(all_risks):
    """Save all selected risks to the client's risk portfolio.

    Converts V2 base_rate_pct (0-100) to probability (0-1) before saving,
    because the exposure formula expects 0-1 scale.
    """
    saved = 0
    for risk_id in st.session_state.selected_risks:
        risk = next((r for r in all_risks if r['id'] == risk_id), None)
        if risk:
            # Probability stored as 0-1 (the exposure formula expects this)
            prob_01 = risk.get('default_probability', 0.5)

            add_client_risk(
                client_id=st.session_state.current_client_id,
                risk_id=risk_id,
                risk_name=risk['name'],
                domain=risk['domain'],
                category=risk.get('family_name', ''),
                probability=prob_01,
                is_prioritized=1,
                notes=risk.get('description', ''),
            )
            saved += 1
    st.success(f"Saved {saved} risks for this client!")


# ====================== Import/Export Tab ======================

def import_export_risks():
    """Import and export risk selections as Excel files."""
    st.subheader("\U0001F4E5 Import / Export Risk Selections")

    if not st.session_state.current_client_id:
        st.warning("Select a client first")
        return

    client = get_client(st.session_state.current_client_id)
    risks = load_risks()

    # --- Download ---
    st.markdown("### \U0001F4E5 Download Risk Selection")

    selected_risks = [r for r in risks if r['id'] in st.session_state.selected_risks]

    if selected_risks:
        export_data = []
        for risk in selected_risks:
            prob = risk.get('default_probability', 0)
            export_data.append({
                'Domain': risk['domain'],
                'Family': risk.get('family_name', ''),
                'Event ID': risk['id'],
                'Event Name': risk['name'],
                'Probability (%)': round(prob * 100, 2),
                'Selected': 'Yes'
            })

        export_df = pd.DataFrame(export_data)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, sheet_name='Risk Selection', index=False)
        output.seek(0)

        st.download_button(
            label="\u2B07\uFE0F Download Risk Selection (XLSX)",
            data=output.getvalue(),
            file_name=f"{client['name']}_Risk_Selection.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("No risks selected. Select risks to download.")

    st.divider()

    # --- Upload ---
    st.markdown("### \U0001F4E4 Upload Risk Selection")

    uploaded_file = st.file_uploader(
        "Select an XLSX file to upload risk selections",
        type=['xlsx'],
        key="risk_upload"
    )

    if uploaded_file:
        try:
            upload_df = pd.read_excel(uploaded_file, sheet_name='Risk Selection')

            valid_ids = {r['id'] for r in risks}
            selected_from_upload = set()
            for _, row in upload_df.iterrows():
                if str(row.get('Selected', '')).lower() == 'yes':
                    event_id = str(row.get('Event ID', ''))
                    if event_id in valid_ids:
                        selected_from_upload.add(event_id)

            st.write(f"Found {len(selected_from_upload)} valid risks in file")

            if selected_from_upload:
                if st.button("\u2713 Import & Update Selection"):
                    st.session_state.selected_risks = selected_from_upload
                    st.success("Risk selection updated!")
                    st.rerun()
            else:
                st.warning("No matching V2 event IDs found in the uploaded file.")

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")


# ====================== Main ======================

def main():
    """Main page layout."""
    client_selector_sidebar()

    if not st.session_state.current_client_id:
        st.warning("\U0001F448 Select a client from the sidebar")
        return

    # Header with navigation
    st.title("\u26A1 Risk Selection")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("\u2190 Process Criticality"):
            st.switch_page("pages/2_Process_Criticality.py")
    with col3:
        if st.button("Next: Assessment \u2192"):
            st.switch_page("pages/4_Risk_Assessment.py")

    st.divider()

    # Tabs — Select first (most common action), then Probabilities, Save, Import/Export
    tab1, tab2, tab3, tab4 = st.tabs([
        "\U0001F3AF Select Risks",
        "\U0001F4CA Probabilities",
        "\U0001F4BE Save",
        "\U0001F4E5 Import / Export"
    ])

    with tab1:
        risk_selection_interface()

    with tab2:
        probability_overview_interface()

    with tab3:
        save_risks_interface()

    with tab4:
        import_export_risks()


if __name__ == "__main__":
    main()
