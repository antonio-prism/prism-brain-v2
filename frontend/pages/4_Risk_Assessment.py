""" PRISM Brain - Risk Assessment Module
=====================================
Input vulnerability, resilience, and downtime for each process-risk combination.
"""

import streamlit as st
import pandas as pd
import io
import sys
from pathlib import Path
from utils.theme import inject_prism_theme

APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))

from utils.constants import CURRENCY_SYMBOLS
from utils.helpers import (
    get_domain_color,
    get_domain_icon,
    format_currency,
    format_percentage,
    get_risk_level,
    get_risk_level_color
)
from modules.database import (
    get_client,
    get_all_clients,
    get_client_processes,
    get_client_risks,
    get_assessments,
    save_assessment,
    calculate_risk_exposure
)

st.set_page_config(
    page_title="Risk Assessment | PRISM Brain",
    page_icon="🎯",
    layout="wide"
)

inject_prism_theme()

if 'current_client_id' not in st.session_state:
    st.session_state.current_client_id = None

if 'assessment_mode' not in st.session_state:
    st.session_state.assessment_mode = 'guided'  # 'guided' or 'table'


# ---- Performance: load data ONCE and reuse across all functions ----
def _load_page_data():
    """Load all data needed by this page once, cache in session_state."""
    cid = st.session_state.current_client_id
    if not cid:
        return None, [], [], []
    client = get_client(cid)
    processes = get_client_processes(cid)
    risks = get_client_risks(cid, prioritized_only=True)
    assessments = get_assessments(cid) or []
    return client, processes, risks, assessments


def client_selector_sidebar(processes, risks, assessments):
    """Sidebar for client selection and progress."""
    st.sidebar.header("🏢 Current Client")
    clients = get_all_clients()

    if not clients:
        st.sidebar.warning("No clients created")
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
        st.rerun()

    # Progress indicator (uses data passed in, not re-fetched)
    if st.session_state.current_client_id:
        total_combinations = len(processes) * len(risks)
        completed = len(assessments)

        st.sidebar.divider()
        st.sidebar.subheader("📊 Progress")

        if total_combinations > 0:
            progress = completed / total_combinations
            st.sidebar.progress(progress)
            st.sidebar.write(f"{completed} / {total_combinations} assessed")
            st.sidebar.write(f"({progress*100:.0f}% complete)")
        else:
            st.sidebar.warning("No combinations to assess")


def assessment_overview(client, processes, risks, assessments):
    """Overview of assessment status."""
    st.subheader("📊 Assessment Overview")

    if not st.session_state.current_client_id:
        st.warning("Please select a client")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Processes", len(processes))
    with col2:
        st.metric("Risks", len(risks))
    with col3:
        total = len(processes) * len(risks)
        st.metric("Combinations", total)
    with col4:
        st.metric("Assessed", len(assessments))

    if not processes:
        st.warning("No processes selected. Please go to Client Setup to select processes.")
        return

    if not risks:
        st.warning("No risks selected. Please go to Risk Selection to choose risks.")
        return

    # Matrix preview
    st.markdown("### Process-Risk Matrix")

    # Create matrix view
    matrix_data = []

    # Use assessments passed in (no re-fetch)
    assessment_lookup = {(a['process_id'], a['risk_id']): a for a in assessments}

    for proc in processes:
        row = {"Process": proc['custom_name'] or proc['process_name']}

        for risk in risks:
            # Check if assessment exists (using backend-aware lookup)
            assessment = assessment_lookup.get((proc['id'], risk['id']))

            if assessment:
                exposure = calculate_risk_exposure(
                    proc['criticality_per_day'],
                    assessment['vulnerability'],
                    assessment['resilience'],
                    assessment['expected_downtime'],
                    risk['probability']
                )
                row[risk['risk_name'][:20]] = f"✅ {format_currency(exposure, client['currency'])}"
            else:
                row[risk['risk_name'][:20]] = "⬜ Pending"

        matrix_data.append(row)

    df_matrix = pd.DataFrame(matrix_data)
    st.dataframe(df_matrix, width="stretch", hide_index=True)


def guided_assessment(client, processes, risks, assessments):
    """Guided step-by-step assessment interface."""
    st.subheader("🎯 Guided Assessment")

    if not st.session_state.current_client_id:
        return

    if not processes or not risks:
        st.warning("Please ensure you have selected both processes and risks.")
        return

    currency = client.get('currency', 'EUR')
    symbol = CURRENCY_SYMBOLS.get(currency, '€')

    # Build list of combinations (using data passed in, no re-fetch)
    assessment_lookup = {(a['process_id'], a['risk_id']): a for a in assessments}

    combinations = []
    for proc in processes:
        for risk in risks:
            existing = assessment_lookup.get((proc['id'], risk['id']))
            combinations.append({
                'process': proc,
                'risk': risk,
                'existing': existing,
                'completed': existing is not None
            })

    # Progress
    completed_count = sum(1 for c in combinations if c['completed'])
    st.progress(completed_count / len(combinations) if combinations else 0)
    st.write(f"Progress: {completed_count} / {len(combinations)} combinations assessed")

    # Select combination to assess
    pending = [c for c in combinations if not c['completed']]
    completed = [c for c in combinations if c['completed']]

    st.divider()

    # Tabs for pending vs editing
    tab1, tab2 = st.tabs(["📝 Pending", "✏️ Edit Completed"])

    with tab1:
        if not pending:
            st.success("🎉 All combinations have been assessed!")
        else:
            st.info(f"{len(pending)} combinations pending assessment")

            # Select next combination
            options = [f"{c['process']['process_name']} × {c['risk']['risk_name']}" for c in pending]
            selected_idx = st.selectbox(
                "Select combination to assess",
                options=range(len(options)),
                format_func=lambda x: options[x]
            )

            if selected_idx is not None:
                combo = pending[selected_idx]
                assessment_form(combo['process'], combo['risk'], None, client)

    with tab2:
        if not completed:
            st.info("No assessments completed yet")
        else:
            options = [f"{c['process']['process_name']} × {c['risk']['risk_name']}" for c in completed]
            selected_idx = st.selectbox(
                "Select combination to edit",
                options=range(len(options)),
                format_func=lambda x: options[x],
                key="edit_select"
            )

            if selected_idx is not None:
                combo = completed[selected_idx]
                assessment_form(combo['process'], combo['risk'], combo['existing'], client)


def assessment_form(process, risk, existing, client):
    """Assessment form for a single process-risk combination."""
    currency = client.get('currency', 'EUR')
    symbol = CURRENCY_SYMBOLS.get(currency, '€')

    st.markdown("---")

    # Process info
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Process")
        st.write(f"**{process['process_name']}**")
        st.write(f"Criticality: {format_currency(process['criticality_per_day'], currency)}/day")

    with col2:
        st.markdown("### ⚠️ Risk")
        domain_color = get_domain_color(risk['domain'])
        st.markdown(f"**{risk['risk_name']}**")
        st.markdown(
            f"<span style='background-color:{domain_color}20; padding:2px 8px; border-radius:4px;'>"
            f"{get_domain_icon(risk['domain'])} {risk['domain']}</span>",
            unsafe_allow_html=True
        )
        st.write(f"Probability: {format_percentage(risk['probability'])}")

    st.divider()

    # Assessment inputs
    with st.form(f"assessment_{process['id']}_{risk['id']}"):
        st.markdown("### 📝 Assessment Inputs")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Vulnerability**")
            st.caption("How likely is this process affected by this risk?")
            vulnerability = st.slider(
                "Vulnerability (%)",
                min_value=0,
                max_value=100,
                value=int(existing['vulnerability'] * 100) if existing else 0,
                step=5,
                key=f"vuln_{process['id']}_{risk['id']}",
                label_visibility="collapsed"
            )
            st.write(f"**{vulnerability}%**")

        with col2:
            st.markdown("**Resilience**")
            st.caption("How quickly can you recover from this risk?")
            resilience = st.slider(
                "Resilience (%)",
                min_value=0,
                max_value=100,
                value=int(existing['resilience'] * 100) if existing else 0,
                step=5,
                key=f"res_{process['id']}_{risk['id']}",
                label_visibility="collapsed"
            )
            st.write(f"**{resilience}%**")

        with col3:
            st.markdown("**Expected Downtime**")
            st.caption("Days until normal operations resume")
            downtime = st.number_input(
                "Downtime (days)",
                min_value=0,
                max_value=365,
                value=existing['expected_downtime'] if existing else 0,
                step=1,
                key=f"down_{process['id']}_{risk['id']}",
                label_visibility="collapsed"
            )
            st.write(f"**{downtime} days**")

        notes = st.text_area(
            "Notes (optional)",
            value=existing['notes'] if existing else "",
            placeholder="Add any notes about this assessment...",
            key=f"notes_{process['id']}_{risk['id']}"
        )

        # Preview calculation
        st.markdown("### 💰 Calculated Exposure")
        exposure = calculate_risk_exposure(
            process['criticality_per_day'],
            vulnerability / 100,
            resilience / 100,
            downtime,
            risk['probability']
        )

        st.metric(
            "Annual Risk Exposure",
            format_currency(exposure, currency),
            help="Criticality × Vulnerability × (1-Resilience) × Downtime × Probability"
        )

        # Formula breakdown
        with st.expander("See calculation breakdown"):
            st.markdown(f"""
            **PRISM Formula:**
            ```
            Exposure = Criticality × Vulnerability × (1-Resilience) × Downtime × Probability
            ```

            **Values:**
            - Criticality: {format_currency(process['criticality_per_day'], currency)}/day
            - Vulnerability: {vulnerability}%
            - Resilience: {resilience}% → Impact factor: {100-resilience}%
            - Downtime: {downtime} days
            - Probability: {format_percentage(risk['probability'])}

            **Calculation:**
            ```
            {symbol}{process['criticality_per_day']:,.0f} × {vulnerability/100:.2f} × {(100-resilience)/100:.2f} × {downtime} × {risk['probability']:.2f} = {symbol}{exposure:,.0f}/year
            ```
            """)

        submitted = st.form_submit_button("💾 Save Assessment", type="primary", width="stretch")

        if submitted:
            save_assessment(
                client_id=st.session_state.current_client_id,
                process_id=process['id'],
                risk_id=risk['id'],
                vulnerability=vulnerability / 100,
                resilience=resilience / 100,
                expected_downtime=downtime,
                notes=notes
            )
            st.success("✅ Assessment saved!")
            st.rerun()


def batch_assessment(client, processes, risks, assessments):
    """Batch assessment table interface."""
    st.subheader("📊 Batch Assessment")

    if not st.session_state.current_client_id:
        return

    if not processes or not risks:
        st.warning("Please ensure you have selected both processes and risks.")
        return

    st.markdown("""
    Enter assessments in bulk using the table below. Edit the values directly in the table, then click Save.
    """)

    # Use assessments passed in (no re-fetch)
    assessment_lookup = {(a['process_id'], a['risk_id']): a for a in assessments}

    data = []
    id_mapping = []  # Store process/risk IDs separately by row index

    for proc in processes:
        for risk in risks:
            existing = assessment_lookup.get((proc['id'], risk['id']))
            data.append({
                'Process': proc['process_name'][:30],
                'Risk': risk['risk_name'][:30],
                'Vulnerability (%)': int(existing['vulnerability'] * 100) if existing else 0,
                'Resilience (%)': int(existing['resilience'] * 100) if existing else 0,
                'Downtime (days)': existing['expected_downtime'] if existing else 0
            })

            # Keep track of IDs separately
            id_mapping.append({
                'proc_id': proc['id'],
                'risk_id': risk['id']
            })

    df = pd.DataFrame(data)

    # Editable dataframe - no hidden columns needed
    edited_df = st.data_editor(
        df,
        column_config={
            'Process': st.column_config.TextColumn(disabled=True),
            'Risk': st.column_config.TextColumn(disabled=True),
            'Vulnerability (%)': st.column_config.NumberColumn(min_value=0, max_value=100, step=5),
            'Resilience (%)': st.column_config.NumberColumn(min_value=0, max_value=100, step=5),
            'Downtime (days)': st.column_config.NumberColumn(min_value=0, max_value=365, step=1)
        },
        hide_index=True,
        width="stretch",
        num_rows="fixed"
    )

    if st.button("💾 Save All Assessments", type="primary", width="stretch"):
        saved_count = 0

        for idx, row in edited_df.iterrows():
            # Get IDs from our separate mapping using the row index
            ids = id_mapping[idx]

            save_assessment(
                client_id=st.session_state.current_client_id,
                process_id=ids['proc_id'],
                risk_id=ids['risk_id'],
                vulnerability=row['Vulnerability (%)'] / 100,
                resilience=row['Resilience (%)'] / 100,
                expected_downtime=int(row['Downtime (days)']),
                notes=""
            )
            saved_count += 1

        st.success(f"✅ Saved {saved_count} assessments!")
        st.rerun()  # Refresh to show updated values


def import_export_assessments(client, processes, risks, assessments):
    """Import/Export assessments via XLSX."""
    st.subheader("📥 Import / Export")

    if not st.session_state.current_client_id:
        st.warning("Please select a client to continue")
        return

    if not processes or not risks:
        st.warning("Please ensure you have selected both processes and risks.")
        return

    currency = client.get('currency', 'EUR')

    # Create tabs for download and upload
    col1, col2 = st.columns(2)

    # Download section
    with col1:
        st.markdown("### 📥 Download Template")
        st.markdown("Export current assessments or download a blank template to fill in.")

        if st.button("📥 Download XLSX Template", width="stretch"):
            # Build export data
            assessment_lookup = {(a['process_id'], a['risk_id']): a for a in assessments}
            export_data = []

            for proc in processes:
                for risk in risks:
                    existing = assessment_lookup.get((proc['id'], risk['id']))
                    export_data.append({
                        'Process Name': proc['process_name'],
                        'Process ID': proc['id'],
                        'Risk Name': risk['risk_name'],
                        'Risk ID': risk['id'],
                        'Criticality/Day': proc['criticality_per_day'],
                        'Probability (%)': risk['probability'] * 100,
                        'Vulnerability (%)': int(existing['vulnerability'] * 100) if existing else '',
                        'Resilience (%)': int(existing['resilience'] * 100) if existing else '',
                        'Downtime (days)': existing['expected_downtime'] if existing else ''
                    })

            df_export = pd.DataFrame(export_data)

            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_export.to_excel(writer, sheet_name='Assessments', index=False)

            output.seek(0)

            st.download_button(
                label="⬇️ Download XLSX",
                data=output.getvalue(),
                file_name=f"risk_assessments_{st.session_state.current_client_id}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # Upload section
    with col2:
        st.markdown("### 📤 Upload XLSX")
        st.markdown("Upload a completed assessment file to import values.")

        uploaded_file = st.file_uploader(
            "Choose XLSX file",
            type=["xlsx", "xls"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            try:
                df_import = pd.read_excel(uploaded_file)

                # Validate required columns
                required_cols = ['Process ID', 'Risk ID', 'Vulnerability (%)', 'Resilience (%)', 'Downtime (days)']
                missing_cols = [col for col in required_cols if col not in df_import.columns]

                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                else:
                    st.markdown("### Preview of imported data")
                    st.dataframe(df_import, width="stretch")

                    if st.button("💾 Save Imported Assessments", type="primary", width="stretch"):
                        saved_count = 0
                        error_count = 0

                        for idx, row in df_import.iterrows():
                            try:
                                process_id = row['Process ID']
                                risk_id = row['Risk ID']
                                vulnerability = float(row['Vulnerability (%)']) / 100 if pd.notna(row['Vulnerability (%)']) else 0
                                resilience = float(row['Resilience (%)']) / 100 if pd.notna(row['Resilience (%)']) else 0
                                downtime = int(row['Downtime (days)']) if pd.notna(row['Downtime (days)']) else 0

                                save_assessment(
                                    client_id=st.session_state.current_client_id,
                                    process_id=process_id,
                                    risk_id=risk_id,
                                    vulnerability=vulnerability,
                                    resilience=resilience,
                                    expected_downtime=downtime,
                                    notes=""
                                )
                                saved_count += 1
                            except Exception as e:
                                error_count += 1
                                st.warning(f"Row {idx + 2}: {str(e)}")

                        st.success(f"✅ Imported {saved_count} assessments!")
                        if error_count > 0:
                            st.warning(f"⚠️ {error_count} rows had errors")
                        st.rerun()

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")


def main():
    """Main page function."""
    st.title("🎯 Risk Assessment")
    st.markdown("Assess vulnerability, resilience, and downtime for each process-risk combination.")

    # Load ALL data once, pass to every function
    client, processes, risks, assessments = _load_page_data()

    client_selector_sidebar(processes, risks, assessments)

    if not st.session_state.current_client_id:
        st.warning("Please select a client to continue")
        if st.button("Go to Client Setup"):
            st.switch_page("pages/1_Client_Setup.py")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📋 Batch Mode",
        "🎯 Guided Assessment",
        "📥 Import / Export"
    ])

    with tab1:
        batch_assessment(client, processes, risks, assessments)

    with tab2:
        guided_assessment(client, processes, risks, assessments)

    with tab3:
        import_export_assessments(client, processes, risks, assessments)

    # Navigation
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("← Risk Selection"):
            st.switch_page("pages/3_Risk_Selection.py")

    with col3:
        if assessments:
            if st.button("Next: Results →", type="primary"):
                st.switch_page("pages/5_Results_Dashboard.py")


if __name__ == "__main__":
    main()
