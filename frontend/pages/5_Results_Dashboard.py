"""
PRISM Brain - Results Dashboard
================================
Visualize and export risk exposure analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import sys
from pathlib import Path
from datetime import datetime
from utils.theme import inject_prism_theme

APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))

from utils.constants import CURRENCY_SYMBOLS, RISK_DOMAINS
from utils.helpers import (
    format_currency, format_percentage, get_domain_color, get_domain_icon,
    get_risk_level, get_risk_level_color, export_timestamp
)
from modules.database import (
    get_client, get_all_clients, get_client_processes, get_client_risks,
    get_assessments, get_risk_exposure_summary, calculate_risk_exposure
)

st.set_page_config(page_title="Results Dashboard | PRISM Brain", page_icon="💰", layout="wide")


inject_prism_theme()
if 'current_client_id' not in st.session_state:
    st.session_state.current_client_id = None


def client_selector_sidebar():
    """Client selection sidebar."""
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
              if st.session_state.current_client_id in client_names else 0
    )

    if selected_id != st.session_state.current_client_id:
        st.session_state.current_client_id = selected_id
        st.rerun()


def executive_summary():
    """Executive summary with key metrics."""
    st.subheader("📊 Executive Summary")

    if not st.session_state.current_client_id:
        st.warning("Please select a client")
        return

    client = get_client(st.session_state.current_client_id)
    summary = get_risk_exposure_summary(st.session_state.current_client_id)

    if not summary:
        st.warning("No assessments completed yet. Please complete risk assessments first.")
        return

    currency = client.get('currency', 'EUR')
    symbol = CURRENCY_SYMBOLS.get(currency, '€')

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Annual Risk Exposure",
            format_currency(summary['total_exposure'], currency),
            help="Sum of all assessed risk exposures"
        )

    with col2:
        revenue = client.get('revenue', 0)
        if revenue > 0:
            risk_ratio = summary['total_exposure'] / revenue * 100
            st.metric(
                "% of Revenue at Risk",
                f"{risk_ratio:.1f}%",
                help="Total exposure as percentage of annual revenue"
            )
        else:
            st.metric("Assessments", len(summary['assessments']))

    with col3:
        # Highest risk process
        if summary['by_process']:
            top_process = max(summary['by_process'].items(), key=lambda x: x[1])
            st.metric(
                "Highest Risk Process",
                top_process[0][:20],
                format_currency(top_process[1], currency)
            )

    with col4:
        # Highest risk domain
        if summary['by_domain']:
            top_domain = max(summary['by_domain'].items(), key=lambda x: x[1])
            st.metric(
                "Highest Risk Domain",
                f"{get_domain_icon(top_domain[0])} {top_domain[0]}",
                format_currency(top_domain[1], currency)
            )

    # Client info card
    st.divider()
    st.markdown(f"""
    **Client:** {client['name']} | **Location:** {client.get('location', 'N/A')} |
    **Industry:** {client.get('industry', 'N/A')} | **Currency:** {currency}
    """)


def domain_breakdown_chart(summary, currency):
    """Create domain breakdown pie/bar chart."""
    if not summary['by_domain']:
        return None

    df = pd.DataFrame([
        {"Domain": domain, "Exposure": exposure,
         "Color": get_domain_color(domain)}
        for domain, exposure in summary['by_domain'].items()
    ])

    # Pie chart
    fig = px.pie(
        df, values='Exposure', names='Domain',
        title='Risk Exposure by Domain',
        color='Domain',
        color_discrete_map={d: get_domain_color(d) for d in RISK_DOMAINS.keys()},
        hole=0.4
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, height=400)

    return fig


def process_exposure_chart(summary, currency, top_n=10):
    """Create top processes bar chart."""
    if not summary['by_process']:
        return None

    # Sort and take top N
    sorted_procs = sorted(summary['by_process'].items(),
                         key=lambda x: x[1], reverse=True)[:top_n]

    df = pd.DataFrame([
        {"Process": name[:25], "Exposure": exposure}
        for name, exposure in sorted_procs
    ])

    fig = px.bar(
        df, x='Exposure', y='Process',
        orientation='h',
        title=f'Top {top_n} Processes by Risk Exposure',
        color='Exposure',
        color_continuous_scale='Reds'
    )

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400,
        showlegend=False
    )

    return fig


def risk_exposure_chart(summary, currency, top_n=10):
    """Create top risks bar chart."""
    if not summary['by_risk']:
        return None

    sorted_risks = sorted(summary['by_risk'].items(),
                         key=lambda x: x[1], reverse=True)[:top_n]

    df = pd.DataFrame([
        {"Risk": name[:30], "Exposure": exposure}
        for name, exposure in sorted_risks
    ])

    fig = px.bar(
        df, x='Exposure', y='Risk',
        orientation='h',
        title=f'Top {top_n} Risks by Exposure',
        color='Exposure',
        color_continuous_scale='Oranges'
    )

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400,
        showlegend=False
    )

    return fig


def heatmap_chart(summary, currency):
    """Create process-domain heatmap."""
    if not summary['assessments']:
        return None

    # Aggregate by process and domain
    heatmap_data = {}
    for a in summary['assessments']:
        process = a['process_name'][:20]
        domain = a['domain']
        exposure = a['exposure']

        if process not in heatmap_data:
            heatmap_data[process] = {}
        if domain not in heatmap_data[process]:
            heatmap_data[process][domain] = 0
        heatmap_data[process][domain] += exposure

    # Convert to matrix
    processes = list(heatmap_data.keys())
    domains = list(RISK_DOMAINS.keys())

    z_data = []
    for proc in processes:
        row = [heatmap_data.get(proc, {}).get(d, 0) for d in domains]
        z_data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=domains,
        y=processes,
        colorscale='RdYlGn_r',
        text=[[f"€{v:,.0f}" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 10},
    ))

    fig.update_layout(
        title='Risk Heatmap: Process × Domain',
        height=max(400, len(processes) * 30),
        xaxis_title='Domain',
        yaxis_title='Process'
    )

    return fig


def visualizations_tab():
    """Visualizations tab content."""
    st.subheader("📈 Risk Visualizations")

    if not st.session_state.current_client_id:
        return

    client = get_client(st.session_state.current_client_id)
    summary = get_risk_exposure_summary(st.session_state.current_client_id)

    if not summary:
        st.warning("No data to visualize. Please complete assessments first.")
        return

    currency = client.get('currency', 'EUR')

    # Charts layout
    col1, col2 = st.columns(2)

    with col1:
        # Domain breakdown
        fig1 = domain_breakdown_chart(summary, currency)
        if fig1:
            st.plotly_chart(fig1, width="stretch")

        # Top risks
        fig3 = risk_exposure_chart(summary, currency)
        if fig3:
            st.plotly_chart(fig3, width="stretch")

    with col2:
        # Top processes
        fig2 = process_exposure_chart(summary, currency)
        if fig2:
            st.plotly_chart(fig2, width="stretch")

        # Domain totals bar
        if summary['by_domain']:
            df_domains = pd.DataFrame([
                {"Domain": d, "Exposure": e, "Icon": get_domain_icon(d)}
                for d, e in summary['by_domain'].items()
            ])

            fig4 = px.bar(
                df_domains, x='Domain', y='Exposure',
                title='Total Exposure by Domain',
                color='Domain',
                color_discrete_map={d: get_domain_color(d) for d in RISK_DOMAINS.keys()}
            )
            fig4.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig4, width="stretch")

    # Full heatmap
    st.divider()
    fig_heatmap = heatmap_chart(summary, currency)
    if fig_heatmap:
        st.plotly_chart(fig_heatmap, width="stretch")


def detailed_results_tab():
    """Detailed results table."""
    st.subheader("📋 Detailed Results")

    if not st.session_state.current_client_id:
        return

    client = get_client(st.session_state.current_client_id)
    summary = get_risk_exposure_summary(st.session_state.current_client_id)

    if not summary:
        st.warning("No results available.")
        return

    currency = client.get('currency', 'EUR')
    symbol = CURRENCY_SYMBOLS.get(currency, '€')

    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        domain_filter = st.selectbox(
            "Filter by Domain",
            options=["All"] + list(RISK_DOMAINS.keys())
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            options=["Exposure (High to Low)", "Exposure (Low to High)",
                    "Process Name", "Risk Name"]
        )
    with col3:
        min_exposure = st.number_input(
            f"Min Exposure ({symbol})",
            min_value=0,
            value=0,
            step=1000
        )

    # Prepare data
    data = []
    for a in summary['assessments']:
        data.append({
            "Process": a['process_name'],
            "Risk": a['risk_name'],
            "Domain": a['domain'],
            f"Criticality ({symbol}/day)": a['criticality_per_day'],
            "Vulnerability (%)": a['vulnerability'] * 100,
            "Resilience (%)": a['resilience'] * 100,
            "Downtime (days)": a['expected_downtime'],
            "Probability (%)": a['probability'] * 100,
            f"Exposure ({symbol}/yr)": a['exposure']
        })

    df = pd.DataFrame(data)

    # Apply filters
    if domain_filter != "All":
        df = df[df['Domain'] == domain_filter]

    df = df[df[f"Exposure ({symbol}/yr)"] >= min_exposure]

    # Sort
    if sort_by == "Exposure (High to Low)":
        df = df.sort_values(f"Exposure ({symbol}/yr)", ascending=False)
    elif sort_by == "Exposure (Low to High)":
        df = df.sort_values(f"Exposure ({symbol}/yr)", ascending=True)
    elif sort_by == "Process Name":
        df = df.sort_values("Process")
    elif sort_by == "Risk Name":
        df = df.sort_values("Risk")

    # Display
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            f"Criticality ({symbol}/day)": st.column_config.NumberColumn(format=f"{symbol}%d"),
            "Vulnerability (%)": st.column_config.NumberColumn(format="%.0f%%"),
            "Resilience (%)": st.column_config.NumberColumn(format="%.0f%%"),
            "Probability (%)": st.column_config.NumberColumn(format="%.1f%%"),
            f"Exposure ({symbol}/yr)": st.column_config.NumberColumn(format=f"{symbol}%,.0f"),
        }
    )

    # Summary stats
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows Shown", len(df))
    with col2:
        st.metric("Total Exposure (Filtered)", format_currency(df[f"Exposure ({symbol}/yr)"].sum(), currency))
    with col3:
        st.metric("Avg Exposure per Combination", format_currency(df[f"Exposure ({symbol}/yr)"].mean(), currency))


def export_tab():
    """Export functionality."""
    st.subheader("📥 Export Results")

    if not st.session_state.current_client_id:
        return

    client = get_client(st.session_state.current_client_id)
    summary = get_risk_exposure_summary(st.session_state.current_client_id)

    if not summary:
        st.warning("No data to export.")
        return

    currency = client.get('currency', 'EUR')
    symbol = CURRENCY_SYMBOLS.get(currency, '€')

    st.markdown("""
    Export your risk assessment results in various formats.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Excel Export")
        st.markdown("Comprehensive workbook with all data and calculations.")

        if st.button("Generate Excel Report", type="primary", width="stretch"):
            with st.spinner("Generating Excel report..."):
                excel_data = generate_excel_report(client, summary, currency)
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"PRISM_Brain_{client['name'].replace(' ', '_')}_{export_timestamp()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch"
                )

    with col2:
        st.markdown("### 📄 CSV Export")
        st.markdown("Raw data for further analysis.")

        if st.button("Generate CSV", width="stretch"):
            csv_data = generate_csv_report(summary, currency)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"PRISM_Brain_{client['name'].replace(' ', '_')}_{export_timestamp()}.csv",
                mime="text/csv",
                width="stretch"
            )


def generate_excel_report(client, summary, currency):
    """Generate comprehensive Excel report."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    symbol = CURRENCY_SYMBOLS.get(currency, '€')

    wb = Workbook()

    # Styles
    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(bold=True, color="FFFFFF")
    currency_format = f'{symbol}#,##0'
    percent_format = '0.0%'

    # Sheet 1: Dashboard
    ws_dash = wb.active
    ws_dash.title = "Dashboard"

    ws_dash['A1'] = f"PRISM Brain Risk Report - {client['name']}"
    ws_dash['A1'].font = Font(bold=True, size=16)

    ws_dash['A3'] = "Total Annual Risk Exposure:"
    ws_dash['B3'] = summary['total_exposure']
    ws_dash['B3'].number_format = currency_format

    ws_dash['A4'] = "Report Generated:"
    ws_dash['B4'] = datetime.now().strftime("%Y-%m-%d %H:%M")

    ws_dash['A6'] = "Exposure by Domain"
    ws_dash['A6'].font = Font(bold=True)
    row = 7
    for domain, exposure in summary['by_domain'].items():
        ws_dash[f'A{row}'] = domain
        ws_dash[f'B{row}'] = exposure
        ws_dash[f'B{row}'].number_format = currency_format
        row += 1

    # Sheet 2: Detailed Results
    ws_detail = wb.create_sheet("Detailed Results")

    headers = ["Process", "Risk", "Domain", f"Criticality ({symbol}/day)",
               "Vulnerability", "Resilience", "Downtime (days)",
               "Probability", f"Exposure ({symbol}/yr)"]

    for col, header in enumerate(headers, 1):
        cell = ws_detail.cell(row=1, column=col, value=header)
        cell.fill = header_fill
        cell.font = header_font

    for row, a in enumerate(summary['assessments'], 2):
        ws_detail.cell(row=row, column=1, value=a['process_name'])
        ws_detail.cell(row=row, column=2, value=a['risk_name'])
        ws_detail.cell(row=row, column=3, value=a['domain'])
        ws_detail.cell(row=row, column=4, value=a['criticality_per_day']).number_format = currency_format
        ws_detail.cell(row=row, column=5, value=a['vulnerability']).number_format = percent_format
        ws_detail.cell(row=row, column=6, value=a['resilience']).number_format = percent_format
        ws_detail.cell(row=row, column=7, value=a['expected_downtime'])
        ws_detail.cell(row=row, column=8, value=a['probability']).number_format = percent_format
        ws_detail.cell(row=row, column=9, value=a['exposure']).number_format = currency_format

    # Auto-width columns
    for ws in [ws_dash, ws_detail]:
        for col in ws.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            ws.column_dimensions[column].width = min(max_length + 2, 40)

    # Save to bytes
    output = BytesIO()
    wb.save(output)
    output.seek(0)

    return output.getvalue()


def generate_csv_report(summary, currency):
    """Generate CSV report."""
    symbol = CURRENCY_SYMBOLS.get(currency, '€')

    data = []
    for a in summary['assessments']:
        data.append({
            "Process": a['process_name'],
            "Risk": a['risk_name'],
            "Domain": a['domain'],
            "Criticality_per_day": a['criticality_per_day'],
            "Vulnerability": a['vulnerability'],
            "Resilience": a['resilience'],
            "Downtime_days": a['expected_downtime'],
            "Probability": a['probability'],
            "Exposure_annual": a['exposure']
        })

    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')


def main():
    """Main page function."""
    st.title("💰 Results Dashboard")
    st.markdown("View and export your risk exposure analysis.")

    client_selector_sidebar()

    if not st.session_state.current_client_id:
        st.warning("Please select a client to view results")
        if st.button("Go to Client Setup"):
            st.switch_page("pages/1_Client_Setup.py")
        return

    # Executive summary at top
    executive_summary()

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "📈 Visualizations",
        "📋 Detailed Results",
        "📥 Export"
    ])

    with tab1:
        visualizations_tab()

    with tab2:
        detailed_results_tab()

    with tab3:
        export_tab()

    # Navigation
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("← Risk Assessment"):
            st.switch_page("pages/4_Risk_Assessment.py")

    with col3:
        if st.button("🏠 Home"):
            st.switch_page("Welcome.py")


if __name__ == "__main__":
    main()
