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
    fetch_events,
    api_get_alerts,
    api_create_alert,
    api_update_alert,
    api_delete_alert,
    api_check_alerts,
    api_get_triggered_alerts
)

# Page configuration
st.set_page_config(
    page_title="Alert Management",
    page_icon="ð¨",
    layout="wide"
)

inject_prism_theme()

# Initialize session state
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'editing_alert_id' not in st.session_state:
    st.session_state.editing_alert_id = None
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

# Page title and description
st.title("ð¨ Alert Management")
st.write("Manage probability threshold alerts for risk events")
st.divider()

# Backend connectivity check
backend_status = check_backend_health()
if not backend_status:
    st.error("â ï¸ Backend service is unavailable. Operating in fallback mode.")
    st.info("Alert management features require a running backend service.")
    st.stop()

# Sidebar: Alert System Controls
with st.sidebar:
    st.header("Alert System")
    
    # Check alerts now button
    if st.button("ð Check Alerts Now", width="stretch"):
        result = api_check_alerts()
        if result:
            st.success(f"â Checked {result.get('alerts_checked', 0)} alerts")
            triggered = result.get('triggered_alerts', [])
            if triggered:
                st.warning(f"â ï¸ {len(triggered)} alert(s) triggered!")
                for alert in triggered[:5]:
                    st.write(f"â¢ {alert.get('alert_name', 'Unknown')}: {alert.get('current_value', 'N/A')}")
                if len(triggered) > 5:
                    st.caption(f"... and {len(triggered) - 5} more")
        else:
            st.error("Failed to check alerts")
    
    st.divider()
    
    # Quick stats
    st.subheader("Quick Stats")
    try:
        active_alerts = api_get_alerts(active_only=True)
        st.metric("Active Alerts", len(active_alerts) if active_alerts else 0)
        
        triggered_this_week = api_get_triggered_alerts(days=7)
        st.metric("Triggered (7 days)", len(triggered_this_week) if triggered_this_week else 0)
    except:
        st.write("Stats unavailable")
    
    st.divider()
    
    # System explanation
    st.subheader("ð How Alerts Work")
    st.caption("""
    **Alert Directions:**
    - **ABOVE**: Triggers when probability exceeds threshold
    - **BELOW**: Triggers when probability drops below threshold
    - **CHANGE**: Triggers when probability changes by more than threshold points
    
    **Severity Levels:**
    - ð´ HIGH: Immediate action required
    - ð  MEDIUM: Monitor closely
    - ðµ LOW: Informational
    """)

# Main content tabs
tab1, tab2, tab3 = st.tabs(["ð Active Alerts", "â Create New Alert", "ð Alert History"])

# TAB 1: Active Alerts
with tab1:
    st.subheader("Active Alerts")
    
    try:
        alerts = api_get_alerts(active_only=True)
        
        if not alerts or len(alerts) == 0:
            st.info("No active alerts yet. Create one to get started!")
        else:
            # Display alerts as cards
            for alert in alerts:
                severity = alert.get('severity', 'LOW')
                severity_color = {
                    'HIGH': 'ð´',
                    'MEDIUM': 'ð ',
                    'LOW': 'ðµ'
                }.get(severity, 'âª')
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(
                        f"### {severity_color} {alert.get('alert_name', 'Unknown Alert')}"
                    )
                    
                    col_details1, col_details2 = st.columns(2)
                    with col_details1:
                        st.caption(f"**Event ID:** {alert.get('event_id', 'N/A')}")
                        st.caption(f"**Threshold:** {alert.get('threshold', 'N/A')}%")
                    
                    with col_details2:
                        st.caption(f"**Direction:** {alert.get('direction', 'ABOVE')}")
                        last_triggered = alert.get('last_triggered', 'Never')
                        st.caption(f"**Last Triggered:** {last_triggered}")
                
                with col2:
                    if st.button("âï¸ Edit", key=f"edit_{alert.get('id', '')}"):
                        st.session_state.edit_mode = True
                        st.session_state.editing_alert_id = alert.get('id')
                        st.rerun()
                
                with col3:
                    if st.button("ðï¸ Delete", key=f"delete_{alert.get('id', '')}"):
                        if st.session_state.get(f"confirm_delete_{alert.get('id', '')}"):
                            result = api_delete_alert(alert.get('id'))
                            if result:
                                st.success("Alert deleted successfully")
                                st.rerun()
                            else:
                                st.error("Failed to delete alert")
                        else:
                            st.session_state[f"confirm_delete_{alert.get('id', '')}"] = True
                            st.warning("Click again to confirm deletion")
                
                st.divider()
    
    except Exception as e:
        st.error(f"Error loading alerts: {str(e)}")

# TAB 2: Create New Alert
with tab2:
    st.subheader("Create New Alert")
    
    try:
        # Get available events for selection
        events = fetch_events()
        if not events:
            st.warning("No events available. Please add events first.")
        else:
            event_options = {f"{e.get('event_id', '')} - {e.get('name', '')}": e for e in events}
            
            with st.form("create_alert_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    alert_name = st.text_input(
                        "Alert Name",
                        placeholder="e.g., High Risk Event Alert"
                    )
                
                with col2:
                    selected_event_str = st.selectbox(
                        "Select Event",
                        options=list(event_options.keys())
                    )
                
                selected_event = event_options[selected_event_str]
                
                # Display current event info
                if selected_event:
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("Event ID", selected_event.get('event_id', 'N/A'))
                    with col_info2:
                        st.metric("Current Probability", f"{selected_event.get('probability', 0):.1f}%")
                    with col_info3:
                        st.metric("Status", selected_event.get('status', 'N/A'))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    threshold = st.number_input(
                        "Threshold (%)",
                        min_value=0.0,
                        max_value=100.0,
                        step=0.1,
                        value=50.0
                    )
                
                with col2:
                    direction = st.selectbox(
                        "Alert Direction",
                        options=['ABOVE', 'BELOW', 'CHANGE'],
                        help="ABOVE: Alert when probability exceeds threshold\nBELOW: Alert when probability drops below threshold\nCHANGE: Alert when probability changes by more than threshold points"
                    )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    severity = st.selectbox(
                        "Severity Level",
                        options=['HIGH', 'MEDIUM', 'LOW'],
                        help="HIGH: Immediate action required\nMEDIUM: Monitor closely\nLOW: Informational"
                    )
                
                with col2:
                    notification_email = st.text_input(
                        "Notification Email (optional)",
                        placeholder="your.email@example.com"
                    )
                
                # Submit button
                if st.form_submit_button("ð¤ Create Alert", width="stretch"):
                    if not alert_name:
                        st.error("Please enter an alert name")
                    else:
                        result = api_create_alert(
                            alert_name=alert_name,
                            event_id=selected_event.get('event_id'),
                            threshold=threshold,
                            direction=direction,
                            severity=severity,
                            notification_email=notification_email if notification_email else None
                        )
                        
                        if result:
                            st.success(f"â Alert '{alert_name}' created successfully!")
                            st.session_state.form_submitted = True
                        else:
                            st.error("Failed to create alert. Please try again.")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# TAB 3: Alert History
with tab3:
    st.subheader("Alert History")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        days_range = st.selectbox(
            "Time Range",
            options=[7, 30, 90],
            format_func=lambda x: f"Last {x} days"
        )
    
    with col2:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=['HIGH', 'MEDIUM', 'LOW'],
            default=['HIGH', 'MEDIUM', 'LOW']
        )
    
    try:
        triggered_alerts = api_get_triggered_alerts(days=days_range)
        
        if not triggered_alerts or len(triggered_alerts) == 0:
            st.info(f"No triggered alerts in the last {days_range} days")
        else:
            # Filter by severity
            filtered_alerts = [
                a for a in triggered_alerts 
                if a.get('severity', 'LOW') in severity_filter
            ]
            
            if not filtered_alerts:
                st.info("No alerts match the selected severity filters")
            else:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Triggered", len(filtered_alerts))
                
                with col2:
                    high_count = len([a for a in filtered_alerts if a.get('severity') == 'HIGH'])
                    st.metric("HIGH Severity", high_count)
                
                with col3:
                    medium_count = len([a for a in filtered_alerts if a.get('severity') == 'MEDIUM'])
                    st.metric("MEDIUM Severity", medium_count)
                
                with col4:
                    low_count = len([a for a in filtered_alerts if a.get('severity') == 'LOW'])
                    st.metric("LOW Severity", low_count)
                
                st.divider()
                
                # Timeline view
                st.subheader("Timeline")
                
                for alert in sorted(filtered_alerts, key=lambda x: x.get('timestamp', ''), reverse=True):
                    severity = alert.get('severity', 'LOW')
                    severity_color = {
                        'HIGH': 'ð´',
                        'MEDIUM': 'ð ',
                        'LOW': 'ðµ'
                    }.get(severity, 'âª')
                    
                    with st.expander(
                        f"{severity_color} {alert.get('alert_name', 'Unknown')} â¢ {alert.get('timestamp', 'N/A')}"
                    ):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Alert Name:** {alert.get('alert_name', 'N/A')}")
                            st.write(f"**Event ID:** {alert.get('event_id', 'N/A')}")
                            st.write(f"**Triggered Value:** {alert.get('triggered_value', 'N/A')}%")
                        
                        with col2:
                            st.write(f"**Timestamp:** {alert.get('timestamp', 'N/A')}")
                            st.write(f"**Direction:** {alert.get('direction', 'N/A')}")
                            st.write(f"**Threshold:** {alert.get('threshold', 'N/A')}%")
    
    except Exception as e:
        st.error(f"Error loading alert history: {str(e)}")

st.divider()
st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
