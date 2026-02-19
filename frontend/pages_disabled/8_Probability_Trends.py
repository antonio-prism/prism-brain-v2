import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import numpy as np
from utils.theme import inject_prism_theme

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.api_client import (
    check_backend_health,
    fetch_events,
    api_get_trend_data,
    api_get_trend_stats,
    api_get_top_movers,
    api_get_trend_summary,
    api_take_snapshot
)

# Page configuration
st.set_page_config(
    page_title="Probability Trends",
    page_icon="ð",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_prism_theme()

# Title and description
st.title("ð Probability Trends")
st.markdown("""
This dashboard tracks how the probability of key events evolves over time. 
Monitor rising risks, falling risks, and overall trend momentum to make informed decisions.
""")

# Sidebar
with st.sidebar:
    st.header("Trend Tracking")
    st.markdown("""
    **What is this page?**
    
    Probability Trends shows historical probability data for tracked events. 
    Each snapshot captures the current probability assessment, creating a 
    time series that reveals momentum and patterns.
    
    **Key Features:**
    - Track probability changes over time
    - Identify rising and falling risks
    - Compare events by trend velocity
    - Take manual snapshots
    """)

# Backend connectivity check
backend_health = check_backend_health()
if backend_health:
    st.success("â Backend Connected")
    is_connected = True
else:
    st.warning("â ï¸ Backend Disconnected - Limited functionality available")
    is_connected = False

# ============================================================================
# MAIN CONTENT (only if backend is connected)
# ============================================================================

if is_connected:
    
    # ========================================================================
    # SECTION A: TREND SUMMARY (Top Metrics)
    # ========================================================================
    st.header("Summary Metrics")
    
    try:
        summary_data = api_get_trend_summary()
        
        if summary_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Events Tracked",
                    summary_data.get("total_events", 0),
                    help="Total number of events with probability tracking"
                )
            
            with col2:
                rising_count = summary_data.get("rising_count", 0)
                st.metric(
                    "Rising Risks",
                    rising_count,
                    delta=f"+{rising_count}" if rising_count > 0 else "0",
                    delta_color="inverse",
                    help="Events with increasing probability"
                )
            
            with col3:
                falling_count = summary_data.get("falling_count", 0)
                st.metric(
                    "Falling Risks",
                    falling_count,
                    delta=f"+{falling_count}" if falling_count > 0 else "0",
                    delta_color="normal",
                    help="Events with decreasing probability"
                )
            
            with col4:
                stable_count = summary_data.get("stable_count", 0)
                st.metric(
                    "Stable Trends",
                    stable_count,
                    help="Events with stable probability"
                )
            
            # Show last snapshot time in sidebar
            if "last_snapshot" in summary_data:
                st.sidebar.metric(
                    "Last Snapshot",
                    summary_data["last_snapshot"].split("T")[0] if "T" in summary_data["last_snapshot"] else summary_data["last_snapshot"]
                )
    except Exception as e:
        st.error(f"Error loading trend summary: {str(e)}")
    
    # ========================================================================
    # SECTION B: TOP MOVERS
    # ========================================================================
    st.header("Top Movers (Last 7 Days)")
    
    try:
        top_movers = api_get_top_movers(days=7, limit=10)
        
        if top_movers and len(top_movers) > 0:
            # Convert to DataFrame for better display
            df_movers = pd.DataFrame(top_movers)
            
            # Style the dataframe with color coding
            def color_trend(val):
                if isinstance(val, str):
                    if "â" in val or "rising" in val.lower():
                        return "color: #ff4b4b"
                    elif "â" in val or "falling" in val.lower():
                        return "color: #09ab3b"
                return ""
            
            # Display as table with styling
            st.dataframe(
                df_movers,
                width="stretch",
                hide_index=True,
                column_config={
                    "event_name": st.column_config.TextColumn("Event"),
                    "current_probability": st.column_config.NumberColumn("Current Prob", format="%.1f%%"),
                    "change_7d": st.column_config.NumberColumn("7-Day Change", format="%.1f%"),
                    "trend_direction": st.column_config.TextColumn("Trend"),
                    "change_percentage": st.column_config.NumberColumn("Change %", format="%.2f%"),
                }
            )
        else:
            st.info("No top movers data available yet. Take a snapshot to start tracking.")
    except Exception as e:
        st.error(f"Error loading top movers: {str(e)}")
    
    # ========================================================================
    # SECTION C: INDIVIDUAL EVENT TREND ANALYSIS
    # ========================================================================
    st.header("Event Trend Analysis")
    
    # Load events with caching
    @st.cache_data(ttl=300)
    def get_events_list():
        try:
            events = fetch_events()
            return events if events else []
        except Exception:
            return []
    
    events = get_events_list()
    
    if events and len(events) > 0:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Event selector
            event_names = [event.get("name", event.get("id", "Unknown")) for event in events]
            selected_event_name = st.selectbox(
                "Select Event",
                event_names,
                key="event_selector"
            )
            
            # Find the selected event's ID
            selected_event_id = None
            for event in events:
                if event.get("name", event.get("id")) == selected_event_name:
                    selected_event_id = event.get("id")
                    break
        
        with col2:
            # Time range selector
            time_range = st.selectbox(
                "Time Range",
                [7, 30, 90, 365],
                format_func=lambda x: f"{x} Days",
                key="time_range"
            )
        
        if selected_event_id:
            try:
                # Fetch trend data
                trend_data = api_get_trend_data(selected_event_id, days=time_range)
                trend_stats = api_get_trend_stats(selected_event_id)
                
                if trend_data and len(trend_data) > 0:
                    # Convert to DataFrame
                    df_trend = pd.DataFrame(trend_data)
                    
                    if "snapshot_date" in df_trend.columns and "probability_pct" in df_trend.columns:
                        # Ensure date column is datetime
                        df_trend["snapshot_date"] = pd.to_datetime(df_trend["snapshot_date"])
                        df_trend = df_trend.sort_values("snapshot_date")
                        
                        # Create trend chart
                        fig = go.Figure()
                        
                        # Add confidence band if available
                        if "confidence_score" in df_trend.columns:
                            fig.add_trace(go.Scatter(
                                x=df_trend["snapshot_date"],
                                y=df_trend["probability_pct"] + (df_trend.get("confidence_score", 0) * 5),
                                fill=None,
                                mode="lines",
                                line_color="rgba(0,0,0,0)",
                                showlegend=False
                            ))
                            fig.add_trace(go.Scatter(
                                x=df_trend["snapshot_date"],
                                y=df_trend["probability_pct"] - (df_trend.get("confidence_score", 0) * 5),
                                fill="tonexty",
                                mode="lines",
                                name="Confidence Band",
                                line_color="rgba(0,0,0,0)",
                                fillcolor="rgba(0,100,200,0.1)"
                            ))
                        
                        # Add main probability line
                        fig.add_trace(go.Scatter(
                            x=df_trend["snapshot_date"],
                            y=df_trend["probability_pct"],
                            mode="lines+markers",
                            name="Probability",
                            line=dict(color="#FF6B6B", width=3),
                            marker=dict(size=6)
                        ))
                        
                        # Add trend line
                        if len(df_trend) >= 2:
                            z = np.polyfit(range(len(df_trend)), df_trend["probability_pct"].values, 1)
                            p = np.poly1d(z)
                            trend_line = p(range(len(df_trend)))
                            fig.add_trace(go.Scatter(
                                x=df_trend["snapshot_date"],
                                y=trend_line,
                                mode="lines",
                                name="Trend Line",
                                line=dict(color="rgba(100,100,100,0.5)", width=2, dash="dash")
                            ))
                        
                        fig.update_layout(
                            title=f"Probability Trend: {selected_event_name}",
                            xaxis_title="Date",
                            yaxis_title="Probability (%)",
                            hovermode="x unified",
                            height=450,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, width="stretch")
                        
                        # Show statistics below chart
                        st.subheader("Trend Statistics")
                        
                        if trend_stats:
                            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                            
                            with stat_col1:
                                st.metric(
                                    "Current Probability",
                                    f"{trend_stats.get('current_probability', 0):.1f}%"
                                )
                            
                            with stat_col2:
                                st.metric(
                                    "30-Day Average",
                                    f"{trend_stats.get('avg_30d', 0):.1f}%"
                                )
                            
                            with stat_col3:
                                change_7d = trend_stats.get("change_7d", 0)
                                st.metric(
                                    "7-Day Change",
                                    f"{change_7d:+.1f}%",
                                    delta_color="inverse" if change_7d > 0 else "normal"
                                )
                            
                            with stat_col4:
                                change_30d = trend_stats.get("change_30d", 0)
                                st.metric(
                                    "30-Day Change",
                                    f"{change_30d:+.1f}%",
                                    delta_color="inverse" if change_30d > 0 else "normal"
                                )
                            
                            # Additional stats in expandable section
                            with st.expander("Advanced Statistics"):
                                adv_col1, adv_col2, adv_col3 = st.columns(3)
                                
                                with adv_col1:
                                    st.metric(
                                        "90-Day Average",
                                        f"{trend_stats.get('avg_90d', 0):.1f}%"
                                    )
                                
                                with adv_col2:
                                    st.metric(
                                        "Min in Period",
                                        f"{trend_stats.get('min_probability', 0):.1f}%"
                                    )
                                
                                with adv_col3:
                                    st.metric(
                                        "Max in Period",
                                        f"{trend_stats.get('max_probability', 0):.1f}%"
                                    )
                    else:
                        st.warning("Trend data format not recognized.")
                else:
                    st.info("No trend data available for this event. Check back after snapshots are taken.")
            except Exception as e:
                st.error(f"Error loading trend data: {str(e)}")
    else:
        st.warning("No events available. Check the Data Sources page to add events.")
    
    # ========================================================================
    # SECTION D: SNAPSHOT MANAGEMENT
    # ========================================================================
    st.header("Snapshot Management")
    
    st.markdown("""
    Snapshots capture the current probability state at a point in time. 
    Regular snapshots create the historical data needed for trend analysis.
    """)
    
    col_snapshot1, col_snapshot2 = st.columns([1, 3])
    
    with col_snapshot1:
        if st.button("ð¸ Take Snapshot Now", width="stretch"):
            try:
                with st.spinner("Taking snapshot..."):
                    result = api_take_snapshot()
                    if result:
                        st.success("â Snapshot taken successfully!")
                        st.balloons()
                    else:
                        st.error("Failed to take snapshot. Please check the backend logs.")
            except Exception as e:
                st.error(f"Error taking snapshot: {str(e)}")
    
    with col_snapshot2:
        st.info("""
        **About Snapshots:**
        - Snapshots are typically taken automatically on a schedule
        - Use this button to take a manual snapshot when needed
        - Snapshots are required to generate probability trend data
        - The more snapshots, the better the trend analysis
        """)

# ============================================================================
# FALLBACK MODE: Backend Not Connected
# ============================================================================

else:
    st.error("ð¡ Backend Connection Required")
    st.markdown("""
    The Probability Trends dashboard requires a connection to the PRISM Brain backend 
    to fetch and display trend data.
    
    **What's needed:**
    - Active backend service running
    - Database with event data and probability snapshots
    - API endpoints for trend data retrieval
    
    **What you can do:**
    1. Check the backend service status
    2. Verify database connectivity
    3. Visit the **Data Sources** page to configure API connections
    4. Return here once the backend is running
    
    **Troubleshooting:**
    - Check if the backend service is running
    - Verify API endpoint configuration
    - Check network connectivity
    - Review backend logs for errors
    """)
    
    if st.button("ð Retry Connection"):
        st.rerun()
