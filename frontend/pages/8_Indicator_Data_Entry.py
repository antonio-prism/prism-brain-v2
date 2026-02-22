"""
PRISM Brain - Indicator Data Entry (Phase II Dynamic Scoring)
=============================================================
Two tabs for entering indicator data that feeds into dynamic probability scoring:

Tab 1 (Tier 2): Annual research indicators from commercial reports
  - Grouped by data source (Gartner, IATA, Drewry, etc.)
  - Analyst enters values once per year when reports are published

Tab 2 (Tier 3): Client-specific operational data
  - Filtered to the client's selected risks
  - Grouped by risk family
  - Can be entered directly or via Excel questionnaire
"""

import streamlit as st
import sys
import json
from pathlib import Path
from collections import defaultdict

from utils.theme import inject_prism_theme

APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR.parent))

from modules.database import is_backend_online

st.set_page_config(
    page_title="Indicator Data Entry | PRISM Brain",
    page_icon=":material/data_thresholding:",
    layout="wide",
)

inject_prism_theme()


# ── Direct engine access (works with or without backend) ──────────────────

def _load_indicator_sources():
    """Load all indicator sources grouped by data source."""
    try:
        from prism_engine.method_c_loader import get_full_research
        overrides_path = APP_DIR.parent / "prism_engine" / "data" / "method_c_overrides.json"
        if not overrides_path.exists():
            return {}
        with open(overrides_path, "r", encoding="utf-8") as f:
            overrides = json.load(f).get("overrides", {})

        sources = defaultdict(list)
        for event_id in overrides:
            research = get_full_research(event_id)
            if not research:
                continue
            sf = research.get("scoring_functions", {})
            for sub_key in ("p_preconditions", "p_trigger", "p_implementation"):
                sub = sf.get(sub_key, {})
                for ind in sub.get("input_indicators", []):
                    src = ind.get("data_source", "unknown")
                    sources[src].append({
                        "event_id": event_id,
                        "event_name": overrides.get(event_id, {}).get("event_name", event_id),
                        "sub_prob": sub_key,
                        "indicator_id": ind.get("indicator_id", ""),
                        "name": ind.get("name", ""),
                        "metric": ind.get("metric", ""),
                        "normalization": ind.get("normalization", ""),
                        "normalization_params": ind.get("normalization_params", {}),
                        "weight": ind.get("weight", 0),
                    })
        return dict(sources)
    except Exception as e:
        st.error(f"Cannot load indicator sources: {e}")
        return {}


def _get_indicator_value(event_id, sub_prob, indicator_id, client_id=None):
    """Get current indicator value from the store."""
    try:
        from prism_engine.indicator_store import get_indicator_value
        return get_indicator_value(event_id, sub_prob, indicator_id, client_id=client_id)
    except Exception:
        return None


def _save_indicators(indicators, client_id=None):
    """Save indicator values to the store."""
    try:
        from prism_engine.indicator_store import (
            set_indicator_value, save_global_store, save_client_store
        )
        saved = 0
        for ind in indicators:
            set_indicator_value(
                event_id=ind["event_id"],
                sub_prob=ind["sub_prob"],
                indicator_id=ind["indicator_id"],
                value=float(ind["value"]),
                tier=int(ind.get("tier", 2)),
                source=ind.get("source", "manual"),
                unit=ind.get("unit", ""),
                client_id=client_id,
            )
            saved += 1
        save_global_store()
        if client_id:
            save_client_store(client_id)
        return saved
    except Exception as e:
        st.error(f"Error saving indicators: {e}")
        return 0


def _get_coverage_summary(client_id=None):
    """Get coverage summary across all events."""
    try:
        from prism_engine.indicator_store import get_coverage_for_event, get_freshness_summary
        from prism_engine.method_c_loader import get_full_research

        overrides_path = APP_DIR.parent / "prism_engine" / "data" / "method_c_overrides.json"
        if not overrides_path.exists():
            return {"total": 0, "with_data": 0}
        with open(overrides_path, "r", encoding="utf-8") as f:
            overrides = json.load(f).get("overrides", {})

        total_indicators = 0
        available_indicators = 0
        for event_id in overrides:
            research = get_full_research(event_id)
            if research and research.get("scoring_functions"):
                cov = get_coverage_for_event(
                    event_id, research["scoring_functions"], client_id=client_id
                )
                total_indicators += cov["total_indicators"]
                available_indicators += cov["available"]

        freshness = get_freshness_summary(client_id)
        return {
            "total_indicators": total_indicators,
            "available": available_indicators,
            "coverage_pct": round(available_indicators / total_indicators * 100, 1) if total_indicators > 0 else 0,
            "freshness": freshness,
        }
    except Exception:
        return {"total_indicators": 0, "available": 0, "coverage_pct": 0}


# ── Classify sources into tiers ────────────────────────────────────────────

# Sources that are auto-fetched (Tier 1) — shown as read-only
TIER_1_SOURCES = {"EIA", "FRED", "NOAA", "World Bank", "USGS", "NVD", "CISA KEV", "UCDP"}

# Sources that are client-internal (Tier 3)
TIER_3_KEYWORDS = {
    "Operations", "Engineering", "Logistics", "Procurement", "IT", "QA",
    "Finance", "Purchasing", "Assessment", "CMMS", "Monitoring", "Maintenance",
    "Inventory", "Security", "Fleet", "Facilities", "Sales", "Planning",
    "Reliability", "Quality", "Grid ops", "Audit", "CRM", "SQM", "R&D",
    "Occupational health", "Labor",
}


def _classify_tier(source_name):
    """Classify a data source name into a tier."""
    if source_name in TIER_1_SOURCES:
        return 1
    if source_name in TIER_3_KEYWORDS:
        return 3
    return 2  # Default: commercial research report


# ── Main page ──────────────────────────────────────────────────────────────

st.title("Indicator Data Entry")
st.caption("Enter indicator values for dynamic probability scoring of Method C events")

# Load data
sources = _load_indicator_sources()
if not sources:
    st.warning("No indicator sources found. Make sure the Method C research has been integrated.")
    st.stop()

# Calculate coverage
coverage = _get_coverage_summary()

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Indicators", f"{coverage.get('total_indicators', 0):,}")
with col2:
    st.metric("Values Entered", f"{coverage.get('available', 0):,}")
with col3:
    st.metric("Coverage", f"{coverage.get('coverage_pct', 0):.1f}%")
with col4:
    freshness = coverage.get("freshness", {})
    live = freshness.get("live", 0)
    stale = freshness.get("stale", 0)
    st.metric("Fresh / Stale", f"{live} / {stale}")

st.divider()

# Tabs
tab1, tab2 = st.tabs(["Tier 2: Research Reports", "Tier 3: Client Data"])


# ── Tab 1: Tier 2 Research Reports ────────────────────────────────────────

with tab1:
    st.markdown("""
    Enter indicator values from commercial research reports (Gartner, IATA, Drewry, SEMI, etc.).
    These values are shared across all clients and typically updated once per year.
    """)

    # Group sources by tier, filter to Tier 2
    tier2_sources = {src: inds for src, inds in sources.items() if _classify_tier(src) == 2}

    if not tier2_sources:
        st.info("No Tier 2 sources found.")
    else:
        # Sort by indicator count (most important first)
        sorted_sources = sorted(tier2_sources.items(), key=lambda x: -len(x[1]))

        # Source selector
        source_names = [f"{src} ({len(inds)} indicators)" for src, inds in sorted_sources]
        selected_idx = st.selectbox(
            "Select data source",
            range(len(sorted_sources)),
            format_func=lambda i: source_names[i],
            key="tier2_source",
        )

        source_name, source_indicators = sorted_sources[selected_idx]

        st.subheader(f"Source: {source_name}")

        # Deduplicate indicators (same indicator_id may appear in multiple events)
        unique_indicators = {}
        for ind in source_indicators:
            ind_id = ind["indicator_id"]
            if ind_id not in unique_indicators:
                unique_indicators[ind_id] = {
                    **ind,
                    "events": [{"event_id": ind["event_id"], "event_name": ind["event_name"], "sub_prob": ind["sub_prob"]}],
                }
            else:
                unique_indicators[ind_id]["events"].append({
                    "event_id": ind["event_id"],
                    "event_name": ind["event_name"],
                    "sub_prob": ind["sub_prob"],
                })

        # Build the entry form
        to_save = []
        for ind_id, ind in unique_indicators.items():
            params = ind.get("normalization_params", {})
            min_val = params.get("min", 0)
            max_val = params.get("max", 100)
            metric = ind.get("metric", "")

            # Get current value
            first_event = ind["events"][0]
            current_val = _get_indicator_value(
                first_event["event_id"], first_event["sub_prob"], ind_id
            )

            col_name, col_range, col_value, col_events = st.columns([3, 1.5, 1.5, 3])

            with col_name:
                st.text(ind.get("name", ind_id))
                st.caption(f"ID: {ind_id}")

            with col_range:
                st.text(f"{min_val} - {max_val}")
                if metric:
                    st.caption(metric)

            with col_value:
                new_val = st.number_input(
                    f"Value for {ind_id}",
                    min_value=float(min_val) if isinstance(min_val, (int, float)) else None,
                    max_value=float(max_val) * 2 if isinstance(max_val, (int, float)) else None,
                    value=float(current_val) if current_val is not None else None,
                    key=f"tier2_{source_name}_{ind_id}",
                    label_visibility="collapsed",
                    placeholder="Enter value",
                )

            with col_events:
                event_names = [f"{e['event_id']}" for e in ind["events"]]
                st.caption(f"Affects: {', '.join(event_names[:3])}")
                if len(event_names) > 3:
                    st.caption(f"  +{len(event_names) - 3} more")

            # Track values that changed
            if new_val is not None and new_val != current_val:
                for evt in ind["events"]:
                    to_save.append({
                        "event_id": evt["event_id"],
                        "sub_prob": evt["sub_prob"],
                        "indicator_id": ind_id,
                        "value": new_val,
                        "tier": 2,
                        "source": source_name,
                        "unit": metric,
                    })

        st.divider()

        col_save, col_count = st.columns([1, 3])
        with col_save:
            if st.button(f"Save {source_name} Data", type="primary", disabled=len(to_save) == 0):
                saved = _save_indicators(to_save)
                st.success(f"Saved {saved} indicator values")
                st.rerun()
        with col_count:
            if to_save:
                st.info(f"{len(to_save)} values to save")
            else:
                st.caption("No changes to save")


# ── Tab 2: Tier 3 Client Data ────────────────────────────────────────────

with tab2:
    st.markdown("""
    Enter client-specific operational data (procurement, engineering, IT, etc.).
    These values are stored per client and affect only that client's risk assessment.
    """)

    # Client selector
    try:
        from modules.api_client import api_get_all_clients
        clients = api_get_all_clients()
    except Exception:
        clients = None

    if not clients:
        st.info("No clients found. Create a client in the Client Setup page first.")
    else:
        client_options = {c["name"]: str(c["id"]) for c in clients}
        selected_client_name = st.selectbox("Select client", list(client_options.keys()), key="tier3_client")
        client_id = client_options[selected_client_name]

        # Get client's selected risks
        try:
            from modules.api_client import api_get_risks
            client_risks = api_get_risks(int(client_id), prioritized_only=True)
        except Exception:
            client_risks = None

        if not client_risks:
            st.info("No prioritized risks found for this client. Select risks in the Risk Selection page first.")
        else:
            # Filter sources to Tier 3 only, matching client's risks
            risk_ids = {r["risk_id"] for r in client_risks if "risk_id" in r}

            tier3_sources = {src: inds for src, inds in sources.items() if _classify_tier(src) == 3}

            # Filter indicators to only those affecting client's selected risks
            filtered_indicators = []
            for src, inds in tier3_sources.items():
                for ind in inds:
                    if ind["event_id"] in risk_ids:
                        filtered_indicators.append({**ind, "source": src})

            if not filtered_indicators:
                st.info("No operational indicators found for this client's selected risks.")
            else:
                # Group by risk family
                by_family = defaultdict(list)
                for ind in filtered_indicators:
                    family = ind["event_id"].rsplit("-", 1)[0]  # e.g., OPS-SUP from OPS-SUP-001
                    by_family[family].append(ind)

                st.markdown(f"**{len(filtered_indicators)} indicators** across "
                           f"**{len(by_family)} families** for {len(risk_ids)} selected risks")

                to_save_t3 = []
                for family, family_inds in sorted(by_family.items()):
                    with st.expander(f"{family} ({len(family_inds)} indicators)", expanded=False):
                        for ind in family_inds:
                            params = ind.get("normalization_params", {})
                            min_val = params.get("min", 0)
                            max_val = params.get("max", 100)
                            metric = ind.get("metric", "")
                            ind_id = ind["indicator_id"]

                            current_val = _get_indicator_value(
                                ind["event_id"], ind["sub_prob"], ind_id, client_id=client_id
                            )

                            col_a, col_b, col_c, col_d = st.columns([3, 1.5, 1.5, 2])

                            with col_a:
                                st.text(ind.get("name", ind_id))
                                st.caption(f"{ind['event_id']} / {ind['sub_prob']}")

                            with col_b:
                                st.text(f"{min_val} - {max_val}")
                                if metric:
                                    st.caption(metric)

                            with col_c:
                                status = "Set" if current_val is not None else "Missing"
                                new_val = st.number_input(
                                    f"Value for {ind_id}",
                                    value=float(current_val) if current_val is not None else None,
                                    key=f"tier3_{client_id}_{ind['event_id']}_{ind_id}",
                                    label_visibility="collapsed",
                                    placeholder="Enter",
                                )

                            with col_d:
                                if current_val is not None:
                                    st.success(status)
                                else:
                                    st.warning(status)

                            if new_val is not None and new_val != current_val:
                                to_save_t3.append({
                                    "event_id": ind["event_id"],
                                    "sub_prob": ind["sub_prob"],
                                    "indicator_id": ind_id,
                                    "value": new_val,
                                    "tier": 3,
                                    "source": ind.get("source", "client"),
                                    "unit": metric,
                                })

                st.divider()

                col_s, col_c = st.columns([1, 3])
                with col_s:
                    if st.button("Save Client Data", type="primary", disabled=len(to_save_t3) == 0):
                        saved = _save_indicators(to_save_t3, client_id=client_id)
                        st.success(f"Saved {saved} client indicator values")
                        st.rerun()
                with col_c:
                    if to_save_t3:
                        st.info(f"{len(to_save_t3)} values to save")
                    else:
                        st.caption("No changes to save")

                # ── Excel Export/Import ───────────────────────────────────────
                st.divider()
                st.subheader("Excel Questionnaire")
                st.markdown("Download a pre-filled questionnaire to share with client teams, then upload the completed file.")

                col_dl, col_ul = st.columns(2)

                with col_dl:
                    # Generate Excel questionnaire
                    try:
                        import pandas as pd
                        from io import BytesIO

                        rows = []
                        for ind in filtered_indicators:
                            params = ind.get("normalization_params", {})
                            current = _get_indicator_value(
                                ind["event_id"], ind["sub_prob"],
                                ind["indicator_id"], client_id=client_id
                            )
                            rows.append({
                                "Event ID": ind["event_id"],
                                "Event Name": ind.get("event_name", ""),
                                "Sub-Probability": ind["sub_prob"],
                                "Indicator ID": ind["indicator_id"],
                                "Indicator Name": ind.get("name", ""),
                                "Data Source": ind.get("source", ""),
                                "Metric": ind.get("metric", ""),
                                "Min": params.get("min", ""),
                                "Max": params.get("max", ""),
                                "Current Value": current if current is not None else "",
                                "New Value": "",
                            })

                        if rows:
                            df_export = pd.DataFrame(rows)
                            buffer = BytesIO()
                            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                                df_export.to_excel(writer, sheet_name="Indicators", index=False)
                            buffer.seek(0)

                            safe_name = selected_client_name.replace(" ", "_")
                            st.download_button(
                                label="Download Questionnaire (.xlsx)",
                                data=buffer,
                                file_name=f"PRISM_Indicators_{safe_name}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                    except ImportError:
                        st.warning("pandas/openpyxl not installed — Excel export unavailable")

                with col_ul:
                    uploaded = st.file_uploader(
                        "Upload completed questionnaire",
                        type=["xlsx", "xls"],
                        key=f"tier3_upload_{client_id}",
                    )

                    if uploaded is not None:
                        try:
                            import pandas as pd
                            df_import = pd.read_excel(uploaded, sheet_name="Indicators")

                            import_indicators = []
                            for _, row in df_import.iterrows():
                                new_val = row.get("New Value")
                                if pd.notna(new_val) and str(new_val).strip() != "":
                                    try:
                                        import_indicators.append({
                                            "event_id": str(row["Event ID"]),
                                            "sub_prob": str(row["Sub-Probability"]),
                                            "indicator_id": str(row["Indicator ID"]),
                                            "value": float(new_val),
                                            "tier": 3,
                                            "source": str(row.get("Data Source", "excel_import")),
                                            "unit": str(row.get("Metric", "")),
                                        })
                                    except (ValueError, TypeError):
                                        pass

                            if import_indicators:
                                if st.button(f"Import {len(import_indicators)} values", type="primary"):
                                    saved = _save_indicators(import_indicators, client_id=client_id)
                                    st.success(f"Imported {saved} indicator values from Excel")
                                    st.rerun()
                            else:
                                st.info("No 'New Value' entries found in the uploaded file")
                        except Exception as e:
                            st.error(f"Error reading Excel file: {e}")
