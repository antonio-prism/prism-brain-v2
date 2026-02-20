"""
PRISM Brain - Process Criticality Module
=========================================
Select business processes and set criticality values.
Processes are organised by Scope (A–D) → Macro-process (1–32) → Sub-process.
Includes import/export functionality via Excel templates.
"""

import streamlit as st
import pandas as pd
import io
import sys
from pathlib import Path
from utils.theme import inject_prism_theme

# Add app directory to path
APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))

from utils.constants import (
    CURRENCY_SYMBOLS,
    PROCESS_SCOPES,
    PROCESS_SCOPE_MAP,
)
from utils.helpers import (
    load_process_framework,
    get_processes_by_level,
    get_process_children,
    get_processes_by_scope,
    format_currency,
    calculate_default_criticality,
)
from modules.database import (
    get_all_clients,
    get_client,
    add_client_process,
    get_client_processes,
    update_client_process,
    delete_client_process,
)

st.set_page_config(
    page_title="Process Criticality | PRISM Brain",
    page_icon="⚙️",
    layout="wide",
)

inject_prism_theme()

# Initialize session state
if "current_client_id" not in st.session_state:
    st.session_state.current_client_id = None

if "selected_processes" not in st.session_state:
    st.session_state.selected_processes = set()

if "active_macro" not in st.session_state:
    st.session_state.active_macro = None

# Track whether we've synced selected_processes from the DB for the current client
if "_processes_synced_for" not in st.session_state:
    st.session_state._processes_synced_for = None


# ── Helper: cached process data ──────────────────────────────────────
@st.cache_data
def _cached_process_framework():
    """Load and cache the process framework (doesn't change during a session)."""
    return load_process_framework()


def _build_process_lookup():
    """Return {process_id: process_dict} for fast lookups."""
    return {p["process_id"]: p for p in _cached_process_framework()}


def _count_selected_accurate():
    """Count selected processes using widget keys (most up-to-date source).

    Reads proc_* widget keys where they exist (Streamlit updates these BEFORE
    the rerun script executes). Falls back to selected_processes for checkboxes
    that haven't been rendered yet (inside collapsed expanders).
    """
    count = 0
    all_procs = _cached_process_framework()
    for p in all_procs:
        pid = p["process_id"]
        if "." not in pid:
            continue  # skip macro-processes
        wkey = f"proc_{pid}"
        if wkey in st.session_state:
            if st.session_state[wkey]:
                count += 1
        elif pid in st.session_state.selected_processes:
            count += 1
    return count


def _count_selected_in_scope(scope_key):
    """Count selected processes in a specific scope."""
    count = 0
    for p in _cached_process_framework():
        pid = p["process_id"]
        if "." not in pid:
            continue
        if PROCESS_SCOPE_MAP.get(pid.split(".")[0]) != scope_key:
            continue
        wkey = f"proc_{pid}"
        if wkey in st.session_state:
            if st.session_state[wkey]:
                count += 1
        elif pid in st.session_state.selected_processes:
            count += 1
    return count


def _set_active_macro(macro_id):
    """Callback: remember which macro-process expander the user is working in."""
    st.session_state.active_macro = macro_id


# ── Sidebar ─────────────────────────────────────────────────────────────
def client_selector_sidebar():
    """Sidebar client selector."""
    st.sidebar.header("🏢 Current Client")
    clients = get_all_clients()

    if not clients:
        st.sidebar.warning("No clients created yet. Go to Client Setup first.")
        return

    client_names = {c["id"]: c["name"] for c in clients}
    client_ids = list(client_names.keys())

    current_idx = 0
    if st.session_state.current_client_id in client_ids:
        current_idx = client_ids.index(st.session_state.current_client_id)

    selected_id = st.sidebar.selectbox(
        "Select Client",
        options=client_ids,
        format_func=lambda x: client_names[x],
        index=current_idx,
        key="process_crit_client_selector",
    )

    if selected_id != st.session_state.current_client_id:
        st.session_state.current_client_id = selected_id
        st.session_state._processes_synced_for = None  # force re-sync
        st.rerun()

    # Sync selected_processes from database on first load for this client
    # (ensures checkboxes match DB even after page navigation or restart)
    if (st.session_state.current_client_id
            and st.session_state._processes_synced_for != st.session_state.current_client_id):
        processes = get_client_processes(st.session_state.current_client_id)
        db_pids = set(p["process_id"] for p in processes)
        st.session_state.selected_processes = db_pids
        # Also update any existing checkbox widget keys to match
        for key in list(st.session_state.keys()):
            if key.startswith("proc_"):
                pid = key[5:]
                st.session_state[key] = pid in db_pids
        st.session_state._processes_synced_for = st.session_state.current_client_id

    # Show progress
    if st.session_state.current_client_id:
        n_selected = _count_selected_accurate()
        saved = get_client_processes(st.session_state.current_client_id)
        saved_selected = [
            p for p in saved
            if p["process_id"] in st.session_state.selected_processes
        ]
        with_crit = sum(
            1
            for p in saved_selected
            if p.get("criticality_per_day") and p["criticality_per_day"] > 0
        )
        st.sidebar.metric("Processes Selected", n_selected)
        st.sidebar.metric("With Criticality Set", with_crit)


# ── Tab 1: Process Selection ────────────────────────────────────────────
def process_selection():
    """Process selection interface organised by Scope → Macro-process → Sub-process."""
    st.subheader("📋 Business Process Selection")
    if not st.session_state.current_client_id:
        st.warning("Please create or select a client first in Client Setup.")
        return

    client = get_client(st.session_state.current_client_id)
    proc_lookup = _build_process_lookup()

    st.markdown(
        "Select the business processes that are relevant to this client. "
        "Processes are grouped by **Scope** and **Macro-process**."
    )

    if st.button("🔄 Clear Selection"):
        st.session_state.selected_processes = set()
        # Also reset every checkbox widget so Streamlit re-reads value=
        for key in list(st.session_state.keys()):
            if key.startswith("proc_"):
                st.session_state[key] = False
        st.rerun()

    st.divider()

    selected_count = _count_selected_accurate()
    st.info(f"📊 {selected_count} processes selected")

    # ── Display by Scope → Macro-process → Sub-processes ──
    for scope_key in ("A", "B", "C", "D"):
        scope_info = PROCESS_SCOPES[scope_key]
        scope_icon = scope_info["icon"]
        scope_name = scope_info["name"]

        scope_selected = _count_selected_in_scope(scope_key)

        st.markdown(
            f"#### {scope_icon} Scope {scope_key}: {scope_name} "
            f"({scope_selected} selected)"
        )

        # Get macro-processes in this scope (depth 1)
        macro_processes = [
            p for p in get_processes_by_level(1)
            if p.get("scope") == scope_key
        ]

        for macro in sorted(macro_processes, key=lambda p: int(p["process_id"])):
            macro_id = macro["process_id"]
            macro_name = macro["name"]
            children = get_process_children(macro_id)

            # Count selected children (use widget keys for accuracy)
            child_selected = 0
            for c in children:
                cpid = c["process_id"]
                wkey = f"proc_{cpid}"
                if wkey in st.session_state:
                    if st.session_state[wkey]:
                        child_selected += 1
                elif cpid in st.session_state.selected_processes:
                    child_selected += 1

            # Keep the expander open while the user is selecting processes inside it
            keep_open = st.session_state.active_macro == macro_id

            with st.expander(
                f"**{macro_id}. {macro_name}** ({child_selected}/{len(children)} selected)",
                expanded=keep_open,
            ):
                if not children:
                    st.caption("No sub-processes defined.")
                    continue

                # "Select all / Deselect all" for this macro-process
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(
                        "✅ Select all",
                        key=f"selall_{macro_id}",
                    ):
                        st.session_state.active_macro = macro_id
                        for c in children:
                            pid = c["process_id"]
                            st.session_state.selected_processes.add(pid)
                            st.session_state[f"proc_{pid}"] = True
                        st.rerun()
                with col_b:
                    if st.button(
                        "❌ Deselect all",
                        key=f"deselall_{macro_id}",
                    ):
                        st.session_state.active_macro = macro_id
                        for c in children:
                            pid = c["process_id"]
                            st.session_state.selected_processes.discard(pid)
                            st.session_state[f"proc_{pid}"] = False
                        st.rerun()

                for child in sorted(children, key=lambda c: c["process_id"]):
                    pid = child["process_id"]
                    is_selected = pid in st.session_state.selected_processes
                    # Only set default value if the widget key doesn't exist yet
                    # (avoids conflict when Select All/Deselect All sets it via session state)
                    checkbox_kwargs = {
                        "label": f"{pid} – {child['name']}",
                        "key": f"proc_{pid}",
                        "on_change": _set_active_macro,
                        "args": (macro_id,),
                    }
                    if f"proc_{pid}" not in st.session_state:
                        checkbox_kwargs["value"] = is_selected
                    if st.checkbox(**checkbox_kwargs):
                        st.session_state.selected_processes.add(pid)
                    else:
                        st.session_state.selected_processes.discard(pid)

    st.divider()

    # ── Save selection to database ──
    if st.button("💾 Save Process Selection", type="primary"):
        saved_processes = get_client_processes(st.session_state.current_client_id)
        saved_ids = {p["process_id"] for p in saved_processes}

        # Add newly selected
        added = 0
        for pid in st.session_state.selected_processes:
            if pid not in saved_ids and pid in proc_lookup:
                proc = proc_lookup[pid]
                add_client_process(
                    client_id=st.session_state.current_client_id,
                    process_id=pid,
                    process_name=proc["name"],
                    category=proc.get("scope", ""),
                    criticality_per_day=0,
                )
                added += 1

        # Remove de-selected
        removed = 0
        for saved in saved_processes:
            if saved["process_id"] not in st.session_state.selected_processes:
                delete_client_process(saved["id"])
                removed += 1

        st.success(
            f"✅ Saved {len(st.session_state.selected_processes)} processes"
            f" (added {added}, removed {removed})"
        )
        st.rerun()


# ── Tab 2: Criticality Input ───────────────────────────────────────────
def criticality_input():
    """Set Daily Downtime Revenue Impact for selected (High criticality) processes."""
    st.subheader("💰 Daily Downtime Revenue Impact")
    if not st.session_state.current_client_id:
        st.warning("Please create or select a client first.")
        return

    client = get_client(st.session_state.current_client_id)

    # Use selected_processes (live session state) as the source of truth,
    # enriched with saved data from the database (for revenue impact values).
    current_selection = st.session_state.selected_processes
    if not current_selection:
        st.info(
            "No processes selected yet. "
            "Please select processes in the Process Selection tab."
        )
        return

    saved_processes_raw = get_client_processes(st.session_state.current_client_id)
    saved_map = {p["process_id"]: p for p in saved_processes_raw}
    proc_lookup = _build_process_lookup()

    # Build the list: only processes that are currently selected
    saved_processes = []
    for pid in sorted(current_selection):
        if pid in saved_map:
            saved_processes.append(saved_map[pid])
        elif pid in proc_lookup:
            # Selected but not yet saved to DB — show it with defaults
            saved_processes.append({
                "id": None,
                "process_id": pid,
                "process_name": proc_lookup[pid]["name"],
                "criticality_per_day": 0.0,
            })

    if not saved_processes:
        st.info("No processes selected. Please select processes in the Process Selection tab.")
        return

    currency = client.get("currency", "EUR")
    symbol = CURRENCY_SYMBOLS.get(currency, "€")

    st.markdown(
        "The processes below have been selected as **High** criticality. "
        f"Set the **Daily Downtime Revenue Impact** — the estimated revenue loss "
        f"per day if this process is disrupted. Values are in **{currency}** ({symbol})."
    )

    # Auto-calculate suggestion
    if client.get("revenue") and client["revenue"] > 0:
        suggested = calculate_default_criticality(
            client["revenue"], len(saved_processes)
        )
        st.info(
            f"💡 Suggested default: {format_currency(suggested, currency)}/day "
            f"(based on {format_currency(client['revenue'], currency)} revenue ÷ "
            f"250 days ÷ {len(saved_processes)} processes)"
        )
        if st.button("Apply Suggested Values to All"):
            for proc in saved_processes:
                update_client_process(proc["id"], criticality_per_day=suggested,
                                      client_id=st.session_state.current_client_id)
            st.success("Applied suggested values")
            st.rerun()

    st.divider()

    # Column headers
    header1, header2, header3 = st.columns([4, 2, 2])
    with header1:
        st.markdown("**Process**")
    with header2:
        st.markdown("**Criticality**")
    with header3:
        st.markdown(f"**Daily Downtime Revenue Impact ({symbol}/day)**")

    st.divider()

    # Criticality input table
    with st.form("criticality_form"):
        updated_values = {}

        for proc in saved_processes:
            col1, col2, col3 = st.columns([4, 2, 2])

            with col1:
                st.write(f"**{proc['process_id']}. {proc['process_name']}**")

            with col2:
                st.write("High")

            with col3:
                # Use process_id for key if DB id is not yet available
                widget_key = f"crit_{proc['id'] or proc['process_id']}"
                value = st.number_input(
                    f"Revenue Impact ({symbol}/day)",
                    min_value=0.0,
                    value=float(proc["criticality_per_day"])
                    if proc["criticality_per_day"]
                    else 0.0,
                    step=1000.0,
                    key=widget_key,
                    label_visibility="collapsed",
                )
                updated_values[proc["id"]] = value

        if st.form_submit_button("💾 Save Values", type="primary"):
            for proc_id, value in updated_values.items():
                if proc_id is not None:
                    update_client_process(proc_id, criticality_per_day=value,
                                          client_id=st.session_state.current_client_id)
            st.success("✅ Daily Downtime Revenue Impact values saved!")
            st.rerun()  # refresh sidebar metrics with updated values

    # Summary (only currently-selected processes)
    total_impact = sum(
        p["criticality_per_day"] or 0 for p in saved_processes
    )
    st.metric(
        "Total Daily Downtime Revenue Impact",
        format_currency(total_impact, currency),
        help="Sum of all process revenue impact values",
    )


# ── Tab 3: Import / Export ──────────────────────────────────────────────

# Path to the PRISM questionnaire template
_TEMPLATE_PATH = Path(__file__).parent.parent / "data" / "PRISM_Process_Criticality_Template.xlsx"

# Scope sheets in the template workbook
_SCOPE_SHEETS = ["SCOPE A", "SCOPE B", "SCOPE C", "SCOPE D"]


def _read_scope_sheet(df):
    """Parse a SCOPE sheet from the questionnaire template.

    Returns list of dicts with keys: process_id, process_name, criticality, revenue_impact.
    Only returns sub-processes (IDs containing a dot, e.g. '4.4').
    """
    results = []
    for _, row in df.iterrows():
        raw_id = row.iloc[0]   # Column A: Process ID
        name = row.iloc[1]     # Column B: Process Title
        crit = row.iloc[3]     # Column D: Criticality Level
        impact = row.iloc[4]   # Column E: Daily Revenue Impact (€)

        if pd.isna(raw_id) or pd.isna(name):
            continue

        pid = str(raw_id).strip()
        # Skip macro-process rows (no dot) and header rows
        if "." not in pid:
            continue

        crit_str = str(crit).strip() if pd.notna(crit) else ""
        try:
            impact_val = float(impact) if pd.notna(impact) else 0.0
        except (ValueError, TypeError):
            impact_val = 0.0

        results.append({
            "process_id": pid,
            "process_name": str(name).strip(),
            "criticality": crit_str,
            "revenue_impact": impact_val,
        })
    return results


def import_export_section():
    """Import/Export processes via the PRISM Questionnaire Excel template."""
    st.subheader("📥 Import / Export Processes")
    if not st.session_state.current_client_id:
        st.warning("Please create or select a client first.")
        return

    client = get_client(st.session_state.current_client_id)
    currency = client.get("currency", "EUR")
    symbol = CURRENCY_SYMBOLS.get(currency, "€")

    col_dl, col_ul = st.columns(2)

    # ── DOWNLOAD TEMPLATE ──
    with col_dl:
        st.markdown("#### ⬇️ Download Questionnaire")
        st.markdown(
            "Download the **Process Criticality Questionnaire** template. "
            "For each process, set:\n"
            "- **Applicable?** (Yes / No / Partial)\n"
            "- **Criticality Level** (Critical / High / Medium / Low / N/A)\n"
            "- **Daily Revenue Impact (€)** for Critical/High processes\n\n"
            "Processes marked **Critical** or **High** will be imported as selected."
        )

        if _TEMPLATE_PATH.exists():
            with open(_TEMPLATE_PATH, "rb") as f:
                template_bytes = f.read()

            st.download_button(
                label="⬇️ Download Questionnaire (.xlsx)",
                data=template_bytes,
                file_name=f"PRISM_Process_Criticality_{client['name'].replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.error("Template file not found. Contact the PRISM team.")

    # ── UPLOAD FILLED TEMPLATE ──
    with col_ul:
        st.markdown("#### ⬆️ Upload Filled Questionnaire")
        st.markdown(
            "Upload the completed questionnaire. Processes with Criticality "
            "set to **Critical** or **High** will be imported as selected "
            "processes. Their Daily Revenue Impact values will also be imported."
        )

        uploaded_file = st.file_uploader(
            "Upload completed questionnaire",
            type=["xlsx", "xls"],
            key="process_upload",
        )

        if uploaded_file is not None:
            try:
                # Read all SCOPE sheets from the uploaded file
                all_parsed = []
                sheets_found = []
                xls = pd.ExcelFile(uploaded_file)

                for sheet_name in _SCOPE_SHEETS:
                    if sheet_name in xls.sheet_names:
                        sheets_found.append(sheet_name)
                        # Skip header rows (row 0 = scope title, row 1 = column headers)
                        df_sheet = pd.read_excel(
                            uploaded_file, sheet_name=sheet_name, header=None,
                            skiprows=2
                        )
                        parsed = _read_scope_sheet(df_sheet)
                        all_parsed.extend(parsed)

                if not sheets_found:
                    st.error(
                        "Could not find SCOPE sheets in the uploaded file. "
                        "Please use the PRISM questionnaire template."
                    )
                    return

                # Filter: only Critical or High criticality processes are "selected"
                high_crit = [
                    p for p in all_parsed
                    if p["criticality"].lower() in ("critical", "high")
                ]

                st.info(
                    f"Read **{len(all_parsed)}** sub-processes from "
                    f"{len(sheets_found)} scope sheets. "
                    f"**{len(high_crit)}** have Critical/High criticality."
                )

                if high_crit:
                    # Preview
                    preview_data = []
                    for p in high_crit:
                        preview_data.append({
                            "Process": f"{p['process_id']}. {p['process_name']}",
                            "Criticality": p["criticality"],
                            f"Daily Revenue Impact ({symbol})": p["revenue_impact"],
                        })
                    st.dataframe(
                        pd.DataFrame(preview_data),
                        hide_index=True,
                        use_container_width=True,
                    )

                if st.button(
                    "💾 Apply Import", type="primary", key="apply_upload"
                ):
                    proc_lookup = _build_process_lookup()
                    saved_processes = get_client_processes(
                        st.session_state.current_client_id
                    )
                    saved_ids = {p["process_id"] for p in saved_processes}
                    saved_map = {
                        p["process_id"]: p for p in saved_processes
                    }

                    new_selected = set()
                    crit_values = {}

                    for p in high_crit:
                        pid = p["process_id"]
                        new_selected.add(pid)
                        if p["revenue_impact"] > 0:
                            crit_values[pid] = p["revenue_impact"]

                    # Add new processes
                    added = 0
                    for pid in new_selected:
                        if pid not in saved_ids and pid in proc_lookup:
                            proc = proc_lookup[pid]
                            add_client_process(
                                client_id=st.session_state.current_client_id,
                                process_id=pid,
                                process_name=proc["name"],
                                category=proc.get("scope", ""),
                                criticality_per_day=crit_values.get(pid, 0.0),
                            )
                            added += 1
                        elif pid in saved_ids:
                            # Update revenue impact if provided
                            if pid in crit_values:
                                update_client_process(
                                    saved_map[pid]["id"],
                                    criticality_per_day=crit_values[pid],
                                    client_id=st.session_state.current_client_id,
                                )

                    # Remove processes no longer Critical/High
                    removed = 0
                    for saved in saved_processes:
                        if saved["process_id"] not in new_selected:
                            delete_client_process(saved["id"])
                            removed += 1

                    # Update session state: set selected_processes AND
                    # reset all checkbox widget keys so Process Selection
                    # tab reflects the import (not stale checkbox values)
                    st.session_state.selected_processes = new_selected
                    for key in list(st.session_state.keys()):
                        if key.startswith("proc_"):
                            pid = key[5:]  # strip "proc_" prefix
                            st.session_state[key] = pid in new_selected

                    st.success(
                        f"✅ Import complete! Added {added}, removed {removed}, "
                        f"total {len(new_selected)} processes selected."
                    )
                    st.rerun()

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    """Main page function."""
    st.title("⚙️ Process Criticality")
    st.markdown(
        "Select business processes and set their criticality values for risk assessment."
    )

    # Sidebar
    client_selector_sidebar()

    # Show current client
    if st.session_state.current_client_id:
        client = get_client(st.session_state.current_client_id)
        if client:
            st.success(f"📁 Working with: **{client['name']}**")

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        ["📋 Process Selection", "💰 Criticality", "📥 Import / Export"]
    )

    with tab1:
        process_selection()

    with tab2:
        criticality_input()

    with tab3:
        import_export_section()

    # Navigation
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("← Client Setup"):
            st.switch_page("pages/1_Client_Setup.py")

    with col3:
        if st.session_state.current_client_id:
            if st.button("Next: Risk Selection →", type="primary"):
                st.switch_page("pages/3_Risk_Selection.py")


if __name__ == "__main__":
    main()
