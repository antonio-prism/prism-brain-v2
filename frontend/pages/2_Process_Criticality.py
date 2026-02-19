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


# ── Helper: build a lookup dict from the flat process list ──────────────
def _build_process_lookup():
    """Return {process_id: process_dict} for fast lookups."""
    return {p["process_id"]: p for p in load_process_framework()}


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
        processes = get_client_processes(selected_id)
        st.session_state.selected_processes = set(
            p["process_id"] for p in processes
        )
        st.rerun()

    # Show progress
    if st.session_state.current_client_id:
        saved = get_client_processes(st.session_state.current_client_id)
        with_crit = sum(
            1
            for p in saved
            if p.get("criticality_per_day") and p["criticality_per_day"] > 0
        )
        st.sidebar.metric("Processes Selected", len(saved))
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

    selected_count = len(st.session_state.selected_processes)
    st.info(f"📊 {selected_count} processes selected")

    # ── Display by Scope → Macro-process → Sub-processes ──
    for scope_key in ("A", "B", "C", "D"):
        scope_info = PROCESS_SCOPES[scope_key]
        scope_icon = scope_info["icon"]
        scope_name = scope_info["name"]

        # Count selected in this scope
        scope_selected = sum(
            1
            for pid in st.session_state.selected_processes
            if PROCESS_SCOPE_MAP.get(pid.split(".")[0]) == scope_key
        )

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

            # Count selected children
            child_selected = sum(
                1 for c in children if c["process_id"] in st.session_state.selected_processes
            )

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

        # Remove de-selected
        for saved in saved_processes:
            if saved["process_id"] not in st.session_state.selected_processes:
                delete_client_process(saved["id"])

        st.success(f"✅ Saved {len(st.session_state.selected_processes)} processes")


# ── Tab 2: Criticality Input ───────────────────────────────────────────
def criticality_input():
    """Set criticality values for selected processes."""
    st.subheader("💰 Process Criticality")
    if not st.session_state.current_client_id:
        st.warning("Please create or select a client first.")
        return

    client = get_client(st.session_state.current_client_id)
    saved_processes = get_client_processes(st.session_state.current_client_id)

    if not saved_processes:
        st.info(
            "No processes selected yet. "
            "Please select processes in the Process Selection tab."
        )
        return

    currency = client.get("currency", "EUR")
    symbol = CURRENCY_SYMBOLS.get(currency, "€")

    st.markdown(
        f"Set the **criticality** for each process — the estimated revenue impact "
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
                update_client_process(proc["id"], criticality_per_day=suggested)
            st.success("Applied suggested values")
            st.rerun()

    st.divider()

    # Criticality input table
    with st.form("criticality_form"):
        updated_values = {}

        for proc in saved_processes:
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.write(f"**{proc['process_name']}**")
                st.caption(f"ID: {proc['process_id']}")

            with col2:
                value = st.number_input(
                    f"Criticality ({symbol}/day)",
                    min_value=0.0,
                    value=float(proc["criticality_per_day"])
                    if proc["criticality_per_day"]
                    else 0.0,
                    step=1000.0,
                    key=f"crit_{proc['id']}",
                    label_visibility="collapsed",
                )
                updated_values[proc["id"]] = value

            with col3:
                st.write(f"{format_currency(value, currency)}/day")

        if st.form_submit_button("💾 Save Criticality Values", type="primary"):
            for proc_id, value in updated_values.items():
                update_client_process(proc_id, criticality_per_day=value)
            st.success("✅ Criticality values saved!")

    # Summary
    total_criticality = sum(
        p["criticality_per_day"] or 0 for p in saved_processes
    )
    st.metric(
        "Total Daily Criticality",
        format_currency(total_criticality, currency),
        help="Sum of all process criticality values",
    )


# ── Tab 3: Import / Export ──────────────────────────────────────────────
def import_export_section():
    """Import/Export processes via Excel template."""
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
        st.markdown("#### ⬇️ Download Template")
        st.markdown(
            "Download an Excel template with **all available sub-processes**. "
            "Mark the relevant ones with **Yes** in the 'Selected' column "
            "and optionally set the criticality value."
        )

        if st.button("📥 Generate Template", key="gen_template"):
            all_processes = load_process_framework()
            saved_processes = get_client_processes(
                st.session_state.current_client_id
            )
            saved_map = {p["process_id"]: p for p in saved_processes}

            rows = []
            for proc in sorted(
                all_processes, key=lambda p: p["process_id"]
            ):
                # Only include sub-processes (depth 2) in the template
                if proc.get("depth") != 2:
                    continue
                pid = proc["process_id"]
                saved = saved_map.get(pid)
                rows.append(
                    {
                        "Scope": proc.get("scope", ""),
                        "Macro-Process": proc.get("parent_id", ""),
                        "Process ID": pid,
                        "Process Name": proc["name"],
                        "Selected": "Yes" if pid in saved_map else "No",
                        f"Revenue Impact/Day ({symbol})": float(
                            saved["criticality_per_day"]
                        )
                        if saved and saved.get("criticality_per_day")
                        else 0.0,
                    }
                )

            df = pd.DataFrame(rows)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Processes")

                # Auto-fit columns
                worksheet = writer.sheets["Processes"]
                for col_idx, col in enumerate(df.columns):
                    max_len = (
                        max(df[col].astype(str).map(len).max(), len(col)) + 2
                    )
                    col_letter = chr(65 + col_idx) if col_idx < 26 else "A"
                    worksheet.column_dimensions[col_letter].width = max_len

            st.download_button(
                label="⬇️ Download Process Template (.xlsx)",
                data=buffer.getvalue(),
                file_name=f"PRISM_Processes_{client['name'].replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ── UPLOAD FILLED TEMPLATE ──
    with col_ul:
        st.markdown("#### ⬆️ Upload Filled Template")
        st.markdown(
            "Upload the completed template. The system will add processes "
            "marked **Yes**, remove those marked **No**, and update criticality values."
        )

        uploaded_file = st.file_uploader(
            "Upload completed process template",
            type=["xlsx", "xls"],
            key="process_upload",
        )

        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                required_cols = ["Process ID", "Selected"]
                if not all(col in df.columns for col in required_cols):
                    st.error(f"Template must contain columns: {required_cols}")
                else:
                    selected_df = df[
                        df["Selected"].str.strip().str.lower() == "yes"
                    ]
                    st.info(
                        f"Found **{len(selected_df)}** processes marked as selected"
                    )

                    if st.button(
                        "💾 Apply Upload", type="primary", key="apply_upload"
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

                        for _, row in df.iterrows():
                            pid = str(row["Process ID"]).strip()
                            is_selected = (
                                str(row.get("Selected", "No"))
                                .strip()
                                .lower()
                                == "yes"
                            )

                            if is_selected:
                                new_selected.add(pid)
                                crit_col = [
                                    c
                                    for c in df.columns
                                    if "revenue impact" in c.lower()
                                    or "criticality" in c.lower()
                                ]
                                if crit_col:
                                    try:
                                        crit_values[pid] = float(
                                            row[crit_col[0]]
                                        )
                                    except (ValueError, TypeError):
                                        crit_values[pid] = 0.0

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
                                    criticality_per_day=crit_values.get(
                                        pid, 0.0
                                    ),
                                )
                                added += 1
                            elif pid in saved_ids:
                                if (
                                    pid in crit_values
                                    and crit_values[pid] > 0
                                ):
                                    update_client_process(
                                        saved_map[pid]["id"],
                                        criticality_per_day=crit_values[pid],
                                    )

                        # Remove deselected
                        removed = 0
                        for saved in saved_processes:
                            if saved["process_id"] not in new_selected:
                                delete_client_process(saved["id"])
                                removed += 1

                        st.session_state.selected_processes = new_selected
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
