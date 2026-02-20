"""
PRISM Brain - Database Module
=============================
Handles all data persistence.
Phase 2: Hybrid mode — tries backend API first, falls back to local SQLite.
"""

import sqlite3
import json
import os
import time
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Simple time-based cache for read operations ---
_data_cache = {}
_DATA_CACHE_TTL = 30  # seconds — long enough to survive rapid Streamlit reruns


def _get_data_cached(key):
    """Get cached data if not expired."""
    if key in _data_cache:
        entry = _data_cache[key]
        if time.time() - entry['ts'] < _DATA_CACHE_TTL:
            return entry['data'], True
    return None, False


def _set_data_cached(key, data):
    """Store data in cache."""
    _data_cache[key] = {'data': data, 'ts': time.time()}


def _clear_data_cache(prefix=None):
    """Clear cached data. Optionally only keys starting with prefix."""
    if prefix:
        keys_to_remove = [k for k in _data_cache if k.startswith(prefix)]
        for k in keys_to_remove:
            del _data_cache[k]
    else:
        _data_cache.clear()

# Get the data directory path
DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "prism_brain.db"

# --- Phase 2: Backend API integration ---
# Try to import the API client for hybrid mode
try:
    from modules.api_client import (
        check_backend_health,
        api_create_client, api_get_all_clients, api_get_client,
        api_update_client, api_delete_client, api_delete_all_clients,
        api_add_process, api_get_processes, api_update_process, api_delete_process,
        api_add_risk, api_get_risks, api_update_risk,
        api_save_assessment, api_get_assessments, api_get_exposure_summary
    )
    API_CLIENT_AVAILABLE = True
except ImportError:
    API_CLIENT_AVAILABLE = False

# Track backend availability (checked once per session, refreshed on demand)
_backend_online = None


def is_backend_online() -> bool:
    """Check if the backend API is available. Caches result for the session."""
    global _backend_online
    if not API_CLIENT_AVAILABLE:
        return False
    if _backend_online is None:
        try:
            health = check_backend_health()
            _backend_online = health.get('status') == 'healthy'
        except Exception:
            _backend_online = False
    return _backend_online


def refresh_backend_status():
    """Force re-check of backend availability."""
    global _backend_online
    _backend_online = None
    return is_backend_online()


def get_data_source():
    """Return 'backend' or 'local' depending on current mode."""
    return 'backend' if is_backend_online() else 'local'


def get_connection():
    """Get a database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def init_database():
    """Initialize the database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Clients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT,
            industry TEXT,
            revenue REAL,
            employees INTEGER,
            currency TEXT DEFAULT 'EUR',
            export_percentage REAL DEFAULT 0,
            primary_markets TEXT,
            sectors TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Client processes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS client_processes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL,
            process_id TEXT NOT NULL,
            process_name TEXT NOT NULL,
            custom_name TEXT,
            category TEXT,
            criticality_per_day REAL DEFAULT 0,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE
        )
    ''')

    # Prioritized risks table (risks selected for a client)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS client_risks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL,
            risk_id TEXT NOT NULL,
            risk_name TEXT NOT NULL,
            domain TEXT,
            category TEXT,
            probability REAL DEFAULT 0.5,
            is_prioritized INTEGER DEFAULT 0,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE
        )
    ''')

    # Risk assessments table (process-risk combinations)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS risk_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL,
            process_id INTEGER NOT NULL,
            risk_id INTEGER NOT NULL,
            vulnerability REAL DEFAULT 0.5,
            resilience REAL DEFAULT 0.3,
            expected_downtime INTEGER DEFAULT 5,
            notes TEXT,
            assessed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE,
            FOREIGN KEY (process_id) REFERENCES client_processes(id) ON DELETE CASCADE,
            FOREIGN KEY (risk_id) REFERENCES client_risks(id) ON DELETE CASCADE,
            UNIQUE(client_id, process_id, risk_id)
        )
    ''')

    # External data cache table
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS external_data_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            category TEXT,
            data_key TEXT NOT NULL,
            data_value TEXT,
            numeric_value REAL,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            UNIQUE(source_name, data_key)
        )
    ''')

    conn.commit()
    conn.close()

    return True


# =============================================================================
# CLIENT OPERATIONS
# =============================================================================

def create_client(name, location="", industry="", revenue=0, employees=0,
                  currency="EUR", export_percentage=0, primary_markets="",
                  sectors="", notes=""):
    """Create a new client. Tries backend API first, falls back to local SQLite."""
    if is_backend_online():
        try:
            result = api_create_client(name, location, industry, revenue,
                                       employees, currency, export_percentage,
                                       primary_markets, sectors, notes)
            if result is not None:
                logger.info(f"Client created on backend: {result}")
                _create_client_local(name, location, industry, revenue,
                                     employees, currency, export_percentage,
                                     primary_markets, sectors, notes)
                _clear_data_cache("all_clients")
                return result
        except Exception as e:
            logger.warning(f"Backend create_client failed, using local: {e}")
    result = _create_client_local(name, location, industry, revenue, employees,
                                currency, export_percentage, primary_markets,
                                sectors, notes)
    _clear_data_cache("all_clients")
    return result


def _create_client_local(name, location="", industry="", revenue=0, employees=0,
                         currency="EUR", export_percentage=0, primary_markets="",
                         sectors="", notes=""):
    """Create a client in local SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO clients (name, location, industry, revenue, employees,
                            currency, export_percentage, primary_markets, sectors, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, location, industry, revenue, employees, currency,
          export_percentage, primary_markets, sectors, notes))
    client_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return client_id


def get_all_clients():
    """Get all clients. Uses cache to avoid repeated API calls on Streamlit reruns."""
    cached, hit = _get_data_cached("all_clients")
    if hit:
        return cached
    if is_backend_online():
        try:
            result = api_get_all_clients()
            if result is not None:
                _set_data_cached("all_clients", result)
                return result
        except Exception as e:
            logger.warning(f"Backend get_all_clients failed, using local: {e}")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM clients ORDER BY updated_at DESC')
    clients = [dict(row) for row in cursor.fetchall()]
    conn.close()
    _set_data_cached("all_clients", clients)
    return clients


def get_client(client_id):
    """Get a specific client by ID. Uses cache to avoid repeated API calls."""
    cache_key = f"client_{client_id}"
    cached, hit = _get_data_cached(cache_key)
    if hit:
        return cached
    if is_backend_online():
        try:
            result = api_get_client(client_id)
            if result is not None:
                _set_data_cached(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"Backend get_client failed, using local: {e}")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM clients WHERE id = ?', (client_id,))
    row = cursor.fetchone()
    conn.close()
    result = dict(row) if row else None
    _set_data_cached(cache_key, result)
    return result


def update_client(client_id, **kwargs):
    """Update client information. Tries backend API first."""
    if is_backend_online():
        try:
            result = api_update_client(client_id, **kwargs)
            if result:
                _update_client_local(client_id, **kwargs)
                _clear_data_cache("all_clients")
                _clear_data_cache(f"client_{client_id}")
                return True
        except Exception as e:
            logger.warning(f"Backend update_client failed, using local: {e}")
    result = _update_client_local(client_id, **kwargs)
    _clear_data_cache("all_clients")
    _clear_data_cache(f"client_{client_id}")
    return result


def _update_client_local(client_id, **kwargs):
    """Update client in local SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    fields = []
    values = []
    for key, value in kwargs.items():
        if key not in ['id', 'created_at']:
            fields.append(f"{key} = ?")
            values.append(value)
    fields.append("updated_at = ?")
    values.append(datetime.now().isoformat())
    values.append(client_id)
    query = f"UPDATE clients SET {', '.join(fields)} WHERE id = ?"
    cursor.execute(query, values)
    conn.commit()
    conn.close()
    return True


def delete_client(client_id):
    """Delete a client and all associated data. Tries backend API first."""
    if is_backend_online():
        try:
            result = api_delete_client(client_id)
            if result:
                _delete_client_local(client_id)
                _clear_data_cache()  # Clear all cache on delete
                return True
        except Exception as e:
            logger.warning(f"Backend delete_client failed, using local: {e}")
    result = _delete_client_local(client_id)
    _clear_data_cache()
    return result


def _delete_client_local(client_id):
    """Delete client from local SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM clients WHERE id = ?', (client_id,))
    conn.commit()
    conn.close()
    return True


def delete_all_clients():
    """Delete ALL clients from backend and local SQLite."""
    result = None
    if is_backend_online():
        try:
            result = api_delete_all_clients()
        except Exception as e:
            logger.warning(f"Backend delete_all_clients failed: {e}")
    # Also clear local SQLite
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM client_risk_assessments')
        cursor.execute('DELETE FROM client_risks')
        cursor.execute('DELETE FROM client_processes')
        cursor.execute('DELETE FROM clients')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Local delete_all_clients failed: {e}")
    return result


# =============================================================================
# PROCESS OPERATIONS
# =============================================================================

def add_client_process(client_id, process_id, process_name, custom_name="",
                       category="", criticality_per_day=0, notes=""):
    """Add a process to a client. Tries backend API first."""
    if is_backend_online():
        try:
            result = api_add_process(client_id, process_id, process_name,
                                     custom_name, category, criticality_per_day, notes)
            if result is not None:
                _add_process_local(client_id, process_id, process_name,
                                   custom_name, category, criticality_per_day, notes)
                _clear_data_cache(f"processes_{client_id}")
                return result
        except Exception as e:
            logger.warning(f"Backend add_process failed, using local: {e}")
    result = _add_process_local(client_id, process_id, process_name,
                              custom_name, category, criticality_per_day, notes)
    _clear_data_cache(f"processes_{client_id}")
    return result


def _add_process_local(client_id, process_id, process_name, custom_name="",
                       category="", criticality_per_day=0, notes=""):
    """Add process to local SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO client_processes (client_id, process_id, process_name,
                                      custom_name, category, criticality_per_day, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (client_id, process_id, process_name, custom_name, category,
          criticality_per_day, notes))
    process_db_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return process_db_id


def get_client_processes(client_id):
    """Get all processes for a client. Uses cache to avoid repeated API calls."""
    cache_key = f"processes_{client_id}"
    cached, hit = _get_data_cached(cache_key)
    if hit:
        return cached
    if is_backend_online():
        try:
            result = api_get_processes(client_id)
            if result is not None:
                _set_data_cached(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"Backend get_processes failed, using local: {e}")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM client_processes
        WHERE client_id = ?
        ORDER BY criticality_per_day DESC
    ''', (client_id,))
    processes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    _set_data_cached(cache_key, processes)
    return processes


def update_client_process(process_db_id, **kwargs):
    """Update a client process. Tries backend API first."""
    # Need client_id for API call and cache clearing — extract from kwargs
    client_id = kwargs.pop('client_id', None)
    if is_backend_online() and client_id:
        try:
            result = api_update_process(client_id, process_db_id, **kwargs)
            if result:
                _update_process_local(process_db_id, **kwargs)
                if client_id:
                    _clear_data_cache(f"processes_{client_id}")
                return True
        except Exception as e:
            logger.warning(f"Backend update_process failed, using local: {e}")
    result = _update_process_local(process_db_id, **kwargs)
    if client_id:
        _clear_data_cache(f"processes_{client_id}")
    return result


def _update_process_local(process_db_id, **kwargs):
    """Update process in local SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    fields = []
    values = []
    for key, value in kwargs.items():
        if key not in ['id', 'client_id', 'created_at']:
            fields.append(f"{key} = ?")
            values.append(value)
    values.append(process_db_id)
    query = f"UPDATE client_processes SET {', '.join(fields)} WHERE id = ?"
    cursor.execute(query, values)
    conn.commit()
    conn.close()
    return True


def delete_client_process(process_db_id, client_id=None):
    """Delete a client process. Tries backend API first."""
    if is_backend_online() and client_id:
        try:
            result = api_delete_process(client_id, process_db_id)
            if result:
                _delete_process_local(process_db_id)
                if client_id:
                    _clear_data_cache(f"processes_{client_id}")
                return True
        except Exception as e:
            logger.warning(f"Backend delete_process failed, using local: {e}")
    result = _delete_process_local(process_db_id)
    if client_id:
        _clear_data_cache(f"processes_{client_id}")
    return result


def _delete_process_local(process_db_id):
    """Delete process from local SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM client_processes WHERE id = ?', (process_db_id,))
    conn.commit()
    conn.close()
    return True


# =============================================================================
# RISK OPERATIONS
# =============================================================================

def add_client_risk(client_id, risk_id, risk_name, domain="", category="",
                    probability=0.5, is_prioritized=0, notes=""):
    """Add a risk to a client's risk portfolio. Tries backend API first."""
    is_prio_bool = bool(is_prioritized)
    if is_backend_online():
        try:
            result = api_add_risk(client_id, risk_id, risk_name, domain,
                                  category, probability, is_prio_bool, notes)
            if result is not None:
                _add_risk_local(client_id, risk_id, risk_name, domain,
                                category, probability, is_prioritized, notes)
                _clear_data_cache(f"risks_{client_id}")
                return result
        except Exception as e:
            logger.warning(f"Backend add_risk failed, using local: {e}")
    result = _add_risk_local(client_id, risk_id, risk_name, domain,
                           category, probability, is_prioritized, notes)
    _clear_data_cache(f"risks_{client_id}")
    return result


def _add_risk_local(client_id, risk_id, risk_name, domain="", category="",
                    probability=0.5, is_prioritized=0, notes=""):
    """Add risk to local SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO client_risks
        (client_id, risk_id, risk_name, domain, category, probability, is_prioritized, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (client_id, risk_id, risk_name, domain, category, probability,
          is_prioritized, notes))
    risk_db_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return risk_db_id


def get_client_risks(client_id, prioritized_only=False):
    """Get all risks for a client. Uses cache to avoid repeated API calls."""
    cache_key = f"risks_{client_id}_{prioritized_only}"
    cached, hit = _get_data_cached(cache_key)
    if hit:
        return cached
    if is_backend_online():
        try:
            result = api_get_risks(client_id, prioritized_only)
            if result is not None:
                _set_data_cached(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"Backend get_risks failed, using local: {e}")
    conn = get_connection()
    cursor = conn.cursor()
    if prioritized_only:
        cursor.execute('''
            SELECT * FROM client_risks
            WHERE client_id = ? AND is_prioritized = 1
            ORDER BY probability DESC
        ''', (client_id,))
    else:
        cursor.execute('''
            SELECT * FROM client_risks
            WHERE client_id = ?
            ORDER BY probability DESC
        ''', (client_id,))
    risks = [dict(row) for row in cursor.fetchall()]
    conn.close()
    _set_data_cached(cache_key, risks)
    return risks


def update_client_risk(risk_db_id, **kwargs):
    """Update a client risk. Tries backend API first."""
    client_id = kwargs.pop('client_id', None)
    if is_backend_online() and client_id:
        try:
            result = api_update_risk(client_id, risk_db_id, **kwargs)
            if result:
                _update_risk_local(risk_db_id, **kwargs)
                return True
        except Exception as e:
            logger.warning(f"Backend update_risk failed, using local: {e}")
    return _update_risk_local(risk_db_id, **kwargs)


def _update_risk_local(risk_db_id, **kwargs):
    """Update risk in local SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    fields = []
    values = []
    for key, value in kwargs.items():
        if key not in ['id', 'client_id', 'created_at']:
            fields.append(f"{key} = ?")
            values.append(value)
    values.append(risk_db_id)
    query = f"UPDATE client_risks SET {', '.join(fields)} WHERE id = ?"
    cursor.execute(query, values)
    conn.commit()
    conn.close()
    return True


# =============================================================================
# ASSESSMENT OPERATIONS
# =============================================================================

def save_assessment(client_id, process_id, risk_id, vulnerability,
                    resilience, expected_downtime, notes=""):
    """Save or update a risk assessment. Tries backend API first."""
    if is_backend_online():
        try:
            result = api_save_assessment(client_id, process_id, risk_id,
                                         vulnerability, resilience,
                                         expected_downtime, notes)
            if result is not None:
                _save_assessment_local(client_id, process_id, risk_id,
                                       vulnerability, resilience,
                                       expected_downtime, notes)
                _clear_data_cache(f"assessments_{client_id}")
                return True
        except Exception as e:
            logger.warning(f"Backend save_assessment failed, using local: {e}")
    result = _save_assessment_local(client_id, process_id, risk_id,
                                  vulnerability, resilience,
                                  expected_downtime, notes)
    _clear_data_cache(f"assessments_{client_id}")
    return result


def _save_assessment_local(client_id, process_id, risk_id, vulnerability,
                           resilience, expected_downtime, notes=""):
    """Save assessment to local SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO risk_assessments
        (client_id, process_id, risk_id, vulnerability, resilience,
         expected_downtime, notes, assessed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (client_id, process_id, risk_id, vulnerability, resilience,
          expected_downtime, notes, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return True


def get_assessments(client_id):
    """Get all assessments for a client with process and risk details. Uses cache."""
    cache_key = f"assessments_{client_id}"
    cached, hit = _get_data_cached(cache_key)
    if hit:
        return cached
    if is_backend_online():
        try:
            result = api_get_assessments(client_id)
            if result is not None:
                _set_data_cached(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"Backend get_assessments failed, using local: {e}")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT
            ra.*,
            cp.process_name,
            cp.custom_name,
            cp.criticality_per_day,
            cp.category as process_category,
            cr.risk_name,
            cr.domain,
            cr.category as risk_category,
            cr.probability
        FROM risk_assessments ra
        JOIN client_processes cp ON ra.process_id = cp.id
        JOIN client_risks cr ON ra.risk_id = cr.id
        WHERE ra.client_id = ?
        ORDER BY cp.criticality_per_day DESC, cr.probability DESC
    ''', (client_id,))
    assessments = [dict(row) for row in cursor.fetchall()]
    conn.close()
    _set_data_cached(cache_key, assessments)
    return assessments


def get_assessment(client_id, process_id, risk_id):
    """Get a specific assessment."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM risk_assessments
        WHERE client_id = ? AND process_id = ? AND risk_id = ?
    ''', (client_id, process_id, risk_id))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# =============================================================================
# CALCULATION HELPERS
# =============================================================================

def calculate_risk_exposure(criticality, vulnerability, resilience,
                           downtime, probability):
    """
    Calculate risk exposure using PRISM formula.

    Risk Exposure (€/yr) = Criticality × Vulnerability × (1 - Resilience) × Downtime × Probability
    """
    return criticality * vulnerability * (1 - resilience) * downtime * probability


def get_risk_exposure_summary(client_id):
    """Get comprehensive risk exposure summary for a client.

    Always computes summary from assessments data to ensure consistency
    across Executive Summary, Visualizations, and Detailed Results.
    Assessments are fetched from backend API when online, local SQLite otherwise.
    """
    assessments = get_assessments(client_id)
    if not assessments:
        return None

    summary = {
        "total_exposure": 0,
        "by_domain": {},
        "by_process": {},
        "by_risk": {},
        "assessments": []
    }

    for a in assessments:
        exposure = calculate_risk_exposure(
            a['criticality_per_day'],
            a['vulnerability'],
            a['resilience'],
            a['expected_downtime'],
            a['probability']
        )

        # Add to total
        summary["total_exposure"] += exposure

        # Add to domain breakdown
        domain = a['domain']
        if domain not in summary["by_domain"]:
            summary["by_domain"][domain] = 0
        summary["by_domain"][domain] += exposure

        # Add to process breakdown
        process_name = a['custom_name'] or a['process_name']
        if process_name not in summary["by_process"]:
            summary["by_process"][process_name] = 0
        summary["by_process"][process_name] += exposure

        # Add to risk breakdown
        risk_name = a['risk_name']
        if risk_name not in summary["by_risk"]:
            summary["by_risk"][risk_name] = 0
        summary["by_risk"][risk_name] += exposure

        # Add to detailed list
        summary["assessments"].append({
            **a,
            "exposure": exposure
        })

    return summary


# NOTE: init_database() is NOT called at import time.
# It is called once via @st.cache_resource in Welcome.py.
# This avoids re-creating/checking tables on every Streamlit rerun.
