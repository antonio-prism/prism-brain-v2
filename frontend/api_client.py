"""
PRISM Brain API Client
Connects the Streamlit frontend to the FastAPI backend on Railway.
Provides cached access to live probabilities, events, and data source health.
Phase 2: Client data CRUD operations.
Phase 3: Trend tracking, alerts, industry profiles, and report management.
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "https://web-production-2cf6.up.railway.app"
API_TIMEOUT = 30  # seconds
CACHE_TTL = 900  # 15 minutes in seconds

# Simple in-memory cache
_cache: Dict[str, Dict[str, Any]] = {}


def _get_cached(key: str) -> Optional[Any]:
    """Get cached value if not expired."""
    if key in _cache:
        entry = _cache[key]
        if time.time() - entry['timestamp'] < CACHE_TTL:
            return entry['data']
        else:
            del _cache[key]
    return None


def _set_cached(key: str, data: Any):
    """Store value in cache with timestamp."""
    _cache[key] = {'data': data, 'timestamp': time.time()}


def clear_cache():
    """Clear all cached data. Call after triggering refresh."""
    _cache.clear()


def check_backend_health() -> Dict:
    """
    Check if the backend API is healthy and responding.
    Returns dict with 'status' ('healthy'|'unhealthy'), 'response_time_ms', 'details'.
    """
    try:
        start = time.time()
        resp = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=10)
        elapsed_ms = round((time.time() - start) * 1000)
        if resp.status_code == 200:
            data = resp.json()
            return {
                'status': 'healthy',
                'response_time_ms': elapsed_ms,
                'details': data,
                'checked_at': datetime.utcnow().isoformat()
            }
        else:
            return {
                'status': 'unhealthy',
                'response_time_ms': elapsed_ms,
                'error': f"HTTP {resp.status_code}",
                'checked_at': datetime.utcnow().isoformat()
            }
    except requests.exceptions.Timeout:
        return {'status': 'unhealthy', 'error': 'Connection timed out', 'checked_at': datetime.utcnow().isoformat()}
    except requests.exceptions.ConnectionError:
        return {'status': 'unhealthy', 'error': 'Cannot reach backend server', 'checked_at': datetime.utcnow().isoformat()}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e), 'checked_at': datetime.utcnow().isoformat()}


def fetch_events(limit: int = 500, skip: int = 0, use_cache: bool = True) -> Optional[List[Dict]]:
    """
    Fetch risk events from the backend.
    Returns list of event dicts or None on error.
    """
    cache_key = f"events_{limit}_{skip}"
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
    
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/v1/events",
            params={'limit': limit, 'skip': skip},
            timeout=API_TIMEOUT
        )
        if resp.status_code == 200:
            data = resp.json()
            # The API returns a list of events or a dict with 'events' key
            events = data if isinstance(data, list) else data.get('events', data.get('data', []))
            _set_cached(cache_key, events)
            return events
        else:
            logger.warning(f"Failed to fetch events: HTTP {resp.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        return None


def fetch_probabilities(limit: int = 1000, skip: int = 0, use_cache: bool = True) -> Optional[Dict[str, Dict]]:
    """
    Fetch latest calculated probabilities from the backend.
    Returns dict mapping event_id -> probability data, or None on error.
    
    Each entry contains:
    - probability_pct: float (0-100)
    - confidence_score: float (0-1)
    - ci_lower_pct, ci_upper_pct: confidence interval
    - precision_band: NARROW|MODERATE|WIDE|VERY_WIDE
    - calculation_date: timestamp
    - flags: list of warning flags
    - data_sources_used: int
    """
    cache_key = f"probabilities_{limit}_{skip}"
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
    
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/v1/probabilities",
            params={'limit': limit, 'skip': skip},
            timeout=API_TIMEOUT
        )
        if resp.status_code == 200:
            data = resp.json()
            # Convert list to dict keyed by event_id for easy lookup
            prob_list = data if isinstance(data, list) else data.get('probabilities', data.get('data', []))
            prob_dict = {}
            for p in prob_list:
                eid = p.get('event_id')
                if eid:
                    prob_dict[eid] = {
                        'probability': p.get('probability_pct', 50.0) / 100.0,  # Convert to 0-1
                        'probability_pct': p.get('probability_pct', 50.0),
                        'confidence_score': p.get('confidence_score', 0.5),
                        'ci_lower_pct': p.get('ci_lower_pct'),
                        'ci_upper_pct': p.get('ci_upper_pct'),
                        'precision_band': p.get('precision_band', 'UNKNOWN'),
                        'calculation_date': p.get('calculation_date'),
                        'flags': p.get('flags', ''),
                        'data_sources_used': p.get('data_sources_used', 0),
                        'baseline_probability_pct': p.get('baseline_probability_pct'),
                        'log_odds': p.get('log_odds'),
                        'total_adjustment': p.get('total_adjustment'),
                        'change_direction': p.get('change_direction'),
                        'attribution': p.get('attribution'),
                        'explanation': p.get('explanation'),
                        'methodology_tier': p.get('methodology_tier'),
                        'signal': p.get('signal'),
                        'momentum': p.get('momentum'),
                        'trend': p.get('trend'),
                        'is_anomaly': p.get('is_anomaly', False)
                    }
            _set_cached(cache_key, prob_dict)
            return prob_dict
        else:
            logger.warning(f"Failed to fetch probabilities: HTTP {resp.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching probabilities: {e}")
        return None


def fetch_data_sources(use_cache: bool = True) -> Optional[List[Dict]]:
    """
    Fetch data source health status from the backend.
    Returns list of data source dicts or None on error.
    """
    cache_key = "data_sources"
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
    
    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/data-sources", timeout=API_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            sources = data if isinstance(data, list) else data.get('sources', data.get('data', []))
            _set_cached(cache_key, sources)
            return sources
        else:
            logger.warning(f"Failed to fetch data sources: HTTP {resp.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching data sources: {e}")
        return None


def trigger_data_refresh(recalculate: bool = True, limit: int = 1000) -> Optional[Dict]:
    """
    Trigger a full data refresh on the backend.
    This fetches new data from all 28 sources and optionally recalculates probabilities.
    
    WARNING: This can take 30-90 seconds. The frontend should show a spinner.
    
    Returns the refresh result dict or None on error.
    """
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/v1/data/refresh",
            params={'recalculate': recalculate, 'limit': limit},
            timeout=120  # Long timeout for refresh
        )
        if resp.status_code == 200:
            clear_cache()  # Invalidate all cached data after refresh
            return resp.json()
        else:
            logger.warning(f"Failed to trigger refresh: HTTP {resp.status_code}")
            return None
    except requests.exceptions.Timeout:
        logger.error("Data refresh timed out after 120 seconds")
        return {'error': 'Refresh timed out. Check backend health.'}
    except Exception as e:
        logger.error(f"Error triggering refresh: {e}")
        return None


def trigger_recalculation(limit: int = 1000) -> Optional[Dict]:
    """
    Trigger probability recalculation without refreshing data sources.
    Faster than full refresh â uses existing indicator data.
    
    Returns the calculation result dict or None on error.
    """
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/v1/calculations/trigger-full",
            params={'limit': limit},
            timeout=90
        )
        if resp.status_code == 200:
            clear_cache()
            return resp.json()
        else:
            logger.warning(f"Failed to trigger recalculation: HTTP {resp.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error triggering recalculation: {e}")
        return None


def get_event_probability(event_id: str, use_cache: bool = True) -> Optional[Dict]:
    """
    Get probability for a single event.
    First tries the cached bulk fetch, then falls back to individual API call.
    """
    # Try bulk cache first
    all_probs = fetch_probabilities(use_cache=use_cache)
    if all_probs and event_id in all_probs:
        return all_probs[event_id]
    
    # No bulk data available, return None (individual endpoint not available)
    return None


# =============================================================================
# Phase 2: Client Data CRUD Operations
# =============================================================================

def _api_request(method: str, path: str, json_data: dict = None,
                 params: dict = None, timeout: int = None) -> Optional[Dict]:
    """Generic API request helper for client data operations."""
    url = f"{API_BASE_URL}{path}"
    t = timeout or API_TIMEOUT
    try:
        if method == "GET":
            resp = requests.get(url, params=params, timeout=t)
        elif method == "POST":
            resp = requests.post(url, json=json_data, params=params, timeout=t)
        elif method == "PUT":
            resp = requests.put(url, json=json_data, params=params, timeout=t)
        elif method == "DELETE":
            resp = requests.delete(url, params=params, timeout=t)
        else:
            return None

        if resp.status_code in (200, 201):
            return resp.json()
        elif resp.status_code == 404:
            logger.warning(f"Not found: {path}")
            return None
        else:
            logger.warning(f"API error {resp.status_code} on {method} {path}")
            return None
    except Exception as e:
        logger.error(f"API request failed: {method} {path}: {e}")
        return None


# --- Client Operations ---

def api_create_client(name: str, location: str = "", industry: str = "",
                      revenue: float = 0, employees: int = 0, currency: str = "EUR",
                      export_percentage: float = 0, primary_markets: str = "",
                      sectors: str = "", notes: str = "") -> Optional[int]:
    """Create a client on the backend. Returns client ID or None."""
    result = _api_request("POST", "/api/v1/clients", json_data={
        "name": name, "location": location, "industry": industry,
        "revenue": revenue, "employees": employees, "currency": currency,
        "export_percentage": export_percentage, "primary_markets": primary_markets,
        "sectors": sectors, "notes": notes
    })
    return result.get("id") if result else None


def api_get_all_clients() -> Optional[List[Dict]]:
    """Get all clients from the backend."""
    result = _api_request("GET", "/api/v1/clients")
    if result:
        return result.get("clients", [])
    return None


def api_get_client(client_id: int) -> Optional[Dict]:
    """Get a specific client by ID."""
    return _api_request("GET", f"/api/v1/clients/{client_id}")


def api_update_client(client_id: int, **kwargs) -> bool:
    """Update a client. Returns True on success."""
    # Filter out None values and internal fields
    update_data = {k: v for k, v in kwargs.items()
                   if v is not None and k not in ('id', 'created_at')}
    result = _api_request("PUT", f"/api/v1/clients/{client_id}", json_data=update_data)
    return result is not None


def api_delete_client(client_id: int) -> bool:
    """Delete a client. Returns True on success."""
    result = _api_request("DELETE", f"/api/v1/clients/{client_id}")
    return result is not None


# --- Process Operations ---

def api_add_process(client_id: int, process_id: str, process_name: str,
                    custom_name: str = "", category: str = "",
                    criticality_per_day: float = 0, notes: str = "") -> Optional[int]:
    """Add a process to a client. Returns process DB ID or None."""
    result = _api_request("POST", f"/api/v1/clients/{client_id}/processes", json_data={
        "process_id": process_id, "process_name": process_name,
        "custom_name": custom_name, "category": category,
        "criticality_per_day": criticality_per_day, "notes": notes
    })
    return result.get("id") if result else None


def api_get_processes(client_id: int) -> Optional[List[Dict]]:
    """Get all processes for a client."""
    result = _api_request("GET", f"/api/v1/clients/{client_id}/processes")
    if result:
        return result.get("processes", [])
    return None


def api_update_process(client_id: int, process_db_id: int, **kwargs) -> bool:
    """Update a process. Returns True on success."""
    update_data = {k: v for k, v in kwargs.items()
                   if v is not None and k not in ('id', 'client_id', 'created_at')}
    result = _api_request("PUT", f"/api/v1/clients/{client_id}/processes/{process_db_id}",
                          json_data=update_data)
    return result is not None


def api_delete_process(client_id: int, process_db_id: int) -> bool:
    """Delete a process. Returns True on success."""
    result = _api_request("DELETE", f"/api/v1/clients/{client_id}/processes/{process_db_id}")
    return result is not None


# --- Risk Operations ---

def api_add_risk(client_id: int, risk_id: str, risk_name: str,
                 domain: str = "", category: str = "", probability: float = 0.5,
                 is_prioritized: bool = False, notes: str = "") -> Optional[int]:
    """Add a risk to a client. Returns risk DB ID or None."""
    result = _api_request("POST", f"/api/v1/clients/{client_id}/risks", json_data={
        "risk_id": risk_id, "risk_name": risk_name, "domain": domain,
        "category": category, "probability": probability,
        "is_prioritized": is_prioritized, "notes": notes
    })
    return result.get("id") if result else None


def api_get_risks(client_id: int, prioritized_only: bool = False) -> Optional[List[Dict]]:
    """Get all risks for a client."""
    params = {"prioritized_only": prioritized_only} if prioritized_only else {}
    result = _api_request("GET", f"/api/v1/clients/{client_id}/risks", params=params)
    if result:
        return result.get("risks", [])
    return None


def api_update_risk(client_id: int, risk_db_id: int, **kwargs) -> bool:
    """Update a risk. Returns True on success."""
    update_data = {k: v for k, v in kwargs.items()
                   if v is not None and k not in ('id', 'client_id', 'created_at')}
    result = _api_request("PUT", f"/api/v1/clients/{client_id}/risks/{risk_db_id}",
                          json_data=update_data)
    return result is not None


# --- Assessment Operations ---

def api_save_assessment(client_id: int, process_id: int, risk_id: int,
                        vulnerability: float = 0.5, resilience: float = 0.3,
                        expected_downtime: int = 5, notes: str = "") -> Optional[int]:
    """Save an assessment. Returns assessment ID or None."""
    result = _api_request("POST", f"/api/v1/clients/{client_id}/assessments", json_data={
        "process_id": process_id, "risk_id": risk_id,
        "vulnerability": vulnerability, "resilience": resilience,
        "expected_downtime": expected_downtime, "notes": notes
    })
    return result.get("id") if result else None


def api_get_assessments(client_id: int) -> Optional[List[Dict]]:
    """Get all assessments for a client with joined details."""
    result = _api_request("GET", f"/api/v1/clients/{client_id}/assessments")
    if result:
        return result.get("assessments", [])
    return None


def api_get_exposure_summary(client_id: int) -> Optional[Dict]:
    """Get comprehensive risk exposure summary."""
    return _api_request("GET", f"/api/v1/clients/{client_id}/exposure-summary")


def get_backend_summary() -> Dict:
    """
    Get a summary of the backend status for display in the frontend.
    Returns a dict with health, event count, probability count, data source info.
    """
    health = check_backend_health()
    summary = {
        'backend_status': health.get('status', 'unknown'),
        'response_time_ms': health.get('response_time_ms'),
        'backend_url': API_BASE_URL,
        'checked_at': health.get('checked_at')
    }

    if health.get('status') == 'healthy':
        details = health.get('details', {})
        summary['total_events'] = details.get('total_events') or details.get('events_count')
        summary['total_probabilities'] = details.get('total_probabilities') or details.get('probabilities_count')
        summary['last_calculation'] = details.get('last_calculation')
        summary['data_sources'] = details.get('data_sources') or details.get('sources_count')

    return summary


# =============================================================================
# Phase 3: Enhanced Dashboard Operations
# =============================================================================

# --- Probability Trend Operations ---

def api_get_trend_data(event_id: str, days: int = 30) -> Optional[List[Dict]]:
    """Get probability trend snapshots for an event. Returns list of snapshot dicts."""
    result = _api_request("GET", f"/api/v1/trends/{event_id}", params={"days": days})
    if result:
        return result.get("snapshots", [])
    return None


def api_get_trend_stats(event_id: str) -> Optional[Dict]:
    """Get trend statistics (averages, changes, direction) for an event."""
    return _api_request("GET", f"/api/v1/trends/{event_id}/stats")


def api_get_top_movers(days: int = 7, limit: int = 20) -> Optional[List[Dict]]:
    """Get events with the biggest probability changes. Returns list of mover dicts."""
    result = _api_request("GET", "/api/v1/trends/movers", params={"days": days, "limit": limit})
    if result:
        return result.get("movers", [])
    return None


def api_take_snapshot() -> Optional[Dict]:
    """Take a snapshot of all current probabilities for trend tracking."""
    return _api_request("POST", "/api/v1/trends/snapshot")


def api_get_trend_summary() -> Optional[Dict]:
    """Get overall trend summary (counts of rising, falling, stable events)."""
    return _api_request("GET", "/api/v1/trends/summary")


# --- Alert Operations ---

def api_create_alert(event_id: str, alert_name: str, threshold_pct: float,
                     direction: str = "ABOVE", severity: str = "MEDIUM",
                     client_id: int = None, notification_email: str = "",
                     is_active: bool = True) -> Optional[int]:
    """Create a probability alert rule. Returns alert ID or None."""
    data = {
        "event_id": event_id, "alert_name": alert_name,
        "threshold_pct": threshold_pct, "direction": direction,
        "severity": severity, "is_active": is_active,
        "notification_email": notification_email
    }
    if client_id is not None:
        data["client_id"] = client_id
    result = _api_request("POST", "/api/v1/alerts", json_data=data)
    return result.get("id") if result else None


def api_get_alerts(client_id: int = None, active_only: bool = True) -> Optional[List[Dict]]:
    """Get all alerts, optionally filtered by client and active status."""
    params = {"active_only": active_only}
    if client_id is not None:
        params["client_id"] = client_id
    result = _api_request("GET", "/api/v1/alerts", params=params)
    if result:
        return result.get("alerts", [])
    return None


def api_get_alert(alert_id: int) -> Optional[Dict]:
    """Get a single alert with recent trigger history."""
    return _api_request("GET", f"/api/v1/alerts/{alert_id}")


def api_update_alert(alert_id: int, **kwargs) -> bool:
    """Update an alert rule. Returns True on success."""
    update_data = {k: v for k, v in kwargs.items()
                   if v is not None and k not in ('id', 'created_at')}
    result = _api_request("PUT", f"/api/v1/alerts/{alert_id}", json_data=update_data)
    return result is not None


def api_delete_alert(alert_id: int) -> bool:
    """Delete an alert rule. Returns True on success."""
    result = _api_request("DELETE", f"/api/v1/alerts/{alert_id}")
    return result is not None


def api_check_alerts() -> Optional[Dict]:
    """Check all active alerts against current probabilities. Returns check results."""
    return _api_request("POST", "/api/v1/alerts/check")


def api_get_triggered_alerts(days: int = 7, acknowledged: bool = None) -> Optional[List[Dict]]:
    """Get recently triggered alert events."""
    params = {"days": days}
    if acknowledged is not None:
        params["acknowledged"] = acknowledged
    result = _api_request("GET", "/api/v1/alerts/triggered", params=params)
    if result:
        return result.get("events", [])
    return None


# --- Industry Profile Operations ---

def api_get_profiles() -> Optional[List[Dict]]:
    """Get all industry risk profiles."""
    result = _api_request("GET", "/api/v1/profiles")
    if result:
        return result.get("profiles", [])
    return None


def api_get_profile(profile_id: int) -> Optional[Dict]:
    """Get a specific profile with its risk events."""
    return _api_request("GET", f"/api/v1/profiles/{profile_id}")


def api_get_profile_by_industry(industry: str) -> Optional[Dict]:
    """Get profile by industry name."""
    return _api_request("GET", f"/api/v1/profiles/industry/{industry}")


def api_create_profile(industry: str, profile_name: str, description: str = "",
                       is_template: bool = True,
                       events: List[Dict] = None) -> Optional[int]:
    """Create an industry profile. events is list of {event_id, relevance_score, weight_multiplier, category}."""
    data = {
        "industry": industry, "profile_name": profile_name,
        "description": description, "is_template": is_template,
        "events": events or []
    }
    result = _api_request("POST", "/api/v1/profiles", json_data=data)
    return result.get("id") if result else None


def api_apply_profile(profile_id: int, client_id: int) -> Optional[Dict]:
    """Apply an industry profile to a client (adds risk events). Returns result dict."""
    return _api_request("POST", f"/api/v1/profiles/{profile_id}/apply/{client_id}")


# --- Report Operations ---

def api_create_report_schedule(report_name: str, client_id: int = None,
                               report_type: str = "WEEKLY", report_format: str = "PDF",
                               recipients: str = "", is_active: bool = True,
                               include_trends: bool = True, include_alerts: bool = True,
                               include_recommendations: bool = True) -> Optional[int]:
    """Create a scheduled report. Returns schedule ID or None."""
    data = {
        "report_name": report_name, "report_type": report_type,
        "report_format": report_format, "recipients": recipients,
        "is_active": is_active, "include_trends": include_trends,
        "include_alerts": include_alerts, "include_recommendations": include_recommendations
    }
    if client_id is not None:
        data["client_id"] = client_id
    result = _api_request("POST", "/api/v1/reports", json_data=data)
    return result.get("id") if result else None


def api_get_report_schedules(client_id: int = None, active_only: bool = False) -> Optional[List[Dict]]:
    """Get all report schedules."""
    params = {"active_only": active_only}
    if client_id is not None:
        params["client_id"] = client_id
    result = _api_request("GET", "/api/v1/reports", params=params)
    if result:
        return result.get("reports", [])
    return None


def api_get_report_schedule(report_id: int) -> Optional[Dict]:
    """Get a report schedule with generation history."""
    return _api_request("GET", f"/api/v1/reports/{report_id}")


def api_update_report_schedule(report_id: int, **kwargs) -> bool:
    """Update a report schedule. Returns True on success."""
    update_data = {k: v for k, v in kwargs.items()
                   if v is not None and k not in ('id', 'created_at')}
    result = _api_request("PUT", f"/api/v1/reports/{report_id}", json_data=update_data)
    return result is not None


def api_generate_report(report_id: int) -> Optional[Dict]:
    """Generate a report on demand. Returns report data dict."""
    return _api_request("POST", f"/api/v1/reports/{report_id}/generate")
