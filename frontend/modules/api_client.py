"""
PRISM Brain API Client
Connects the Streamlit frontend to the FastAPI backend on Railway.
Provides cached access to live probabilities, events, and data source health.
"""

import requests
import time
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

logger = logging.getLogger(__name__)

# Configuration — URL is configurable via Streamlit secrets or environment variable
# IMPORTANT: Use 127.0.0.1 instead of "localhost" — on Windows, "localhost" triggers
# a slow DNS lookup (IPv6 → IPv4 fallback) that adds 200-500ms per HTTP call.
_DEFAULT_API_URL = "http://127.0.0.1:8000"

def _get_api_base_url() -> str:
    """Get API base URL from Streamlit secrets, env var, or default."""
    # Try Streamlit secrets first
    try:
        import streamlit as st
        url = st.secrets.get("PRISM_BACKEND_URL", "")
        if url:
            return url.rstrip("/")
    except Exception:
        pass
    # Try environment variable
    url = os.environ.get("PRISM_BACKEND_URL", "")
    if url:
        return url.rstrip("/")
    return _DEFAULT_API_URL

API_BASE_URL = _get_api_base_url()
API_TIMEOUT = 2  # seconds (local backend should respond in <1s)
CACHE_TTL = 300  # 5 minutes in seconds

# Reusable HTTP session — keeps TCP connections alive between calls,
# avoiding the overhead of creating a new connection for every request.
_http_session = requests.Session()

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
        resp = _http_session.get(f"{API_BASE_URL}/health", timeout=0.5)
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
        resp = _http_session.get(
            f"{API_BASE_URL}/api/v1/events",
            params={'limit': limit, 'skip': skip},
            timeout=API_TIMEOUT
        )
        if resp.status_code == 200:
            data = resp.json()
            events = data if isinstance(data, list) else data.get('events', data.get('data', []))
            _set_cached(cache_key, events)
            return events
        else:
            logger.warning(f"Failed to fetch events: HTTP {resp.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        return None


def fetch_probabilities(limit: int = 0, skip: int = 0, use_cache: bool = True) -> Optional[Dict[str, Dict]]:
    """
    Fetch latest calculated probabilities from the backend.
    Uses pagination (max 500 per request) to retrieve all entries.
    Returns dict mapping event_id -> probability data, or None on error.
    """
    cache_key = "probabilities_all"
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
    
    PAGE_SIZE = 500  # Backend maximum limit per request
    prob_dict = {}
    current_skip = 0
    total = None
    
    try:
        while True:
            resp = _http_session.get(
                f"{API_BASE_URL}/api/v1/probabilities",
                params={'limit': PAGE_SIZE, 'skip': current_skip},
                timeout=API_TIMEOUT
            )
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch probabilities page at skip={current_skip}: HTTP {resp.status_code}")
                break
            
            data = resp.json()
            if total is None:
                total = data.get('total', 0)
            
            prob_list = data if isinstance(data, list) else data.get('probabilities', data.get('data', []))
            if not prob_list:
                break
            
            for p in prob_list:
                eid = p.get('event_id')
                if eid and eid not in prob_dict:  # Keep first (newest) record per event
                    prob_dict[eid] = {
                        'probability': p.get('probability_pct', 50.0) / 100.0,
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
            
            current_skip += len(prob_list)
            if total and current_skip >= total:
                break
            if len(prob_list) < PAGE_SIZE:
                break
        
        if prob_dict:
            logger.info(f"Fetched {len(prob_dict)} probabilities from backend (paginated, {current_skip} total entries)")
            _set_cached(cache_key, prob_dict)
            return prob_dict
        else:
            logger.warning("No probabilities returned from backend after pagination")
            return None
    except Exception as e:
        logger.error(f"Error fetching probabilities: {e}")
        if prob_dict:
            logger.info(f"Returning {len(prob_dict)} partial probabilities despite error")
            return prob_dict
        return None


def fetch_data_sources(use_cache: bool = True) -> Optional[List[Dict]]:
    cache_key = "data_sources"
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
    try:
        resp = _http_session.get(f"{API_BASE_URL}/api/v1/data-sources/health", timeout=API_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            sources = data if isinstance(data, list) else data.get('data_sources', data.get('sources', data.get('data', [])))
            _set_cached(cache_key, sources)
            return sources
        else:
            return None
    except Exception as e:
        return None


def trigger_data_refresh(recalculate: bool = True, limit: int = 1000) -> Optional[Dict]:
    try:
        resp = _http_session.post(f"{API_BASE_URL}/api/v1/data/refresh", params={'recalculate': recalculate, 'limit': limit}, timeout=120)
        if resp.status_code == 200:
            clear_cache()
            return resp.json()
        return None
    except Exception as e:
        return None


def trigger_recalculation(limit: int = 1000) -> Optional[Dict]:
    try:
        resp = _http_session.post(f"{API_BASE_URL}/api/v1/calculations/trigger-full", params={'limit': limit}, timeout=90)
        if resp.status_code == 200:
            clear_cache()
            return resp.json()
        return None
    except Exception as e:
        return None


def get_event_probability(event_id: str, use_cache: bool = True) -> Optional[Dict]:
    all_probs = fetch_probabilities(use_cache=use_cache)
    if all_probs and event_id in all_probs:
        return all_probs[event_id]
    return None


# =============================================================================
# Phase 2: Client Data CRUD Operations
# =============================================================================

def _api_request(method: str, path: str, json_data: dict = None,
                 params: dict = None, timeout: int = None) -> Optional[Dict]:
    url = f"{API_BASE_URL}{path}"
    t = timeout or API_TIMEOUT
    try:
        if method == "GET":
            resp = _http_session.get(url, params=params, timeout=t)
        elif method == "POST":
            resp = _http_session.post(url, json=json_data, params=params, timeout=t)
        elif method == "PUT":
            resp = _http_session.put(url, json=json_data, params=params, timeout=t)
        elif method == "DELETE":
            resp = _http_session.delete(url, params=params, timeout=t)
        else:
            return None
        if resp.status_code in (200, 201):
            return resp.json()
        elif resp.status_code == 404:
            return None
        else:
            return None
    except Exception as e:
        return None


def api_create_client(name, location="", industry="", revenue=0, employees=0, currency="EUR", export_percentage=0, primary_markets="", sectors="", notes=""):
    result = _api_request("POST", "/api/v1/clients", json_data={"name": name, "location": location, "industry": industry, "revenue": revenue, "employees": employees, "currency": currency, "export_percentage": export_percentage, "primary_markets": primary_markets, "sectors": sectors, "notes": notes})
    return result.get("id") if result else None

def api_get_all_clients():
    result = _api_request("GET", "/api/v1/clients")
    return result.get("clients", []) if result else None

def api_get_client(client_id):
    return _api_request("GET", f"/api/v1/clients/{client_id}")

def api_update_client(client_id, **kwargs):
    update_data = {k: v for k, v in kwargs.items() if v is not None and k not in ('id', 'created_at')}
    result = _api_request("PUT", f"/api/v1/clients/{client_id}", json_data=update_data)
    return result is not None

def api_delete_client(client_id):
    result = _api_request("DELETE", f"/api/v1/clients/{client_id}")
    return result is not None

def api_delete_all_clients():
    """Delete ALL clients and associated data from the backend."""
    result = _api_request("DELETE", "/api/v1/clients")
    return result

def api_add_process(client_id, process_id, process_name, custom_name="", category="", criticality_per_day=0, notes=""):
    result = _api_request("POST", f"/api/v1/clients/{client_id}/processes", json_data={"process_id": process_id, "process_name": process_name, "custom_name": custom_name, "category": category, "criticality_per_day": criticality_per_day, "notes": notes})
    return result.get("id") if result else None

def api_get_processes(client_id):
    result = _api_request("GET", f"/api/v1/clients/{client_id}/processes")
    return result.get("processes", []) if result else None

def api_update_process(client_id, process_db_id, **kwargs):
    update_data = {k: v for k, v in kwargs.items() if v is not None and k not in ('id', 'client_id', 'created_at')}
    return _api_request("PUT", f"/api/v1/clients/{client_id}/processes/{process_db_id}", json_data=update_data) is not None

def api_delete_process(client_id, process_db_id):
    return _api_request("DELETE", f"/api/v1/clients/{client_id}/processes/{process_db_id}") is not None

def api_add_risk(client_id, risk_id, risk_name, domain="", category="", probability=0.5, is_prioritized=False, notes=""):
    result = _api_request("POST", f"/api/v1/clients/{client_id}/risks", json_data={"risk_id": risk_id, "risk_name": risk_name, "domain": domain, "category": category, "probability": probability, "is_prioritized": is_prioritized, "notes": notes})
    return result.get("id") if result else None

def api_get_risks(client_id, prioritized_only=False):
    params = {"prioritized_only": prioritized_only} if prioritized_only else {}
    result = _api_request("GET", f"/api/v1/clients/{client_id}/risks", params=params)
    return result.get("risks", []) if result else None

def api_update_risk(client_id, risk_db_id, **kwargs):
    update_data = {k: v for k, v in kwargs.items() if v is not None and k not in ('id', 'client_id', 'created_at')}
    return _api_request("PUT", f"/api/v1/clients/{client_id}/risks/{risk_db_id}", json_data=update_data) is not None

def api_save_assessment(client_id, process_id, risk_id, vulnerability=0.5, resilience=0.3, expected_downtime=5, notes=""):
    result = _api_request("POST", f"/api/v1/clients/{client_id}/assessments", json_data={"process_id": process_id, "risk_id": risk_id, "vulnerability": vulnerability, "resilience": resilience, "expected_downtime": expected_downtime, "notes": notes})
    return result.get("id") if result else None

def api_get_assessments(client_id):
    result = _api_request("GET", f"/api/v1/clients/{client_id}/assessments")
    return result.get("assessments", []) if result else None

def api_get_exposure_summary(client_id):
    return _api_request("GET", f"/api/v1/clients/{client_id}/exposure-summary")

def get_backend_summary():
    health = check_backend_health()
    summary = {'backend_status': health.get('status', 'unknown'), 'response_time_ms': health.get('response_time_ms'), 'backend_url': API_BASE_URL, 'checked_at': health.get('checked_at')}
    if health.get('status') == 'healthy':
        details = health.get('details', {})
        summary['total_events'] = details.get('total_events') or details.get('events_count')
        summary['total_probabilities'] = details.get('total_probabilities') or details.get('probabilities_count')
        summary['last_calculation'] = details.get('last_calculation')
        summary['data_sources'] = details.get('data_sources') or details.get('sources_count')
    return summary


# =============================================================================
# V2 API: New Taxonomy-Based Endpoints
# =============================================================================

def api_v2_get_taxonomy(use_cache: bool = True) -> Optional[Dict]:
    """Get the full V2 taxonomy overview (domains → families → event counts)."""
    cache_key = "v2_taxonomy"
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
    result = _api_request("GET", "/api/v2/taxonomy")
    if result:
        _set_cached(cache_key, result)
    return result


def api_v2_get_events(domain: str = None, family_code: str = None,
                       search: str = None, use_cache: bool = True) -> Optional[List[Dict]]:
    """Get V2 events with optional filters."""
    params = {}
    if domain:
        params["domain"] = domain
    if family_code:
        params["family_code"] = family_code
    if search:
        params["search"] = search

    cache_key = f"v2_events_{domain}_{family_code}_{search}"
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

    result = _api_request("GET", "/api/v2/events", params=params)
    if result is not None:
        _set_cached(cache_key, result)
    return result


def api_v2_get_event(event_id: str) -> Optional[Dict]:
    """Get full detail for a single V2 event."""
    return _api_request("GET", f"/api/v2/events/{event_id}")


def api_v2_get_domain(domain: str, use_cache: bool = True) -> Optional[Dict]:
    """Get all families and events within a domain."""
    cache_key = f"v2_domain_{domain}"
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
    result = _api_request("GET", f"/api/v2/domains/{domain}")
    if result:
        _set_cached(cache_key, result)
    return result


def api_v2_get_family(family_code: str) -> Optional[Dict]:
    """Get all events within a specific family."""
    return _api_request("GET", f"/api/v2/families/{family_code}")


def api_v2_get_probabilities(domain: str = None, family_code: str = None,
                              use_cache: bool = True) -> Optional[List[Dict]]:
    """Get latest V2 probabilities for all active events."""
    params = {}
    if domain:
        params["domain"] = domain
    if family_code:
        params["family_code"] = family_code

    cache_key = f"v2_probs_{domain}_{family_code}"
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

    result = _api_request("GET", "/api/v2/probabilities", params=params)
    if result is not None:
        _set_cached(cache_key, result)
    return result


def api_v2_get_stats(use_cache: bool = True) -> Optional[Dict]:
    """Get V2 system statistics."""
    cache_key = "v2_stats"
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
    result = _api_request("GET", "/api/v2/stats")
    if result:
        _set_cached(cache_key, result)
    return result


def api_v2_health(use_cache: bool = True) -> Optional[Dict]:
    """Check V2 health."""
    cache_key = "v2_health"
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
    result = _api_request("GET", "/api/v2/health")
    if result:
        _set_cached(cache_key, result)
    return result


# =============================================================================
# Engine API: New probability engine endpoints
# =============================================================================

def api_engine_compute_all(use_cache: bool = True) -> Optional[Dict]:
    """Compute probabilities for all 174 events via the prism_engine.

    Returns dict keyed by event_id, e.g.:
        {"PHY-GEO-001": {"event_id": ..., "layer1": {"p_global": 0.86}, ...}, ...}

    Uses a longer timeout (120s) because the engine calls external APIs
    for 174 events on the first request.  Results are cached for 5 minutes.
    """
    cache_key = "engine_compute_all"
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
    result = _api_request("GET", "/api/v2/engine/compute-all", timeout=120)
    if result and "events" in result:
        _set_cached(cache_key, result["events"])
        return result["events"]
    return None


def api_engine_get_fallback_rates(use_cache: bool = True) -> Optional[Dict]:
    """Get all 174 fallback rates from the engine.

    Returns dict keyed by event_id with decimal probabilities, e.g.:
        {"PHY-GEO-001": 0.035, "STR-ECO-001": 0.15, ...}
    """
    cache_key = "engine_fallback_rates"
    if use_cache:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached
    result = _api_request("GET", "/api/v2/engine/fallback-rates", timeout=10)
    if result and "rates" in result:
        _set_cached(cache_key, result["rates"])
        return result["rates"]
    return None


def get_engine_probability(event_id: str) -> Optional[float]:
    """Get engine-computed P_global for a single event.

    Returns probability as a decimal (0-1), or None if the event
    is not a Phase 1 event or the engine is unavailable.
    """
    all_results = api_engine_compute_all()
    if all_results and event_id in all_results:
        result = all_results[event_id]
        if isinstance(result, dict):
            return result.get("layer1", {}).get("p_global")
    return None


def get_best_probability(event_id: str, fallback: float = 0.5) -> float:
    """Get the best available probability for an event.

    Priority: engine-computed P_global > fallback rate > provided default.
    Returns probability as a decimal (0-1).
    """
    # Try engine first (only works for Phase 1 events)
    engine_prob = get_engine_probability(event_id)
    if engine_prob is not None:
        return engine_prob
    # Fall back to the provided default (typically base_rate_pct / 100)
    return fallback


# =============================================================================
# V2 Helpers: Normalize V2 events for the rest of the frontend
# =============================================================================

def normalize_v2_event(event: Dict) -> Dict:
    """Convert a V2 event dict into the format the frontend expects.

    The Risk Selection, Risk Assessment, and Results Dashboard pages
    all expect events with these keys:
        id, name, domain, description, default_probability (0-1), risk_name

    V2 events come from the API with different keys and a 0-100 probability
    scale.  This function bridges the gap so the rest of the code doesn't
    need to know which version it's dealing with.
    """
    base_rate = event.get("base_rate_pct", 0)
    return {
        "id": event.get("event_id", ""),
        "name": event.get("event_name", ""),
        "domain": event.get("domain", ""),
        "family_code": event.get("family_code", ""),
        "family_name": event.get("family_name", ""),
        "description": event.get("description", ""),
        "confidence_level": event.get("confidence_level", "MEDIUM"),
        "is_super_risk": event.get("super_risk", False),
        "default_probability": base_rate / 100.0,   # convert 0-100 → 0-1
        "base_rate_pct": base_rate,                  # keep original for display
        "impact_level": "Medium",
        "risk_name": event.get("event_name", ""),
        # Extra V2 fields (passed through for pages that want them)
        "event_id": event.get("event_id", ""),
        "event_name": event.get("event_name", ""),
    }


def fetch_v2_events_normalized(domain: str = None,
                                family_code: str = None,
                                search: str = None) -> List[Dict]:
    """Fetch V2 events from backend and normalize for frontend use.
    Results are cached via the in-memory cache in api_v2_get_events."""
    events = api_v2_get_events(domain=domain, family_code=family_code,
                                search=search)
    if not events:
        return []
    return [normalize_v2_event(e) for e in events]


# Streamlit-cached version for pages that load events repeatedly
if _HAS_STREAMLIT:
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_v2_events_cached(domain: str = None,
                                family_code: str = None,
                                search: str = None) -> List[Dict]:
        """Streamlit-cached wrapper for fetch_v2_events_normalized."""
        return fetch_v2_events_normalized(domain=domain,
                                           family_code=family_code,
                                           search=search)
