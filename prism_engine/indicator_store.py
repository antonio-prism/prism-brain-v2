"""
PRISM Engine — Indicator value store.

Central store for all indicator values from the three data tiers:
  Tier 1: Auto-fetched from public APIs (FRED, EIA, NOAA, etc.)
  Tier 2: Manually entered from commercial research reports (Gartner, IATA, etc.)
  Tier 3: Client-specific operational data (procurement, engineering, etc.)

Storage:
  - Global store (Tier 1 + 2): prism_engine/data/indicator_store_global.json
  - Client stores (Tier 3):     prism_engine/data/indicator_store_client_{id}.json

Key format: "{event_id}/{sub_probability}/{indicator_id}"
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_STORE_PATH = DATA_DIR / "indicator_store_global.json"

# Default TTLs by tier (hours)
DEFAULT_TTL = {1: 24, 2: 8760, 3: 4380}  # 24h, 1 year, 6 months

# In-memory caches
_global_cache: dict | None = None
_client_caches: dict[str, dict] = {}


def _client_store_path(client_id: str) -> Path:
    return DATA_DIR / f"indicator_store_client_{client_id}.json"


def _empty_store() -> dict:
    return {
        "metadata": {
            "last_updated": datetime.utcnow().isoformat(),
            "version": "1.0",
        },
        "indicators": {},
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        return _empty_store()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return _empty_store()


def _save_json(path: Path, data: dict) -> None:
    data["metadata"]["last_updated"] = datetime.utcnow().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_global_store() -> dict:
    """Load the global indicator store (Tier 1 + 2). Cached in memory."""
    global _global_cache
    if _global_cache is None:
        _global_cache = _load_json(GLOBAL_STORE_PATH)
    return _global_cache


def load_client_store(client_id: str) -> dict:
    """Load a client-specific indicator store (Tier 3). Cached in memory."""
    if client_id not in _client_caches:
        _client_caches[client_id] = _load_json(_client_store_path(client_id))
    return _client_caches[client_id]


def save_global_store() -> None:
    """Persist the global store to disk."""
    global _global_cache
    if _global_cache is None:
        return
    _save_json(GLOBAL_STORE_PATH, _global_cache)


def save_client_store(client_id: str) -> None:
    """Persist a client store to disk."""
    if client_id not in _client_caches:
        return
    _save_json(_client_store_path(client_id), _client_caches[client_id])


def invalidate_caches() -> None:
    """Clear all in-memory caches."""
    global _global_cache, _client_caches
    _global_cache = None
    _client_caches = {}


def _make_key(event_id: str, sub_prob: str, indicator_id: str) -> str:
    """Build the standard indicator key."""
    return f"{event_id}/{sub_prob}/{indicator_id}"


def set_indicator_value(
    event_id: str,
    sub_prob: str,
    indicator_id: str,
    value: float,
    tier: int,
    source: str,
    unit: str = "",
    client_id: str | None = None,
    ttl_hours: int | None = None,
) -> None:
    """Set an indicator value in the appropriate store.

    Tier 1 and 2 go to global store.
    Tier 3 goes to client-specific store (client_id required).
    """
    key = _make_key(event_id, sub_prob, indicator_id)
    entry = {
        "value": value,
        "unit": unit,
        "tier": tier,
        "source": source,
        "updated_at": datetime.utcnow().isoformat(),
        "ttl_hours": ttl_hours or DEFAULT_TTL.get(tier, 24),
    }

    if tier in (1, 2):
        store = load_global_store()
        store["indicators"][key] = entry
    elif tier == 3 and client_id:
        store = load_client_store(client_id)
        store["indicators"][key] = entry
    else:
        logger.warning(f"Cannot set indicator: tier={tier}, client_id={client_id}")


def get_indicator_value(
    event_id: str,
    sub_prob: str,
    indicator_id: str,
    client_id: str | None = None,
) -> float | None:
    """Get an indicator value, checking client store first, then global."""
    key = _make_key(event_id, sub_prob, indicator_id)

    # Check client store first (Tier 3 overrides global for same key)
    if client_id:
        client_store = load_client_store(client_id)
        entry = client_store.get("indicators", {}).get(key)
        if entry is not None:
            return entry["value"]

    global_store = load_global_store()
    entry = global_store.get("indicators", {}).get(key)
    if entry is not None:
        return entry["value"]

    return None


def get_all_values_for_event(
    event_id: str,
    scoring_functions: dict | None = None,
    client_id: str | None = None,
) -> dict[str, float]:
    """Get all indicator values for an event, keyed by indicator_id.

    If scoring_functions is provided, only returns values for indicators
    defined in those functions. Otherwise returns all values matching
    the event_id prefix.
    """
    result = {}
    prefix = f"{event_id}/"

    # Collect from global store
    global_store = load_global_store()
    for key, entry in global_store.get("indicators", {}).items():
        if key.startswith(prefix):
            # Extract indicator_id from key: event_id/sub_prob/indicator_id
            parts = key.split("/")
            if len(parts) == 3:
                ind_id = parts[2]
                result[ind_id] = entry["value"]

    # Overlay client store (Tier 3 values override global for same indicator)
    if client_id:
        client_store = load_client_store(client_id)
        for key, entry in client_store.get("indicators", {}).items():
            if key.startswith(prefix):
                parts = key.split("/")
                if len(parts) == 3:
                    ind_id = parts[2]
                    result[ind_id] = entry["value"]

    return result


def get_freshness_summary(client_id: str | None = None) -> dict:
    """Get a summary of indicator freshness across all events.

    Returns: {total, live, stale, missing_count, by_tier: {1: {live, stale}, ...}}
    """
    now = datetime.utcnow()
    summary = {"total": 0, "live": 0, "stale": 0, "by_tier": {}}

    stores = [load_global_store()]
    if client_id:
        stores.append(load_client_store(client_id))

    for store in stores:
        for key, entry in store.get("indicators", {}).items():
            summary["total"] += 1
            tier = entry.get("tier", 0)
            ttl = entry.get("ttl_hours", 24)
            updated = datetime.fromisoformat(entry.get("updated_at", "2000-01-01"))

            is_live = (now - updated) < timedelta(hours=ttl)

            tier_key = str(tier)
            if tier_key not in summary["by_tier"]:
                summary["by_tier"][tier_key] = {"live": 0, "stale": 0}

            if is_live:
                summary["live"] += 1
                summary["by_tier"][tier_key]["live"] += 1
            else:
                summary["stale"] += 1
                summary["by_tier"][tier_key]["stale"] += 1

    return summary


def get_coverage_for_event(
    event_id: str,
    scoring_functions: dict,
    client_id: str | None = None,
) -> dict:
    """Get indicator coverage for a specific event.

    Returns: {total_indicators, available, missing, coverage_ratio, by_sub_prob}
    """
    values = get_all_values_for_event(event_id, client_id=client_id)
    total = 0
    available = 0
    by_sub = {}

    for sub_key in ("p_preconditions", "p_trigger", "p_implementation"):
        sf = scoring_functions.get(sub_key, {})
        indicators = sf.get("input_indicators", [])
        sub_total = len(indicators)
        sub_avail = sum(1 for ind in indicators if ind.get("indicator_id") in values)

        total += sub_total
        available += sub_avail
        by_sub[sub_key] = {
            "total": sub_total,
            "available": sub_avail,
            "coverage": round(sub_avail / sub_total, 3) if sub_total > 0 else 0.0,
        }

    return {
        "total_indicators": total,
        "available": available,
        "missing": total - available,
        "coverage_ratio": round(available / total, 3) if total > 0 else 0.0,
        "by_sub_probability": by_sub,
    }
