"""
PRISM Engine — Tier 1 indicator fetch orchestrator.

Calls the mapped data connectors and populates the indicator store with
auto-fetched values. Runs before dynamic scoring to ensure fresh data.

Usage:
    from prism_engine.indicator_fetch import fetch_tier1_indicators

    stats = fetch_tier1_indicators()          # Fetch all Tier 1 indicators
    stats = fetch_tier1_indicators("OPS-AIR-004")  # Fetch for one event only
"""

import logging

from .config.indicator_mapping import INDICATOR_MAPPINGS
from .indicator_store import set_indicator_value, save_global_store

logger = logging.getLogger(__name__)

# Cache connector results within a single fetch run to avoid duplicate API calls.
# E.g., "stocks" indicator appears in both OPS-AIR-004 and OPS-RLD-005 but uses
# the same EIA petroleum stocks call.
_connector_cache: dict[str, dict] = {}


def _call_connector(connector_ref: str, mapping: dict) -> dict | None:
    """Call a connector function and return its result data, or None on failure.

    Uses a per-run cache so the same connector+function isn't called twice.
    """
    cache_key = connector_ref
    if cache_key in _connector_cache:
        return _connector_cache[cache_key]

    try:
        module_name, func_name = connector_ref.rsplit(".", 1)

        if module_name == "eia":
            from .connectors import eia
            func = getattr(eia, func_name)
            result = func()
        elif module_name == "fred":
            from .connectors import fred
            func = getattr(fred, func_name)
            # FRED fetch_series needs a series_id argument
            series_id = mapping.get("series_id")
            if series_id:
                result = func(series_id)
            else:
                result = func()
        else:
            logger.warning(f"Unknown connector module: {module_name}")
            return None

        if result.success:
            _connector_cache[cache_key] = result.data
            return result.data
        else:
            logger.warning(f"Connector {connector_ref} failed: {result.error}")
            return None
    except Exception as e:
        logger.error(f"Error calling connector {connector_ref}: {e}")
        return None


def _extract_value(data: dict, mapping: dict) -> float | None:
    """Extract the indicator value from connector result data."""
    # Simple key extraction
    extract_key = mapping.get("extract")
    if extract_key and extract_key in data:
        try:
            return float(data[extract_key])
        except (ValueError, TypeError):
            return None

    # Custom extraction function (for FRED series where we need latest value)
    extract_func = mapping.get("extract_func")
    if extract_func == "_extract_latest_value":
        observations = data.get("observations", [])
        if observations:
            try:
                return float(observations[-1]["value"])
            except (KeyError, ValueError, IndexError):
                pass
    return None


def fetch_tier1_indicators(event_id: str | None = None) -> dict:
    """Fetch all Tier 1 (auto-fetchable) indicators and populate the store.

    Args:
        event_id: If provided, only fetch indicators for this event.
                  If None, fetch all mapped indicators.

    Returns:
        Stats dict: {fetched, failed, skipped, details}
    """
    global _connector_cache
    _connector_cache = {}  # Reset per-run cache

    stats = {"fetched": 0, "failed": 0, "skipped": 0, "details": []}

    for ind_id, mapping in INDICATOR_MAPPINGS.items():
        connector_ref = mapping.get("connector", "")
        events = mapping.get("events", [])

        # Filter to specific event if requested
        if event_id:
            events = [e for e in events if e["event_id"] == event_id]
            if not events:
                continue

        # Call the connector (cached within this run)
        data = _call_connector(connector_ref, mapping)
        if data is None:
            stats["failed"] += 1
            stats["details"].append({
                "indicator_id": ind_id,
                "status": "failed",
                "connector": connector_ref,
            })
            continue

        # Extract the value
        value = _extract_value(data, mapping)
        if value is None:
            stats["failed"] += 1
            stats["details"].append({
                "indicator_id": ind_id,
                "status": "extract_failed",
                "connector": connector_ref,
            })
            continue

        # Write to indicator store for each event that uses this indicator
        for evt in events:
            set_indicator_value(
                event_id=evt["event_id"],
                sub_prob=evt["sub_prob"],
                indicator_id=ind_id,
                value=value,
                tier=1,
                source=connector_ref,
                unit=mapping.get("unit", ""),
            )

        stats["fetched"] += 1
        stats["details"].append({
            "indicator_id": ind_id,
            "status": "ok",
            "value": value,
            "events_updated": len(events),
        })

    # Persist to disk
    save_global_store()
    _connector_cache = {}  # Clean up

    logger.info(
        f"Tier 1 fetch complete: {stats['fetched']} fetched, "
        f"{stats['failed']} failed, {stats['skipped']} skipped"
    )
    return stats
