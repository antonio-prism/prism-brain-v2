"""
PRISM Engine — Indicator-to-connector mapping.

Maps indicator_ids from the Method C research scoring functions to actual
data sources (API connectors, FRED series, EIA endpoints, etc.).

Only ~5% of the 1,072 indicators can be auto-fetched from free public APIs.
The rest require manual entry (Tier 2 analyst reports, Tier 3 client data).
This file maps the auto-fetchable subset.

Each mapping defines:
  - connector: which connector module to call
  - fetch_func: function name within the connector
  - extract: how to extract the indicator value from the connector result
  - events: which event_id/sub_prob combinations use this indicator

Adding new auto-fetch indicators:
  1. Add the connector function if needed
  2. Add a mapping entry here
  3. The indicator_fetch orchestrator will pick it up automatically
"""

import logging

logger = logging.getLogger(__name__)


# Mapping: indicator_id -> how to fetch it automatically
# Only Tier 1 indicators (auto-fetchable from free APIs) are listed here.
#
# Format:
#   "indicator_id": {
#       "connector": "module.function" to call,
#       "extract": "key_path" in connector result data dict,
#       "unit": display unit,
#       "description": human-readable description,
#       "events": [  # which events use this indicator
#           {"event_id": "...", "sub_prob": "p_preconditions"},
#       ],
#   }

INDICATOR_MAPPINGS: dict[str, dict] = {
    # ── EIA Petroleum Indicators ─────────────────────────────────────
    "stocks": {
        "connector": "eia.fetch_petroleum_stocks",
        "extract": "days_of_supply",
        "unit": "days",
        "description": "U.S. petroleum days of supply",
        "events": [
            {"event_id": "OPS-AIR-004", "sub_prob": "p_preconditions"},
            {"event_id": "OPS-RLD-005", "sub_prob": "p_preconditions"},
        ],
    },
    "vol": {
        "connector": "eia.fetch_crude_price",
        "extract": "volatility_pct",
        "unit": "%",
        "description": "Crude oil weekly price volatility",
        "events": [
            {"event_id": "OPS-AIR-004", "sub_prob": "p_preconditions"},
        ],
    },
    "spike": {
        "connector": "eia.fetch_crude_price",
        "extract": "yoy_change_pct",
        "unit": "%",
        "description": "Crude oil year-over-year price change",
        "events": [
            {"event_id": "OPS-RLD-005", "sub_prob": "p_trigger"},
        ],
    },
    "outages": {
        "connector": "eia.fetch_refinery_outages",
        "extract": "outage_weeks",
        "unit": "count",
        "description": "Refinery outage weeks (utilization < 85%)",
        "events": [
            {"event_id": "OPS-AIR-004", "sub_prob": "p_trigger"},
            {"event_id": "OPS-RLD-005", "sub_prob": "p_trigger"},
        ],
    },
    # ── FRED Economic Indicators (already fetched by FRED connector) ──
    "ig_spread": {
        "connector": "fred.fetch_series",
        "extract_func": "_extract_latest_value",
        "series_id": "BAMLC0A4CBBB",
        "unit": "bp",
        "description": "ICE BofA BBB Corporate Bond OAS",
        "events": [
            {"event_id": "STR-FIN-003", "sub_prob": "p_preconditions"},
        ],
    },
}


def get_auto_fetch_indicators() -> dict[str, dict]:
    """Return all indicator mappings that can be auto-fetched."""
    return INDICATOR_MAPPINGS.copy()


def get_indicators_for_event(event_id: str) -> list[dict]:
    """Get all auto-fetchable indicators for a specific event.

    Returns list of dicts with indicator_id, sub_prob, connector info.
    """
    result = []
    for ind_id, mapping in INDICATOR_MAPPINGS.items():
        for evt in mapping.get("events", []):
            if evt["event_id"] == event_id:
                result.append({
                    "indicator_id": ind_id,
                    "sub_prob": evt["sub_prob"],
                    **{k: v for k, v in mapping.items() if k != "events"},
                })
    return result


def get_events_with_auto_fetch() -> set[str]:
    """Get the set of event IDs that have at least one auto-fetchable indicator."""
    events = set()
    for mapping in INDICATOR_MAPPINGS.values():
        for evt in mapping.get("events", []):
            events.add(evt["event_id"])
    return events
