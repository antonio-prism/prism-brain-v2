"""
PRISM Engine — Method C research output loader.

Integrates event-specific sub-probabilities from the Method C research
output JSON file into the engine's event mapping configuration.

Expected input: method_c_research_output.json (produced by research session)
Schema: See docs/PRD_Method_C_Research.md Section 5.

Usage:
    from prism_engine.method_c_loader import load_research_output, integrate_research

    # Load and validate
    data, errors = load_research_output("path/to/method_c_research_output.json")

    # Integrate into event_mapping (writes to event-level overrides)
    stats = integrate_research(data)
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Where the research output file is expected
DEFAULT_RESEARCH_PATH = Path(__file__).parent.parent / "method_c_research_output.json"
OVERRIDES_PATH = Path(__file__).parent / "data" / "method_c_overrides.json"

VALID_EVIDENCE_TYPES = {
    "legislative_pipeline",
    "historical_frequency",
    "election_cycle",
    "expert_survey",
    "market_data",
    "incident_count",
    "regulatory_timeline",
    "DEFAULT_0.50_NO_DATA",
}


def load_research_output(path: str | Path | None = None) -> tuple[dict | None, list[str]]:
    """
    Load and validate the Method C research output JSON.

    Returns (data, errors). If errors is non-empty, data may be partial.
    """
    if path is None:
        path = DEFAULT_RESEARCH_PATH
    path = Path(path)

    if not path.exists():
        return None, [f"File not found: {path}"]

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return None, [f"Invalid JSON: {e}"]

    errors = []
    events = data.get("events", {})

    if not events:
        errors.append("No 'events' key found in JSON")
        return data, errors

    for event_id, event in events.items():
        prefix = f"[{event_id}]"

        # Check required fields
        for field in ("p_pre", "p_trig", "p_impl", "prior_computed"):
            if field not in event:
                errors.append(f"{prefix} Missing field: {field}")

        # Validate probability ranges
        for field in ("p_pre", "p_trig", "p_impl"):
            val = event.get(field, 0)
            if not isinstance(val, (int, float)):
                errors.append(f"{prefix} {field} is not a number: {val}")
            elif not 0.05 <= val <= 0.95:
                errors.append(f"{prefix} {field}={val} outside [0.05, 0.95]")

        # Validate prior_computed = p_pre * p_trig * p_impl
        p_pre = event.get("p_pre", 0)
        p_trig = event.get("p_trig", 0)
        p_impl = event.get("p_impl", 0)
        expected = round(p_pre * p_trig * p_impl, 4)
        actual = event.get("prior_computed", 0)
        if abs(expected - actual) > 0.001:
            errors.append(
                f"{prefix} prior_computed={actual} != p_pre*p_trig*p_impl={expected}"
            )

        # Validate evidence structure
        evidence = event.get("evidence", {})
        for sub in ("p_preconditions", "p_trigger", "p_implementation"):
            sub_ev = evidence.get(sub)
            if sub_ev is None:
                errors.append(f"{prefix} Missing evidence for {sub}")
                continue
            if "value" not in sub_ev:
                errors.append(f"{prefix} evidence.{sub} missing 'value'")
            ev_type = sub_ev.get("type", "")
            if ev_type and ev_type not in VALID_EVIDENCE_TYPES:
                errors.append(f"{prefix} evidence.{sub} invalid type: '{ev_type}'")
            if not sub_ev.get("justification"):
                errors.append(f"{prefix} evidence.{sub} missing 'justification'")

    logger.info(f"Loaded {len(events)} events from {path}, {len(errors)} validation errors")
    return data, errors


def integrate_research(data: dict) -> dict:
    """
    Integrate validated research output into the engine configuration.

    Writes event-level overrides to method_c_overrides.json, which is
    loaded by engine.py when computing Method C priors.

    Returns stats dict: {total, integrated, skipped, errors}.
    """
    events = data.get("events", {})
    overrides = {}
    stats = {"total": len(events), "integrated": 0, "skipped": 0, "errors": []}

    for event_id, event in events.items():
        try:
            p_pre = float(event["p_pre"])
            p_trig = float(event["p_trig"])
            p_impl = float(event["p_impl"])

            overrides[event_id] = {
                "p_pre": p_pre,
                "p_trig": p_trig,
                "p_impl": p_impl,
                "prior_computed": round(p_pre * p_trig * p_impl, 4),
                "evidence": event.get("evidence", {}),
                "event_name": event.get("event_name", ""),
            }
            stats["integrated"] += 1
        except (KeyError, ValueError, TypeError) as e:
            stats["errors"].append(f"[{event_id}] {e}")
            stats["skipped"] += 1

    # Write overrides file
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": data.get("metadata", {}),
            "overrides": overrides,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Integrated {stats['integrated']} Method C overrides to {OVERRIDES_PATH}")
    return stats


def get_method_c_override(event_id: str) -> dict | None:
    """
    Get event-specific Method C override if available.

    Returns dict with p_pre, p_trig, p_impl, evidence, or None if
    no override exists for this event.
    """
    if not OVERRIDES_PATH.exists():
        return None

    try:
        with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("overrides", {}).get(event_id)
    except Exception:
        return None
