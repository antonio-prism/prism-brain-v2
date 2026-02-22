"""
PRISM Engine — Method C research output loader.

Integrates event-specific sub-probabilities from the Method C research
output JSON file into the engine's event mapping configuration.

Supports two schema versions:
  - v2 (flat): evidence at event["evidence"]
  - v3 (nested): evidence at event["layer1"]["derivation"]["sub_probabilities"]

Usage:
    from prism_engine.method_c_loader import load_and_integrate

    # One-step: load, validate, and integrate
    stats = load_and_integrate("path/to/method_c_v3_complete.json")

    # Or step-by-step:
    from prism_engine.method_c_loader import load_research_output, integrate_research
    data, errors = load_research_output("path/to/research.json")
    stats = integrate_research(data)
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Where the research output file is expected
DEFAULT_RESEARCH_PATH = Path(__file__).parent.parent / "method_c_v3_complete.json"
OVERRIDES_PATH = Path(__file__).parent / "data" / "method_c_overrides.json"
FULL_RESEARCH_PATH = Path(__file__).parent / "data" / "method_c_full_research.json"

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

# In-memory cache for overrides (avoid 115+ file reads during compute-all)
_overrides_cache: dict | None = None


def _detect_schema_version(data: dict) -> str:
    """Detect v2 (flat evidence) vs v3 (nested layer1) schema."""
    sample_event = next(iter(data.get("events", {}).values()), {})
    if "layer1" in sample_event:
        return "v3"
    return "v2"


def _extract_evidence_v3(event: dict) -> dict:
    """Extract evidence from v3 nested schema into the flat format the engine expects."""
    sub_probs = (
        event.get("layer1", {})
        .get("derivation", {})
        .get("sub_probabilities", {})
    )
    evidence = {}
    for key in ("p_preconditions", "p_trigger", "p_implementation"):
        sp = sub_probs.get(key, {})
        evidence[key] = {
            "value": sp.get("value"),
            "type": sp.get("evidence_type", "DEFAULT_0.50_NO_DATA"),
            "justification": sp.get("justification", ""),
            "sources": sp.get("sources", []),
            "confidence": sp.get("confidence", "Low"),
        }
    return evidence


def _get_event_confidence(event: dict, schema: str) -> str:
    """Extract overall confidence level for an event."""
    if schema == "v3":
        return (
            event.get("layer1", {})
            .get("derivation", {})
            .get("confidence", "Medium")
        )
    return "Medium"


def load_research_output(path: str | Path | None = None) -> tuple[dict | None, list[str]]:
    """
    Load and validate the Method C research output JSON.

    Supports both v2 (flat evidence) and v3 (nested layer1) schemas.
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

    schema = _detect_schema_version(data)
    logger.info(f"Detected schema version: {schema}")

    for event_id, event in events.items():
        prefix = f"[{event_id}]"

        # Check required fields
        for field in ("p_pre", "p_trig", "p_impl"):
            if field not in event:
                errors.append(f"{prefix} Missing field: {field}")

        # Validate probability ranges
        for field in ("p_pre", "p_trig", "p_impl"):
            val = event.get(field, 0)
            if not isinstance(val, (int, float)):
                errors.append(f"{prefix} {field} is not a number: {val}")
            elif not 0.05 <= val <= 0.95:
                errors.append(f"{prefix} {field}={val} outside [0.05, 0.95]")

        # Validate prior_computed = p_pre * p_trig * p_impl (warn only, don't block)
        p_pre = event.get("p_pre", 0)
        p_trig = event.get("p_trig", 0)
        p_impl = event.get("p_impl", 0)
        if isinstance(p_pre, (int, float)) and isinstance(p_trig, (int, float)) and isinstance(p_impl, (int, float)):
            expected = round(p_pre * p_trig * p_impl, 4)
            actual = event.get("prior_computed", 0)
            if actual and abs(expected - actual) > 0.005:
                logger.debug(
                    f"{prefix} prior_computed={actual} vs calculated={expected} (diff={abs(expected-actual):.4f})"
                )

        # Validate evidence structure (schema-aware)
        if schema == "v3":
            evidence = _extract_evidence_v3(event)
        else:
            evidence = event.get("evidence", {})

        for sub in ("p_preconditions", "p_trigger", "p_implementation"):
            sub_ev = evidence.get(sub)
            if sub_ev is None:
                errors.append(f"{prefix} Missing evidence for {sub}")
                continue
            if "value" not in sub_ev:
                errors.append(f"{prefix} evidence.{sub} missing 'value'")
            ev_type = sub_ev.get("type", sub_ev.get("evidence_type", ""))
            if ev_type and ev_type not in VALID_EVIDENCE_TYPES:
                errors.append(f"{prefix} evidence.{sub} invalid type: '{ev_type}'")
            if not sub_ev.get("justification"):
                errors.append(f"{prefix} evidence.{sub} missing 'justification'")

    logger.info(f"Loaded {len(events)} events from {path} (schema {schema}), {len(errors)} validation errors")
    return data, errors


def integrate_research(data: dict) -> dict:
    """
    Integrate validated research output into the engine configuration.

    Writes:
      - method_c_overrides.json  (engine consumption: p_pre/p_trig/p_impl + evidence)
      - method_c_full_research.json  (Phase II: scoring_functions, geographic, correlation)

    Returns stats dict: {total, integrated, skipped, errors, schema}.
    """
    events = data.get("events", {})
    schema = _detect_schema_version(data)
    overrides = {}
    full_research = {}
    stats = {"total": len(events), "integrated": 0, "skipped": 0, "errors": [], "schema": schema}

    for event_id, event in events.items():
        try:
            p_pre = float(event["p_pre"])
            p_trig = float(event["p_trig"])
            p_impl = float(event["p_impl"])

            # Extract evidence based on schema
            if schema == "v3":
                evidence = _extract_evidence_v3(event)
                confidence = _get_event_confidence(event, schema)
            else:
                evidence = event.get("evidence", {})
                confidence = "Medium"

            overrides[event_id] = {
                "p_pre": p_pre,
                "p_trig": p_trig,
                "p_impl": p_impl,
                "prior_computed": round(p_pre * p_trig * p_impl, 4),
                "evidence": evidence,
                "confidence": confidence,
                "event_name": event.get("event_name", ""),
            }

            # Store full research data for Phase II (scoring functions, etc.)
            if schema == "v3":
                full_research[event_id] = {
                    "event_name": event.get("event_name", ""),
                    "scoring_functions": event.get("scoring_functions", {}),
                    "geographic_adjustment": event.get("geographic_adjustment", {}),
                    "correlation": event.get("correlation", {}),
                    "tail_risk": event.get("tail_risk", {}),
                    "cascade_participation": event.get("cascade_participation", {}),
                }

            stats["integrated"] += 1
        except (KeyError, ValueError, TypeError) as e:
            stats["errors"].append(f"[{event_id}] {e}")
            stats["skipped"] += 1

    # Write overrides file (engine consumption)
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": data.get("metadata", {}),
            "overrides": overrides,
        }, f, indent=2, ensure_ascii=False)

    # Write full research file (Phase II)
    if full_research:
        with open(FULL_RESEARCH_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": data.get("metadata", {}),
                "events": full_research,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved Phase II research data to {FULL_RESEARCH_PATH}")

    # Invalidate the in-memory cache so next get_method_c_override() reads fresh data
    invalidate_cache()

    logger.info(f"Integrated {stats['integrated']} Method C overrides to {OVERRIDES_PATH}")
    return stats


def get_method_c_override(event_id: str) -> dict | None:
    """
    Get event-specific Method C override if available.

    Returns dict with p_pre, p_trig, p_impl, evidence, confidence,
    or None if no override exists for this event.

    Uses an in-memory cache to avoid repeated file reads during compute-all.
    """
    global _overrides_cache

    if _overrides_cache is None:
        if not OVERRIDES_PATH.exists():
            return None
        try:
            with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            _overrides_cache = data.get("overrides", {})
        except Exception:
            return None

    return _overrides_cache.get(event_id)


def invalidate_cache() -> None:
    """Clear the in-memory overrides cache (call after re-integration)."""
    global _overrides_cache
    _overrides_cache = None


def load_and_integrate(path: str | Path | None = None) -> dict:
    """
    One-step: load, validate, and integrate research output.

    Returns stats dict with integration results.
    """
    data, errors = load_research_output(path)

    if data is None:
        return {"total": 0, "integrated": 0, "skipped": 0,
                "errors": errors, "schema": "unknown"}

    if errors:
        logger.warning(f"Validation produced {len(errors)} warnings (proceeding with integration)")

    stats = integrate_research(data)
    stats["validation_warnings"] = errors
    return stats
