"""
PRISM Engine — Validation rules for computed values.

Every computed value must pass bounds checking before being included
in the output JSON.
"""

import logging

logger = logging.getLogger(__name__)

VALIDATION_RULES = {
    "prior": (0.001, 0.95),
    "modifier": (0.50, 3.00),
    "p_global": (0.001, 0.95),
    "p_client": (0.001, 0.95),
    "geographic_exposure": (0.0, 3.0),
    "industry_exposure": (0.0, 3.0),
    "scale_factor": (0.5, 2.0),
    "vulnerability_score": (0, 100),
    "resilience_score": (0, 100),
}


def validate_value(field: str, value: float) -> tuple[bool, str]:
    """
    Validate a computed value against its bounds.

    Returns: (is_valid, message)
    """
    if field not in VALIDATION_RULES:
        return True, f"No validation rule for '{field}'"

    low, high = VALIDATION_RULES[field]
    if low <= value <= high:
        return True, "OK"
    return False, f"{field}={value} is outside [{low}, {high}]"


def clip_value(field: str, value: float) -> float:
    """Clip a value to its valid range, logging if clamped."""
    if field not in VALIDATION_RULES:
        return value

    low, high = VALIDATION_RULES[field]
    if value < low:
        logger.warning(f"Clipping {field}={value} to floor {low}")
        return low
    if value > high:
        logger.warning(f"Clipping {field}={value} to ceiling {high}")
        return high
    return value


def validate_event_output(event_output: dict) -> list[str]:
    """
    Validate a complete event output JSON against all rules.

    Returns list of validation errors (empty = all good).
    """
    errors = []

    # Check Layer 1
    layer1 = event_output.get("layer1", {})
    prior = layer1.get("prior")
    if prior is not None:
        ok, msg = validate_value("prior", prior)
        if not ok:
            errors.append(f"Layer 1 prior: {msg}")

    p_global = layer1.get("p_global")
    if p_global is not None:
        ok, msg = validate_value("p_global", p_global)
        if not ok:
            errors.append(f"Layer 1 p_global: {msg}")

    # Check modifiers
    for mod in layer1.get("modifiers", []):
        mod_val = mod.get("modifier_value")
        if mod_val is not None:
            ok, msg = validate_value("modifier", mod_val)
            if not ok:
                errors.append(f"Modifier '{mod.get('name', '?')}': {msg}")

    # Check derivation exists
    derivation = layer1.get("derivation", {})
    if not derivation.get("formula"):
        errors.append("Missing derivation.formula")
    if not derivation.get("data_source"):
        errors.append("Missing derivation.data_source")

    # Check Layer 2 (if present)
    layer2 = event_output.get("layer2", {})
    if layer2:
        for field in ["geographic_exposure", "industry_exposure", "scale_factor"]:
            val = layer2.get(field)
            if val is not None:
                ok, msg = validate_value(field, val)
                if not ok:
                    errors.append(f"Layer 2 {field}: {msg}")

    return errors
