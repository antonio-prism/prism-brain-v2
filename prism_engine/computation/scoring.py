"""
PRISM Engine — Phase II Dynamic Scoring.

Pure math functions for computing sub-probabilities from indicator data.
No I/O, no side effects — all data is passed in as arguments.

Pipeline per sub-probability:
  For each indicator: raw_value -> normalize(min, max, type) -> [0, 1]
  weighted_score = sum(normalized_i * weight_i)  [re-weighted if partial]
  P_sub = sigmoid(weighted_score, midpoint, steepness), clamped [0.05, 0.95]

Pipeline per event:
  P_composite = P_pre * P_trig * P_impl
  Each sub-prob independently decides dynamic vs static fallback.
"""

import math
import logging

logger = logging.getLogger(__name__)

# Minimum fraction of indicators that must have values to use dynamic scoring
# for a sub-probability. Below this, fall back to the static research value.
MIN_COVERAGE_FOR_DYNAMIC = 0.30

# Sub-probability clamp range (from research architecture spec)
P_SUB_FLOOR = 0.05
P_SUB_CEILING = 0.95


def normalize_value(raw: float, norm_type: str, params: dict) -> float:
    """Normalize a raw indicator value to [0, 1].

    Supported types:
      - linear_scale: (value - min) / (max - min), clamped [0,1]
      - inverse_linear_scale: 1 - linear_scale
      - log_scale: log(value - min + 1) / log(max - min + 1)
      - discrete_map: lookup table {value -> score}
      - threshold: 1 if value >= threshold, else 0
    """
    if norm_type == "linear_scale":
        lo = params.get("min", 0)
        hi = params.get("max", 1)
        if hi == lo:
            return 0.5
        result = (raw - lo) / (hi - lo)
        return max(0.0, min(1.0, result))

    if norm_type == "inverse_linear_scale":
        lo = params.get("min", 0)
        hi = params.get("max", 1)
        if hi == lo:
            return 0.5
        result = 1.0 - (raw - lo) / (hi - lo)
        return max(0.0, min(1.0, result))

    if norm_type == "log_scale":
        lo = params.get("min", 0)
        hi = params.get("max", 1)
        if hi == lo:
            return 0.5
        numerator = math.log(max(raw - lo, 0) + 1)
        denominator = math.log(hi - lo + 1)
        if denominator == 0:
            return 0.5
        result = numerator / denominator
        return max(0.0, min(1.0, result))

    if norm_type == "discrete_map":
        mapping = params.get("map", params.get("mapping", {}))
        # Try exact match first, then closest key
        str_raw = str(int(raw)) if raw == int(raw) else str(raw)
        if str_raw in mapping:
            return float(mapping[str_raw])
        # Find closest numeric key
        try:
            numeric_keys = [(float(k), v) for k, v in mapping.items()]
            numeric_keys.sort(key=lambda x: abs(x[0] - raw))
            return float(numeric_keys[0][1])
        except (ValueError, IndexError):
            return 0.5

    if norm_type == "threshold":
        thresh = params.get("threshold", 0.5)
        return 1.0 if raw >= thresh else 0.0

    # Unknown normalization type — return 0.5 as neutral
    logger.warning(f"Unknown normalization type: {norm_type}")
    return 0.5


def sigmoid(x: float, midpoint: float = 0.5, steepness: float = 6.0) -> float:
    """Standard sigmoid: P = 1 / (1 + exp(-steepness * (x - midpoint)))."""
    z = -steepness * (x - midpoint)
    # Prevent overflow
    if z > 500:
        return 0.0
    if z < -500:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def compute_weighted_score(indicators: list[dict],
                           values: dict[str, float]) -> tuple[float, dict]:
    """Compute weighted score from available indicator values.

    Args:
        indicators: List of indicator defs from scoring_functions, each with
                    indicator_id, weight, normalization, normalization_params.
        values: Dict of {indicator_id: raw_value} from the indicator store.

    Returns:
        (weighted_score, metadata) where metadata has:
          - coverage_ratio: fraction of indicators with values
          - indicators_used: list of dicts with id, raw, normalized, weight
          - indicators_missing: list of indicator IDs without values
          - total_weight_used: sum of original weights of available indicators
    """
    used = []
    missing = []
    total_weight_available = 0.0

    for ind in indicators:
        ind_id = ind.get("indicator_id", "")
        raw = values.get(ind_id)

        if raw is None:
            missing.append(ind_id)
            continue

        norm_type = ind.get("normalization", "linear_scale")
        norm_params = ind.get("normalization_params", {})
        weight = float(ind.get("weight", 0))

        normalized = normalize_value(float(raw), norm_type, norm_params)
        total_weight_available += weight

        used.append({
            "indicator_id": ind_id,
            "raw_value": float(raw),
            "normalized": round(normalized, 4),
            "original_weight": weight,
        })

    total_indicators = len(indicators)
    coverage = len(used) / total_indicators if total_indicators > 0 else 0.0

    if not used:
        return 0.5, {
            "coverage_ratio": 0.0,
            "indicators_used": [],
            "indicators_missing": missing,
            "total_weight_used": 0.0,
        }

    # Re-normalize weights so they sum to 1.0 across available indicators only
    weighted_score = 0.0
    for u in used:
        renorm_weight = u["original_weight"] / total_weight_available
        u["renormalized_weight"] = round(renorm_weight, 4)
        weighted_score += u["normalized"] * renorm_weight

    return round(weighted_score, 6), {
        "coverage_ratio": round(coverage, 3),
        "indicators_used": used,
        "indicators_missing": missing,
        "total_weight_used": round(total_weight_available, 4),
    }


def compute_sub_probability(scoring_func: dict,
                            values: dict[str, float]) -> dict:
    """Compute one sub-probability from its scoring function and indicator values.

    Args:
        scoring_func: Scoring function def from research (has input_indicators,
                      midpoint, steepness).
        values: Dict of {indicator_id: raw_value}.

    Returns dict with:
        value: computed probability [0.05, 0.95]
        weighted_score: raw weighted score before sigmoid
        coverage_ratio: fraction of indicators with data
        is_dynamic: True if enough coverage to use dynamic value
        indicators_used: details of each indicator used
        indicators_missing: IDs of indicators without values
    """
    indicators = scoring_func.get("input_indicators", [])
    midpoint = float(scoring_func.get("midpoint", 0.5))
    steepness = float(scoring_func.get("steepness", 6.0))

    ws, meta = compute_weighted_score(indicators, values)

    coverage = meta["coverage_ratio"]
    is_dynamic = coverage >= MIN_COVERAGE_FOR_DYNAMIC

    if is_dynamic:
        p_sub = sigmoid(ws, midpoint, steepness)
        p_sub = max(P_SUB_FLOOR, min(P_SUB_CEILING, p_sub))
    else:
        p_sub = None  # Caller should use static fallback

    return {
        "value": round(p_sub, 4) if p_sub is not None else None,
        "weighted_score": ws,
        "coverage_ratio": coverage,
        "is_dynamic": is_dynamic,
        "indicators_used": meta["indicators_used"],
        "indicators_missing": meta["indicators_missing"],
    }


def compute_dynamic_prior(event_scoring: dict,
                          values: dict[str, float],
                          static_fallback: dict) -> dict:
    """Compute dynamic prior for an event using scoring functions.

    For each sub-probability:
      - If enough indicators have values, compute dynamically
      - Otherwise, use the static research value as fallback

    Args:
        event_scoring: The scoring_functions dict from method_c_full_research.json
                       (has p_preconditions, p_trigger, p_implementation).
        values: Dict of {indicator_id: raw_value} from indicator store.
        static_fallback: Dict with p_pre, p_trig, p_impl from research overrides.

    Returns dict compatible with engine's prior_data format:
        prior, method, sub_probabilities, data_source, etc.
    """
    sub_prob_map = {
        "p_preconditions": "p_pre",
        "p_trigger": "p_trig",
        "p_implementation": "p_impl",
    }

    results = {}
    any_dynamic = False
    total_coverage = 0.0

    for scoring_key, prior_key in sub_prob_map.items():
        sf = event_scoring.get(scoring_key, {})
        static_val = float(static_fallback.get(prior_key, 0.50))

        if not sf or not sf.get("input_indicators"):
            # No scoring function defined — use static
            results[scoring_key] = {
                "value": static_val,
                "is_dynamic": False,
                "coverage_ratio": 0.0,
                "source": "static_research",
                "indicators_used": [],
                "indicators_missing": [],
            }
            continue

        sub_result = compute_sub_probability(sf, values)
        total_coverage += sub_result["coverage_ratio"]

        if sub_result["is_dynamic"] and sub_result["value"] is not None:
            results[scoring_key] = {
                "value": sub_result["value"],
                "is_dynamic": True,
                "coverage_ratio": sub_result["coverage_ratio"],
                "weighted_score": sub_result["weighted_score"],
                "source": "dynamic_scoring",
                "indicators_used": sub_result["indicators_used"],
                "indicators_missing": sub_result["indicators_missing"],
            }
            any_dynamic = True
        else:
            results[scoring_key] = {
                "value": static_val,
                "is_dynamic": False,
                "coverage_ratio": sub_result["coverage_ratio"],
                "source": "static_fallback",
                "indicators_used": sub_result.get("indicators_used", []),
                "indicators_missing": sub_result.get("indicators_missing", []),
            }

    if not any_dynamic:
        return {"is_dynamic": False}

    p_pre = results["p_preconditions"]["value"]
    p_trig = results["p_trigger"]["value"]
    p_impl = results["p_implementation"]["value"]
    prior = round(p_pre * p_trig * p_impl, 4)
    avg_coverage = round(total_coverage / 3, 3)

    # Determine confidence from coverage
    if avg_coverage >= 0.70:
        confidence = "High"
    elif avg_coverage >= 0.40:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Count dynamic vs static sub-probs
    dynamic_count = sum(1 for r in results.values() if r["is_dynamic"])

    # Build evidence strings for method_c_prior() compatibility
    evidence = {}
    for scoring_key in ("p_preconditions", "p_trigger", "p_implementation"):
        r = results[scoring_key]
        if r["is_dynamic"]:
            used_names = [u["indicator_id"] for u in r["indicators_used"]]
            evidence[scoring_key] = (
                f"Dynamic scoring ({len(used_names)} indicators: "
                f"{', '.join(used_names)}). "
                f"Coverage: {r['coverage_ratio']:.0%}"
            )
        else:
            evidence[scoring_key] = "Static research value (insufficient indicator coverage)"

    return {
        "is_dynamic": True,
        "prior": prior,
        "method": "C",
        "formula": f"{p_pre} x {p_trig} x {p_impl} = {prior}",
        "data_source": f"Dynamic scoring ({dynamic_count}/3 sub-probs dynamic, "
                       f"coverage {avg_coverage:.0%})",
        "source_id": "dynamic",
        "confidence": confidence,
        "sub_probabilities": {
            "p_preconditions": {
                "value": p_pre,
                "evidence": evidence["p_preconditions"],
                "is_dynamic": results["p_preconditions"]["is_dynamic"],
                "coverage": results["p_preconditions"]["coverage_ratio"],
            },
            "p_trigger": {
                "value": p_trig,
                "evidence": evidence["p_trigger"],
                "is_dynamic": results["p_trigger"]["is_dynamic"],
                "coverage": results["p_trigger"]["coverage_ratio"],
            },
            "p_implementation": {
                "value": p_impl,
                "evidence": evidence["p_implementation"],
                "is_dynamic": results["p_implementation"]["is_dynamic"],
                "coverage": results["p_implementation"]["coverage_ratio"],
            },
        },
        "dynamic_metadata": {
            "dynamic_sub_probs": dynamic_count,
            "avg_coverage": avg_coverage,
            "details": results,
        },
    }
