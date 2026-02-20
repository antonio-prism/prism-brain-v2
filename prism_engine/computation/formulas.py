"""
PRISM Engine — P_global and P_client calculation formulas.

Layer 1: P_global = prior × ∏(modifiers), clipped to [floor, 0.95]
Layer 2: P_client = P_global × exposure × net_vulnerability (Phase 2+)
"""

import logging
from functools import reduce
from operator import mul

logger = logging.getLogger(__name__)


def _product(values: list[float]) -> float:
    """Compute product of a list of floats."""
    return reduce(mul, values, 1.0)


def calculate_p_global(prior: float, modifiers: list[float]) -> dict:
    """
    Calculate P_global (Layer 1 probability).

    P_global = prior × ∏(modifiers), with floor and ceiling.
    Floor = max(0.001, 0.1 × prior)  — never drops below 10% of the prior
    Ceiling = 0.95
    """
    raw = prior
    for m in modifiers:
        raw *= m

    floor = max(0.001, 0.1 * prior)
    ceiling = 0.95

    p_global = round(max(floor, min(ceiling, raw)), 4)
    was_capped = raw > ceiling or raw < floor

    return {
        "p_global": p_global,
        "p_global_raw": round(raw, 4),
        "prior": prior,
        "modifiers_applied": len(modifiers),
        "modifier_product": round(
            _product(modifiers) if modifiers else 1.0, 4
        ),
        "floor": round(floor, 4),
        "ceiling": ceiling,
        "was_capped": was_capped,
        "capped_at": ceiling if raw > ceiling else (floor if raw < floor else None),
    }


def calculate_p_client(
    p_global: float,
    geographic_exposure: float = 1.0,
    industry_exposure: float = 1.0,
    scale_factor: float = 1.0,
    vulnerability_score: float = 50.0,
    resilience_score: float = 50.0,
) -> dict:
    """
    Calculate P_client (Layer 2 probability). Phase 2+ only.

    P_client = P_global × exposure × net_vulnerability
    where:
      exposure = geographic_exposure × industry_exposure × scale_factor
      net_vulnerability = (vulnerability_score/100) × (1 - resilience_score/100)
    """
    exposure = geographic_exposure * industry_exposure * scale_factor
    net_vulnerability = (vulnerability_score / 100.0) * (1.0 - resilience_score / 100.0)
    raw = p_global * exposure * net_vulnerability

    p_client = round(max(0.001, min(0.95, raw)), 4)

    return {
        "p_client": p_client,
        "p_global": p_global,
        "exposure": round(exposure, 4),
        "geographic_exposure": geographic_exposure,
        "industry_exposure": industry_exposure,
        "scale_factor": scale_factor,
        "vulnerability_score": vulnerability_score,
        "resilience_score": resilience_score,
        "net_vulnerability": round(net_vulnerability, 4),
    }
