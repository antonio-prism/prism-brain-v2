"""
PRISM Engine — Modifier calibration.

Two methods:
  Standard ratio method: for continuous indicators (time series)
  Categorical method: for binary/on-off conditions
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Pre-defined categorical modifier conditions
CATEGORICAL_MODIFIERS = {
    "us_election_year": {
        "if_true": 1.25,
        "justification": (
            "WTO data: Trade restrictive measures increase ~25% in US election years "
            "(2016: +28%, 2020: +22%, 2024: +31% vs non-election year baseline)"
        ),
    },
    "active_military_conflict_oecd": {
        "if_true": 1.40,
        "justification": (
            "GPR Index averaged 40% above 5yr mean during 2022-2024 "
            "(Ukraine conflict period)"
        ),
    },
    "who_pheic_active": {
        "if_true": 1.50,
        "justification": (
            "During COVID PHEIC (2020-2023), supply chain disruption frequency "
            "was ~50% above historical average per Munich Re"
        ),
    },
    "is_pheic_active": {
        "if_true": 1.50,
        "justification": (
            "During COVID PHEIC (2020-2023), supply chain disruption frequency "
            "was ~50% above historical average per Munich Re"
        ),
    },
    "enso_el_nino_active": {
        "if_true": 1.20,
        "justification": (
            "EM-DAT data: 20% more climate-related disasters globally "
            "during El Niño years"
        ),
    },
    "ecb_rate_hiking": {
        "if_true": 1.15,
        "justification": (
            "During rate-hiking periods, Allianz insolvency index rises ~15% "
            "(observed 2022-2023)"
        ),
    },
}


def calibrate_ratio_modifier(time_series: pd.Series, window: int = 60) -> dict:
    """
    Standard modifier calibration from a monthly time series.

    Args:
        time_series: Monthly values of the indicator.
        window: Rolling window for baseline (months). Default 60 = 5 years.

    Returns:
        Dict with p5, p50, p95, floor, ceiling, current_modifier.
    """
    if len(time_series) < window // 2:
        return {
            "status": "INSUFFICIENT_DATA",
            "n_observations": len(time_series),
            "current_modifier": 1.0,
        }

    baseline = time_series.rolling(window, min_periods=window // 2).mean()
    ratio = time_series / baseline
    ratio = ratio.dropna()

    if ratio.empty:
        return {
            "status": "NO_VALID_RATIOS",
            "current_modifier": 1.0,
        }

    p5 = float(ratio.quantile(0.05))
    p50 = float(ratio.quantile(0.50))
    p95 = float(ratio.quantile(0.95))

    floor = max(0.50, round(p5, 2))
    ceiling = min(3.00, round(p95, 2))

    current_value = float(ratio.iloc[-1])
    current_modifier = round(max(floor, min(ceiling, current_value)), 2)

    return {
        "method": "ratio",
        "n_observations": len(ratio),
        "p5": round(p5, 2),
        "p50": round(p50, 2),
        "p95": round(p95, 2),
        "floor": floor,
        "ceiling": ceiling,
        "current_value": round(current_value, 2),
        "current_modifier": current_modifier,
        "status": "COMPUTED",
    }


def categorical_modifier(condition_name: str, condition_met: bool) -> dict:
    """
    Categorical modifier for binary/on-off conditions.

    Args:
        condition_name: Key into CATEGORICAL_MODIFIERS table.
        condition_met: Is the condition currently true?

    Returns:
        Dict with modifier value and justification.
    """
    preset = CATEGORICAL_MODIFIERS.get(condition_name)
    if preset is None:
        logger.warning(f"Unknown categorical modifier: {condition_name}")
        return {
            "type": "categorical",
            "condition_name": condition_name,
            "condition_met": condition_met,
            "modifier": 1.0,
            "justification": "Unknown condition — defaulting to 1.0",
        }

    return {
        "type": "categorical",
        "condition_name": condition_name,
        "condition_met": condition_met,
        "modifier": preset["if_true"] if condition_met else 1.00,
        "justification": preset["justification"],
    }


def scaling_modifier(raw_value: float, scaling_constant: float,
                     floor: float = 0.50, ceiling: float = 3.00,
                     formula_name: str = "") -> dict:
    """
    Convert a raw indicator value to a modifier using a linear scaling formula.
    modifier = 1.0 + (raw_value × scaling_constant)

    Args:
        raw_value: The raw indicator value (e.g., temperature anomaly in σ).
        scaling_constant: The conversion factor.
        floor/ceiling: Bounds for the modifier.
        formula_name: Human-readable formula description.
    """
    modifier_raw = 1.0 + (raw_value * scaling_constant)
    modifier = round(max(floor, min(ceiling, modifier_raw)), 2)

    return {
        "method": "scaling",
        "raw_value": round(raw_value, 2),
        "scaling_constant": scaling_constant,
        "modifier_raw": round(modifier_raw, 2),
        "modifier": modifier,
        "floor": floor,
        "ceiling": ceiling,
        "formula": formula_name or f"1.0 + ({raw_value:.2f} × {scaling_constant})",
        "status": "COMPUTED",
    }
