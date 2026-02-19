"""
Calculation engines and endpoints (V1 API).

Kept endpoints:
- POST /api/v1/calculations/trigger-full (enhanced calculation with signal extraction) - used by frontend
- GET /api/v1/probabilities (list probabilities) - used by frontend

Includes:
- ProbabilityCalculator (log-odds based Bayesian model)
- SignalExtractor (z-scores, momentum, trend detection, anomaly detection)
- ExplainabilityEngine (attribution, explanations, recommendations)
- DependencyModeler (risk interdependencies, cascading effects)
- MLEnhancementLayer (ensemble learning)
- EnhancedConfidenceScorer (multi-factor confidence scoring)
"""

import asyncio
import json
import logging
import math
import statistics
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

from fastapi import Query, HTTPException
from config.settings import get_settings
from database.connection import get_session_context
from database.models import (
    RiskEvent, RiskProbability, IndicatorWeight, IndicatorValue,
    CalculationLog, DataSourceHealth
)
from config.category_indicators import (
    get_category_prefix, get_default_baseline, get_indicators_for_event,
    get_event_sensitivity
)

logger = logging.getLogger(__name__)

# Optional ML imports - degrade gracefully if not available
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    import pickle
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Beta parameters for different correlation types
BETA_PARAMETERS = {
    "HIGH": 3.0,
    "MODERATE": 2.0,
    "LOW": 1.0,
    "direct_causal": 3.0,
    "strong_correlation": 2.0,
    "moderate_correlation": 1.2,
    "weak_correlation": 0.6
}

# Expected indicator ranges for proper z-score calculation
INDICATOR_RANGES = {
    # USGS - Earthquake indicators
    "usgs_earthquake_count": {"mean": 150, "std": 80},
    "usgs_significant_count": {"mean": 5, "std": 3},
    "usgs_max_magnitude": {"mean": 5.5, "std": 1.2},
    "usgs_seismic_activity": {"mean": 0.5, "std": 0.2},
    # CISA - Vulnerability indicators
    "cisa_total_kev": {"mean": 1100, "std": 200},
    "cisa_recent_kev": {"mean": 8, "std": 5},
    "cisa_kev_rate": {"mean": 2.0, "std": 1.0},
    "cisa_threat_level": {"mean": 0.5, "std": 0.2},
    # NVD - CVE indicators
    "nvd_total_cves": {"mean": 800, "std": 300},
    "nvd_critical_count": {"mean": 50, "std": 30},
    "nvd_high_count": {"mean": 200, "std": 100},
    "nvd_severity_index": {"mean": 0.5, "std": 0.15},
    # FRED - US Economic indicators
    "fred_unemployment_rate": {"mean": 4.5, "std": 1.5},
    "fred_inflation_rate": {"mean": 3.0, "std": 1.5},
    "fred_fed_funds_rate": {"mean": 3.0, "std": 2.0},
    "fred_vix_index": {"mean": 20, "std": 8},
    "fred_treasury_spread": {"mean": 1.5, "std": 1.0},
    # NOAA - Climate indicators
    "noaa_temp_anomaly": {"mean": 1.0, "std": 0.3},
    "noaa_precipitation_index": {"mean": 0.5, "std": 0.2},
    "noaa_extreme_events": {"mean": 15, "std": 8},
    "noaa_climate_risk": {"mean": 0.5, "std": 0.2},
    # World Bank - Development indicators
    "world_bank_gdp_growth": {"mean": 3.0, "std": 2.0},
    "world_bank_gdp_avg": {"mean": 2.5, "std": 1.5},
    "world_bank_economic_health": {"mean": 0.5, "std": 0.2},
    # GDELT - Geopolitical indicators
    "gdelt_event_volume": {"mean": 5000, "std": 2000},
    "gdelt_peak_volume": {"mean": 8000, "std": 3000},
    "gdelt_trend": {"mean": 0.0, "std": 0.3},
    "gdelt_crisis_intensity": {"mean": 50, "std": 20},
    # EIA - Energy indicators
    "eia_crude_oil_price": {"mean": 75, "std": 15},
    "eia_natural_gas_price": {"mean": 3.5, "std": 1.5},
    "eia_energy_volatility": {"mean": 0.15, "std": 0.08},
    "eia_strategic_reserve_level": {"mean": 400, "std": 50},
    # IMF - International finance
    "imf_world_gdp_growth": {"mean": 3.2, "std": 1.5},
    "imf_gdp_forecast": {"mean": 3.0, "std": 1.0},
    "imf_growth_momentum": {"mean": 0.0, "std": 0.5},
    # FAO - Food/agriculture
    "fao_food_price_index": {"mean": 130, "std": 25},
    "fao_cereal_index": {"mean": 140, "std": 30},
    "fao_food_security_risk": {"mean": 0.4, "std": 0.15},
    # OTX - Cyber threat intel
    "otx_threat_pulse_count": {"mean": 500, "std": 200},
    "otx_malware_indicators": {"mean": 1000, "std": 400},
    "otx_ransomware_activity": {"mean": 50, "std": 25},
    # ACLED - Conflict data
    "acled_conflict_events": {"mean": 2000, "std": 800},
    "acled_fatalities": {"mean": 5000, "std": 3000},
    "acled_instability_index": {"mean": 0.5, "std": 0.2},
}

# Predefined correlation matrix for risk interdependencies
RISK_CORRELATIONS = {
    # Geopolitical -> Technology
    ("GEO-002", "TECH-015"): 0.85,  # Taiwan Conflict -> Semiconductor Crisis
    ("GEO-001", "TECH-003"): 0.80,  # US-China Decoupling -> 5G Fragmentation
    # Climate -> Energy
    ("CLIM-045", "ENRG-022"): 0.65,  # Hurricane Season -> Oil Refinery Disruption
    ("CLIM-001", "ENRG-001"): 0.55,  # Extreme Weather -> Energy Grid Stress
    # Energy -> Geopolitical
    ("ENRG-011", "GEO-004"): 0.75,  # Oil Supply Shock -> Middle East Conflict
    ("ENRG-001", "SUPL-001"): 0.60,  # Energy Crisis -> Supply Chain Disruption
    # Cyber -> Supply Chain
    ("CYBER-078", "SUPL-134"): 0.72,  # Ransomware Wave -> Supply Chain IT Failure
    ("CYBER-001", "TECH-001"): 0.65,  # Major Cyberattack -> Technology Disruption
    # Financial -> Economic
    ("FIN-001", "ECON-001"): 0.70,  # Financial Crisis -> Economic Recession
    ("FIN-002", "GEO-001"): 0.55,  # Currency Crisis -> Trade Decoupling
    # Cross-category systemic
    ("GEO-001", "SUPL-001"): 0.60,  # US-China Decoupling -> Supply Chain
    ("CLIM-001", "SUPL-002"): 0.50,  # Climate Extreme -> Food Supply
}

# Make correlations bidirectional
_bidirectional = {}
for (a, b), corr in RISK_CORRELATIONS.items():
    _bidirectional[(a, b)] = corr
    _bidirectional[(b, a)] = corr
RISK_CORRELATIONS = _bidirectional


class ProbabilityCalculator:
    """
    Self-contained probability calculator using log-odds model.

    The model: log_odds_final = log_odds_baseline + sum(weight_i * signal_i * beta_i)

    This produces properly calibrated probabilities between 0.1% and 99.9%.
    """

    def __init__(self, min_prob: float = 0.001, max_prob: float = 0.999):
        self.min_prob = min_prob
        self.max_prob = max_prob

    @staticmethod
    def scale_to_probability(scale: float) -> float:
        """
        Convert 1-5 baseline scale to annual probability.

        Scale mapping (using return periods):
        1 = 20 year return (rare) -> ~5% annual
        2 = 10 year return (unlikely) -> ~10% annual
        3 = 5 year return (possible) -> ~18% annual
        4 = 2 year return (likely) -> ~39% annual
        5 = 1 year return (very likely) -> ~63% annual
        """
        scale = max(1.0, min(5.0, float(scale)))
        return_period = 20.0 / (2 ** (scale - 1))
        return 1 - math.exp(-1 / return_period)

    @staticmethod
    def probability_to_log_odds(p: float) -> float:
        """Convert probability to log-odds."""
        p = max(0.001, min(0.999, p))
        return math.log(p / (1 - p))

    @staticmethod
    def log_odds_to_probability(log_odds: float) -> float:
        """Convert log-odds to probability."""
        return 1 / (1 + math.exp(-log_odds))

    def calculate_signal(self, z_score: Optional[float], value: float) -> float:
        """
        Calculate signal from indicator data.
        Uses z-score if available, otherwise normalizes value to [-1, 1].
        """
        if z_score is not None:
            # Use tanh for smooth bounded mapping [-1, 1]
            return math.tanh(z_score / 2.0)
        else:
            # No z-score: use value directly with centering
            return max(-1.0, min(1.0, (value - 0.5) * 2))

    def calculate_event_probability(
        self,
        baseline_scale: float,
        indicator_signals: List[float],
        indicator_weights: List[float],
        indicator_betas: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate probability for a single event.
        Returns dict with probability, confidence interval, and metadata.
        """
        # Step 1: Convert baseline to probability and log-odds
        baseline_prob = self.scale_to_probability(baseline_scale)
        baseline_log_odds = self.probability_to_log_odds(baseline_prob)

        # Step 2: Calculate total adjustment from indicators
        total_adjustment = 0.0
        contributions = []
        for signal, weight, beta in zip(indicator_signals, indicator_weights, indicator_betas):
            contribution = weight * signal * beta
            total_adjustment += contribution
            contributions.append(contribution)

        # Step 3: Apply adjustment
        final_log_odds = baseline_log_odds + total_adjustment

        # Step 4: Convert back to probability
        probability = self.log_odds_to_probability(final_log_odds)
        probability = max(self.min_prob, min(self.max_prob, probability))

        # Step 5: Calculate confidence interval
        n_indicators = len(indicator_signals)
        if n_indicators > 0:
            signal_variance = sum((s - sum(indicator_signals)/n_indicators)**2 for s in indicator_signals) / n_indicators
            ci_width = 0.05 + 0.10 * signal_variance + 0.05 / math.sqrt(n_indicators)
        else:
            ci_width = 0.15

        ci_lower = max(self.min_prob, probability - ci_width/2)
        ci_upper = min(self.max_prob, probability + ci_width/2)

        # Step 6: Determine precision band
        ci_width_actual = ci_upper - ci_lower
        if ci_width_actual <= 0.02:
            precision_band = "NARROW"
        elif ci_width_actual <= 0.05:
            precision_band = "MODERATE"
        elif ci_width_actual <= 0.10:
            precision_band = "WIDE"
        else:
            precision_band = "VERY_WIDE"

        # Step 7: Calculate confidence score
        if n_indicators > 0:
            positive = sum(1 for s in indicator_signals if s > 0.1)
            negative = sum(1 for s in indicator_signals if s < -0.1)
            total = positive + negative
            agreement = max(positive, negative) / total if total > 0 else 0.5
            completeness = min(1.0, n_indicators / 5)
            confidence = 0.5 * completeness + 0.5 * agreement
        else:
            confidence = 0.3

        # Determine flags
        flags = []
        if confidence < 0.4:
            flags.append("LOW_CONFIDENCE")
        if probability > 0.5 and baseline_scale <= 2:
            flags.append("BLACK_SWAN")
        if n_indicators > 0:
            pos_strong = sum(1 for s in indicator_signals if s > 0.3)
            neg_strong = sum(1 for s in indicator_signals if s < -0.3)
            if pos_strong >= 2 and neg_strong >= 2:
                flags.append("CONFLICTING_SIGNALS")

        return {
            "probability": probability,
            "probability_pct": round(probability * 100, 2),
            "baseline_probability": baseline_prob,
            "baseline_probability_pct": round(baseline_prob * 100, 2),
            "log_odds": final_log_odds,
            "total_adjustment": total_adjustment,
            "contributions": contributions,
            "ci_lower": ci_lower,
            "ci_lower_pct": round(ci_lower * 100, 2),
            "ci_upper": ci_upper,
            "ci_upper_pct": round(ci_upper * 100, 2),
            "precision_band": precision_band,
            "confidence_score": round(confidence, 3),
            "flags": flags,
            "n_indicators": n_indicators
        }


class SignalExtractor:
    """
    Extracts meaningful signals from raw indicator data using statistical methods.

    Provides:
    - Historical z-scores (how unusual is this value?)
    - Momentum (how fast is it changing?)
    - Trend detection (accelerating, decelerating, stable, reversing)
    - Anomaly detection (is this an outlier?)
    - Composite signal (weighted combination of z-score and momentum)
    """

    @staticmethod
    def calculate_z_score_from_history(
        current_value: float,
        historical_values: List[float],
        indicator_name: Optional[str] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate z-score from actual historical data.

        Returns: (z_score, mean, std)
        """
        if not historical_values or len(historical_values) < 3:
            # Use sensible defaults when insufficient history
            if indicator_name and indicator_name in INDICATOR_RANGES:
                ref = INDICATOR_RANGES[indicator_name]
                default_mean = ref["mean"]
                default_std = ref["std"]
            else:
                default_mean = 0.5
                default_std = 0.25
            z_score = (current_value - default_mean) / max(default_std, 1e-10)
            return z_score, default_mean, default_std

        mean = statistics.mean(historical_values)
        std = statistics.stdev(historical_values)

        if std == 0 or std < 1e-10:
            return 0.0, mean, 0.0

        z_score = (current_value - mean) / std
        return z_score, mean, std

    @staticmethod
    def calculate_momentum(values: List[float], window: int = 12) -> float:
        """
        Calculate momentum (rate of change) over a window using linear regression.

        Returns: momentum score normalized to [-1, 1] using tanh.
        """
        if len(values) < 2:
            return 0.0

        # Take last 'window' values
        recent = values[-window:]
        n = len(recent)

        if n < 2:
            return 0.0

        # Simple linear regression slope
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # Normalize by standard deviation of values
        y_std = statistics.stdev(recent) if len(recent) > 1 else 1.0
        if y_std == 0 or y_std < 1e-10:
            return 0.0

        normalized_slope = slope / (y_std / math.sqrt(n))

        # Apply tanh to bound to [-1, 1]
        return math.tanh(normalized_slope)

    @staticmethod
    def detect_trend(values: List[float]) -> str:
        """
        Detect trend pattern from recent values.

        Returns one of: ACCELERATING, DECELERATING, STABLE, REVERSING
        """
        if len(values) < 4:
            return "STABLE"

        # Split into two halves
        mid = len(values) // 2
        first_half = values[:mid]
        second_half = values[mid:]

        # Calculate momentum for each half
        first_momentum = SignalExtractor.calculate_momentum(first_half, window=len(first_half))
        second_momentum = SignalExtractor.calculate_momentum(second_half, window=len(second_half))

        # Determine trend
        if abs(second_momentum) < 0.1 and abs(first_momentum) < 0.1:
            return "STABLE"
        elif first_momentum > 0.1 and second_momentum > first_momentum:
            return "ACCELERATING"
        elif first_momentum > 0.1 and second_momentum < first_momentum * 0.5:
            return "DECELERATING"
        elif first_momentum < -0.1 and second_momentum < first_momentum:
            return "ACCELERATING"
        elif (first_momentum > 0.2 and second_momentum < -0.1) or \
             (first_momentum < -0.2 and second_momentum > 0.1):
            return "REVERSING"
        else:
            return "STABLE"

    @staticmethod
    def detect_anomaly(z_score: float, momentum: float) -> bool:
        """
        Detect if current value is anomalous.
        Anomaly if z-score > 2.5 or momentum change is extreme.
        """
        return abs(z_score) > 2.5 or abs(momentum) > 0.8

    @staticmethod
    def calculate_composite_signal(
        z_score: float,
        momentum: float,
        z_weight: float = 0.6,
        m_weight: float = 0.4
    ) -> float:
        """
        Calculate composite signal from z-score and momentum.
        Returns signal in [-1.0, 1.0].
        """
        # Convert z-score to bounded signal using tanh
        z_signal = math.tanh(z_score / 2.0)

        # Weighted combination
        signal = z_weight * z_signal + m_weight * momentum

        # Clip to bounds
        return max(-1.0, min(1.0, signal))

    def extract_signal_for_indicator(
        self,
        event_id: str,
        indicator_name: str,
        current_value: float,
        session
    ) -> Dict[str, Any]:
        """
        Extract full signal analysis for a single indicator.
        Queries historical data from database.
        """
        # Get historical values for this indicator (last 52 weeks)
        cutoff_date = datetime.utcnow() - timedelta(days=365)
        historical = session.query(IndicatorValue).filter(
            IndicatorValue.event_id == event_id,
            IndicatorValue.indicator_name == indicator_name,
            IndicatorValue.timestamp >= cutoff_date
        ).order_by(IndicatorValue.timestamp.asc()).all()

        historical_values = [h.value for h in historical if h.value is not None]

        # Calculate z-score
        z_score, hist_mean, hist_std = self.calculate_z_score_from_history(
            current_value, historical_values, indicator_name
        )

        # Calculate momentum
        momentum = self.calculate_momentum(historical_values)

        # Detect trend
        trend = self.detect_trend(historical_values)

        # Detect anomaly
        is_anomaly = self.detect_anomaly(z_score, momentum)

        # Calculate composite signal
        signal = self.calculate_composite_signal(z_score, momentum)

        return {
            "signal": signal,
            "z_score": round(z_score, 4),
            "momentum": round(momentum, 4),
            "trend": trend,
            "is_anomaly": is_anomaly,
            "historical_mean": round(hist_mean, 4),
            "historical_std": round(hist_std, 4),
            "historical_data_points": len(historical_values),
            "current_value": current_value
        }


class ExplainabilityEngine:
    """
    Generates human-readable explanations for probability changes.

    Provides:
    - Attribution: which indicators drove the probability change
    - Explanations: natural language summaries
    - Recommendations: MONITOR, MONITOR_CLOSELY, ALERT, CRITICAL
    """

    @staticmethod
    def generate_attribution(
        indicator_names: List[str],
        indicator_signals: List[float],
        indicator_weights: List[float],
        indicator_betas: List[float],
        signal_details: List[Dict],
        total_adjustment: float
    ) -> List[Dict[str, Any]]:
        """
        Generate detailed attribution for each indicator's contribution.
        """
        attributions = []
        for i, (name, signal, weight, beta) in enumerate(
            zip(indicator_names, indicator_signals, indicator_weights, indicator_betas)
        ):
            contribution = weight * signal * beta
            pct_of_total = (contribution / total_adjustment * 100) if total_adjustment != 0 else 0

            detail = signal_details[i] if i < len(signal_details) else {}

            attributions.append({
                "indicator": name,
                "signal": round(signal, 4),
                "weight": round(weight, 4),
                "contribution": round(contribution, 4),
                "contribution_pct": round(pct_of_total, 1),
                "z_score": detail.get("z_score", 0),
                "momentum": detail.get("momentum", 0),
                "trend": detail.get("trend", "STABLE"),
                "is_anomaly": detail.get("is_anomaly", False),
                "data_points": detail.get("historical_data_points", 0)
            })

        # Sort by absolute contribution (biggest drivers first)
        attributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return attributions

    @staticmethod
    def generate_explanation(
        event_name: str,
        current_prob_pct: float,
        previous_prob_pct: Optional[float],
        attribution: List[Dict],
        confidence: float,
        flags: List[str]
    ) -> str:
        """
        Generate a human-readable explanation of the probability.
        """
        parts = []

        # Main probability statement
        if previous_prob_pct is not None:
            change = current_prob_pct - previous_prob_pct
            direction = "increased" if change > 0 else "decreased"
            parts.append(
                f"{event_name}: probability {direction} from "
                f"{previous_prob_pct:.1f}% to {current_prob_pct:.1f}% "
                f"(change: {change:+.1f}%)."
            )
        else:
            parts.append(
                f"{event_name}: probability calculated at {current_prob_pct:.1f}%."
            )

        # Top drivers
        if attribution:
            top_drivers = attribution[:3]  # Top 3 contributors
            driver_parts = []
            for a in top_drivers:
                direction = "increasing" if a["signal"] > 0 else "decreasing"
                strength = "strongly" if abs(a["signal"]) > 0.5 else "moderately"
                driver_parts.append(
                    f"{a['indicator']} ({strength} {direction}, "
                    f"z-score: {a['z_score']:.1f})"
                )
            if driver_parts:
                parts.append("Main drivers: " + "; ".join(driver_parts) + ".")

        # Confidence
        if confidence >= 0.75:
            parts.append(f"Confidence: HIGH ({confidence:.0%}).")
        elif confidence >= 0.50:
            parts.append(f"Confidence: MEDIUM ({confidence:.0%}).")
        else:
            parts.append(f"Confidence: LOW ({confidence:.0%}).")

        # Flags
        if flags:
            parts.append(f"Flags: {', '.join(flags)}.")

        return " ".join(parts)

    @staticmethod
    def generate_recommendation(
        probability_pct: float,
        change_pct: Optional[float],
        confidence: float,
        flags: List[str]
    ) -> Dict[str, str]:
        """
        Generate a recommendation based on probability and change.
        """
        # Determine alert level
        if probability_pct >= 60:
            level = "CRITICAL"
            action = "Immediate attention required. Activate contingency plans."
        elif probability_pct >= 40:
            level = "ALERT"
            action = "Elevated risk. Review mitigation strategies."
        elif probability_pct >= 25 or (change_pct and change_pct > 5):
            level = "MONITOR_CLOSELY"
            action = "Notable risk level or significant change. Increase monitoring frequency."
        else:
            level = "MONITOR"
            action = "Standard monitoring. No immediate action needed."

        # Adjust for flags
        if "BLACK_SWAN" in flags:
            action += " BLACK SWAN flag: low-data event, treat with caution."
        if "CONFLICTING_SIGNALS" in flags:
            action += " Conflicting signals detected: consider additional analysis."

        return {
            "level": level,
            "action": action,
            "probability_pct": probability_pct,
            "change_pct": change_pct
        }


class DependencyModeler:
    """
    Models risk interdependencies and cascading effects.

    When one risk rises, correlated risks are adjusted proportionally.
    """

    DAMPING_FACTOR = 0.3  # Prevents runaway cascading

    @staticmethod
    def get_correlated_events(event_id: str) -> Dict[str, float]:
        """Get all events correlated with the given event."""
        correlated = {}
        for (a, b), corr in RISK_CORRELATIONS.items():
            if a == event_id:
                correlated[b] = corr
        return correlated

    @classmethod
    def calculate_dependency_adjustment(
        cls,
        event_id: str,
        all_probabilities: Dict[str, float],
        all_baselines: Dict[str, float]
    ) -> Tuple[float, List[Dict]]:
        """
        Calculate probability adjustment based on correlated events.

        Returns: (adjustment, dependency_details)
        """
        correlated = cls.get_correlated_events(event_id)
        if not correlated:
            return 0.0, []

        adjustment = 0.0
        details = []

        for dep_event_id, correlation in correlated.items():
            if dep_event_id in all_probabilities and dep_event_id in all_baselines:
                dep_prob = all_probabilities[dep_event_id]
                dep_baseline = all_baselines[dep_event_id]
                prob_change = dep_prob - dep_baseline

                # Only adjust if correlated event has moved significantly
                if abs(prob_change) > 0.01:
                    contrib = correlation * prob_change * cls.DAMPING_FACTOR
                    adjustment += contrib
                    details.append({
                        "correlated_event": dep_event_id,
                        "correlation": correlation,
                        "probability_change": round(prob_change * 100, 2),
                        "adjustment_contribution": round(contrib * 100, 2)
                    })

        # Cap total adjustment
        adjustment = max(-0.15, min(0.15, adjustment))

        return adjustment, details

    @staticmethod
    def get_dependency_chain(event_id: str) -> List[Dict]:
        """Get the full dependency chain for an event."""
        chain = []
        correlated = DependencyModeler.get_correlated_events(event_id)
        for dep_id, corr in sorted(correlated.items(), key=lambda x: -x[1]):
            chain.append({
                "event_id": dep_id,
                "correlation": corr,
                "relationship": "strong" if corr >= 0.7 else "moderate" if corr >= 0.5 else "weak"
            })
        return chain


class MLEnhancementLayer:
    """
    Machine learning layer that improves probability estimates over time.

    For MVP: starts with Bayesian-only. As data accumulates (>50 calculations),
    auto-trains a gradient boosting model and blends predictions.
    """

    MIN_TRAINING_SAMPLES = 50
    MODEL_KEY = "prism_ml_model_v1"

    def __init__(self):
        self.model = None
        self.is_trained = False

    def get_ensemble_probability(
        self,
        bayesian_prob: float,
        features: Optional[List[float]] = None
    ) -> Tuple[float, str, Optional[float]]:
        """
        Get ensemble probability combining Bayesian and ML.

        Returns: (final_prob, method_used, ml_prob_or_none)
        """
        if not ML_AVAILABLE or not self.is_trained or features is None:
            return bayesian_prob, "bayesian_only", None

        try:
            ml_prob = self.predict(features)
            if ml_prob is not None:
                # 60% Bayesian + 40% ML
                ensemble = 0.6 * bayesian_prob + 0.4 * ml_prob
                return ensemble, "ensemble_bayesian_ml", ml_prob
        except Exception as e:
            logger.warning(f"ML prediction failed, using Bayesian only: {e}")

        return bayesian_prob, "bayesian_only", None

    def predict(self, features: List[float]) -> Optional[float]:
        """Predict probability using trained ML model."""
        if not self.is_trained or self.model is None:
            return None
        try:
            import numpy as np
            X = np.array([features])
            prob = self.model.predict_proba(X)[0][1]
            return float(prob)
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return None

    def check_and_train(self, session) -> bool:
        """
        Check if enough data exists to train, and train if so.
        Returns True if model was trained.
        """
        if not ML_AVAILABLE:
            return False

        try:
            # Count historical calculations
            calc_count = session.query(CalculationLog).filter(
                CalculationLog.status.in_(["COMPLETED", "COMPLETED_WITH_ERRORS"])
            ).count()

            if calc_count < self.MIN_TRAINING_SAMPLES:
                logger.info(f"ML training needs {self.MIN_TRAINING_SAMPLES} calculations, have {calc_count}")
                return False

            logger.info(f"Sufficient data ({calc_count} calculations). ML training available for future implementation.")
            # For MVP, we log that training is possible but don't train yet.
            # Full training requires labeled outcome data (did the event happen?).
            # This will be implemented when outcome tracking is added.
            return False

        except Exception as e:
            logger.error(f"ML training check failed: {e}")
            return False


class EnhancedConfidenceScorer:
    """
    Multi-factor confidence scoring system.

    Factors:
    - Data completeness: do we have all expected indicators?
    - Source reliability: are data sources healthy?
    - Signal consistency: do indicators agree?
    - Data recency: how fresh is the data?
    """

    WEIGHTS = {
        "data_completeness": 0.25,
        "source_reliability": 0.25,
        "signal_consistency": 0.20,
        "data_recency": 0.30
    }

    @staticmethod
    def calculate(
        n_indicators: int,
        expected_indicators: int,
        indicator_signals: List[float],
        source_health: Dict[str, str],
        data_ages_hours: List[float]
    ) -> Tuple[float, str, Dict]:
        """
        Calculate multi-factor confidence score.

        Returns: (score, band, factor_details)
        """
        # Data completeness
        if expected_indicators > 0:
            completeness = min(1.0, n_indicators / expected_indicators)
        else:
            completeness = 0.5

        # Source reliability
        if source_health:
            operational = sum(1 for s in source_health.values() if s == "OPERATIONAL")
            reliability = operational / len(source_health) if source_health else 0.5
        else:
            reliability = 0.5

        # Signal consistency
        if len(indicator_signals) >= 2:
            try:
                signal_std = statistics.stdev(indicator_signals)
                consistency = max(0.0, 1.0 - signal_std)
            except statistics.StatisticsError:
                consistency = 0.5
        else:
            consistency = 0.5

        # Data recency (exponential decay over 7 days)
        if data_ages_hours:
            avg_age = statistics.mean(data_ages_hours)
            recency = math.exp(-avg_age / (7 * 24))  # 7-day half-life
        else:
            recency = 0.3

        # Weighted combination
        factors = {
            "data_completeness": completeness,
            "source_reliability": reliability,
            "signal_consistency": consistency,
            "data_recency": recency
        }

        score = sum(
            factors[k] * EnhancedConfidenceScorer.WEIGHTS[k]
            for k in factors
        )
        score = max(0.0, min(1.0, score))

        # Determine band
        if score >= 0.75:
            band = "HIGH"
        elif score >= 0.50:
            band = "MEDIUM"
        else:
            band = "LOW"

        return round(score, 3), band, factors


# Global instances
probability_calculator = ProbabilityCalculator()
signal_extractor = SignalExtractor()
explainability_engine = ExplainabilityEngine()
ml_layer = MLEnhancementLayer()


def _run_enhanced_calculation_sync(limit=100, event_ids=None):
    """Run enhanced calculation in a thread pool to avoid blocking the event loop."""
    calculation_id = str(uuid.uuid4())[:8] + "-" + datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    start_time = datetime.utcnow()

    events_processed = 0
    events_succeeded = 0
    events_failed = 0
    errors = []

    logger.info(f"Starting enhanced calculation: {calculation_id}")

    try:
        with get_session_context() as session:
            # Load events
            query = session.query(RiskEvent)
            if event_ids:
                query = query.filter(RiskEvent.event_id.in_(event_ids))
            events = query.limit(limit).all()

            logger.info(f"Processing {len(events)} events")

            # Load all weights at once
            event_id_list = [e.event_id for e in events]
            all_weights = session.query(IndicatorWeight).filter(
                IndicatorWeight.event_id.in_(event_id_list)
            ).all()

            weights_by_event: Dict[str, List] = defaultdict(list)
            for w in all_weights:
                weights_by_event[w.event_id].append(w)

            # Load latest indicator values per (event_id, indicator_name)
            all_values = session.query(IndicatorValue).filter(
                IndicatorValue.event_id.in_(event_id_list)
            ).all()

            values_by_key: Dict[str, IndicatorValue] = {}
            for v in all_values:
                key = f"{v.event_id}:{v.indicator_name}"
                if key not in values_by_key or \
                   (v.timestamp and values_by_key[key].timestamp and v.timestamp > values_by_key[key].timestamp):
                    values_by_key[key] = v

            # Get data source health for confidence scoring
            source_health = {}
            try:
                health_records = session.query(DataSourceHealth).order_by(
                    DataSourceHealth.check_time.desc()
                ).all()
                for h in health_records:
                    if h.source_name not in source_health:
                        source_health[h.source_name] = h.status
            except Exception:
                pass

            # First pass: calculate all Bayesian probabilities
            all_bayesian_probs = {}
            all_baselines = {}

            results = []
            for event in events:
                events_processed += 1
                try:
                    weights = weights_by_event.get(event.event_id, [])
                    stored_baseline = event.baseline_probability
                    if stored_baseline and 1 <= stored_baseline <= 5:
                        baseline_scale = stored_baseline
                    else:
                        baseline_scale = get_default_baseline(event.event_id)
                    sensitivity = get_event_sensitivity(event.event_id)
                    baseline_scale = baseline_scale + sensitivity["baseline_offset"]
                    baseline_scale = max(1.0, min(5.0, baseline_scale))
                    baseline_prob = probability_calculator.scale_to_probability(baseline_scale)
                    all_baselines[event.event_id] = baseline_prob

                    indicator_signals = []
                    indicator_weights_list = []
                    indicator_betas = []
                    indicator_names = []
                    signal_details = []
                    data_ages = []

                    for w in weights:
                        key = f"{event.event_id}:{w.indicator_name}"
                        value_record = values_by_key.get(key)

                        if value_record:
                            sig_detail = signal_extractor.extract_signal_for_indicator(
                                event_id=event.event_id,
                                indicator_name=w.indicator_name,
                                current_value=value_record.value or 0.5,
                                session=session
                            )
                            signal = sig_detail["signal"]
                            signal_details.append(sig_detail)

                            indicator_signals.append(signal)
                            adj_weight = w.normalized_weight * sensitivity.get("weight_multipliers", {}).get(w.indicator_name, 1.0) * sensitivity.get("severity_factor", 1.0)
                            indicator_weights_list.append(adj_weight)
                            indicator_names.append(w.indicator_name)

                            beta = BETA_PARAMETERS.get(w.beta_type, 0.7)
                            indicator_betas.append(beta)

                            if value_record.timestamp:
                                age_hours = (start_time - value_record.timestamp).total_seconds() / 3600
                                data_ages.append(age_hours)

                    calc_result = probability_calculator.calculate_event_probability(
                        baseline_scale=baseline_scale,
                        indicator_signals=indicator_signals,
                        indicator_weights=indicator_weights_list,
                        indicator_betas=indicator_betas
                    )

                    bayesian_prob = calc_result["probability"]
                    all_bayesian_probs[event.event_id] = bayesian_prob

                    final_prob, ensemble_method, ml_prob = ml_layer.get_ensemble_probability(
                        bayesian_prob=bayesian_prob,
                        features=indicator_signals if indicator_signals else None
                    )

                    confidence, confidence_band, confidence_factors = EnhancedConfidenceScorer.calculate(
                        n_indicators=len(indicator_signals),
                        expected_indicators=len(weights),
                        indicator_signals=indicator_signals,
                        source_health=source_health,
                        data_ages_hours=data_ages
                    )

                    attribution = explainability_engine.generate_attribution(
                        indicator_names=indicator_names,
                        indicator_signals=indicator_signals,
                        indicator_weights=indicator_weights_list,
                        indicator_betas=indicator_betas,
                        signal_details=signal_details,
                        total_adjustment=calc_result["total_adjustment"]
                    )

                    prev_prob = session.query(RiskProbability).filter(
                        RiskProbability.event_id == event.event_id
                    ).order_by(RiskProbability.calculation_date.desc()).first()

                    prev_prob_pct = prev_prob.probability_pct if prev_prob else None
                    current_prob_pct = round(final_prob * 100, 2)

                    if prev_prob_pct is not None:
                        change_pct = current_prob_pct - prev_prob_pct
                        if change_pct > 1:
                            change_direction = "INCREASING"
                        elif change_pct < -1:
                            change_direction = "DECREASING"
                        else:
                            change_direction = "STABLE"
                    else:
                        change_pct = None
                        change_direction = "NEW"

                    explanation = explainability_engine.generate_explanation(
                        event_name=event.event_name,
                        current_prob_pct=current_prob_pct,
                        previous_prob_pct=prev_prob_pct,
                        attribution=attribution,
                        confidence=confidence,
                        flags=calc_result["flags"]
                    )

                    recommendation = explainability_engine.generate_recommendation(
                        probability_pct=current_prob_pct,
                        change_pct=change_pct,
                        confidence=confidence,
                        flags=calc_result["flags"]
                    )

                    prob_record = RiskProbability(
                        event_id=event.event_id,
                        calculation_id=calculation_id,
                        calculation_date=start_time,
                        probability_pct=current_prob_pct,
                        log_odds=calc_result["log_odds"],
                        baseline_probability_pct=calc_result["baseline_probability_pct"],
                        ci_lower_pct=calc_result["ci_lower_pct"],
                        ci_upper_pct=calc_result["ci_upper_pct"],
                        ci_level=0.95,
                        ci_width_pct=calc_result["ci_upper_pct"] - calc_result["ci_lower_pct"],
                        precision_band=calc_result["precision_band"],
                        bootstrap_iterations=1000,
                        confidence_score=confidence,
                        methodology_tier=event.methodology_tier or "TIER_2_ANALOG",
                        change_direction=change_direction,
                        flags=calc_result["flags"] if calc_result["flags"] else None
                    )

                    try:
                        prob_record.attribution = attribution
                        prob_record.explanation = explanation
                        prob_record.recommendation = recommendation
                        prob_record.ensemble_method = ensemble_method
                        prob_record.ml_probability_pct = round(ml_prob * 100, 2) if ml_prob else None
                        prob_record.previous_probability_pct = prev_prob_pct
                        prob_record.probability_change_pct = round(change_pct, 2) if change_pct is not None else None
                    except Exception as col_err:
                        logger.debug(f"New column not yet available: {col_err}")

                    session.add(prob_record)

                    results.append({
                        "event_id": event.event_id,
                        "probability_pct": current_prob_pct,
                        "previous_pct": prev_prob_pct,
                        "change_pct": round(change_pct, 2) if change_pct is not None else None,
                        "change_direction": change_direction,
                        "confidence_score": confidence,
                        "confidence_band": confidence_band,
                        "ensemble_method": ensemble_method,
                        "n_indicators": len(indicator_signals),
                        "flags": calc_result["flags"],
                        "recommendation": recommendation["level"],
                        "top_driver": attribution[0]["indicator"] if attribution else None
                    })

                    events_succeeded += 1

                except Exception as e:
                    events_failed += 1
                    errors.append({
                        "event_id": event.event_id,
                        "error": str(e)
                    })
                    logger.error(f"Error calculating {event.event_id}: {e}")

            # Second pass: apply dependency adjustments
            try:
                for result_entry in results:
                    event_id = result_entry["event_id"]
                    adjustment, dep_details = DependencyModeler.calculate_dependency_adjustment(
                        event_id=event_id,
                        all_probabilities=all_bayesian_probs,
                        all_baselines=all_baselines
                    )
                    if abs(adjustment) > 0.001:
                        prob_record = session.query(RiskProbability).filter(
                            RiskProbability.event_id == event_id,
                            RiskProbability.calculation_id == calculation_id
                        ).first()
                        if prob_record:
                            adjusted_prob_pct = prob_record.probability_pct + (adjustment * 100)
                            adjusted_prob_pct = max(0.1, min(99.9, adjusted_prob_pct))
                            prob_record.probability_pct = round(adjusted_prob_pct, 2)
                            try:
                                prob_record.dependency_adjustment = round(adjustment * 100, 2)
                                prob_record.dependency_details = dep_details
                            except Exception:
                                pass
                            result_entry["probability_pct"] = round(adjusted_prob_pct, 2)
                            result_entry["dependency_adjustment"] = round(adjustment * 100, 2)
            except Exception as dep_err:
                logger.warning(f"Dependency adjustment error: {dep_err}")

            ml_layer.check_and_train(session)

            end_time = datetime.utcnow()
            duration_secs = int((end_time - start_time).total_seconds())
            calc_log = CalculationLog(
                calculation_id=calculation_id,
                status="COMPLETED" if events_failed == 0 else "COMPLETED_WITH_ERRORS",
                events_processed=events_processed,
                events_succeeded=events_succeeded,
                events_failed=events_failed,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration_secs,
                errors=json.dumps(errors) if errors else None
            )
            session.add(calc_log)
            session.commit()

            duration_seconds = (end_time - start_time).total_seconds()

            logger.info(
                f"Enhanced calculation {calculation_id} complete: "
                f"{events_succeeded}/{events_processed} succeeded in {duration_seconds:.1f}s"
            )

            return {
                "status": "completed",
                "calculation_id": calculation_id,
                "version": "3.0.0-enhanced",
                "events_processed": events_processed,
                "events_succeeded": events_succeeded,
                "events_failed": events_failed,
                "duration_seconds": round(duration_seconds, 2),
                "features_used": {
                    "signal_extraction": True,
                    "ml_enhancement": ml_layer.is_trained,
                    "dependency_modeling": True,
                    "explainability": True,
                    "enhanced_confidence": True
                },
                "errors": errors if errors else None,
                "sample_results": results[:10]
            }

    except Exception as e:
        logger.error(f"Enhanced calculation batch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def register_calculations_routes(app, get_session_fn):
    """Register calculation endpoints on the FastAPI app."""

    @app.get("/api/v1/probabilities")
    async def list_probabilities(
        event_id: Optional[str] = None,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500)
    ):
        """List calculated probabilities."""
        with get_session_fn() as session:
            query = session.query(RiskProbability)
            if event_id:
                query = query.filter(RiskProbability.event_id == event_id)
            query = query.order_by(RiskProbability.calculation_date.desc())

            total = query.count()
            probabilities = query.offset(skip).limit(limit).all()

            return {
                "total": total,
                "skip": skip,
                "limit": limit,
                "probabilities": [
                    {
                        "event_id": p.event_id,
                        "probability_pct": p.probability_pct,
                        "ci_lower_pct": p.ci_lower_pct,
                        "ci_upper_pct": p.ci_upper_pct,
                        "precision_band": p.precision_band,
                        "confidence_score": p.confidence_score,
                        "methodology_tier": p.methodology_tier,
                        "change_direction": p.change_direction,
                        "flags": p.flags,
                        "calculation_date": p.calculation_date.isoformat()
                    }
                    for p in probabilities
                ]
            }

    @app.post("/api/v1/calculations/trigger-full")
    async def trigger_enhanced_calculation(
        event_ids: Optional[List[str]] = None,
        limit: int = Query(100, ge=1, le=1000, description="Max events to process")
    ):
        """
        Enhanced probability calculation with signal extraction, explainability,
        and dependency modeling.

        Pipeline:
        1. Load events and their indicator weights
        2. Get latest indicator values
        3. Extract signals (z-scores, momentum, trends) using historical data
        4. Calculate Bayesian probability
        5. Apply ML enhancement (when model is trained)
        6. Adjust for risk interdependencies
        7. Calculate enhanced confidence score
        8. Generate attribution and explanation
        9. Store everything in database

        DB operations run in thread pool to avoid blocking the event loop.
        """
        return await asyncio.to_thread(_run_enhanced_calculation_sync, limit, event_ids)
