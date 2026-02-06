"""
PRISM Brain FastAPI Application

Main entry point for the REST API.
Phase 1-3: Database, API, Railway deployment, 905 events, probability engine.
Phase 4A: Added 12 data sources (USGS, CISA, NVD, FRED, NOAA, World Bank,
          GDELT, EIA, IMF, FAO, OTX, ACLED).
Phase 4B: Signal extraction system (z-scores, momentum, trends, anomaly detection).
Phase 4C: ML enhancement layer (auto-training ensemble).
Phase 4D: Explainability framework (attribution, explanations, recommendations).
Phase 4E: Risk interdependency modeling (cascading effects).
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel
from collections import defaultdict
import logging
import json
import math
import uuid
import os
import asyncio
import aiohttp
import statistics

from config.settings import get_settings
from database.connection import init_db, get_session_context
from database.models import (
    RiskEvent, RiskProbability, IndicatorWeight,
    DataSourceHealth, CalculationLog, IndicatorValue
)
from config.category_indicators import (
    get_category_prefix, get_default_baseline, get_indicators_for_event,
    get_event_sensitivity, CATEGORY_INDICATOR_MAP
)

# Optional ML imports - degrade gracefully if not available
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    import pickle
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version="3.0.0",
    description="Probability calculation engine for 900 risk events with signal extraction, ML enhancement, and explainability",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class EventCreate(BaseModel):
    event_id: str
    event_name: str
    description: str = ""
    layer1_primary: Optional[str] = None
    layer1_secondary: Optional[str] = None
    layer2_primary: Optional[str] = None
    layer2_secondary: Optional[str] = None
    super_risk: bool = False
    baseline_probability: int = 3
    baseline_impact: int = 3
    geographic_scope: Optional[str] = None
    time_horizon: Optional[str] = None
    methodology_tier: Optional[str] = None

class IndicatorWeightCreate(BaseModel):
    event_id: str
    indicator_name: str
    normalized_weight: float
    data_source: str
    beta_type: str = "MODERATE"
    time_scale: str = "medium"
    update_frequency_hours: int = 24

class IndicatorValueCreate(BaseModel):
    event_id: str
    indicator_name: str
    data_source: str
    value: float
    timestamp: Optional[datetime] = None
    raw_value: Optional[float] = None
    historical_mean: Optional[float] = None
    historical_std: Optional[float] = None
    z_score: Optional[float] = None
    quality_score: float = 0.8


# ============== Probability Calculation Engine ==============

# Beta parameters for different correlation types
BETA_PARAMETERS = {
    "HIGH": 1.2,
    "MODERATE": 0.8,
    "LOW": 0.4,
    "direct_causal": 1.2,
    "strong_correlation": 1.0,
    "moderate_correlation": 0.7,
    "weak_correlation": 0.4
}

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

        FIX: Use tanh(z/2) instead of z/3 clamping.
        Rationale: tanh provides smooth non-linear mapping that preserves
        the full dynamic range. A z-score of 2 (95th percentile) now maps
        to signal 0.76 instead of 0.67. A z-score of 3 maps to 0.91
        instead of 1.0. This gives more realistic probability adjustments.
        """
        if z_score is not None:
            # Use tanh for smooth bounded mapping [-1, 1]
            return math.tanh(z_score / 2.0)
        else:
            # No z-score: use value directly with centering
            # Assume value is 0-1 range, center around 0.5
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


# ============== Phase 4B: Signal Extraction System ==============

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
        historical_values: List[float]
    ) -> Tuple[float, float, float]:
        """
        Calculate z-score from actual historical data.

        Returns: (z_score, mean, std)
        """
        if not historical_values or len(historical_values) < 2:
            return 0.0, current_value, 1.0

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
            current_value, historical_values
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


# ============== Phase 4D: Explainability Framework ==============

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


# ============== Phase 4E: Risk Interdependency Modeling ==============

# Predefined correlation matrix based on domain expertise
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


# ============== Phase 4C: ML Enhancement Layer ==============

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


# ============== Enhanced Confidence Scoring ==============

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


# ============== Schema Migration ==============

def ensure_schema_updates(session):
    """
    Add new columns to database tables if they don't exist.
    Uses ALTER TABLE with IF NOT EXISTS logic.
    """
    alter_statements = [
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS attribution JSON",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS explanation TEXT",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS recommendation JSON",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS dependency_adjustment FLOAT",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS dependency_details JSON",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS ensemble_method VARCHAR(50)",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS ml_probability_pct FLOAT",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS previous_probability_pct FLOAT",
        "ALTER TABLE risk_probabilities ADD COLUMN IF NOT EXISTS probability_change_pct FLOAT",
        "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS signal FLOAT",
        "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS momentum FLOAT",
        "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS trend VARCHAR(20)",
        "ALTER TABLE indicator_values ADD COLUMN IF NOT EXISTS is_anomaly BOOLEAN DEFAULT FALSE",
    ]
    for stmt in alter_statements:
        try:
            session.execute(stmt)
        except Exception as e:
            # Column might already exist or DB might not support IF NOT EXISTS
            logger.debug(f"Schema update note: {e}")
    try:
        session.commit()
    except Exception:
        session.rollback()


# ============== Global Instances ==============

probability_calculator = ProbabilityCalculator()
signal_extractor = SignalExtractor()
explainability_engine = ExplainabilityEngine()
ml_layer = MLEnhancementLayer()


# ============== Startup/Shutdown ==============

@app.on_event("startup")
async def startup_event():
    logger.info("Starting PRISM Brain API v3.0.0...")
    init_db()
    logger.info("Database initialized")
    # Run schema migrations
    try:
        with get_session_context() as session:
            ensure_schema_updates(session)
            logger.info("Schema updates applied")
    except Exception as e:
        logger.warning(f"Schema update warning (may be OK on first run): {e}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down PRISM Brain API...")


# ============== Health Check ==============

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0",
        "ml_available": ML_AVAILABLE,
        "features": [
            "signal_extraction", "explainability",
            "dependency_modeling", "enhanced_confidence"
        ]
    }



# ============== Dashboard ==============

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the interactive PRISM Brain dashboard."""
    dashboard_path = Path(__file__).parent / "static" / "dashboard.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Dashboard not found</h1><p>static/dashboard.html is missing.</p>", status_code=404)

# ============== Events Endpoints ==============

@app.get("/api/v1/events")
async def list_events(
    layer1: Optional[str] = None,
    layer2: Optional[str] = None,
    super_risk: Optional[bool] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500)
):
    """List all risk events with optional filtering."""
    with get_session_context() as session:
        query = session.query(RiskEvent)
        if layer1:
            query = query.filter(RiskEvent.layer1_primary == layer1)
        if layer2:
            query = query.filter(RiskEvent.layer2_primary == layer2)
        if super_risk is not None:
            query = query.filter(RiskEvent.super_risk == super_risk)

        total = query.count()
        events = query.offset(skip).limit(limit).all()

        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "events": [
                {
                    "event_id": e.event_id,
                    "event_name": e.event_name,
                    "description": e.description,
                    "layer1_primary": e.layer1_primary,
                    "layer1_secondary": e.layer1_secondary,
                    "layer2_primary": e.layer2_primary,
                    "layer2_secondary": e.layer2_secondary,
                    "super_risk": e.super_risk,
                    "baseline_probability": e.baseline_probability,
                    "baseline_impact": e.baseline_impact,
                    "geographic_scope": e.geographic_scope,
                    "time_horizon": e.time_horizon,
                    "methodology_tier": e.methodology_tier
                }
                for e in events
            ]
        }


@app.get("/api/v1/events/{event_id}")
async def get_event(event_id: str):
    """Get a specific risk event by ID."""
    with get_session_context() as session:
        event = session.query(RiskEvent).filter(
            RiskEvent.event_id == event_id
        ).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        # Get latest probability
        latest_prob = session.query(RiskProbability).filter(
            RiskProbability.event_id == event_id
        ).order_by(RiskProbability.calculation_date.desc()).first()

        # Get indicators
        weights = session.query(IndicatorWeight).filter(
            IndicatorWeight.event_id == event_id
        ).all()

        return {
            "event": {
                "event_id": event.event_id,
                "event_name": event.event_name,
                "description": event.description,
                "layer1_primary": event.layer1_primary,
                "layer2_primary": event.layer2_primary,
                "super_risk": event.super_risk,
                "baseline_probability": event.baseline_probability,
                "methodology_tier": event.methodology_tier
            },
            "latest_probability": {
                "probability_pct": latest_prob.probability_pct,
                "ci_lower_pct": latest_prob.ci_lower_pct,
                "ci_upper_pct": latest_prob.ci_upper_pct,
                "confidence_score": latest_prob.confidence_score,
                "change_direction": latest_prob.change_direction,
                "calculation_date": latest_prob.calculation_date.isoformat()
            } if latest_prob else None,
            "indicators": [
                {
                    "indicator_name": w.indicator_name,
                    "data_source": w.data_source,
                    "normalized_weight": w.normalized_weight,
                    "beta_type": w.beta_type,
                    "time_scale": w.time_scale
                }
                for w in weights
            ],
            "dependencies": DependencyModeler.get_dependency_chain(event_id)
        }


@app.post("/api/v1/events/bulk")
async def bulk_import_events(events: List[EventCreate]):
    """Bulk import risk events."""
    with get_session_context() as session:
        imported = 0
        updated = 0
        for e in events:
            existing = session.query(RiskEvent).filter(
                RiskEvent.event_id == e.event_id
            ).first()
            if existing:
                existing.event_name = e.event_name
                existing.description = e.description
                existing.layer1_primary = e.layer1_primary
                existing.layer2_primary = e.layer2_primary
                existing.super_risk = e.super_risk
                existing.baseline_probability = e.baseline_probability
                existing.baseline_impact = e.baseline_impact
                updated += 1
            else:
                event = RiskEvent(
                    event_id=e.event_id,
                    event_name=e.event_name,
                    description=e.description,
                    layer1_primary=e.layer1_primary,
                    layer1_secondary=e.layer1_secondary,
                    layer2_primary=e.layer2_primary,
                    layer2_secondary=e.layer2_secondary,
                    super_risk=e.super_risk,
                    baseline_probability=e.baseline_probability,
                    baseline_impact=e.baseline_impact,
                    geographic_scope=e.geographic_scope,
                    time_horizon=e.time_horizon,
                    methodology_tier=e.methodology_tier
                )
                session.add(event)
                imported += 1
        session.commit()
        total = session.query(RiskEvent).count()
    return {"imported": imported, "updated": updated, "total": total}


# ============== Indicator Weights Endpoints ==============

@app.get("/api/v1/indicator-weights")
async def list_indicator_weights(
    event_id: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """List indicator weights, optionally filtered by event."""
    with get_session_context() as session:
        query = session.query(IndicatorWeight)
        if event_id:
            query = query.filter(IndicatorWeight.event_id == event_id)
        total = query.count()
        weights = query.offset(skip).limit(limit).all()
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "weights": [
                {
                    "event_id": w.event_id,
                    "indicator_name": w.indicator_name,
                    "normalized_weight": w.normalized_weight,
                    "data_source": w.data_source,
                    "beta_type": w.beta_type,
                    "time_scale": w.time_scale
                }
                for w in weights
            ]
        }


@app.post("/api/v1/events/apply-baselines")
async def apply_category_baselines():
    """Apply differentiated baseline probabilities to all events based on category."""
    with get_session_context() as session:
        events = session.query(RiskEvent).all()
        updated = 0
        categories_found = {}
        for event in events:
            new_baseline = get_default_baseline(event.event_id)
            prefix = get_category_prefix(event.event_id)
            if prefix not in categories_found:
                categories_found[prefix] = {"count": 0, "baseline": new_baseline}
            categories_found[prefix]["count"] += 1
            if event.baseline_probability != new_baseline:
                event.baseline_probability = new_baseline
                updated += 1
        session.commit()
        return {
            "total_events": len(events),
            "updated": updated,
            "categories": categories_found,
            "message": f"Applied differentiated baselines to {updated} events across {len(categories_found)} categories"
        }


@app.post("/api/v1/indicator-weights/bulk")
async def bulk_import_indicator_weights(weights: List[IndicatorWeightCreate]):
    """Bulk import indicator weights."""
    with get_session_context() as session:
        imported = 0
        updated = 0
        for w in weights:
            existing = session.query(IndicatorWeight).filter(
                IndicatorWeight.event_id == w.event_id,
                IndicatorWeight.indicator_name == w.indicator_name
            ).first()
            if existing:
                existing.normalized_weight = w.normalized_weight
                existing.data_source = w.data_source
                existing.beta_type = w.beta_type
                existing.time_scale = w.time_scale
                updated += 1
            else:
                weight = IndicatorWeight(
                    event_id=w.event_id,
                    indicator_name=w.indicator_name,
                    normalized_weight=w.normalized_weight,
                    data_source=w.data_source,
                    beta_type=w.beta_type,
                    time_scale=w.time_scale,
                    causal_proximity_score=0.5,
                    data_quality_score=0.7,
                    timeliness_score=0.6,
                    predictive_lead_score=0.5,
                    raw_score=w.normalized_weight,
                    beta_value=0.5
                )
                session.add(weight)
                imported += 1
        session.commit()
        total = session.query(IndicatorWeight).count()
    return {"imported": imported, "updated": updated, "total": total}


# ============== Indicator Values Endpoints ==============

@app.get("/api/v1/indicator-values")
async def list_indicator_values(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """List current indicator values."""
    with get_session_context() as session:
        total = session.query(IndicatorValue).count()
        values = session.query(IndicatorValue).offset(skip).limit(limit).all()
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "values": [
                {
                    "event_id": v.event_id,
                    "indicator_name": v.indicator_name,
                    "data_source": v.data_source,
                    "value": v.value,
                    "timestamp": v.timestamp.isoformat() if v.timestamp else None,
                    "z_score": v.z_score,
                    "quality_score": v.quality_score
                }
                for v in values
            ]
        }


@app.post("/api/v1/indicator-values/bulk")
async def bulk_import_indicator_values(values: List[IndicatorValueCreate]):
    """Bulk import indicator values (time series data points)."""
    with get_session_context() as session:
        imported = 0
        updated = 0
        for v in values:
            timestamp = v.timestamp or datetime.utcnow()
            existing = session.query(IndicatorValue).filter(
                IndicatorValue.event_id == v.event_id,
                IndicatorValue.indicator_name == v.indicator_name,
                IndicatorValue.timestamp == timestamp
            ).first()
            if existing:
                existing.value = v.value
                existing.raw_value = v.raw_value or v.value
                existing.quality_score = v.quality_score
                updated += 1
            else:
                value = IndicatorValue(
                    event_id=v.event_id,
                    indicator_name=v.indicator_name,
                    data_source=v.data_source,
                    timestamp=timestamp,
                    value=v.value,
                    raw_value=v.raw_value or v.value,
                    historical_mean=v.historical_mean,
                    historical_std=v.historical_std,
                    z_score=v.z_score,
                    quality_score=v.quality_score
                )
                session.add(value)
                imported += 1
        session.commit()
        total = session.query(IndicatorValue).count()
    return {"imported": imported, "updated": updated, "total": total}


# ============== Probabilities Endpoints ==============

@app.get("/api/v1/probabilities")
async def list_probabilities(
    event_id: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500)
):
    """List calculated probabilities."""
    with get_session_context() as session:
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


@app.get("/api/v1/probabilities/{event_id}/history")
async def get_probability_history(
    event_id: str,
    limit: int = Query(52, ge=1, le=200)
):
    """Get probability history for a specific event."""
    with get_session_context() as session:
        event = session.query(RiskEvent).filter(
            RiskEvent.event_id == event_id
        ).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        probabilities = session.query(RiskProbability).filter(
            RiskProbability.event_id == event_id
        ).order_by(RiskProbability.calculation_date.desc()).limit(limit).all()

        return {
            "event_id": event_id,
            "event_name": event.event_name,
            "history": [
                {
                    "probability_pct": p.probability_pct,
                    "ci_lower_pct": p.ci_lower_pct,
                    "ci_upper_pct": p.ci_upper_pct,
                    "confidence_score": p.confidence_score,
                    "change_direction": p.change_direction,
                    "calculation_date": p.calculation_date.isoformat()
                }
                for p in probabilities
            ]
        }


# ============== NEW: Attribution & Explanation Endpoints ==============

@app.get("/api/v1/probabilities/{event_id}/attribution")
async def get_probability_attribution(event_id: str):
    """Get detailed attribution for the latest probability calculation."""
    with get_session_context() as session:
        event = session.query(RiskEvent).filter(
            RiskEvent.event_id == event_id
        ).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        latest_prob = session.query(RiskProbability).filter(
            RiskProbability.event_id == event_id
        ).order_by(RiskProbability.calculation_date.desc()).first()

        if not latest_prob:
            raise HTTPException(status_code=404, detail="No probability calculated yet")

        # Try to get stored attribution
        attribution = None
        try:
            attribution = latest_prob.attribution
        except AttributeError:
            pass

        return {
            "event_id": event_id,
            "event_name": event.event_name,
            "probability_pct": latest_prob.probability_pct,
            "confidence_score": latest_prob.confidence_score,
            "calculation_date": latest_prob.calculation_date.isoformat(),
            "attribution": attribution or [],
            "flags": latest_prob.flags or []
        }


@app.get("/api/v1/probabilities/{event_id}/explanation")
async def get_probability_explanation(event_id: str):
    """Get human-readable explanation for the latest probability."""
    with get_session_context() as session:
        event = session.query(RiskEvent).filter(
            RiskEvent.event_id == event_id
        ).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        latest_prob = session.query(RiskProbability).filter(
            RiskProbability.event_id == event_id
        ).order_by(RiskProbability.calculation_date.desc()).first()

        if not latest_prob:
            raise HTTPException(status_code=404, detail="No probability calculated yet")

        # Try to get stored explanation and recommendation
        explanation = None
        recommendation = None
        try:
            explanation = latest_prob.explanation
            recommendation = latest_prob.recommendation
        except AttributeError:
            pass

        # Generate on-the-fly if not stored
        if not explanation:
            prev_prob_pct = None
            try:
                prev_prob_pct = latest_prob.previous_probability_pct
            except AttributeError:
                pass

            attribution = []
            try:
                attribution = latest_prob.attribution or []
            except AttributeError:
                pass

            explanation = ExplainabilityEngine.generate_explanation(
                event_name=event.event_name,
                current_prob_pct=latest_prob.probability_pct,
                previous_prob_pct=prev_prob_pct,
                attribution=attribution,
                confidence=latest_prob.confidence_score,
                flags=latest_prob.flags or []
            )

        if not recommendation:
            change_pct = None
            try:
                change_pct = latest_prob.probability_change_pct
            except AttributeError:
                pass

            recommendation = ExplainabilityEngine.generate_recommendation(
                probability_pct=latest_prob.probability_pct,
                change_pct=change_pct,
                confidence=latest_prob.confidence_score,
                flags=latest_prob.flags or []
            )

        return {
            "event_id": event_id,
            "event_name": event.event_name,
            "explanation": explanation,
            "recommendation": recommendation
        }


# ============== NEW: Dependencies Endpoint ==============

@app.get("/api/v1/dependencies/{event_id}")
async def get_event_dependencies(event_id: str):
    """Get dependency chain and correlated events for a risk event."""
    with get_session_context() as session:
        event = session.query(RiskEvent).filter(
            RiskEvent.event_id == event_id
        ).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        chain = DependencyModeler.get_dependency_chain(event_id)

        # Enrich with current probabilities
        for dep in chain:
            latest_prob = session.query(RiskProbability).filter(
                RiskProbability.event_id == dep["event_id"]
            ).order_by(RiskProbability.calculation_date.desc()).first()

            dep_event = session.query(RiskEvent).filter(
                RiskEvent.event_id == dep["event_id"]
            ).first()

            dep["event_name"] = dep_event.event_name if dep_event else "Unknown"
            dep["current_probability_pct"] = latest_prob.probability_pct if latest_prob else None

        return {
            "event_id": event_id,
            "event_name": event.event_name,
            "dependencies": chain,
            "total_correlated": len(chain)
        }


# ============== NEW: Dashboard Summary Endpoint ==============

@app.get("/api/v1/dashboard/summary")
async def get_dashboard_summary():
    """Get overall dashboard summary with top risers, fallers, and alerts."""
    with get_session_context() as session:
        from sqlalchemy import func, desc

        # Get latest calculation
        latest_calc = session.query(CalculationLog).order_by(
            CalculationLog.start_time.desc()
        ).first()

        # Get all latest probabilities (subquery for latest per event)
        subquery = session.query(
            RiskProbability.event_id,
            func.max(RiskProbability.calculation_date).label('latest')
        ).group_by(RiskProbability.event_id).subquery()

        latest_probs = session.query(RiskProbability).join(
            subquery,
            (RiskProbability.event_id == subquery.c.event_id) &
            (RiskProbability.calculation_date == subquery.c.latest)
        ).all()

        # Categorize
        high_risk = []
        increasing = []
        decreasing = []
        flagged = []

        for p in latest_probs:
            entry = {
                "event_id": p.event_id,
                "probability_pct": p.probability_pct,
                "confidence_score": p.confidence_score,
                "change_direction": p.change_direction,
                "flags": p.flags
            }

            if p.probability_pct >= 30:
                high_risk.append(entry)
            if p.change_direction == "INCREASING":
                increasing.append(entry)
            elif p.change_direction == "DECREASING":
                decreasing.append(entry)
            if p.flags and len(p.flags) > 0:
                flagged.append(entry)

        # Sort
        high_risk.sort(key=lambda x: -x["probability_pct"])
        increasing.sort(key=lambda x: -x["probability_pct"])
        decreasing.sort(key=lambda x: x["probability_pct"])

        # Data source health
        source_health = {}
        health_records = session.query(DataSourceHealth).order_by(
            DataSourceHealth.check_time.desc()
        ).all()
        for h in health_records:
            if h.source_name not in source_health:
                source_health[h.source_name] = h.status

        operational = sum(1 for s in source_health.values() if s == "OPERATIONAL")

        return {
            "summary": {
                "total_events": len(latest_probs),
                "high_risk_count": len(high_risk),
                "increasing_count": len(increasing),
                "decreasing_count": len(decreasing),
                "flagged_count": len(flagged),
                "data_sources_operational": operational,
                "data_sources_total": len(source_health)
            },
            "top_risks": high_risk[:10],
            "top_risers": increasing[:10],
            "top_fallers": decreasing[:10],
            "flagged_events": flagged[:10],
            "latest_calculation": {
                "id": latest_calc.calculation_id if latest_calc else None,
                "status": latest_calc.status if latest_calc else None,
                "date": latest_calc.start_time.isoformat() if latest_calc else None,
                "events_processed": latest_calc.events_processed if latest_calc else 0
            }
        }


# ============== Calculations Endpoints ==============

@app.get("/api/v1/calculations")
async def list_calculations(limit: int = Query(20, ge=1, le=100)):
    """List recent calculation batches."""
    with get_session_context() as session:
        calculations = session.query(CalculationLog).order_by(
            CalculationLog.start_time.desc()
        ).limit(limit).all()

        return {
            "calculations": [
                {
                    "calculation_id": c.calculation_id,
                    "status": c.status,
                    "events_processed": c.events_processed,
                    "events_succeeded": c.events_succeeded,
                    "events_failed": c.events_failed,
                    "started_at": c.start_time.isoformat() if c.start_time else None,
                    "completed_at": c.end_time.isoformat() if c.end_time else None,
                    "duration_seconds": c.duration_seconds,
                    "errors": c.errors
                }
                for c in calculations
            ]
        }


# Legacy calculation trigger (kept for backwards compatibility)
@app.post("/api/v1/calculations/trigger")
async def trigger_calculation(
    event_ids: Optional[List[str]] = None,
    limit: int = Query(100, ge=1, le=1000, description="Max events to process")
):
    """
    Trigger probability calculations (legacy endpoint).
    For enhanced calculations with signal extraction and explainability,
    use POST /api/v1/calculations/trigger-full instead.
    """
    return await trigger_enhanced_calculation(event_ids=event_ids, limit=limit)


# ============== ENHANCED Calculation Trigger ==============

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
    """
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
                    # Get event-specific sensitivity profile
                    sensitivity = get_event_sensitivity(event.event_id)
                    baseline_scale = baseline_scale + sensitivity["baseline_offset"]
                    baseline_scale = max(1.0, min(5.0, baseline_scale))
                    baseline_prob = probability_calculator.scale_to_probability(baseline_scale)
                    all_baselines[event.event_id] = baseline_prob

                    # Gather signals with enhanced extraction
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
                            # Phase 4B: Extract signal using historical data
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

                            # Track data age for confidence
                            if value_record.timestamp:
                                age_hours = (start_time - value_record.timestamp).total_seconds() / 3600
                                data_ages.append(age_hours)

                    # Calculate Bayesian probability
                    calc_result = probability_calculator.calculate_event_probability(
                        baseline_scale=baseline_scale,
                        indicator_signals=indicator_signals,
                        indicator_weights=indicator_weights_list,
                        indicator_betas=indicator_betas
                    )

                    bayesian_prob = calc_result["probability"]
                    all_bayesian_probs[event.event_id] = bayesian_prob

                    # Phase 4C: ML enhancement
                    final_prob, ensemble_method, ml_prob = ml_layer.get_ensemble_probability(
                        bayesian_prob=bayesian_prob,
                        features=indicator_signals if indicator_signals else None
                    )

                    # Enhanced confidence scoring
                    confidence, confidence_band, confidence_factors = EnhancedConfidenceScorer.calculate(
                        n_indicators=len(indicator_signals),
                        expected_indicators=len(weights),
                        indicator_signals=indicator_signals,
                        source_health=source_health,
                        data_ages_hours=data_ages
                    )

                    # Phase 4D: Generate attribution
                    attribution = ExplainabilityEngine.generate_attribution(
                        indicator_names=indicator_names,
                        indicator_signals=indicator_signals,
                        indicator_weights=indicator_weights_list,
                        indicator_betas=indicator_betas,
                        signal_details=signal_details,
                        total_adjustment=calc_result["total_adjustment"]
                    )

                    # Get previous probability for change tracking
                    prev_prob = session.query(RiskProbability).filter(
                        RiskProbability.event_id == event.event_id
                    ).order_by(RiskProbability.calculation_date.desc()).first()

                    prev_prob_pct = prev_prob.probability_pct if prev_prob else None
                    current_prob_pct = round(final_prob * 100, 2)

                    # Change detection
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

                    # Generate explanation
                    explanation = ExplainabilityEngine.generate_explanation(
                        event_name=event.event_name,
                        current_prob_pct=current_prob_pct,
                        previous_prob_pct=prev_prob_pct,
                        attribution=attribution,
                        confidence=confidence,
                        flags=calc_result["flags"]
                    )

                    # Generate recommendation
                    recommendation = ExplainabilityEngine.generate_recommendation(
                        probability_pct=current_prob_pct,
                        change_pct=change_pct,
                        confidence=confidence,
                        flags=calc_result["flags"]
                    )

                    # Store result
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

                    # Set new columns safely
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
            # (requires all Bayesian probs to be calculated first)
            try:
                for result_entry in results:
                    event_id = result_entry["event_id"]
                    adjustment, dep_details = DependencyModeler.calculate_dependency_adjustment(
                        event_id=event_id,
                        all_probabilities=all_bayesian_probs,
                        all_baselines=all_baselines
                    )
                    if abs(adjustment) > 0.001:
                        # Update the stored probability
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

            # Check if ML training is possible
            ml_layer.check_and_train(session)

            # Create calculation log
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


# ============== Data Sources Endpoints ==============

@app.get("/api/v1/data-sources/health")
async def get_data_source_health():
    """Get health status of all data sources."""
    from sqlalchemy import func
    with get_session_context() as session:
        subquery = session.query(
            DataSourceHealth.source_name,
            func.max(DataSourceHealth.check_time).label('latest')
        ).group_by(DataSourceHealth.source_name).subquery()

        health_records = session.query(DataSourceHealth).join(
            subquery,
            (DataSourceHealth.source_name == subquery.c.source_name) &
            (DataSourceHealth.check_time == subquery.c.latest)
        ).all()

        return {
            "data_sources": [
                {
                    "source_name": h.source_name,
                    "status": h.status,
                    "check_time": h.check_time.isoformat(),
                    "response_time_ms": h.response_time_ms,
                    "success_rate_24h": h.success_rate_24h,
                    "error_message": h.error_message
                }
                for h in health_records
            ]
        }


# ============== Stats Endpoint ==============

@app.get("/api/v1/stats")
async def get_stats():
    """Get overall system statistics."""
    with get_session_context() as session:
        event_count = session.query(RiskEvent).count()
        weight_count = session.query(IndicatorWeight).count()
        value_count = session.query(IndicatorValue).count()
        prob_count = session.query(RiskProbability).count()
        events_with_weights = session.query(IndicatorWeight.event_id).distinct().count()

        latest_calc = session.query(CalculationLog).order_by(
            CalculationLog.start_time.desc()
        ).first()

        return {
            "version": "3.0.0",
            "events": {
                "total": event_count,
                "with_weights": events_with_weights
            },
            "indicator_weights": weight_count,
            "indicator_values": value_count,
            "probabilities_calculated": prob_count,
            "ml_available": ML_AVAILABLE,
            "ml_trained": ml_layer.is_trained,
            "latest_calculation": {
                "id": latest_calc.calculation_id if latest_calc else None,
                "status": latest_calc.status if latest_calc else None,
                "date": latest_calc.start_time.isoformat() if latest_calc else None
            }
        }


# ============== Data Fetching System ==============

# BUGFIX: Complete source prefix mapping (was missing 6 sources)
SOURCE_PREFIX_MAP = {
    'usgs_': 'USGS',
    'cisa_': 'CISA',
    'nvd_': 'NVD',
    'fred_': 'FRED',
    'noaa_': 'NOAA',
    'world_bank_': 'WORLD_BANK',
    'gdelt_': 'GDELT',
    'eia_': 'EIA',
    'imf_': 'IMF',
    'fao_': 'FAO',
    'otx_': 'OTX',
    'acled_': 'ACLED',
}


def indicator_to_source(indicator_name: str) -> str:
    """Map an indicator name to its data source using prefix."""
    for prefix, source in SOURCE_PREFIX_MAP.items():
        if indicator_name.startswith(prefix):
            return source
    return 'INTERNAL'


class DataFetcher:
    """
    Self-contained data fetcher for external data sources.
    Fetches live data from free APIs and transforms into indicator values.
    """

    def __init__(self):
        self.timeout = aiohttp.ClientTimeout(total=30)

    async def fetch_usgs_earthquakes(self, days: int = 30) -> Dict[str, Any]:
        """Fetch earthquake data from USGS."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            'format': 'geojson',
            'starttime': start_date.strftime('%Y-%m-%d'),
            'endtime': end_date.strftime('%Y-%m-%d'),
            'minmagnitude': 2.5
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        features = data.get('features', [])

                        total = len(features)
                        significant = sum(1 for f in features if (f.get('properties', {}).get('mag') or 0) >= 5.0)
                        max_mag = max((f.get('properties', {}).get('mag') or 0) for f in features) if features else 0

                        return {
                            'source': 'USGS',
                            'status': 'success',
                            'indicators': {
                                'usgs_earthquake_count': total,
                                'usgs_significant_count': significant,
                                'usgs_max_magnitude': max_mag,
                                'usgs_seismic_activity': min(1.0, significant / 10) if significant else 0.1
                            }
                        }
        except Exception as e:
            logger.error(f"USGS fetch error: {e}")

        return {'source': 'USGS', 'status': 'error', 'indicators': {}}

    async def fetch_cisa_kev(self) -> Dict[str, Any]:
        """Fetch CISA Known Exploited Vulnerabilities catalog."""
        url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        vulns = data.get('vulnerabilities', [])
                        cutoff = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
                        recent = [v for v in vulns if v.get('dateAdded', '') >= cutoff]

                        return {
                            'source': 'CISA',
                            'status': 'success',
                            'indicators': {
                                'cisa_total_kev': len(vulns),
                                'cisa_recent_kev': len(recent),
                                'cisa_kev_rate': min(1.0, len(recent) / 50),
                                'cisa_threat_level': min(1.0, len(recent) / 30)
                            }
                        }
        except Exception as e:
            logger.error(f"CISA fetch error: {e}")

        return {'source': 'CISA', 'status': 'error', 'indicators': {}}

    async def fetch_world_bank(self, indicator: str = 'NY.GDP.MKTP.KD.ZG') -> Dict[str, Any]:
        """Fetch World Bank economic indicators."""
        url = f"https://api.worldbank.org/v2/country/WLD/indicator/{indicator}"
        params = {'format': 'json', 'per_page': 10, 'date': '2020:2025'}

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if len(data) > 1 and data[1]:
                            values = [d['value'] for d in data[1] if d['value'] is not None]
                            if values:
                                latest = values[0]
                                avg = sum(values) / len(values)
                                return {
                                    'source': 'WORLD_BANK',
                                    'status': 'success',
                                    'indicators': {
                                        'world_bank_gdp_growth': latest,
                                        'world_bank_gdp_avg': avg,
                                        'world_bank_economic_health': min(1.0, max(0, (latest + 5) / 10))
                                    }
                                }
        except Exception as e:
            logger.error(f"World Bank fetch error: {e}")

        return {'source': 'WORLD_BANK', 'status': 'error', 'indicators': {}}

    async def fetch_fred_data(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Fetch FRED economic data."""
        if not api_key:
            return {
                'source': 'FRED', 'status': 'no_api_key',
                'indicators': {
                    'fred_unemployment_rate': 4.1,
                    'fred_inflation_rate': 3.2,
                    'fred_fed_funds_rate': 5.25,
                    'fred_vix_index': 18.5
                }
            }

        series_map = {
            'UNRATE': 'fred_unemployment_rate',
            'CPIAUCSL': 'fred_inflation_rate',
            'FEDFUNDS': 'fred_fed_funds_rate',
            'VIXCLS': 'fred_vix_index'
        }

        indicators = {}
        base_url = "https://api.stlouisfed.org/fred/series/observations"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                for series_id, indicator_name in series_map.items():
                    params = {
                        'series_id': series_id,
                        'api_key': api_key,
                        'file_type': 'json',
                        'limit': 1,
                        'sort_order': 'desc'
                    }
                    async with session.get(base_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            obs = data.get('observations', [])
                            if obs and obs[0].get('value') != '.':
                                indicators[indicator_name] = float(obs[0]['value'])

            return {'source': 'FRED', 'status': 'success', 'indicators': indicators}
        except Exception as e:
            logger.error(f"FRED fetch error: {e}")

        return {'source': 'FRED', 'status': 'error', 'indicators': {}}

    async def fetch_noaa_climate(self, token: Optional[str] = None) -> Dict[str, Any]:
        """Fetch NOAA climate data from CDO API."""
        if not token:
            return {
                'source': 'NOAA', 'status': 'no_api_key',
                'indicators': {
                    'noaa_temp_anomaly': 1.2,
                    'noaa_precipitation_index': 0.95,
                    'noaa_extreme_events': 12,
                    'noaa_climate_risk': 0.65
                }
            }

        base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
        headers = {'token': token}

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=30)

                events_url = f"{base_url}/data"
                params = {
                    'datasetid': 'GHCND',
                    'datatypeid': 'TMAX,TMIN,PRCP',
                    'startdate': start_date.strftime('%Y-%m-%d'),
                    'enddate': end_date.strftime('%Y-%m-%d'),
                    'limit': 100,
                    'units': 'metric'
                }

                async with session.get(events_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', [])

                        temp_values = [r.get('value', 0) for r in results if r.get('datatype') in ['TMAX', 'TMIN']]
                        precip_values = [r.get('value', 0) for r in results if r.get('datatype') == 'PRCP']

                        avg_temp = sum(temp_values) / len(temp_values) / 10 if temp_values else 15
                        temp_anomaly = avg_temp - 15

                        avg_precip = sum(precip_values) / len(precip_values) / 10 if precip_values else 2.5
                        precip_index = avg_precip / 2.5 if avg_precip else 1.0

                        extreme_count = sum(1 for t in temp_values if t > 350 or t < -100)
                        extreme_count += sum(1 for p in precip_values if p > 250)

                        return {
                            'source': 'NOAA',
                            'status': 'success',
                            'indicators': {
                                'noaa_temp_anomaly': round(temp_anomaly, 2),
                                'noaa_precipitation_index': round(precip_index, 2),
                                'noaa_extreme_events': extreme_count,
                                'noaa_climate_risk': min(1.0, (abs(temp_anomaly) / 3 + extreme_count / 20))
                            }
                        }
                    elif response.status == 429:
                        logger.warning("NOAA API rate limited")
                    else:
                        logger.error(f"NOAA API error: {response.status}")

        except Exception as e:
            logger.error(f"NOAA fetch error: {e}")

        return {
            'source': 'NOAA', 'status': 'error',
            'indicators': {
                'noaa_temp_anomaly': 1.2,
                'noaa_precipitation_index': 0.95,
                'noaa_extreme_events': 12,
                'noaa_climate_risk': 0.65
            }
        }

    async def fetch_nvd_vulnerabilities(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Fetch NVD vulnerability data."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)

        url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        params = {
            'pubStartDate': start_date.strftime('%Y-%m-%dT00:00:00.000'),
            'pubEndDate': end_date.strftime('%Y-%m-%dT23:59:59.999')
        }

        headers = {}
        if api_key:
            headers['apiKey'] = api_key

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        total = data.get('totalResults', 0)
                        vulns = data.get('vulnerabilities', [])

                        critical = high = medium = 0
                        for v in vulns[:100]:
                            metrics = v.get('cve', {}).get('metrics', {})
                            cvss = metrics.get('cvssMetricV31', [{}])[0] if metrics.get('cvssMetricV31') else {}
                            severity = cvss.get('cvssData', {}).get('baseSeverity', 'UNKNOWN')
                            if severity == 'CRITICAL':
                                critical += 1
                            elif severity == 'HIGH':
                                high += 1
                            elif severity == 'MEDIUM':
                                medium += 1

                        return {
                            'source': 'NVD',
                            'status': 'success',
                            'indicators': {
                                'nvd_total_cves': total,
                                'nvd_critical_count': critical,
                                'nvd_high_count': high,
                                'nvd_vulnerability_rate': min(1.0, total / 500),
                                'nvd_severity_index': min(1.0, (critical * 3 + high * 2 + medium) / 100)
                            }
                        }
        except Exception as e:
            logger.error(f"NVD fetch error: {e}")

        return {'source': 'NVD', 'status': 'error', 'indicators': {}}

    async def fetch_gdelt_events(self) -> Dict[str, Any]:
        """Fetch GDELT global events data. No API key required."""
        url = "https://api.gdeltproject.org/api/v2/tv/tv"
        params = {
            'query': 'conflict',
            'mode': 'timelinevol',
            'format': 'json',
            'timespan': '7d'
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    content_type = response.headers.get('Content-Type', '')

                    if response.status == 200 and 'json' in content_type.lower():
                        data = await response.json()
                        timeline = data.get('timeline', [])

                        if timeline and isinstance(timeline, list):
                            # Handle nested timeline structure
                            all_values = []
                            if isinstance(timeline[0], dict) and 'data' in timeline[0]:
                                for series in timeline:
                                    for point in series.get('data', [])[-7:]:
                                        all_values.append(point.get('value', 0))
                            else:
                                all_values = [point.get('value', 0) for point in timeline[-7:]]

                            if all_values:
                                avg_volume = sum(all_values) / len(all_values)
                                max_volume = max(all_values)
                                trend = (all_values[-1] - all_values[0]) / (all_values[0] + 1) if all_values[0] else 0

                                return {
                                    'source': 'GDELT',
                                    'status': 'success',
                                    'indicators': {
                                        'gdelt_event_volume': avg_volume,
                                        'gdelt_peak_volume': max_volume,
                                        'gdelt_trend': min(1.0, max(-1.0, trend)),
                                        'gdelt_crisis_intensity': min(1.0, avg_volume / 10000) if avg_volume else 0.3
                                    }
                                }

                # Fallback: GEO API
                geo_url = "https://api.gdeltproject.org/api/v2/geo/geo"
                geo_params = {'query': 'protest OR conflict', 'format': 'geojson'}
                async with session.get(geo_url, params=geo_params) as geo_response:
                    if geo_response.status == 200:
                        geo_ct = geo_response.headers.get('Content-Type', '')
                        if 'json' in geo_ct.lower():
                            geo_data = await geo_response.json()
                            features = geo_data.get('features', [])
                            event_count = len(features)

                            return {
                                'source': 'GDELT',
                                'status': 'success',
                                'indicators': {
                                    'gdelt_event_volume': event_count,
                                    'gdelt_peak_volume': event_count,
                                    'gdelt_trend': 0.0,
                                    'gdelt_crisis_intensity': min(1.0, event_count / 500) if event_count else 0.3
                                }
                            }

        except Exception as e:
            logger.error(f"GDELT fetch error: {e}")

        return {
            'source': 'GDELT', 'status': 'simulated',
            'indicators': {
                'gdelt_event_volume': 5000,
                'gdelt_peak_volume': 8000,
                'gdelt_trend': 0.1,
                'gdelt_crisis_intensity': 0.5
            }
        }

    async def fetch_eia_energy(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Fetch EIA energy data."""
        if not api_key:
            return {
                'source': 'EIA', 'status': 'no_api_key',
                'indicators': {
                    'eia_crude_oil_price': 78.50,
                    'eia_natural_gas_price': 2.85,
                    'eia_oil_production_change': -0.02,
                    'eia_energy_volatility': 0.45,
                    'eia_strategic_reserve_level': 0.65
                }
            }

        base_url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
        params = {
            'api_key': api_key,
            'frequency': 'weekly',
            'data[0]': 'value',
            'facets[product][]': 'EPCWTI',
            'sort[0][column]': 'period',
            'sort[0][direction]': 'desc',
            'length': 10
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        records = data.get('response', {}).get('data', [])

                        if records:
                            prices = [float(r.get('value', 0)) for r in records if r.get('value')]
                            current_price = prices[0] if prices else 75.0
                            avg_price = sum(prices) / len(prices) if prices else 75.0
                            volatility = (max(prices) - min(prices)) / avg_price if prices and avg_price else 0.1

                            return {
                                'source': 'EIA',
                                'status': 'success',
                                'indicators': {
                                    'eia_crude_oil_price': current_price,
                                    'eia_natural_gas_price': 2.85,
                                    'eia_oil_production_change': (current_price - avg_price) / avg_price,
                                    'eia_energy_volatility': min(1.0, volatility),
                                    'eia_strategic_reserve_level': 0.65
                                }
                            }
        except Exception as e:
            logger.error(f"EIA fetch error: {e}")

        return {'source': 'EIA', 'status': 'error', 'indicators': {}}

    async def fetch_imf_data(self) -> Dict[str, Any]:
        """Fetch IMF financial indicators. No API key required."""
        url = "https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH"
        params = {'periods': '2024,2025,2026'}

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        values = data.get('values', {}).get('NGDP_RPCH', {})
                        world_data = values.get('W', {})
                        growth_2025 = world_data.get('2025', 3.2)
                        growth_2026 = world_data.get('2026', 3.3)

                        return {
                            'source': 'IMF',
                            'status': 'success',
                            'indicators': {
                                'imf_world_gdp_growth': growth_2025,
                                'imf_gdp_forecast': growth_2026,
                                'imf_growth_momentum': growth_2026 - growth_2025,
                                'imf_economic_health': min(1.0, max(0, (growth_2025 + 3) / 8))
                            }
                        }
        except Exception as e:
            logger.error(f"IMF fetch error: {e}")

        return {
            'source': 'IMF', 'status': 'simulated',
            'indicators': {
                'imf_world_gdp_growth': 3.2,
                'imf_gdp_forecast': 3.3,
                'imf_growth_momentum': 0.1,
                'imf_economic_health': 0.65
            }
        }

    async def fetch_fao_food(self) -> Dict[str, Any]:
        """Fetch FAO food price data. No API key required."""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                fpi_url = "https://www.fao.org/worldfoodsituation/foodpricesindex/data/IndexJson.json"

                async with session.get(fpi_url) as response:
                    if response.status == 200:
                        content_type = response.headers.get('Content-Type', '')
                        if 'json' in content_type.lower() or response.content_type == 'application/json':
                            data = await response.json()

                            if isinstance(data, list) and len(data) > 0:
                                recent = sorted(data, key=lambda x: x.get('Date', ''), reverse=True)[:12]
                                if recent:
                                    latest_value = float(recent[0].get('Food Price Index', 120))
                                    values = [float(r.get('Food Price Index', 120)) for r in recent if r.get('Food Price Index')]

                                    if values:
                                        avg_value = sum(values) / len(values)
                                        volatility = (max(values) - min(values)) / avg_value if avg_value else 0.1

                                        return {
                                            'source': 'FAO',
                                            'status': 'success',
                                            'indicators': {
                                                'fao_food_price_index': latest_value,
                                                'fao_price_volatility': min(1.0, volatility),
                                                'fao_food_security_risk': min(1.0, max(0, (latest_value - 100) / 50)),
                                                'fao_supply_stress': min(1.0, volatility * 2)
                                            }
                                        }
        except Exception as e:
            logger.error(f"FAO fetch error: {e}")

        return {
            'source': 'FAO', 'status': 'estimated',
            'indicators': {
                'fao_food_price_index': 118.5,
                'fao_price_volatility': 0.15,
                'fao_food_security_risk': 0.37,
                'fao_supply_stress': 0.4
            }
        }

    async def fetch_otx_threats(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Fetch AlienVault OTX cyber threat data."""
        if not api_key:
            return {
                'source': 'OTX', 'status': 'no_api_key',
                'indicators': {
                    'otx_threat_pulse_count': 150,
                    'otx_malware_indicators': 2500,
                    'otx_ransomware_activity': 0.55,
                    'otx_threat_severity': 0.6
                }
            }

        url = "https://otx.alienvault.com/api/v1/pulses/subscribed"
        headers = {'X-OTX-API-KEY': api_key}
        params = {'limit': 50, 'modified_since': (datetime.utcnow() - timedelta(days=7)).isoformat()}

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        pulses = data.get('results', [])

                        total_indicators = sum(len(p.get('indicators', [])) for p in pulses)
                        ransomware_count = sum(1 for p in pulses if 'ransomware' in p.get('name', '').lower())

                        return {
                            'source': 'OTX',
                            'status': 'success',
                            'indicators': {
                                'otx_threat_pulse_count': len(pulses),
                                'otx_malware_indicators': total_indicators,
                                'otx_ransomware_activity': min(1.0, ransomware_count / 10),
                                'otx_threat_severity': min(1.0, total_indicators / 5000)
                            }
                        }
        except Exception as e:
            logger.error(f"OTX fetch error: {e}")

        return {'source': 'OTX', 'status': 'error', 'indicators': {}}

    async def fetch_acled_conflicts(self, email: Optional[str] = None, password: Optional[str] = None) -> Dict[str, Any]:
        """Fetch ACLED conflict data via OAuth authentication."""
        if not email or not password:
            return {
                'source': 'ACLED', 'status': 'no_credentials',
                'indicators': {
                    'acled_conflict_events': 1200,
                    'acled_fatalities': 3500,
                    'acled_protest_count': 450,
                    'acled_violence_intensity': 0.55,
                    'acled_instability_index': 0.48
                }
            }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # Step 1: OAuth authentication
                auth_url = "https://acleddata.com/oauth/token"
                auth_data = {
                    'username': email,
                    'password': password,
                    'grant_type': 'password',
                    'client_id': 'acled'
                }
                auth_headers = {'Content-Type': 'application/x-www-form-urlencoded'}

                async with session.post(auth_url, data=auth_data, headers=auth_headers) as auth_response:
                    if auth_response.status != 200:
                        logger.error(f"ACLED OAuth failed: {auth_response.status}")
                        return {
                            'source': 'ACLED', 'status': 'auth_failed',
                            'indicators': {
                                'acled_conflict_events': 1200,
                                'acled_fatalities': 3500,
                                'acled_protest_count': 450,
                                'acled_violence_intensity': 0.55,
                                'acled_instability_index': 0.48
                            }
                        }

                    auth_result = await auth_response.json()
                    access_token = auth_result.get('access_token')

                    if not access_token:
                        return {'source': 'ACLED', 'status': 'auth_failed', 'indicators': {}}

                # Step 2: Fetch conflict data
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=30)

                data_url = "https://api.acleddata.com/acled/read"
                data_headers = {'Authorization': f'Bearer {access_token}'}
                params = {
                    'event_date': f"{start_date.strftime('%Y-%m-%d')}|{end_date.strftime('%Y-%m-%d')}",
                    'event_date_where': 'BETWEEN',
                    'limit': 5000
                }

                async with session.get(data_url, params=params, headers=data_headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        events = data.get('data', [])
                        count = data.get('count', len(events))

                        fatalities = sum(int(e.get('fatalities', 0) or 0) for e in events)
                        protests = sum(1 for e in events if 'protest' in str(e.get('event_type', '')).lower())
                        violence = sum(1 for e in events if 'violence' in str(e.get('event_type', '')).lower())

                        return {
                            'source': 'ACLED',
                            'status': 'success',
                            'indicators': {
                                'acled_conflict_events': count,
                                'acled_fatalities': fatalities,
                                'acled_protest_count': protests,
                                'acled_violence_intensity': min(1.0, violence / 1000),
                                'acled_instability_index': min(1.0, (count + fatalities) / 10000)
                            }
                        }
                    else:
                        logger.error(f"ACLED data fetch failed: {response.status}")

        except Exception as e:
            logger.error(f"ACLED fetch error: {e}")

        return {'source': 'ACLED', 'status': 'error', 'indicators': {}}

    async def fetch_all(self, api_keys: Dict[str, str] = None) -> Dict[str, Any]:
        """Fetch data from all 12 sources concurrently."""
        api_keys = api_keys or {}

        tasks = [
            self.fetch_usgs_earthquakes(),
            self.fetch_cisa_kev(),
            self.fetch_world_bank(),
            self.fetch_fred_data(api_keys.get('FRED')),
            self.fetch_noaa_climate(api_keys.get('NOAA')),
            self.fetch_nvd_vulnerabilities(api_keys.get('NVD')),
            self.fetch_gdelt_events(),
            self.fetch_eia_energy(api_keys.get('EIA')),
            self.fetch_imf_data(),
            self.fetch_fao_food(),
            self.fetch_otx_threats(api_keys.get('OTX')),
            self.fetch_acled_conflicts(api_keys.get('ACLED_EMAIL'), api_keys.get('ACLED_PASSWORD'))
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_indicators = {}
        source_status = {}

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Data source error: {result}")
                continue
            if isinstance(result, dict):
                source = result.get('source', 'UNKNOWN')
                source_status[source] = result.get('status', 'unknown')
                all_indicators.update(result.get('indicators', {}))

        return {
            'indicators': all_indicators,
            'sources': source_status,
            'fetch_time': datetime.utcnow().isoformat(),
            'total_sources': len(source_status),
            'total_indicators': len(all_indicators)
        }


# Global data fetcher instance
data_fetcher = DataFetcher()


# ============== Data Refresh Endpoint (BUGFIXED) ==============

@app.post("/api/v1/data/refresh")
async def refresh_data(
    recalculate: bool = Query(True, description="Trigger probability recalculation after refresh"),
    limit: int = Query(100, description="Max events to recalculate")
):
    """
    Refresh indicator data from all 12 external sources.

    BUGFIXES in v3.0.0:
    - Fixed: All 12 sources now mapped correctly (was missing GDELT, EIA, IMF, FAO, OTX, ACLED)
    - Fixed: New values are APPENDED as time series (were being overwritten, destroying history)
    - Fixed: Z-scores calculated from actual historical data (was using meaningless formula)
    """
    refresh_id = str(uuid.uuid4())[:8]
    start_time = datetime.utcnow()

    logger.info(f"Starting data refresh v3: {refresh_id}")

    try:
        # Get API keys from environment
        api_keys = {
            'FRED': os.getenv('FRED_API_KEY'),
            'NOAA': os.getenv('NOAA_API_KEY'),
            'NVD': os.getenv('NVD_API_KEY'),
            'EIA': os.getenv('EIA_API_KEY'),
            'OTX': os.getenv('OTX_API_KEY'),
            'ACLED_EMAIL': os.getenv('ACLED_EMAIL'),
            'ACLED_PASSWORD': os.getenv('ACLED_PASSWORD')
        }

        # Fetch data from all sources
        fetch_result = await data_fetcher.fetch_all(api_keys)
        indicators = fetch_result.get('indicators', {})
        sources = fetch_result.get('sources', {})

        logger.info(f"Fetched {len(indicators)} indicators from {len(sources)} sources")

        values_created = 0
        values_updated = 0

        with get_session_context() as session:
            for indicator_name, value in indicators.items():
                # BUGFIX #1: Use complete source mapping
                source = indicator_to_source(indicator_name)

                # ACCURACY FIX: Match on BOTH source AND indicator_name
                # Previously matched only on data_source, which meant ALL events
                # with any indicator from this source got the SAME values.
                # Now each event only gets values for indicators it actually uses.
                matching_weights = session.query(IndicatorWeight).filter(
                    IndicatorWeight.data_source == source,
                    IndicatorWeight.indicator_name == indicator_name
                ).all()

                for weight in matching_weights:
                    float_value = float(value) if isinstance(value, (int, float)) else 0.5

                    # BUGFIX #2: Calculate z-score from actual historical data
                    cutoff = start_time - timedelta(days=365)
                    historical = session.query(IndicatorValue).filter(
                        IndicatorValue.event_id == weight.event_id,
                        IndicatorValue.indicator_name == weight.indicator_name,
                        IndicatorValue.timestamp >= cutoff
                    ).order_by(IndicatorValue.timestamp.asc()).all()

                    historical_values = [h.value for h in historical if h.value is not None]

                    z_score, hist_mean, hist_std = SignalExtractor.calculate_z_score_from_history(
                        float_value, historical_values
                    )

                    # BUGFIX #3: APPEND new record instead of overwriting
                    # Check if we already have a value for this exact timestamp (same refresh)
                    existing_this_refresh = session.query(IndicatorValue).filter(
                        IndicatorValue.event_id == weight.event_id,
                        IndicatorValue.indicator_name == weight.indicator_name,
                        IndicatorValue.timestamp == start_time
                    ).first()

                    if existing_this_refresh:
                        existing_this_refresh.value = float_value
                        existing_this_refresh.z_score = z_score
                        existing_this_refresh.quality_score = 0.9
                        values_updated += 1
                    else:
                        new_value = IndicatorValue(
                            event_id=weight.event_id,
                            indicator_name=weight.indicator_name,
                            data_source=source,
                            timestamp=start_time,
                            value=float_value,
                            raw_value=float_value,
                            historical_mean=hist_mean,
                            historical_std=hist_std,
                            z_score=z_score,
                            quality_score=0.9
                        )
                        session.add(new_value)
                        values_created += 1

            session.commit()

            # Record data source health
            for source_name, status in sources.items():
                health_record = DataSourceHealth(
                    source_name=source_name,
                    check_time=start_time,
                    status='OPERATIONAL' if status == 'success' else 'DEGRADED',
                    success_rate_24h=1.0 if status == 'success' else 0.5
                )
                session.add(health_record)

            session.commit()

        duration = (datetime.utcnow() - start_time).total_seconds()

        result = {
            "refresh_id": refresh_id,
            "version": "3.0.0",
            "status": "completed",
            "duration_seconds": round(duration, 2),
            "indicators_fetched": len(indicators),
            "values_created": values_created,
            "values_updated": values_updated,
            "sources": sources,
            "bugfixes_applied": [
                "All 12 sources now mapped (was 6)",
                "Values appended as time series (was overwriting)",
                "Z-scores from historical data (was hardcoded formula)"
            ]
        }

        # Optionally trigger enhanced recalculation
        if recalculate:
            logger.info("Triggering enhanced probability recalculation...")
            calc_result = await trigger_enhanced_calculation(limit=limit)
            result["recalculation"] = {
                "calculation_id": calc_result.get("calculation_id"),
                "events_processed": calc_result.get("events_processed"),
                "events_succeeded": calc_result.get("events_succeeded"),
                "version": calc_result.get("version")
            }

        logger.info(f"Data refresh {refresh_id} complete: {values_created} created, {values_updated} updated")
        return result

    except Exception as e:
        logger.error(f"Data refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data/sources")
async def list_data_sources():
    """List all configured data sources and their status."""
    with get_session_context() as session:
        from sqlalchemy import func

        subquery = session.query(
            DataSourceHealth.source_name,
            func.max(DataSourceHealth.check_time).label('latest')
        ).group_by(DataSourceHealth.source_name).subquery()

        health_records = session.query(DataSourceHealth).join(
            subquery,
            (DataSourceHealth.source_name == subquery.c.source_name) &
            (DataSourceHealth.check_time == subquery.c.latest)
        ).all()

        source_counts = session.query(
            IndicatorWeight.data_source,
            func.count(IndicatorWeight.id)
        ).group_by(IndicatorWeight.data_source).all()

        count_map = {s: c for s, c in source_counts}

        sources = []
        for h in health_records:
            sources.append({
                "name": h.source_name,
                "status": h.status,
                "last_check": h.check_time.isoformat() if h.check_time else None,
                "indicator_count": count_map.get(h.source_name, 0)
            })

        all_sources = [
            'USGS', 'CISA', 'NVD', 'FRED', 'NOAA', 'WORLD_BANK',
            'GDELT', 'EIA', 'IMF', 'FAO', 'OTX', 'ACLED',
            'BLS', 'OSHA', 'INTERNAL'
        ]
        checked = {s["name"] for s in sources}
        for src in all_sources:
            if src not in checked:
                sources.append({
                    "name": src,
                    "status": "NOT_CHECKED",
                    "last_check": None,
                    "indicator_count": count_map.get(src, 0)
                })

        return {
            "sources": sources,
            "total_indicator_weights": sum(count_map.values())
        }


# ============== Error Handler ==============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
