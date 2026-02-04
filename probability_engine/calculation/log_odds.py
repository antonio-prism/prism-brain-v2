"""
Log-Odds Probability Calculator

Implements the rigorous log-odds linear model for probability calculation.
Based on logistic regression framework - mathematically sound and widely accepted.
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from config.settings import BETA_PARAMETERS, get_settings


@dataclass
class ProbabilityResult:
    """Result of probability calculation with all components."""
    probability: float
    log_odds: float
    baseline_probability: float
    baseline_log_odds: float
    total_adjustment: float
    dependency_adjustment: float
    confidence_score: float
    attribution: List[Dict]


class LogOddsProbabilityCalculator:
    """
    Calculate risk probabilities using the log-odds linear model.

    The model: log_odds_final = log_odds_baseline + Σ(weight_i × signal_i × beta_i)

    This is mathematically equivalent to logistic regression and produces
    properly calibrated probabilities.
    """

    def __init__(self):
        self.settings = get_settings()
        self.min_prob = self.settings.min_probability
        self.max_prob = self.settings.max_probability

    @staticmethod
    def scale_to_return_period(scale: float) -> float:
        """
        Convert 1-5 scale to return period in years.

        Scale mapping:
        1 = 20 year return period (rare)
        2 = 10 year return period (unlikely)
        3 = 5 year return period (possible)
        4 = 2 year return period (likely)
        5 = 1 year return period (very likely)

        Formula: T = 20 / 2^(scale-1)
        """
        return 20.0 / (2 ** (scale - 1))

    @staticmethod
    def return_period_to_probability(return_period: float) -> float:
        """
        Convert return period to annual probability using Poisson process.

        Formula: P(at least one in 1 year) = 1 - e^(-1/T)

        This is the standard model for rare event occurrence.
        """
        if return_period <= 0:
            return 0.5
        return 1 - math.exp(-1 / return_period)

    def baseline_to_probability(self, scale: float) -> float:
        """
        Convert 1-5 scale to annual probability.

        Args:
            scale: Baseline scale value (1-5, can be fractional)

        Returns:
            Annual probability (0 to 1)
        """
        # Clamp to valid range
        scale = max(1.0, min(5.0, scale))

        # Get return period
        return_period = self.scale_to_return_period(scale)

        # Convert to probability
        return self.return_period_to_probability(return_period)

    @staticmethod
    def probability_to_log_odds(probability: float) -> float:
        """
        Convert probability to log-odds.

        Formula: log_odds = ln(p / (1-p))
        """
        # Clamp to avoid infinity
        p = max(0.001, min(0.999, probability))
        return math.log(p / (1 - p))

    @staticmethod
    def log_odds_to_probability(log_odds: float) -> float:
        """
        Convert log-odds back to probability.

        Formula: p = 1 / (1 + e^(-log_odds))
        """
        return 1 / (1 + math.exp(-log_odds))

    def calculate_indicator_contribution(
        self,
        signal: float,
        weight: float,
        beta: float
    ) -> float:
        """
        Calculate a single indicator's contribution to log-odds.

        Args:
            signal: Indicator signal strength (-1 to +1)
            weight: Indicator weight (0 to 1, should sum to 1 across indicators)
            beta: Maximum log-odds shift for this indicator type

        Returns:
            Log-odds contribution
        """
        # Contribution = weight × signal × beta
        return weight * signal * beta

    def calculate_probability(
        self,
        baseline_scale: float,
        indicator_signals: Dict[str, float],
        indicator_weights: Dict[str, float],
        indicator_betas: Dict[str, float],
        dependency_adjustment: float = 0.0
    ) -> ProbabilityResult:
        """
        Calculate probability using the log-odds model.

        Args:
            baseline_scale: Baseline probability scale (1-5)
            indicator_signals: Dict of indicator_name -> signal [-1, 1]
            indicator_weights: Dict of indicator_name -> weight [0, 1]
            indicator_betas: Dict of indicator_name -> beta value
            dependency_adjustment: Additional adjustment from correlated events

        Returns:
            ProbabilityResult with full attribution
        """
        # Step 1: Convert baseline to probability and log-odds
        baseline_prob = self.baseline_to_probability(baseline_scale)
        baseline_log_odds = self.probability_to_log_odds(baseline_prob)

        # Step 2: Calculate contributions from each indicator
        attribution = []
        total_adjustment = 0.0

        for indicator_name in indicator_signals:
            signal = indicator_signals.get(indicator_name, 0.0)
            weight = indicator_weights.get(indicator_name, 0.0)
            beta = indicator_betas.get(indicator_name, 0.7)

            contribution = self.calculate_indicator_contribution(signal, weight, beta)
            total_adjustment += contribution

            attribution.append({
                'indicator': indicator_name,
                'signal': signal,
                'weight': weight,
                'beta': beta,
                'contribution': contribution,
                'contribution_pct': 0  # Will be calculated after
            })

        # Step 3: Apply log-odds adjustment
        log_odds_final = baseline_log_odds + total_adjustment + dependency_adjustment

        # Step 4: Convert back to probability
        probability = self.log_odds_to_probability(log_odds_final)

        # Step 5: Apply bounds
        probability = max(self.min_prob, min(self.max_prob, probability))

        # Step 6: Calculate contribution percentages
        prob_change = probability - baseline_prob
        for attr in attribution:
            if abs(prob_change) > 0.0001:
                # Approximate contribution to probability change
                attr['contribution_pct'] = (attr['contribution'] / (total_adjustment + dependency_adjustment + 0.0001)) * prob_change * 100
            else:
                attr['contribution_pct'] = 0

        # Calculate confidence score (simplified)
        confidence = self._calculate_confidence(indicator_signals, indicator_weights)

        return ProbabilityResult(
            probability=probability,
            log_odds=log_odds_final,
            baseline_probability=baseline_prob,
            baseline_log_odds=baseline_log_odds,
            total_adjustment=total_adjustment,
            dependency_adjustment=dependency_adjustment,
            confidence_score=confidence,
            attribution=attribution
        )

    def _calculate_confidence(
        self,
        signals: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calculate confidence score based on data completeness and agreement."""
        if not signals:
            return 0.5

        # Factor 1: Data completeness (how many indicators have signals)
        expected_indicators = len(weights)
        actual_indicators = sum(1 for s in signals.values() if s != 0)
        completeness = actual_indicators / max(expected_indicators, 1)

        # Factor 2: Signal agreement (do signals point in same direction)
        positive = sum(1 for s in signals.values() if s > 0.1)
        negative = sum(1 for s in signals.values() if s < -0.1)
        total = positive + negative
        if total > 0:
            agreement = max(positive, negative) / total
        else:
            agreement = 0.5

        # Combine factors
        confidence = 0.5 * completeness + 0.5 * agreement
        return min(1.0, max(0.0, confidence))


# Convenience functions
def calculate_baseline_probability(scale: float) -> float:
    """Quick conversion from scale to probability."""
    calc = LogOddsProbabilityCalculator()
    return calc.baseline_to_probability(scale)


def calculate_probability_simple(
    baseline_scale: float,
    weighted_signal: float,
    avg_beta: float = 0.8
) -> float:
    """
    Simplified probability calculation with single aggregated signal.

    Args:
        baseline_scale: 1-5 baseline scale
        weighted_signal: Pre-aggregated weighted signal (-1 to +1)
        avg_beta: Average beta parameter

    Returns:
        Final probability
    """
    calc = LogOddsProbabilityCalculator()
    baseline_prob = calc.baseline_to_probability(baseline_scale)
    baseline_log_odds = calc.probability_to_log_odds(baseline_prob)

    # Apply adjustment
    adjustment = weighted_signal * avg_beta
    final_log_odds = baseline_log_odds + adjustment

    # Convert back
    probability = calc.log_odds_to_probability(final_log_odds)

    # Apply bounds
    return max(0.001, min(0.999, probability))
