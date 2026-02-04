"""
Bootstrap Confidence Interval Calculator

Estimates uncertainty in probability calculations using bootstrap resampling.
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from config.settings import PRECISION_BANDS


@dataclass
class ConfidenceIntervalResult:
    """Result of confidence interval calculation."""
    point_estimate: float  # Central probability estimate
    ci_lower: float  # Lower bound
    ci_upper: float  # Upper bound
    ci_level: float  # Confidence level (e.g., 0.80)
    ci_width: float  # Width in percentage points
    precision_band: str  # HIGH, MODERATE, LOW, VERY_LOW
    bootstrap_iterations: int
    std_error: float


class BootstrapConfidenceCalculator:
    """
    Calculate confidence intervals using bootstrap resampling.

    The bootstrap approach:
    1. Resample indicator signals with replacement
    2. Recalculate probability for each resample
    3. Use percentiles of bootstrap distribution for CI
    """

    def __init__(self, n_iterations: int = 1000, confidence_level: float = 0.80):
        """
        Initialize calculator.

        Args:
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level (default 80%)
        """
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level

    def calculate_ci(
        self,
        point_estimate: float,
        signals: List[float],
        weights: List[float],
        betas: List[float],
        baseline_log_odds: float,
        dependency_adjustment: float = 0.0,
        signal_uncertainties: Optional[List[float]] = None
    ) -> ConfidenceIntervalResult:
        """
        Calculate bootstrap confidence interval.

        Args:
            point_estimate: Central probability estimate
            signals: List of indicator signals (-1 to +1)
            weights: Normalized weights (sum to 1)
            betas: Beta parameters for each indicator
            baseline_log_odds: Log-odds of baseline probability
            dependency_adjustment: Additional adjustment from dependencies
            signal_uncertainties: Optional std dev for each signal

        Returns:
            ConfidenceIntervalResult with CI bounds
        """
        if not signals or len(signals) == 0:
            # No signals - use wider default interval
            return self._create_default_ci(point_estimate)

        # Set default uncertainties if not provided
        if signal_uncertainties is None:
            # Assume moderate uncertainty proportional to signal magnitude
            signal_uncertainties = [0.2 * (1 + abs(s)) for s in signals]

        # Run bootstrap
        bootstrap_probs = []

        for _ in range(self.n_iterations):
            # Perturb signals based on uncertainties
            perturbed_signals = [
                np.clip(s + np.random.normal(0, u), -1, 1)
                for s, u in zip(signals, signal_uncertainties)
            ]

            # Calculate log-odds adjustment
            total_adjustment = sum(
                w * s * b
                for w, s, b in zip(weights, perturbed_signals, betas)
            )

            # Add some uncertainty to dependency adjustment too
            dep_adj_perturbed = dependency_adjustment * (1 + np.random.normal(0, 0.1))

            # Calculate probability
            log_odds = baseline_log_odds + total_adjustment + dep_adj_perturbed
            prob = 1 / (1 + math.exp(-log_odds))

            # Bound probability
            prob = max(0.001, min(0.999, prob))
            bootstrap_probs.append(prob * 100)  # Convert to percentage

        # Calculate percentiles
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_probs, lower_percentile)
        ci_upper = np.percentile(bootstrap_probs, upper_percentile)
        ci_width = ci_upper - ci_lower

        # Determine precision band
        precision_band = self._get_precision_band(ci_width)

        # Calculate standard error
        std_error = np.std(bootstrap_probs)

        return ConfidenceIntervalResult(
            point_estimate=point_estimate * 100,  # Convert to percentage
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.confidence_level,
            ci_width=ci_width,
            precision_band=precision_band,
            bootstrap_iterations=self.n_iterations,
            std_error=std_error
        )

    def _create_default_ci(self, point_estimate: float) -> ConfidenceIntervalResult:
        """Create a default wide CI when no signals available."""
        prob_pct = point_estimate * 100

        # Use wider interval for no-data case
        ci_half_width = 20  # 20 percentage points each side
        ci_lower = max(0.1, prob_pct - ci_half_width)
        ci_upper = min(99.9, prob_pct + ci_half_width)

        return ConfidenceIntervalResult(
            point_estimate=prob_pct,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.confidence_level,
            ci_width=ci_upper - ci_lower,
            precision_band="VERY_LOW",
            bootstrap_iterations=0,
            std_error=ci_half_width / 2
        )

    def _get_precision_band(self, ci_width: float) -> str:
        """Classify precision based on CI width."""
        if ci_width < PRECISION_BANDS["HIGH"]:
            return "HIGH"
        elif ci_width < PRECISION_BANDS["MODERATE"]:
            return "MODERATE"
        elif ci_width < PRECISION_BANDS["LOW"]:
            return "LOW"
        else:
            return "VERY_LOW"


class ConfidenceScorer:
    """
    Calculate confidence score based on data quality metrics.

    Three components:
    1. Data completeness (% of expected indicators with data)
    2. Source agreement (consistency across sources)
    3. Data recency (freshness of data)
    """

    def calculate_confidence_score(
        self,
        expected_indicators: int,
        actual_indicators: int,
        signal_values: List[float],
        data_ages_hours: List[float],
        max_age_hours: float = 168  # 1 week
    ) -> Tuple[float, float, float, float]:
        """
        Calculate overall confidence score.

        Args:
            expected_indicators: Number of indicators expected
            actual_indicators: Number with actual data
            signal_values: List of signal values
            data_ages_hours: Age of each data point in hours
            max_age_hours: Maximum acceptable age

        Returns:
            Tuple of (overall_score, completeness, agreement, recency)
        """
        # Data completeness (0 to 1)
        if expected_indicators > 0:
            completeness = actual_indicators / expected_indicators
        else:
            completeness = 0.0

        # Source agreement (inverse of signal variance)
        if len(signal_values) >= 2:
            signal_variance = np.var(signal_values)
            # Map variance to agreement score (high variance = low agreement)
            # Variance of uniform(-1,1) is 0.33, so normalize by that
            agreement = max(0, 1 - (signal_variance / 0.33))
        else:
            agreement = 0.5  # Neutral when insufficient data

        # Data recency (penalize old data)
        if data_ages_hours:
            avg_age = np.mean(data_ages_hours)
            recency = max(0, 1 - (avg_age / max_age_hours))
        else:
            recency = 0.5  # Neutral when no age data

        # Weighted combination
        # Completeness is most important, then recency, then agreement
        overall = (
            0.5 * completeness +
            0.3 * recency +
            0.2 * agreement
        )

        return overall, completeness, agreement, recency


def quick_confidence_interval(
    probability: float,
    n_indicators: int = 5,
    data_quality: float = 0.7
) -> Tuple[float, float]:
    """
    Quick confidence interval estimation without full bootstrap.

    Uses a heuristic based on number of indicators and data quality.

    Args:
        probability: Point probability estimate (0-1)
        n_indicators: Number of indicators used
        data_quality: Data quality score (0-1)

    Returns:
        Tuple of (ci_lower, ci_upper) in percentages
    """
    prob_pct = probability * 100

    # Base width depends on probability (wider near 50%, narrower at extremes)
    base_width = 10 * (1 - abs(probability - 0.5) * 2) + 5

    # Adjust for number of indicators (more = narrower)
    indicator_factor = 1.5 / math.sqrt(max(1, n_indicators))

    # Adjust for data quality (higher = narrower)
    quality_factor = 2 - data_quality

    half_width = base_width * indicator_factor * quality_factor

    ci_lower = max(0.1, prob_pct - half_width)
    ci_upper = min(99.9, prob_pct + half_width)

    return ci_lower, ci_upper
