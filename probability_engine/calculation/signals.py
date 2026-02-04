"""
Signal Calculation Module

Calculates indicator signals from raw values using statistical methods.
Combines z-score (deviation from norm) with momentum (rate of change).
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config.settings import TIME_SCALE_WEIGHTS


@dataclass
class SignalResult:
    """Result of signal calculation with components."""
    signal: float  # Final signal (-1 to +1)
    z_score: float
    z_signal: float
    momentum: float
    current_value: float
    historical_mean: float
    historical_std: float


class SignalCalculator:
    """
    Calculate indicator signals from time series data.

    Signal = (z_weight × z_signal) + (momentum_weight × momentum_signal)

    Where:
    - z_signal = tanh(z_score / 2) - deviation from historical mean
    - momentum_signal = normalized trend direction
    """

    def calculate_z_score(
        self,
        current_value: float,
        historical_values: List[float]
    ) -> float:
        """
        Calculate z-score (standard deviations from mean).

        Args:
            current_value: Most recent value
            historical_values: Historical values for comparison

        Returns:
            Z-score (unbounded)
        """
        if not historical_values or len(historical_values) < 2:
            return 0.0

        mean = np.mean(historical_values)
        std = np.std(historical_values)

        if std < 0.0001:  # Avoid division by zero
            return 0.0

        return (current_value - mean) / std

    def calculate_momentum(
        self,
        values: List[float],
        window: int = 12
    ) -> float:
        """
        Calculate momentum (rate of change) using linear regression.

        Args:
            values: Time series (oldest to newest)
            window: Number of periods to consider

        Returns:
            Normalized momentum (-1 to +1)
        """
        if len(values) < 2:
            return 0.0

        # Take last N values
        recent = values[-window:] if len(values) >= window else values

        if len(recent) < 2:
            return 0.0

        # Linear regression slope
        x = np.arange(len(recent))
        y = np.array(recent)

        # Normalize
        if np.std(y) < 0.0001:
            return 0.0

        # Calculate slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

        # Normalize by standard deviation
        normalized_slope = slope / (np.std(y) / len(y))

        # Bound to (-1, 1) using tanh
        return math.tanh(normalized_slope)

    def calculate_signal(
        self,
        current_value: float,
        historical_values: List[float],
        time_scale: str = 'medium'
    ) -> SignalResult:
        """
        Calculate indicator signal combining z-score and momentum.

        Args:
            current_value: Most recent indicator value
            historical_values: Historical time series (oldest to newest)
            time_scale: 'fast', 'medium', or 'slow' (affects weighting)

        Returns:
            SignalResult with all components
        """
        # Get weights for this time scale
        weights = TIME_SCALE_WEIGHTS.get(time_scale, TIME_SCALE_WEIGHTS['medium'])
        z_weight = weights['z_score']
        m_weight = weights['momentum']

        # Calculate z-score
        z_score = self.calculate_z_score(current_value, historical_values)

        # Convert z-score to bounded signal using tanh
        # z-score of 2 -> signal of ~0.76
        # z-score of 3 -> signal of ~0.91
        z_signal = math.tanh(z_score / 2)

        # Calculate momentum
        momentum = self.calculate_momentum(historical_values)

        # Combine with weights
        signal = z_weight * z_signal + m_weight * momentum

        # Ensure bounds
        signal = max(-1.0, min(1.0, signal))

        # Calculate historical stats for reporting
        hist_mean = np.mean(historical_values) if historical_values else current_value
        hist_std = np.std(historical_values) if len(historical_values) > 1 else 0

        return SignalResult(
            signal=signal,
            z_score=z_score,
            z_signal=z_signal,
            momentum=momentum,
            current_value=current_value,
            historical_mean=hist_mean,
            historical_std=hist_std
        )

    def batch_calculate_signals(
        self,
        indicators: Dict[str, Dict],
        time_scales: Dict[str, str]
    ) -> Dict[str, SignalResult]:
        """
        Calculate signals for multiple indicators.

        Args:
            indicators: Dict of indicator_name -> {current: float, history: List[float]}
            time_scales: Dict of indicator_name -> time_scale

        Returns:
            Dict of indicator_name -> SignalResult
        """
        results = {}
        for name, data in indicators.items():
            current = data.get('current', 0)
            history = data.get('history', [])
            time_scale = time_scales.get(name, 'medium')

            results[name] = self.calculate_signal(current, history, time_scale)

        return results


def calculate_simple_signal(current: float, historical: List[float]) -> float:
    """
    Quick signal calculation with default settings.

    Args:
        current: Current value
        historical: Historical values

    Returns:
        Signal value (-1 to +1)
    """
    calc = SignalCalculator()
    result = calc.calculate_signal(current, historical)
    return result.signal
