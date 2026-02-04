"""
Calculation Engine Module

Core probability calculation components.
"""

from .log_odds import LogOddsCalculator, ProbabilityResult
from .signals import SignalCalculator, SignalResult, calculate_simple_signal
from .confidence_interval import (
    BootstrapConfidenceCalculator,
    ConfidenceScorer,
    ConfidenceIntervalResult,
    quick_confidence_interval
)
from .dependencies import (
    DependencyNetwork,
    DependencyAdjuster,
    DependencyAdjustment,
    load_dependencies_from_db
)
from .orchestrator import (
    ProbabilityOrchestrator,
    EventCalculationResult,
    CalculationBatch,
    run_weekly_calculation
)

__all__ = [
    # Log-odds
    'LogOddsCalculator',
    'ProbabilityResult',
    # Signals
    'SignalCalculator',
    'SignalResult',
    'calculate_simple_signal',
    # Confidence intervals
    'BootstrapConfidenceCalculator',
    'ConfidenceScorer',
    'ConfidenceIntervalResult',
    'quick_confidence_interval',
    # Dependencies
    'DependencyNetwork',
    'DependencyAdjuster',
    'DependencyAdjustment',
    'load_dependencies_from_db',
    # Orchestrator
    'ProbabilityOrchestrator',
    'EventCalculationResult',
    'CalculationBatch',
    'run_weekly_calculation',
]
