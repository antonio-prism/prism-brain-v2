"""
Probability Calculation Orchestrator

Main entry point for probability calculations.
Coordinates data acquisition, signal calculation, and probability estimation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .log_odds import LogOddsCalculator, ProbabilityResult
from .signals import SignalCalculator, SignalResult
from .confidence_interval import BootstrapConfidenceCalculator, ConfidenceScorer
from .dependencies import DependencyNetwork, DependencyAdjuster, load_dependencies_from_db

from config.settings import (
    get_settings,
    BETA_PARAMETERS,
    DATA_SOURCES,
    PRECISION_BANDS
)
from database.connection import get_session_context
from database.models import (
    RiskEvent, RiskProbability, IndicatorValue, IndicatorWeight,
    CalculationLog, MethodologyTier
)

logger = logging.getLogger(__name__)


@dataclass
class EventCalculationResult:
    """Complete result for a single event calculation."""
    event_id: str
    event_name: str
    probability_pct: float
    ci_lower_pct: float
    ci_upper_pct: float
    precision_band: str
    confidence_score: float
    methodology_tier: str

    # Components
    baseline_probability_pct: float
    log_odds: float
    signals: Dict[str, SignalResult] = field(default_factory=dict)
    dependency_adjustment: float = 0.0

    # Metadata
    calculation_duration_ms: int = 0
    flags: List[str] = field(default_factory=list)
    attribution: Dict[str, float] = field(default_factory=dict)

    # Change tracking
    previous_probability_pct: Optional[float] = None
    change_direction: str = "STABLE"


@dataclass
class CalculationBatch:
    """Results of a batch calculation run."""
    calculation_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    events_processed: int = 0
    events_succeeded: int = 0
    events_failed: int = 0
    results: List[EventCalculationResult] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)


class ProbabilityOrchestrator:
    """
    Main orchestrator for probability calculations.

    Coordinates:
    1. Loading event configuration and weights
    2. Fetching indicator data from APIs
    3. Calculating signals from raw data
    4. Computing probabilities with confidence intervals
    5. Applying dependency adjustments
    6. Storing results in database
    """

    def __init__(self, data_clients: Optional[Dict] = None):
        """
        Initialize orchestrator.

        Args:
            data_clients: Optional dict of data source name -> client instance
        """
        self.settings = get_settings()
        self.data_clients = data_clients or {}

        # Initialize calculators
        self.signal_calc = SignalCalculator()
        self.log_odds_calc = LogOddsCalculator(
            max_probability=self.settings.max_probability,
            min_probability=self.settings.min_probability
        )
        self.ci_calc = BootstrapConfidenceCalculator(
            n_iterations=self.settings.bootstrap_iterations,
            confidence_level=self.settings.confidence_level
        )
        self.confidence_scorer = ConfidenceScorer()

        # Dependency network (loaded on first use)
        self._dependency_network: Optional[DependencyNetwork] = None
        self._dependency_adjuster: Optional[DependencyAdjuster] = None

    def _load_dependency_network(self):
        """Load dependency network from database."""
        if self._dependency_network is None:
            with get_session_context() as session:
                self._dependency_network = load_dependencies_from_db(session)
                self._dependency_adjuster = DependencyAdjuster(self._dependency_network)

    def generate_calculation_id(self) -> str:
        """Generate a unique calculation ID based on date."""
        now = datetime.utcnow()
        week_num = now.isocalendar()[1]
        return f"{now.year}-W{week_num:02d}"

    async def calculate_single_event(
        self,
        event: RiskEvent,
        weights: List[IndicatorWeight],
        indicator_data: Dict[str, Any],
        dependency_adjustment: float = 0.0
    ) -> EventCalculationResult:
        """
        Calculate probability for a single event.

        Args:
            event: RiskEvent instance
            weights: List of IndicatorWeight for this event
            indicator_data: Dict of indicator_name -> {current, history}
            dependency_adjustment: Pre-calculated dependency adjustment

        Returns:
            EventCalculationResult
        """
        start_time = datetime.utcnow()
        flags = []

        # Prepare signals
        signals: Dict[str, SignalResult] = {}
        signal_values = []
        weight_values = []
        beta_values = []

        for weight in weights:
            indicator_name = weight.indicator_name
            data = indicator_data.get(indicator_name, {})

            if not data:
                continue

            current = data.get('current', 0)
            history = data.get('history', [])

            # Calculate signal
            signal_result = self.signal_calc.calculate_signal(
                current_value=current,
                historical_values=history,
                time_scale=weight.time_scale or 'medium'
            )

            signals[indicator_name] = signal_result
            signal_values.append(signal_result.signal)
            weight_values.append(weight.normalized_weight)

            # Get beta from config
            beta_type = weight.beta_type or 'moderate_correlation'
            beta_values.append(BETA_PARAMETERS.get(beta_type, 0.7))

        # Check for conflicting signals
        if signal_values:
            positive_signals = sum(1 for s in signal_values if s > 0.3)
            negative_signals = sum(1 for s in signal_values if s < -0.3)
            if positive_signals > 0 and negative_signals > 0:
                if min(positive_signals, negative_signals) >= 2:
                    flags.append("CONFLICTING_SIGNALS")

        # Calculate probability
        baseline_scale = event.baseline_probability or 3
        prob_result = self.log_odds_calc.calculate_probability(
            baseline_scale=baseline_scale,
            indicator_signals=signal_values,
            indicator_weights=weight_values,
            indicator_betas=beta_values,
            dependency_adjustment=dependency_adjustment
        )

        # Check for black swan
        if prob_result.probability > 0.5 and baseline_scale <= 2:
            flags.append("BLACK_SWAN")

        # Calculate confidence interval
        ci_result = self.ci_calc.calculate_ci(
            point_estimate=prob_result.probability,
            signals=signal_values,
            weights=weight_values,
            betas=beta_values,
            baseline_log_odds=prob_result.baseline_log_odds,
            dependency_adjustment=dependency_adjustment
        )

        # Calculate confidence score
        expected_indicators = len(weights)
        actual_indicators = len(signal_values)
        data_ages = [1.0] * len(signal_values)  # Placeholder - would come from actual data

        confidence_score, completeness, agreement, recency = \
            self.confidence_scorer.calculate_confidence_score(
                expected_indicators=expected_indicators,
                actual_indicators=actual_indicators,
                signal_values=signal_values,
                data_ages_hours=data_ages
            )

        if confidence_score < 0.3:
            flags.append("LOW_CONFIDENCE")

        # Calculate attribution (which indicators contributed most)
        attribution = {}
        total_contribution = 0
        for i, (signal, weight, beta) in enumerate(zip(signal_values, weight_values, beta_values)):
            contribution = abs(weight * signal * beta)
            total_contribution += contribution

        if total_contribution > 0:
            idx = 0
            for weight_obj in weights:
                if weight_obj.indicator_name in signals:
                    signal = signals[weight_obj.indicator_name].signal
                    w = weight_obj.normalized_weight
                    b = BETA_PARAMETERS.get(weight_obj.beta_type or 'moderate_correlation', 0.7)
                    contribution = abs(w * signal * b) / total_contribution
                    attribution[weight_obj.indicator_name] = round(contribution * 100, 1)
                    idx += 1

        # Calculate duration
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return EventCalculationResult(
            event_id=event.event_id,
            event_name=event.event_name,
            probability_pct=prob_result.probability_pct,
            ci_lower_pct=ci_result.ci_lower,
            ci_upper_pct=ci_result.ci_upper,
            precision_band=ci_result.precision_band,
            confidence_score=confidence_score,
            methodology_tier=event.methodology_tier or "TIER_2_ANALOG",
            baseline_probability_pct=prob_result.baseline_probability_pct,
            log_odds=prob_result.log_odds_final,
            signals=signals,
            dependency_adjustment=dependency_adjustment,
            calculation_duration_ms=duration_ms,
            flags=flags,
            attribution=attribution
        )

    async def run_batch_calculation(
        self,
        event_ids: Optional[List[str]] = None,
        include_dependencies: bool = True
    ) -> CalculationBatch:
        """
        Run probability calculations for multiple events.

        Args:
            event_ids: Optional list of specific events (None = all events)
            include_dependencies: Whether to apply dependency adjustments

        Returns:
            CalculationBatch with all results
        """
        calculation_id = self.generate_calculation_id()
        batch = CalculationBatch(
            calculation_id=calculation_id,
            start_time=datetime.utcnow()
        )

        logger.info(f"Starting batch calculation: {calculation_id}")

        # Load dependency network if needed
        if include_dependencies:
            self._load_dependency_network()

        with get_session_context() as session:
            # Load events
            query = session.query(RiskEvent)
            if event_ids:
                query = query.filter(RiskEvent.event_id.in_(event_ids))
            events = query.all()

            logger.info(f"Processing {len(events)} events")

            # Load all weights
            weights_query = session.query(IndicatorWeight)
            if event_ids:
                weights_query = weights_query.filter(IndicatorWeight.event_id.in_(event_ids))
            all_weights = weights_query.all()

            # Group weights by event
            weights_by_event: Dict[str, List[IndicatorWeight]] = {}
            for w in all_weights:
                if w.event_id not in weights_by_event:
                    weights_by_event[w.event_id] = []
                weights_by_event[w.event_id].append(w)

            # First pass: calculate probabilities without dependencies
            first_pass_results: Dict[str, EventCalculationResult] = {}
            first_pass_probs: Dict[str, float] = {}
            baseline_probs: Dict[str, float] = {}

            for event in events:
                batch.events_processed += 1

                try:
                    weights = weights_by_event.get(event.event_id, [])

                    # Get indicator data (would come from data clients in production)
                    indicator_data = await self._fetch_indicator_data(event, weights)

                    result = await self.calculate_single_event(
                        event=event,
                        weights=weights,
                        indicator_data=indicator_data,
                        dependency_adjustment=0.0
                    )

                    first_pass_results[event.event_id] = result
                    first_pass_probs[event.event_id] = result.probability_pct / 100
                    baseline_probs[event.event_id] = result.baseline_probability_pct / 100
                    batch.events_succeeded += 1

                except Exception as e:
                    logger.error(f"Error calculating {event.event_id}: {e}")
                    batch.events_failed += 1
                    batch.errors.append({
                        "event_id": event.event_id,
                        "error": str(e)
                    })

            # Second pass: apply dependency adjustments
            if include_dependencies and self._dependency_adjuster:
                for event_id, result in first_pass_results.items():
                    try:
                        adjustment = self._dependency_adjuster.calculate_adjustment(
                            event_id=event_id,
                            driver_probabilities=first_pass_probs,
                            driver_baselines=baseline_probs
                        )

                        if abs(adjustment.log_odds_adjustment) > 0.01:
                            # Recalculate with dependency adjustment
                            event = next(e for e in events if e.event_id == event_id)
                            weights = weights_by_event.get(event_id, [])
                            indicator_data = await self._fetch_indicator_data(event, weights)

                            result = await self.calculate_single_event(
                                event=event,
                                weights=weights,
                                indicator_data=indicator_data,
                                dependency_adjustment=adjustment.log_odds_adjustment
                            )
                            first_pass_results[event_id] = result

                    except Exception as e:
                        logger.warning(f"Error applying dependencies to {event_id}: {e}")

            # Compile final results
            batch.results = list(first_pass_results.values())
            batch.end_time = datetime.utcnow()

            # Store results
            await self._store_results(session, batch)

        logger.info(
            f"Batch {calculation_id} complete: "
            f"{batch.events_succeeded}/{batch.events_processed} succeeded"
        )

        return batch

    async def _fetch_indicator_data(
        self,
        event: RiskEvent,
        weights: List[IndicatorWeight]
    ) -> Dict[str, Any]:
        """
        Fetch indicator data from data sources.

        In production, this would call the actual API clients.
        For now, returns simulated data based on weights.
        """
        indicator_data = {}

        for weight in weights:
            # Simulate data based on indicator configuration
            # In production, this would call the appropriate data client
            indicator_data[weight.indicator_name] = {
                'current': 0.0,  # Would be actual current value
                'history': [0.0] * 52  # Would be actual historical values
            }

        return indicator_data

    async def _store_results(
        self,
        session,
        batch: CalculationBatch
    ):
        """Store calculation results in database."""
        # Create calculation log
        calc_log = CalculationLog(
            calculation_id=batch.calculation_id,
            start_time=batch.start_time,
            end_time=batch.end_time,
            duration_seconds=int((batch.end_time - batch.start_time).total_seconds())
            if batch.end_time else None,
            events_processed=batch.events_processed,
            events_succeeded=batch.events_succeeded,
            events_failed=batch.events_failed,
            errors=batch.errors if batch.errors else None,
            status="COMPLETED" if batch.events_failed == 0 else "COMPLETED_WITH_ERRORS"
        )
        session.add(calc_log)

        # Store individual probability results
        for result in batch.results:
            prob_record = RiskProbability(
                event_id=result.event_id,
                calculation_id=batch.calculation_id,
                calculation_date=batch.start_time,
                probability_pct=result.probability_pct,
                log_odds=result.log_odds,
                baseline_probability_pct=result.baseline_probability_pct,
                ci_lower_pct=result.ci_lower_pct,
                ci_upper_pct=result.ci_upper_pct,
                ci_level=self.settings.confidence_level,
                ci_width_pct=result.ci_upper_pct - result.ci_lower_pct,
                precision_band=result.precision_band,
                bootstrap_iterations=self.settings.bootstrap_iterations,
                confidence_score=result.confidence_score,
                methodology_tier=result.methodology_tier,
                change_direction=result.change_direction,
                flags=result.flags if result.flags else None,
                attribution=result.attribution if result.attribution else None,
                calculation_duration_ms=result.calculation_duration_ms
            )
            session.add(prob_record)

        session.commit()
        logger.info(f"Stored {len(batch.results)} probability records")


async def run_weekly_calculation():
    """Entry point for scheduled weekly calculations."""
    orchestrator = ProbabilityOrchestrator()
    batch = await orchestrator.run_batch_calculation()
    return batch
