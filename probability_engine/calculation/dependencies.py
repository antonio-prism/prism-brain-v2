"""
Causal Dependency Adjustment Module

Handles the cascading effects when one risk event affects another.
Implements the dependency network logic for probability adjustments.
"""

import math
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from database.models import RelationshipType, ConfidenceLevel


# Multiplier caps by relationship type
MULTIPLIER_CAPS = {
    "CAUSAL": 3.0,
    "ENABLING": 2.0,
    "CORRELATED": 1.5,
    "WEAK": 1.2
}

# Confidence discounts
CONFIDENCE_DISCOUNTS = {
    "HIGH": 1.0,
    "MEDIUM": 0.7,
    "LOW": 0.4
}


@dataclass
class DependencyAdjustment:
    """Result of dependency adjustment calculation."""
    log_odds_adjustment: float
    applied_dependencies: List[Dict]
    total_multiplier: float
    capped: bool


@dataclass
class DependencyRelation:
    """Represents a single dependency relationship."""
    driver_id: str
    dependent_id: str
    relationship_type: str
    multiplier: float
    confidence: str
    bidirectional: bool
    rationale: Optional[str] = None


class DependencyNetwork:
    """
    Manages the causal dependency network between risk events.

    The network is a directed graph where:
    - Nodes are risk events
    - Edges represent causal/enabling/correlated relationships
    - Edge weights are multipliers (how much driver affects dependent)
    """

    def __init__(self):
        """Initialize empty dependency network."""
        # Forward map: driver_id -> list of (dependent_id, relation)
        self.forward_deps: Dict[str, List[DependencyRelation]] = defaultdict(list)

        # Reverse map: dependent_id -> list of (driver_id, relation)
        self.reverse_deps: Dict[str, List[DependencyRelation]] = defaultdict(list)

        # All relationships
        self.all_relations: List[DependencyRelation] = []

    def add_dependency(
        self,
        driver_id: str,
        dependent_id: str,
        relationship_type: str,
        multiplier: float,
        confidence: str,
        bidirectional: bool = False,
        rationale: Optional[str] = None
    ):
        """Add a dependency relationship to the network."""
        # Cap multiplier by relationship type
        max_mult = MULTIPLIER_CAPS.get(relationship_type, 1.5)
        multiplier = min(multiplier, max_mult)

        relation = DependencyRelation(
            driver_id=driver_id,
            dependent_id=dependent_id,
            relationship_type=relationship_type,
            multiplier=multiplier,
            confidence=confidence,
            bidirectional=bidirectional,
            rationale=rationale
        )

        self.forward_deps[driver_id].append(relation)
        self.reverse_deps[dependent_id].append(relation)
        self.all_relations.append(relation)

        # Handle bidirectional relationships
        if bidirectional:
            reverse_relation = DependencyRelation(
                driver_id=dependent_id,
                dependent_id=driver_id,
                relationship_type=relationship_type,
                multiplier=multiplier,
                confidence=confidence,
                bidirectional=True,
                rationale=rationale
            )
            self.forward_deps[dependent_id].append(reverse_relation)
            self.reverse_deps[driver_id].append(reverse_relation)

    def get_drivers(self, event_id: str) -> List[DependencyRelation]:
        """Get all events that drive (affect) this event."""
        return self.reverse_deps.get(event_id, [])

    def get_dependents(self, event_id: str) -> List[DependencyRelation]:
        """Get all events affected by this event."""
        return self.forward_deps.get(event_id, [])

    def get_all_upstream(
        self,
        event_id: str,
        max_depth: int = 3,
        visited: Optional[Set[str]] = None
    ) -> List[Tuple[str, int]]:
        """
        Get all upstream events (recursive drivers) with their depth.

        Args:
            event_id: Target event
            max_depth: Maximum recursion depth
            visited: Set of already visited events (cycle prevention)

        Returns:
            List of (event_id, depth) tuples
        """
        if visited is None:
            visited = set()

        if event_id in visited or max_depth <= 0:
            return []

        visited.add(event_id)
        result = []

        for relation in self.get_drivers(event_id):
            driver_id = relation.driver_id
            result.append((driver_id, 1))

            # Recurse
            upstream = self.get_all_upstream(
                driver_id,
                max_depth - 1,
                visited.copy()
            )
            for up_id, up_depth in upstream:
                result.append((up_id, up_depth + 1))

        return result


class DependencyAdjuster:
    """
    Calculates probability adjustments based on dependency network.

    When a driver event has elevated probability, dependent events
    should also see increased probability (and vice versa).
    """

    def __init__(self, network: DependencyNetwork):
        """
        Initialize with dependency network.

        Args:
            network: The dependency network to use
        """
        self.network = network

    def calculate_adjustment(
        self,
        event_id: str,
        driver_probabilities: Dict[str, float],
        driver_baselines: Dict[str, float],
        max_total_adjustment: float = 2.0
    ) -> DependencyAdjustment:
        """
        Calculate log-odds adjustment from driver events.

        The adjustment is based on:
        1. How much each driver's probability exceeds its baseline
        2. The multiplier and confidence of the relationship
        3. Dampening to prevent runaway cascades

        Args:
            event_id: The event being calculated
            driver_probabilities: Current probabilities of all drivers
            driver_baselines: Baseline probabilities of all drivers
            max_total_adjustment: Cap on total log-odds adjustment

        Returns:
            DependencyAdjustment with details
        """
        drivers = self.network.get_drivers(event_id)

        if not drivers:
            return DependencyAdjustment(
                log_odds_adjustment=0.0,
                applied_dependencies=[],
                total_multiplier=1.0,
                capped=False
            )

        total_adjustment = 0.0
        applied = []
        total_multiplier = 1.0

        for relation in drivers:
            driver_id = relation.driver_id

            # Skip if we don't have probability for this driver
            if driver_id not in driver_probabilities:
                continue

            current_prob = driver_probabilities[driver_id]
            baseline_prob = driver_baselines.get(driver_id, current_prob)

            # Calculate deviation from baseline
            # Positive if current > baseline (elevated risk)
            prob_ratio = current_prob / max(baseline_prob, 0.001)

            # Convert to log scale
            if prob_ratio > 1:
                # Elevated risk - positive adjustment
                log_ratio = math.log(prob_ratio)
            else:
                # Reduced risk - negative adjustment (but dampened)
                log_ratio = math.log(prob_ratio) * 0.5

            # Apply multiplier and confidence discount
            confidence_discount = CONFIDENCE_DISCOUNTS.get(relation.confidence, 0.5)
            adjustment = log_ratio * relation.multiplier * confidence_discount

            # Dampening factor based on relationship type
            if relation.relationship_type == "CAUSAL":
                dampening = 0.8
            elif relation.relationship_type == "ENABLING":
                dampening = 0.6
            elif relation.relationship_type == "CORRELATED":
                dampening = 0.4
            else:  # WEAK
                dampening = 0.2

            final_adjustment = adjustment * dampening

            total_adjustment += final_adjustment

            # Track for reporting
            applied.append({
                "driver_id": driver_id,
                "relationship_type": relation.relationship_type,
                "multiplier": relation.multiplier,
                "confidence": relation.confidence,
                "driver_prob": current_prob,
                "driver_baseline": baseline_prob,
                "adjustment": final_adjustment
            })

            # Track multiplicative effect
            if final_adjustment > 0:
                total_multiplier *= (1 + final_adjustment * 0.5)

        # Cap total adjustment
        capped = False
        if abs(total_adjustment) > max_total_adjustment:
            total_adjustment = max_total_adjustment * (1 if total_adjustment > 0 else -1)
            capped = True

        return DependencyAdjustment(
            log_odds_adjustment=total_adjustment,
            applied_dependencies=applied,
            total_multiplier=total_multiplier,
            capped=capped
        )

    def propagate_changes(
        self,
        changed_events: Dict[str, Tuple[float, float]],
        all_probabilities: Dict[str, float],
        all_baselines: Dict[str, float],
        max_iterations: int = 3
    ) -> Dict[str, float]:
        """
        Propagate probability changes through the network.

        When an event's probability changes significantly, this
        propagates the effect to dependent events.

        Args:
            changed_events: Dict of event_id -> (old_prob, new_prob)
            all_probabilities: Current probabilities
            all_baselines: Baseline probabilities
            max_iterations: Maximum propagation iterations

        Returns:
            Dict of event_id -> adjustment for affected events
        """
        adjustments = {}

        # Find all events affected by the changes
        affected_events: Set[str] = set()
        for event_id in changed_events:
            for relation in self.network.get_dependents(event_id):
                affected_events.add(relation.dependent_id)

        # Calculate adjustments for affected events
        for event_id in affected_events:
            if event_id in changed_events:
                continue  # Don't adjust events that were just calculated

            adjustment = self.calculate_adjustment(
                event_id,
                all_probabilities,
                all_baselines
            )

            if abs(adjustment.log_odds_adjustment) > 0.01:
                adjustments[event_id] = adjustment.log_odds_adjustment

        return adjustments


def load_dependencies_from_db(session) -> DependencyNetwork:
    """
    Load dependency network from database.

    Args:
        session: SQLAlchemy session

    Returns:
        Populated DependencyNetwork
    """
    from database.models import CausalDependency

    network = DependencyNetwork()

    dependencies = session.query(CausalDependency).all()

    for dep in dependencies:
        network.add_dependency(
            driver_id=dep.driver_event_id,
            dependent_id=dep.dependent_event_id,
            relationship_type=dep.relationship_type,
            multiplier=dep.multiplier,
            confidence=dep.confidence,
            bidirectional=dep.bidirectional,
            rationale=dep.rationale
        )

    return network
