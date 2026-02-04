#!/usr/bin/env python3
"""
Railway Initialization Script

This script initializes the database and loads data when deployed to Railway.
Run this once after deployment to set up the database.
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import create_tables, get_session_context
from database.models import RiskEvent, IndicatorWeight, CausalDependency


def init_database():
    """Create all database tables."""
    print("Creating database tables...")
    create_tables()
    print("✓ Database tables created")


def load_sample_events():
    """Load a sample of risk events for testing."""
    print("Loading sample risk events...")

    # Sample events for testing (subset of the full 900)
    sample_events = [
        {
            "event_id": "GEO_001",
            "event_name": "Major Earthquake (Magnitude 7.0+)",
            "description": "A significant seismic event causing widespread damage",
            "layer1_primary": "Natural Catastrophes",
            "layer2_primary": "Geological Events",
            "baseline_probability": 3,
            "baseline_impact": 4,
            "super_risk": False,
            "methodology_tier": "TIER_1_ML_ENHANCED"
        },
        {
            "event_id": "CYB_001",
            "event_name": "Major Ransomware Attack on Critical Infrastructure",
            "description": "Widespread ransomware affecting essential services",
            "layer1_primary": "Cyber & Technology",
            "layer2_primary": "Cyber Attacks",
            "baseline_probability": 4,
            "baseline_impact": 4,
            "super_risk": True,
            "methodology_tier": "TIER_1_ML_ENHANCED"
        },
        {
            "event_id": "ECO_001",
            "event_name": "Global Recession",
            "description": "Worldwide economic downturn with GDP contraction",
            "layer1_primary": "Economic & Financial",
            "layer2_primary": "Macroeconomic",
            "baseline_probability": 3,
            "baseline_impact": 5,
            "super_risk": True,
            "methodology_tier": "TIER_2_ANALOG"
        },
        {
            "event_id": "POL_001",
            "event_name": "Major Geopolitical Conflict",
            "description": "Armed conflict between major powers",
            "layer1_primary": "Geopolitical",
            "layer2_primary": "International Conflict",
            "baseline_probability": 2,
            "baseline_impact": 5,
            "super_risk": True,
            "methodology_tier": "TIER_3_SCENARIO"
        },
        {
            "event_id": "CLI_001",
            "event_name": "Extreme Heat Event",
            "description": "Prolonged extreme temperatures affecting large population",
            "layer1_primary": "Climate & Environment",
            "layer2_primary": "Climate Events",
            "baseline_probability": 4,
            "baseline_impact": 3,
            "super_risk": False,
            "methodology_tier": "TIER_1_ML_ENHANCED"
        }
    ]

    with get_session_context() as session:
        # Check if events already exist
        existing = session.query(RiskEvent).count()
        if existing > 0:
            print(f"  ⚠ {existing} events already exist, skipping load")
            return

        for event_data in sample_events:
            event = RiskEvent(**event_data)
            session.add(event)

        session.commit()
        print(f"✓ Loaded {len(sample_events)} sample risk events")


def load_sample_weights():
    """Load sample indicator weights for testing."""
    print("Loading sample indicator weights...")

    sample_weights = [
        # GEO_001 - Earthquake
        {"event_id": "GEO_001", "indicator_name": "usgs_seismic_activity", "data_source": "USGS", "normalized_weight": 0.40, "beta_type": "direct_causal", "time_scale": "fast"},
        {"event_id": "GEO_001", "indicator_name": "historical_earthquake_frequency", "data_source": "NOAA", "normalized_weight": 0.35, "beta_type": "strong_correlation", "time_scale": "slow"},
        {"event_id": "GEO_001", "indicator_name": "plate_boundary_stress", "data_source": "NASA", "normalized_weight": 0.25, "beta_type": "moderate_correlation", "time_scale": "slow"},

        # CYB_001 - Ransomware
        {"event_id": "CYB_001", "indicator_name": "ransomware_attack_volume", "data_source": "OTX", "normalized_weight": 0.35, "beta_type": "direct_causal", "time_scale": "fast"},
        {"event_id": "CYB_001", "indicator_name": "vulnerability_count", "data_source": "NVD", "normalized_weight": 0.30, "beta_type": "strong_correlation", "time_scale": "fast"},
        {"event_id": "CYB_001", "indicator_name": "threat_intelligence_alerts", "data_source": "OTX", "normalized_weight": 0.35, "beta_type": "direct_causal", "time_scale": "fast"},

        # ECO_001 - Recession
        {"event_id": "ECO_001", "indicator_name": "gdp_growth_rate", "data_source": "World Bank", "normalized_weight": 0.30, "beta_type": "direct_causal", "time_scale": "slow"},
        {"event_id": "ECO_001", "indicator_name": "unemployment_rate", "data_source": "FRED", "normalized_weight": 0.25, "beta_type": "strong_correlation", "time_scale": "medium"},
        {"event_id": "ECO_001", "indicator_name": "yield_curve_inversion", "data_source": "FRED", "normalized_weight": 0.25, "beta_type": "strong_correlation", "time_scale": "medium"},
        {"event_id": "ECO_001", "indicator_name": "consumer_confidence", "data_source": "FRED", "normalized_weight": 0.20, "beta_type": "moderate_correlation", "time_scale": "medium"},

        # POL_001 - Geopolitical Conflict
        {"event_id": "POL_001", "indicator_name": "conflict_events", "data_source": "ACLED", "normalized_weight": 0.35, "beta_type": "direct_causal", "time_scale": "medium"},
        {"event_id": "POL_001", "indicator_name": "diplomatic_tone", "data_source": "GDELT", "normalized_weight": 0.30, "beta_type": "strong_correlation", "time_scale": "fast"},
        {"event_id": "POL_001", "indicator_name": "military_expenditure", "data_source": "World Bank", "normalized_weight": 0.20, "beta_type": "moderate_correlation", "time_scale": "slow"},
        {"event_id": "POL_001", "indicator_name": "sanctions_activity", "data_source": "IMF", "normalized_weight": 0.15, "beta_type": "weak_contextual", "time_scale": "medium"},

        # CLI_001 - Extreme Heat
        {"event_id": "CLI_001", "indicator_name": "temperature_anomaly", "data_source": "NOAA", "normalized_weight": 0.40, "beta_type": "direct_causal", "time_scale": "slow"},
        {"event_id": "CLI_001", "indicator_name": "heat_wave_frequency", "data_source": "NASA", "normalized_weight": 0.35, "beta_type": "strong_correlation", "time_scale": "slow"},
        {"event_id": "CLI_001", "indicator_name": "co2_concentration", "data_source": "NOAA", "normalized_weight": 0.25, "beta_type": "moderate_correlation", "time_scale": "slow"},
    ]

    with get_session_context() as session:
        existing = session.query(IndicatorWeight).count()
        if existing > 0:
            print(f"  ⚠ {existing} weights already exist, skipping load")
            return

        for weight_data in sample_weights:
            weight = IndicatorWeight(**weight_data)
            session.add(weight)

        session.commit()
        print(f"✓ Loaded {len(sample_weights)} sample indicator weights")


def load_sample_dependencies():
    """Load sample causal dependencies for testing."""
    print("Loading sample dependencies...")

    sample_deps = [
        {"driver_event_id": "ECO_001", "dependent_event_id": "POL_001", "relationship_type": "ENABLING", "multiplier": 1.5, "confidence": "MEDIUM", "rationale": "Economic stress increases geopolitical tensions"},
        {"driver_event_id": "CYB_001", "dependent_event_id": "ECO_001", "relationship_type": "CAUSAL", "multiplier": 1.8, "confidence": "HIGH", "rationale": "Major cyber attacks can trigger economic disruption"},
        {"driver_event_id": "CLI_001", "dependent_event_id": "ECO_001", "relationship_type": "CORRELATED", "multiplier": 1.3, "confidence": "MEDIUM", "rationale": "Climate events can affect economic output"},
        {"driver_event_id": "GEO_001", "dependent_event_id": "ECO_001", "relationship_type": "CAUSAL", "multiplier": 1.6, "confidence": "HIGH", "rationale": "Major earthquakes cause economic damage"},
    ]

    with get_session_context() as session:
        existing = session.query(CausalDependency).count()
        if existing > 0:
            print(f"  ⚠ {existing} dependencies already exist, skipping load")
            return

        for dep_data in sample_deps:
            dep = CausalDependency(**dep_data)
            session.add(dep)

        session.commit()
        print(f"✓ Loaded {len(sample_deps)} sample dependencies")


def main():
    """Run all initialization steps."""
    print("\n" + "="*50)
    print("PRISM Brain - Railway Initialization")
    print("="*50 + "\n")

    try:
        init_database()
        load_sample_events()
        load_sample_weights()
        load_sample_dependencies()

        print("\n" + "="*50)
        print("✓ Initialization complete!")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\n✗ Error during initialization: {e}")
        raise


if __name__ == "__main__":
    main()
