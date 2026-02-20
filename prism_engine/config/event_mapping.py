"""
PRISM Engine — Event → Source → Method mapping.

Phase 1: 10 prototype events with full hand-crafted computation specs.
Phase 2: Auto-loads all 174 events from seed files with method/source
assignment based on available connector mappings.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SEED_DIR = Path(__file__).parent.parent.parent / "frontend" / "data" / "seeds"

# ── Known connector coverage ─────────────────────────────────────────────
# Events that have EM-DAT type patterns (handled in emdat.py EMDAT_EVENT_MAPPINGS)
EMDAT_EVENTS = {
    "PHY-CLI-001", "PHY-CLI-002", "PHY-CLI-003", "PHY-CLI-004",
    "PHY-CLI-005", "PHY-CLI-006", "PHY-GEO-001", "PHY-GEO-002",
    "PHY-GEO-003", "PHY-GEO-004", "PHY-GEO-005", "PHY-POL-001",
    "PHY-POL-002", "PHY-POL-004",
    "PHY-WAT-001", "PHY-WAT-006",
    "PHY-BIO-001", "PHY-BIO-002", "PHY-BIO-004",
    "PHY-ENE-001",
    "OPS-MAR-006", "OPS-SUP-002", "OPS-MFG-002", "OPS-MFG-005",
    "OPS-WHS-001",
}

# Events that have DBIR decomposition (handled in priors.py DBIR_EVENT_MAPPING)
DBIR_EVENTS = {
    "DIG-RDE-001", "DIG-RDE-002", "DIG-RDE-003", "DIG-RDE-004",
    "DIG-RDE-005", "DIG-RDE-006", "DIG-RDE-007", "DIG-RDE-008",
    "DIG-FSD-001", "DIG-FSD-002", "DIG-FSD-003", "DIG-FSD-004",
    "DIG-FSD-005", "DIG-FSD-006", "DIG-FSD-007",
    "DIG-SCC-001", "DIG-SCC-002", "DIG-SCC-003", "DIG-SCC-004",
    "DIG-SCC-005",
    "DIG-CIC-001", "DIG-CIC-003", "DIG-CIC-004", "DIG-CIC-005",
    "DIG-CIC-006",
}

# Events that use FRED threshold counting
FRED_THRESHOLD_EVENTS = {
    "STR-ECO-002": {"series": "BAMLH0A0HYM2", "threshold": 600, "comparison": "above",
                     "label": "HY credit spread > 600bp"},
    "STR-ECO-005": {"series": "CPIAUCSL", "threshold": 10.0, "comparison": "yoy_above",
                     "label": "CPI YoY > 10%"},
    "STR-ECO-006": {"series": "CSUSHPISA", "threshold": -10.0, "comparison": "yoy_below",
                     "label": "House price index YoY drop > 10%"},
    "PHY-ENE-003": {"series": "DCOILWTICO", "threshold": 100.0, "comparison": "yoy_above",
                     "label": "WTI crude oil YoY change > 100%"},
    "PHY-MAT-003": {"series": "PPIACO", "threshold": 30.0, "comparison": "yoy_above",
                     "label": "PPI all commodities YoY > 30%"},
}

# ── Method C family-level calibration ────────────────────────────────────
# Evidence-based sub-probabilities by event family prefix.
# Events not in PHASE1_EVENTS, EMDAT, DBIR, or FRED get these values.
METHOD_C_FAMILY_DEFAULTS = {
    "STR-REG": {
        "p_pre": 0.70, "p_trig": 0.50, "p_impl": 0.60,
        "evidence": {
            "p_preconditions": "EU regulatory pipeline consistently active (Green Deal, AI Act, CSRD). High probability of new requirements being proposed.",
            "p_trigger": "Legislative cycles, elections, and crises trigger new regulation. ~50% chance per year.",
            "p_implementation": "EU regulations typically enter force 12-24 months after adoption. ~60% implemented within the year.",
        },
    },
    "STR-ENP": {
        "p_pre": 0.65, "p_trig": 0.45, "p_impl": 0.55,
        "evidence": {
            "p_preconditions": "Climate targets established (Paris Agreement, Fit for 55). Policy momentum strong.",
            "p_trigger": "COP summits, extreme weather events, or elections can trigger acceleration.",
            "p_implementation": "Energy transition policies face industrial pushback. ~55% implemented on schedule.",
        },
    },
    "STR-TEC": {
        "p_pre": 0.60, "p_trig": 0.50, "p_impl": 0.50,
        "evidence": {
            "p_preconditions": "US-China tech competition ongoing. EU Digital Markets Act in force.",
            "p_trigger": "Geopolitical events or technology breakthroughs trigger new controls.",
            "p_implementation": "Export controls and tech restrictions face enforcement challenges. ~50% effective.",
        },
    },
    "STR-TRD": {
        "p_pre": 0.70, "p_trig": 0.55, "p_impl": 0.65,
        "evidence": {
            "p_preconditions": "WTO: trade-restrictive stockpile at record $2.4T. Protectionism trend persistent.",
            "p_trigger": "Elections, bilateral disputes, or economic downturns trigger new measures.",
            "p_implementation": "~65% of announced trade measures are implemented (WTO TMR data 2016-2024).",
        },
    },
    "STR-GEO": {
        "p_pre": 0.55, "p_trig": 0.40, "p_impl": 0.50,
        "evidence": {
            "p_preconditions": "Multiple active conflict zones globally. GPR index elevated since 2022.",
            "p_trigger": "Escalation events (military actions, regime changes) difficult to predict. ~40% annually.",
            "p_implementation": "Conflict outcomes depend on military dynamics. ~50% reach supply-chain-impacting severity.",
        },
    },
    "STR-ECO": {
        "p_pre": 0.50, "p_trig": 0.35, "p_impl": 0.50,
        "evidence": {
            "p_preconditions": "Global debt-to-GDP elevated. Financial imbalances present but variable.",
            "p_trigger": "Financial crises require specific triggers (bank failures, currency runs). ~35% annually.",
            "p_implementation": "When triggered, financial shocks typically propagate. ~50% reach systemic level.",
        },
    },
    "STR-FIN": {
        "p_pre": 0.40, "p_trig": 0.30, "p_impl": 0.50,
        "evidence": {
            "p_preconditions": "Financial system generally stable post-2008 reforms, but vulnerabilities exist.",
            "p_trigger": "Financial disruptions are low-probability events. ~30% chance of trigger annually.",
            "p_implementation": "When financial disruptions occur, they tend to be significant. ~50% reach impactful level.",
        },
    },
    "DIG-CLS": {
        "p_pre": 0.60, "p_trig": 0.50, "p_impl": 0.70,
        "evidence": {
            "p_preconditions": "Cloud market dominated by 3 hyperscalers. Lock-in risk structurally high.",
            "p_trigger": "Pricing changes, outages, or regulatory requirements trigger sovereignty concerns.",
            "p_implementation": "Cloud dependency is difficult to reverse. ~70% of triggers lead to realized impact.",
        },
    },
    "DIG-HWS": {
        "p_pre": 0.55, "p_trig": 0.45, "p_impl": 0.60,
        "evidence": {
            "p_preconditions": "Semiconductor supply geographically concentrated (TSMC ~60% advanced). Hardware dependencies structural.",
            "p_trigger": "Geopolitical events, demand surges, or natural disasters can trigger shortages.",
            "p_implementation": "Hardware shortages persist due to long manufacturing lead times. ~60% last >3 months.",
        },
    },
    "DIG-SWS": {
        "p_pre": 0.50, "p_trig": 0.45, "p_impl": 0.55,
        "evidence": {
            "p_preconditions": "Software dependencies on proprietary platforms and open-source maintainers.",
            "p_trigger": "License changes, maintainer burnout, or regulatory actions can trigger issues.",
            "p_implementation": "Software migration is possible but costly. ~55% of triggers lead to material impact.",
        },
    },
    "PHY-ENE": {
        "p_pre": 0.55, "p_trig": 0.45, "p_impl": 0.55,
        "evidence": {
            "p_preconditions": "Aging grid infrastructure in EU. Increasing renewable intermittency.",
            "p_trigger": "Extreme weather, demand peaks, or equipment failure can trigger disruptions.",
            "p_implementation": "Grid disruptions typically resolved in hours-days. ~55% exceed 12-hour threshold.",
        },
    },
    "PHY-MAT": {
        "p_pre": 0.55, "p_trig": 0.45, "p_impl": 0.55,
        "evidence": {
            "p_preconditions": "Commodity markets subject to geopolitical and supply chain concentration risks.",
            "p_trigger": "Trade restrictions, natural disasters, or demand shocks trigger price spikes.",
            "p_implementation": "Price volatility often temporary. ~55% of spikes persist >3 months.",
        },
    },
    "PHY-WAT": {
        "p_pre": 0.50, "p_trig": 0.45, "p_impl": 0.55,
        "evidence": {
            "p_preconditions": "Water stress increasing in southern EU. WEI+ indicators rising.",
            "p_trigger": "Droughts, contamination events, or infrastructure failures.",
            "p_implementation": "Water disruptions typically manageable. ~55% reach industrial-impact threshold.",
        },
    },
    "PHY-POL": {
        "p_pre": 0.45, "p_trig": 0.40, "p_impl": 0.50,
        "evidence": {
            "p_preconditions": "Environmental regulations tightening. Industrial pollution risk ongoing.",
            "p_trigger": "Accidents, inspections, or extreme weather reveal contamination.",
            "p_implementation": "Pollution events often localized. ~50% reach reportable-incident threshold.",
        },
    },
    "PHY-BIO": {
        "p_pre": 0.50, "p_trig": 0.40, "p_impl": 0.50,
        "evidence": {
            "p_preconditions": "Zoonotic spillover risk increasing (deforestation, wildlife trade, AMR).",
            "p_trigger": "Novel pathogen emergence or antimicrobial resistance events.",
            "p_implementation": "Outbreak containment varies. ~50% reach supply-chain-impacting severity.",
        },
    },
    "OPS-MAR": {
        "p_pre": 0.55, "p_trig": 0.45, "p_impl": 0.55,
        "evidence": {
            "p_preconditions": "Maritime trade concentrated through key chokepoints. Port congestion structural.",
            "p_trigger": "Weather, strikes, geopolitical events, or vessel incidents.",
            "p_implementation": "Maritime disruptions cascade through supply chains. ~55% exceed 2-week threshold.",
        },
    },
    "OPS-AIR": {
        "p_pre": 0.50, "p_trig": 0.45, "p_impl": 0.55,
        "evidence": {
            "p_preconditions": "Air freight capacity tight post-COVID. Fuel price volatility structural.",
            "p_trigger": "Weather closures, strikes, airline failures, or fuel spikes.",
            "p_implementation": "Air disruptions resolve faster than maritime. ~55% exceed 24-hour threshold.",
        },
    },
    "OPS-RLD": {
        "p_pre": 0.55, "p_trig": 0.50, "p_impl": 0.55,
        "evidence": {
            "p_preconditions": "Driver shortages structural in EU. Infrastructure aging.",
            "p_trigger": "Strikes, weather, border closures, or fuel shortages.",
            "p_implementation": "Road/rail disruptions moderately impactful. ~55% exceed 48-hour threshold.",
        },
    },
    "OPS-CMP": {
        "p_pre": 0.55, "p_trig": 0.45, "p_impl": 0.55,
        "evidence": {
            "p_preconditions": "Component supply chains geographically concentrated. Single-source risks.",
            "p_trigger": "Demand surges, geopolitical events, or natural disasters.",
            "p_implementation": "Component shortages cascade through manufacturing. ~55% last >3 months.",
        },
    },
    "OPS-SUP": {
        "p_pre": 0.50, "p_trig": 0.40, "p_impl": 0.55,
        "evidence": {
            "p_preconditions": "Supplier concentration varies by industry. Single-source dependencies common.",
            "p_trigger": "Financial stress, quality failures, or capacity constraints.",
            "p_implementation": "Supplier disruptions require qualification of alternatives. ~55% exceed 1-month recovery.",
        },
    },
    "OPS-MFG": {
        "p_pre": 0.55, "p_trig": 0.45, "p_impl": 0.55,
        "evidence": {
            "p_preconditions": "Manufacturing operations subject to equipment aging and process complexity.",
            "p_trigger": "Equipment failures, fires, or quality deviations.",
            "p_implementation": "Production disruptions depend on spare parts availability. ~55% exceed 48-hour threshold.",
        },
    },
    "OPS-WHS": {
        "p_pre": 0.45, "p_trig": 0.40, "p_impl": 0.50,
        "evidence": {
            "p_preconditions": "Warehouse operations increasingly automated. Single-point-of-failure risk.",
            "p_trigger": "Fires, system failures, theft, or capacity constraints.",
            "p_implementation": "Warehouse disruptions vary by severity. ~50% exceed material-loss threshold.",
        },
    },
}

# ── Default modifier sources by domain/family prefix ─────────────────────
DEFAULT_MODIFIER_SOURCES = {
    "PHY-CLI": ["A01"],        # ERA5 temperature anomaly
    "PHY-GEO": ["A02"],        # USGS seismicity
    "PHY-ENE": ["A03"],        # FRED AMTMNO manufacturing orders (demand pressure)
    "PHY-MAT": ["A03"],        # FRED commodity prices
    "PHY-WAT": ["A01"],        # ERA5 precipitation
    "PHY-POL": [],             # No live modifier
    "PHY-BIO": [],             # Only PHY-BIO-001 (Phase 1) has PHEIC categorical
    "DIG-CIC": ["A04", "A05"], # NVD + CISA KEV
    "DIG-RDE": ["A05"],        # CISA KEV
    "DIG-SCC": ["A05"],        # CISA KEV
    "DIG-FSD": ["A05"],        # CISA KEV
    "DIG-CLS": [],             # No live modifier
    "DIG-HWS": ["A03"],        # FRED durable goods
    "DIG-SWS": ["A05"],        # CISA KEV
    "STR-GEO": ["A06"],        # GPR Index
    "STR-TRD": ["A06"],        # GPR Index
    "STR-REG": [],             # No live modifier
    "STR-ECO": ["A03"],        # FRED yield curve
    "STR-ENP": [],             # No live modifier
    "STR-TEC": [],             # No live modifier
    "STR-FIN": ["A03"],        # FRED credit spread
    "OPS-MAR": ["A06"],        # GPR Index
    "OPS-AIR": [],             # No live modifier
    "OPS-RLD": [],             # No live modifier
    "OPS-CMP": ["A03"],        # FRED durable goods
    "OPS-SUP": [],             # No live modifier
    "OPS-MFG": [],             # No live modifier
    "OPS-WHS": [],             # No live modifier
}


# ── The 10 Phase 1 prototype events (hand-crafted, detailed configs) ─────
PHASE1_EVENTS = {
    "PHY-CLI-003": {
        "name": "Extreme heat wave affecting production/logistics",
        "domain": "Physical",
        "family": "Climate Extremes & Weather Events",
        "method": "A",
        "prior_source": "emdat",
        "primary_sources": ["A01", "A08"],
        "modifier_sources": ["A01"],
        "observation_window": "2000-2024",
        "observation_years": 25,
        "emdat_type": "Heat wave",
        "emdat_countries": "EEA_EXTENDED",
        "description": "Count years with at least 1 heatwave event in EU region (ERA5 anomaly >2σ AND EM-DAT logged event)",
    },
    "PHY-GEO-001": {
        "name": "Major earthquake disrupting operations/supply chain",
        "domain": "Physical",
        "family": "Geophysical Disasters",
        "method": "A",
        "prior_source": "usgs",
        "primary_sources": ["A02", "A08"],
        "modifier_sources": ["A02"],
        "observation_window": "2000-2024",
        "observation_years": 25,
        "usgs_min_magnitude": 6.0,
        "region": "ALL_SEISMIC",
        "description": "P(at least 1 M6.0+ earthquake in any major seismic zone) = 1 - ∏(1 - P_zone_i)",
    },
    "STR-GEO-001": {
        "name": "Armed conflict erupting in key supplier country",
        "domain": "Structural",
        "family": "Geopolitical Conflict & Instability",
        "method": "A",
        "prior_source": "ucdp",
        "primary_sources": ["A07"],
        "modifier_sources": ["A06"],
        "observation_window": "2000-2024",
        "observation_years": 25,
        "description": "Count years with at least 1 armed conflict in any TOP20 supplier country (UCDP, 25yr window)",
    },
    "STR-TRD-001": {
        "name": "Major tariff increases >25% on key inputs",
        "domain": "Structural",
        "family": "Trade & Economic Policy Shifts",
        "method": "C",
        "prior_source": "manual_c",
        "primary_sources": ["B08"],
        "modifier_sources": ["A06", "categorical"],
        "observation_window": "2000-2024",
        "observation_years": 25,
        "method_c_components": {
            "p_preconditions": {
                "description": "Trade tensions exist (WTO stockpile rising, GPR trade component elevated)",
                "evidence_type": "historical_policy_frequency",
            },
            "p_trigger": {
                "description": "Election year or geopolitical escalation triggers new measures",
                "evidence_type": "election_cycle",
            },
            "p_implementation": {
                "description": "Announced tariffs are actually implemented within 12 months",
                "evidence_type": "historical_policy_frequency",
            },
        },
        "description": "Structural calibration: P(preconditions) × P(trigger) × P(implementation)",
    },
    "DIG-RDE-001": {
        "name": "ERP system encrypted by ransomware",
        "domain": "Digital",
        "family": "Ransomware, Data Breaches & Exfiltration",
        "method": "B",
        "prior_source": "dbir",
        "primary_sources": ["B01"],
        "modifier_sources": ["A05"],
        "dbir_share": "ransomware",
        "dbir_subsplit": 0.50,
        "dark_figure": 1.0,
        "description": "DBIR base breach rate × ransomware share × ERP subsplit × dark figure",
    },
    "DIG-CIC-002": {
        "name": "SCADA/ICS system compromised in manufacturing",
        "domain": "Digital",
        "family": "Critical Infrastructure Cyberattacks",
        "method": "B",
        "prior_source": "dragos",
        "primary_sources": ["B04", "B01"],
        "modifier_sources": ["A04", "A05"],
        "dragos_incidents_2025": 3300,
        "dragos_manufacturing_pct": 0.67,
        "dragos_total_mfg_orgs": 300000,
        "dark_figure": 3.0,
        "description": "Dragos: 3300 × 0.67 manufacturing / 300000 orgs × 3.0 dark figure",
    },
    "OPS-MAR-002": {
        "name": "Major canal or strait closure disrupting trade",
        "domain": "Operational",
        "family": "Port & Maritime Logistics",
        "method": "A",
        "prior_source": "manual_events",
        "primary_sources": ["A08"],
        "modifier_sources": ["A06"],
        "observation_window": "2000-2024",
        "observation_years": 25,
        "manual_event_list": [
            {"year": 2021, "description": "Ever Given Suez Canal blockage (6 days)"},
            {"year": 2023, "description": "Panama Canal drought restrictions (6+ months)"},
            {"year": 2024, "description": "Houthi attacks on Red Sea shipping (ongoing)"},
        ],
        "description": "Count years with major chokepoint disruption from manual event list + EM-DAT",
    },
    "OPS-CMP-001": {
        "name": "Critical semiconductor chip shortage",
        "domain": "Operational",
        "family": "Component & Materials Shortages",
        "method": "C",
        "prior_source": "manual_c",
        "primary_sources": ["B04"],
        "modifier_sources": ["A03"],
        "proxy": "C01",
        "fred_series": "ACDGNO",
        "method_c_components": {
            "p_preconditions": {
                "description": "Supply-demand structural imbalance exists",
                "evidence_type": "expert_survey",
            },
            "p_trigger": {
                "description": "Demand surge or supply shock triggers shortage",
                "evidence_type": "historical_policy_frequency",
            },
            "p_implementation": {
                "description": "Shortage persists for >3 months (lead time expansion)",
                "evidence_type": "historical_policy_frequency",
            },
        },
        "description": "Structural calibration with FRED durable goods proxy modifier",
    },
    "STR-ECO-001": {
        "name": "Recession in major trading partner economy",
        "domain": "Structural",
        "family": "Macroeconomic Shocks",
        "method": "A",
        "prior_source": "world_bank",
        "primary_sources": ["A03", "A09"],
        "modifier_sources": ["A03"],
        "observation_window": "2000-2024",
        "observation_years": 25,
        "fred_series": {
            "yield_curve": "T10Y2Y",
            "credit_spread": "BAMLH0A0HYM2",
            "unemployment": "UNRATE",
        },
        "description": "Count years with recession in any OECD major economy (FRED + World Bank GDP data)",
    },
    "PHY-BIO-001": {
        "name": "Major zoonotic disease outbreak affecting workforce/supply chain",
        "domain": "Physical",
        "family": "Biological & Pandemic Risks",
        "method": "A",
        "prior_source": "manual_events",
        "primary_sources": ["A11"],
        "modifier_sources": ["categorical"],
        "observation_window": "2000-2024",
        "observation_years": 25,
        "manual_event_list": [
            {"year": 2003, "description": "SARS outbreak"},
            {"year": 2009, "description": "H1N1 pandemic"},
            {"year": 2012, "description": "MERS emergence"},
            {"year": 2014, "description": "Ebola West Africa epidemic"},
            {"year": 2015, "description": "Zika virus epidemic"},
            {"year": 2019, "description": "Ebola DRC outbreak"},
            {"year": 2020, "description": "COVID-19 pandemic"},
            {"year": 2022, "description": "Mpox international outbreak"},
            {"year": 2024, "description": "H5N1 avian flu (mammalian spillover)"},
        ],
        "categorical_modifiers": [
            {
                "name": "WHO PHEIC active",
                "condition_check": "is_pheic_active",
                "if_true": 1.50,
                "justification": "During COVID PHEIC (2020-2023), supply chain disruption frequency was ~50% above historical average per Munich Re",
            }
        ],
        "description": "Count years with major zoonotic outbreak from WHO DON + manual event list",
    },
}


# ── Auto-load all 174 events from seed files ─────────────────────────────

def _get_family_prefix(event_id: str) -> str:
    """Extract family prefix like 'STR-GEO' from event_id like 'STR-GEO-001'."""
    parts = event_id.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return event_id


def _assign_method(event_id: str) -> str:
    """Determine computation method for an auto-loaded event."""
    if event_id in EMDAT_EVENTS:
        return "A"
    if event_id in DBIR_EVENTS:
        return "B"
    if event_id in FRED_THRESHOLD_EVENTS:
        return "A"
    return "C"


def _assign_prior_source(event_id: str) -> str:
    """Determine the prior data source for an auto-loaded event."""
    if event_id in EMDAT_EVENTS:
        return "emdat"
    if event_id in DBIR_EVENTS:
        return "dbir"
    if event_id in FRED_THRESHOLD_EVENTS:
        return "fred_threshold"
    return "family_defaults"


def _assign_modifier_sources(event_id: str) -> list[str]:
    """Assign modifier sources based on event family prefix."""
    prefix = _get_family_prefix(event_id)
    return DEFAULT_MODIFIER_SOURCES.get(prefix, [])


def _load_all_events() -> dict[str, dict]:
    """Load all 174 events from seed JSON files and assign computation configs."""
    all_events = {}
    seed_files = [
        "physical_domain_seed.json",
        "digital_domain_seed.json",
        "structural_domain_seed.json",
        "operational_domain_seed.json",
    ]

    for filename in seed_files:
        path = SEED_DIR / filename
        if not path.exists():
            logger.warning(f"Seed file not found: {path}")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                events = json.load(f)
            for event in events:
                eid = event.get("event_id")
                if not eid:
                    continue
                # Skip if it's a Phase 1 event (those have hand-crafted configs)
                if eid in PHASE1_EVENTS:
                    continue

                config = {
                    "name": event.get("event_name", eid),
                    "domain": event.get("domain", "Unknown").title(),
                    "family": event.get("family_name", "Unknown"),
                    "method": _assign_method(eid),
                    "prior_source": _assign_prior_source(eid),
                    "modifier_sources": _assign_modifier_sources(eid),
                    "observation_window": "2000-2024",
                    "observation_years": 25,
                    "base_rate_pct": event.get("base_rate_pct"),
                }

                # Add FRED threshold config if applicable
                if eid in FRED_THRESHOLD_EVENTS:
                    config["fred_threshold"] = FRED_THRESHOLD_EVENTS[eid]

                all_events[eid] = config
        except Exception as e:
            logger.error(f"Error loading seed file {filename}: {e}")

    logger.info(f"Auto-loaded {len(all_events)} events from seed files "
                f"(+ {len(PHASE1_EVENTS)} Phase 1 = {len(all_events) + len(PHASE1_EVENTS)} total)")
    return all_events


# Load at import time
ALL_EVENTS = _load_all_events()


def get_phase1_event_ids() -> list[str]:
    """Return the 10 Phase 1 prototype event IDs."""
    return list(PHASE1_EVENTS.keys())


def get_all_event_ids() -> list[str]:
    """Return all event IDs (Phase 1 + auto-loaded)."""
    ids = list(PHASE1_EVENTS.keys())
    ids.extend(sorted(ALL_EVENTS.keys()))
    return ids


def get_event_config(event_id: str) -> dict | None:
    """Get the computation configuration for any event.

    Priority: Phase 1 (hand-crafted) > auto-loaded from seeds.
    """
    config = PHASE1_EVENTS.get(event_id)
    if config is not None:
        return config
    return ALL_EVENTS.get(event_id)
