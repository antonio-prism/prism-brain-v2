"""
PRISM Engine — Prior derivation methods.

Three methods for computing base probability (prior):
  Method A: Frequency count from historical databases
  Method B: Incidence rate from industry surveys × dark figure
  Method C: Structural calibration (preconditions × trigger × implementation)
"""

import logging

logger = logging.getLogger(__name__)

# DBIR-derived constants (update annually from Type B manual entry)
DBIR_BASE_BREACH_RATE = 0.18  # 18% annual probability of any breach type

DBIR_ATTACK_SHARES = {
    "ransomware": 0.44,
    "social_engineering": 0.25,
    "credential_theft": 0.38,
    "third_party": 0.30,
    "web_app_exploit": 0.26,
    "insider_misuse": 0.08,
}

DBIR_EVENT_MAPPING = {
    # Ransomware, Data Breaches & Exfiltration (DIG-RDE)
    "DIG-RDE-001": {"share": "ransomware", "subsplit": 0.50, "subsplit_source": "ESTIMATE"},
    "DIG-RDE-002": {"share": "ransomware", "subsplit": 0.30, "subsplit_source": "ESTIMATE"},
    "DIG-RDE-003": {"share": "ransomware", "subsplit": 0.62, "subsplit_source": "DBIR_2025_double_extortion"},
    "DIG-RDE-004": {"share": "credential_theft", "subsplit": 0.40, "subsplit_source": "ESTIMATE"},
    "DIG-RDE-005": {"share": "insider_misuse", "subsplit": 0.35, "subsplit_source": "ESTIMATE"},
    "DIG-RDE-006": {"share": "web_app_exploit", "subsplit": 0.25, "subsplit_source": "ESTIMATE"},
    "DIG-RDE-007": {"share": "credential_theft", "subsplit": 1.0, "subsplit_source": "DBIR_2025"},
    "DIG-RDE-008": {"share": "third_party", "subsplit": 0.30, "subsplit_source": "ESTIMATE"},
    # Fraud, Social Engineering & Denial-of-Service (DIG-FSD)
    "DIG-FSD-001": {"share": "social_engineering", "subsplit": 0.40, "subsplit_source": "ESTIMATE"},
    "DIG-FSD-002": {"share": "social_engineering", "subsplit": 0.25, "subsplit_source": "ESTIMATE"},
    "DIG-FSD-003": {"share": "credential_theft", "subsplit": 0.60, "subsplit_source": "ESTIMATE"},
    "DIG-FSD-004": {"share": "social_engineering", "subsplit": 0.05, "subsplit_source": "ESTIMATE_deepfake_subset"},
    "DIG-FSD-005": {"share": None, "fixed_rate": 0.12, "subsplit_source": "Netscout_data"},
    "DIG-FSD-006": {"share": None, "fixed_rate": 0.08, "subsplit_source": "Netscout_app_layer"},
    "DIG-FSD-007": {"share": None, "fixed_rate": 0.03, "subsplit_source": "ESTIMATE_dns_bgp"},
    # Supply Chain Cyberattacks (DIG-SCC)
    "DIG-SCC-001": {"share": "third_party", "subsplit": 0.35, "subsplit_source": "ENISA_ETL_2024"},
    "DIG-SCC-002": {"share": "third_party", "subsplit": 0.25, "subsplit_source": "ESTIMATE_MSP"},
    "DIG-SCC-003": {"share": "third_party", "subsplit": 0.40, "subsplit_source": "ESTIMATE_opensource"},
    "DIG-SCC-004": {"share": "third_party", "subsplit": 0.20, "subsplit_source": "ESTIMATE_cloud"},
    "DIG-SCC-005": {"share": "third_party", "subsplit": 0.15, "subsplit_source": "ESTIMATE_saas"},
    # Critical Infrastructure Cyberattacks (DIG-CIC) — not DIG-CIC-002 (uses Dragos)
    "DIG-CIC-001": {"share": "ransomware", "subsplit": 0.15, "subsplit_source": "ENISA_healthcare"},
    "DIG-CIC-003": {"share": "web_app_exploit", "subsplit": 0.10, "subsplit_source": "ESTIMATE_energy_OT"},
    "DIG-CIC-004": {"share": "web_app_exploit", "subsplit": 0.08, "subsplit_source": "ESTIMATE_water_OT"},
    "DIG-CIC-005": {"share": "web_app_exploit", "subsplit": 0.06, "subsplit_source": "ESTIMATE_transport_OT"},
    "DIG-CIC-006": {"share": "web_app_exploit", "subsplit": 0.03, "subsplit_source": "ESTIMATE_nuclear_OT"},
}

# Dark figure multiplier table (with empirical sources)
DARK_FIGURES = {
    "ransomware_enterprise": {
        "multiplier": 1.0,
        "source": "DBIR includes forensic firm + insurance data. No adjustment needed.",
    },
    "bec_wire_fraud": {
        "multiplier": 1.5,
        "source": "FBI IC3: enterprise BEC reporting rate ~67%. Multiplier = 1/0.67 ≈ 1.5",
    },
    "ics_ot_compromise": {
        "multiplier": 3.0,
        "source": "Dragos 2026 YiR: 'persistent mischaracterization' of OT incidents as IT. Conservative 3×.",
    },
    "supply_chain_software": {
        "multiplier": 2.0,
        "source": "ENISA ETL 2024: supply chain attacks difficult to attribute. Conservative 2×.",
    },
    "general_data_breaches": {
        "multiplier": 1.0,
        "source": "DBIR + IC3 + HIBP comprehensive coverage. No multiplier.",
    },
    "ddos": {
        "multiplier": 1.0,
        "source": "Netscout/Akamai telemetry near-complete. No multiplier.",
    },
}


def method_a_prior(event_years: int, total_years: int) -> dict:
    """
    Method A: Frequency count.
    Used for discrete events logged in structured databases (EM-DAT, USGS, ACLED).

    prior = event_years / total_years
    """
    if total_years <= 0:
        return {
            "prior": 0.0,
            "method": "A",
            "formula": "0 / 0 (no data)",
            "confidence": "Low",
            "error": "total_years is 0",
        }

    prior = round(event_years / total_years, 4)
    confidence = "High" if total_years >= 20 and event_years >= 5 else "Medium"

    return {
        "prior": prior,
        "method": "A",
        "formula": f"{event_years} event-years / {total_years} total years",
        "confidence": confidence,
        "event_years": event_years,
        "total_years": total_years,
    }


def method_b_prior(event_id: str, incident_rate: float | None = None,
                   dark_figure: float | None = None) -> dict:
    """
    Method B: Incidence rate.
    Used for per-organization risk from surveys (DBIR, IC3, Dragos).

    prior = incident_rate × dark_figure (capped at 0.95)
    """
    if incident_rate is None:
        # Derive from DBIR mapping
        mapping = DBIR_EVENT_MAPPING.get(event_id)
        if mapping is None:
            return {
                "prior": 0.05,
                "method": "B",
                "formula": "DEFAULT — no DBIR mapping",
                "confidence": "Low",
                "data_status": "NO_MAPPING",
            }
        if mapping.get("fixed_rate") is not None:
            incident_rate = mapping["fixed_rate"]
            formula_detail = f"fixed rate {incident_rate} from {mapping['subsplit_source']}"
        else:
            share_key = mapping["share"]
            share_value = DBIR_ATTACK_SHARES[share_key]
            subsplit = mapping["subsplit"]
            incident_rate = DBIR_BASE_BREACH_RATE * share_value * subsplit
            formula_detail = (
                f"{DBIR_BASE_BREACH_RATE} base × {share_value} {share_key} share "
                f"× {subsplit} subsplit = {incident_rate:.4f}"
            )
    else:
        formula_detail = f"provided rate: {incident_rate}"

    if dark_figure is None:
        dark_figure = 1.0

    prior = min(0.95, incident_rate * dark_figure)
    prior = round(prior, 4)

    return {
        "prior": prior,
        "method": "B",
        "formula": f"{incident_rate:.4f} × {dark_figure} dark figure = {prior}",
        "formula_detail": formula_detail,
        "confidence": "Medium",
        "incident_rate": round(incident_rate, 4),
        "dark_figure": dark_figure,
        "subsplit_source": DBIR_EVENT_MAPPING.get(event_id, {}).get("subsplit_source", "N/A"),
    }


def method_b_dragos_prior(incidents: int, sector_pct: float,
                          total_orgs: int, dark_figure: float) -> dict:
    """
    Method B variant for Dragos ICS/OT data.
    prior = (incidents × sector_pct / total_orgs) × dark_figure
    """
    if total_orgs <= 0:
        return {
            "prior": 0.05,
            "method": "B",
            "formula": "DEFAULT — no org count",
            "confidence": "Low",
        }

    rate = (incidents * sector_pct) / total_orgs
    prior = min(0.95, rate * dark_figure)
    prior = round(prior, 4)

    return {
        "prior": prior,
        "method": "B",
        "formula": f"({incidents} × {sector_pct} / {total_orgs}) × {dark_figure} = {prior}",
        "confidence": "Medium",
        "incident_rate": round(rate, 6),
        "dark_figure": dark_figure,
        "dark_figure_source": DARK_FIGURES.get("ics_ot_compromise", {}).get("source", ""),
    }


def method_c_prior(p_preconditions: float, p_trigger: float,
                   p_implementation: float, evidence: dict) -> dict:
    """
    Method C: Structural calibration.
    Used for policy/regulatory/market outcomes without long frequency histories.

    prior = P(preconditions) × P(trigger) × P(implementation)
    """
    for name, value in [("p_preconditions", p_preconditions),
                        ("p_trigger", p_trigger),
                        ("p_implementation", p_implementation)]:
        if not 0.0 <= value <= 1.0:
            logger.warning(f"Method C {name} = {value} is out of [0,1] range, clipping")

    p_pre = max(0.0, min(1.0, p_preconditions))
    p_trig = max(0.0, min(1.0, p_trigger))
    p_impl = max(0.0, min(1.0, p_implementation))

    prior = round(p_pre * p_trig * p_impl, 4)

    return {
        "prior": prior,
        "method": "C",
        "formula": f"{p_pre} × {p_trig} × {p_impl} = {prior}",
        "confidence": "Low",
        "sub_probabilities": {
            "p_preconditions": {
                "value": p_pre,
                "evidence": evidence.get("p_preconditions", "DEFAULT_0.50_NO_DATA"),
            },
            "p_trigger": {
                "value": p_trig,
                "evidence": evidence.get("p_trigger", "DEFAULT_0.50_NO_DATA"),
            },
            "p_implementation": {
                "value": p_impl,
                "evidence": evidence.get("p_implementation", "DEFAULT_0.50_NO_DATA"),
            },
        },
    }
