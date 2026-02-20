"""
PRISM Probability Engine — Main orchestrator.

Central entry point: compute(event_id) → full JSON output per Section 9.2.

Dispatches to the correct data connectors, computation method, and modifiers
based on the event mapping configuration.

Phase 2: Config-driven routing handles all 174 events.
Phase 1 events retain their hand-crafted prior functions.
"""

import logging
from datetime import datetime

from .config.event_mapping import (
    PHASE1_EVENTS, ALL_EVENTS, METHOD_C_FAMILY_DEFAULTS,
    get_event_config, get_all_event_ids,
)
from .computation.priors import (
    method_a_prior, method_b_prior, method_b_dragos_prior, method_c_prior,
    DBIR_EVENT_MAPPING,
)
from .computation.formulas import calculate_p_global
from .computation.validation import validate_event_output, clip_value
from .fallback import get_fallback_rate

logger = logging.getLogger(__name__)


def compute(event_id: str) -> dict:
    """
    Compute the full probability output for a single event.

    Returns a JSON-serializable dict matching the Section 9.2 schema:
    {event_id, event_name, domain, family, layer1: {prior, derivation, modifiers, p_global}, ...}

    Fallback chain: computed → cached → hardcoded base rate.
    """
    config = get_event_config(event_id)
    if config is None:
        return _fallback_result(event_id, reason="Unknown event — not in seed files")

    try:
        method = config["method"]
        if method == "A":
            result = _compute_method_a(event_id, config)
        elif method == "B":
            result = _compute_method_b(event_id, config)
        elif method == "C":
            result = _compute_method_c(event_id, config)
        else:
            return _fallback_result(event_id, reason=f"Unknown method: {method}")
    except Exception as e:
        logger.error(f"Computation failed for {event_id}: {e}", exc_info=True)
        return _fallback_result(event_id, reason=f"Computation error: {e}")

    # Validate
    errors = validate_event_output(result)
    if errors:
        logger.warning(f"Validation issues for {event_id}: {errors}")
        result["metadata"]["validation_warnings"] = errors

    return result


def compute_all() -> dict[str, dict]:
    """Compute all 174 events (Phase 1 hand-crafted + auto-loaded)."""
    results = {}
    all_ids = get_all_event_ids()
    for event_id in all_ids:
        config = get_event_config(event_id)
        name = config.get("name", event_id) if config else event_id
        logger.info(f"Computing {event_id}: {name}")
        results[event_id] = compute(event_id)
    return results


def compute_all_phase1() -> dict[str, dict]:
    """Compute only the 10 Phase 1 prototype events (backward compat)."""
    results = {}
    for event_id in PHASE1_EVENTS:
        logger.info(f"Computing {event_id}: {PHASE1_EVENTS[event_id]['name']}")
        results[event_id] = compute(event_id)
    return results


# ---------------------------------------------------------------------------
#  Method A: Frequency count
# ---------------------------------------------------------------------------

def _compute_method_a(event_id: str, config: dict) -> dict:
    """Compute using Method A (frequency count from historical databases)."""

    prior_data = _get_method_a_prior(event_id, config)
    modifiers = _get_modifiers(event_id, config)

    modifier_values = [m["modifier_value"] for m in modifiers]
    p_global_result = calculate_p_global(prior_data["prior"], modifier_values)

    return _build_output(event_id, config, prior_data, modifiers, p_global_result)


def _get_method_a_prior(event_id: str, config: dict) -> dict:
    """Get the Method A prior — config-driven routing."""

    prior_source = config.get("prior_source", "")

    # ── EM-DAT: disaster frequency counting ──────────────────────────
    if prior_source == "emdat":
        return _prior_emdat(event_id, config)

    # ── USGS: earthquake Poisson ─────────────────────────────────────
    if prior_source == "usgs":
        return _prior_earthquake(config)

    # ── ACLED: armed conflict ────────────────────────────────────────
    if prior_source == "acled":
        return _prior_armed_conflict(config)

    # ── World Bank: recession ────────────────────────────────────────
    if prior_source == "world_bank":
        return _prior_recession(config)

    # ── FRED threshold: economic indicator crossing ──────────────────
    if prior_source == "fred_threshold":
        return _prior_fred_threshold(event_id, config)

    # ── Manual event list ────────────────────────────────────────────
    if prior_source == "manual_events":
        return _prior_manual_events(event_id, config)

    # ── Fallback for unmapped Method A events ────────────────────────
    fallback = get_fallback_rate(event_id)
    return {
        "prior": fallback,
        "method": "A",
        "formula": f"Fallback rate: {fallback}",
        "confidence": "Low",
        "data_status": "FALLBACK_HARDCODED",
        "data_source": "hardcoded_seed",
        "source_id": "fallback",
    }


def _prior_emdat(event_id: str, config: dict) -> dict:
    """Method A prior from EM-DAT disaster database."""
    from .connectors.emdat import count_event_years

    start = int(config.get("observation_window", "2000-2024").split("-")[0])
    end = int(config.get("observation_window", "2000-2024").split("-")[1])
    emdat_result = count_event_years(event_id, start, end)

    if emdat_result.success:
        d = emdat_result.data
        return {
            "prior": clip_value("prior", d["prior"]),
            "method": "A",
            "formula": d["formula"],
            "data_source": "EM-DAT",
            "source_id": "A08",
            "observation_window": d["observation_window"],
            "calculation_steps": (
                f"Counted years with EM-DAT events matching '{d['type_pattern']}' "
                f"in {d['region']} region. "
                f"Result: {d['event_years']} of {d['total_years']} years qualify."
            ),
            "confidence": "High" if d["event_years"] >= 5 else "Medium",
            "n_observations": d["total_years"],
        }

    # Fallback
    fb = get_fallback_rate(event_id)
    return {
        "prior": fb, "method": "A", "formula": f"Fallback: {fb}",
        "data_source": "hardcoded_seed", "source_id": "fallback",
        "confidence": "Low", "data_status": "FALLBACK_HARDCODED",
        "error": emdat_result.error,
    }


def _prior_earthquake(config: dict) -> dict:
    """PHY-GEO-001: Major earthquake prior from USGS ComCat."""
    from .connectors.usgs import compute_earthquake_prior

    result = compute_earthquake_prior(
        start="2000-01-01", end="2024-12-31",
        min_mag=config.get("usgs_min_magnitude", 6.0),
    )

    if result.success:
        prior = result.data["combined_prior"]
        return {
            "prior": clip_value("prior", prior),
            "method": "A",
            "formula": result.data["formula"],
            "data_source": "USGS ComCat",
            "source_id": "A02",
            "observation_window": result.data["observation_window"],
            "calculation_steps": (
                f"Counted M{config.get('usgs_min_magnitude', 6.0)}+ earthquakes in "
                f"{len(result.data['zones'])} seismic zones, computed P(at least 1 in any zone) "
                f"using inclusion-exclusion."
            ),
            "confidence": "High",
            "zone_details": result.data["zones"],
        }

    # Fallback
    fb = get_fallback_rate("PHY-GEO-001")
    return {
        "prior": fb, "method": "A", "formula": f"Fallback: {fb}",
        "data_source": "hardcoded_seed", "source_id": "fallback",
        "confidence": "Low", "data_status": "FALLBACK_HARDCODED",
        "error": result.error,
    }


def _prior_armed_conflict(config: dict) -> dict:
    """STR-GEO-001: Armed conflict in supplier country from ACLED."""
    from .connectors.acled import count_conflict_years

    result = count_conflict_years(start_year=2018, end_year=2024)

    if result.success:
        d = result.data
        return {
            "prior": clip_value("prior", d["prior"]),
            "method": "A",
            "formula": d["formula"],
            "data_source": "ACLED",
            "source_id": "A07",
            "observation_window": d["observation_window"],
            "calculation_steps": (
                f"Counted years with at least 1 battle event in TOP20 supplier countries. "
                f"Result: {d['conflict_year_count']} of {d['total_years']} years."
            ),
            "confidence": "Medium",
            "n_observations": d["total_years"],
            "note": d.get("note", ""),
        }

    # Fallback
    fb = get_fallback_rate("STR-GEO-001")
    return {
        "prior": fb, "method": "A", "formula": f"Fallback: {fb}",
        "data_source": "hardcoded_seed", "source_id": "fallback",
        "confidence": "Low", "data_status": "FALLBACK_HARDCODED",
        "error": result.error,
    }


def _prior_recession(config: dict) -> dict:
    """STR-ECO-001: Recession prior from World Bank GDP + FRED yield curve."""
    from .connectors.world_bank import fetch_gdp_growth

    wb_result = fetch_gdp_growth(start_year=2000, end_year=2024)
    if wb_result.success:
        d = wb_result.data
        return {
            "prior": clip_value("prior", d["prior"]),
            "method": "A",
            "formula": d["formula"],
            "data_source": "World Bank GDP growth + FRED yield curve",
            "source_id": "A09+A03",
            "observation_window": d["observation_window"],
            "calculation_steps": (
                f"Counted years with negative GDP growth in any OECD major economy. "
                f"Result: {d['recession_count']} of {d['total_years']} years. "
                f"Recession years: {d['recession_years']}"
            ),
            "confidence": "High",
            "n_observations": d["total_years"],
        }

    # Fallback
    fb = get_fallback_rate("STR-ECO-001")
    return {
        "prior": fb, "method": "A", "formula": f"Fallback: {fb}",
        "data_source": "hardcoded_seed", "source_id": "fallback",
        "confidence": "Low", "data_status": "FALLBACK_HARDCODED",
        "error": wb_result.error,
    }


def _prior_fred_threshold(event_id: str, config: dict) -> dict:
    """Method A prior from FRED series threshold crossing."""
    from .connectors.fred import count_threshold_years

    ft = config.get("fred_threshold", {})
    if not ft:
        fb = get_fallback_rate(event_id)
        return {
            "prior": fb, "method": "A", "formula": f"Fallback: {fb}",
            "data_source": "hardcoded_seed", "source_id": "fallback",
            "confidence": "Low", "data_status": "FALLBACK_HARDCODED",
        }

    result = count_threshold_years(
        series_id=ft["series"],
        threshold=ft["threshold"],
        comparison=ft["comparison"],
        label=ft.get("label", ""),
    )

    if result.success:
        d = result.data
        return {
            "prior": clip_value("prior", d["prior"]),
            "method": "A",
            "formula": d["formula"],
            "data_source": f"FRED ({d['series_id']})",
            "source_id": "A03",
            "observation_window": d["observation_window"],
            "calculation_steps": (
                f"Counted years where {d['label']} "
                f"Result: {d['event_years']} of {d['total_years']} years. "
                f"Qualifying: {d['qualifying_years']}"
            ),
            "confidence": "High" if d["event_years"] >= 3 else "Medium",
            "n_observations": d["total_years"],
        }

    fb = get_fallback_rate(event_id)
    return {
        "prior": fb, "method": "A", "formula": f"Fallback: {fb}",
        "data_source": "hardcoded_seed", "source_id": "fallback",
        "confidence": "Low", "data_status": "FALLBACK_HARDCODED",
        "error": result.error,
    }


def _prior_manual_events(event_id: str, config: dict) -> dict:
    """Method A prior from a hand-curated event list."""
    events = config.get("manual_event_list", [])
    event_years = len(set(e["year"] for e in events)) if events else 0
    total_years = config.get("observation_years", 25)
    prior_data = method_a_prior(event_years, total_years)

    prior_data.update({
        "data_source": "Manual event compilation",
        "source_id": "manual",
        "observation_window": config.get("observation_window", "2000-2024"),
        "calculation_steps": (
            f"Counted {event_years} distinct years with qualifying events: "
            + ", ".join(f"{e['year']} ({e['description']})" for e in events[:5])
            + ("..." if len(events) > 5 else "")
        ),
        "n_observations": total_years,
    })
    prior_data["prior"] = clip_value("prior", prior_data["prior"])
    return prior_data


# ---------------------------------------------------------------------------
#  Method B: Incidence rate
# ---------------------------------------------------------------------------

def _compute_method_b(event_id: str, config: dict) -> dict:
    """Compute using Method B (incidence rate from surveys)."""

    prior_data = _get_method_b_prior(event_id, config)
    modifiers = _get_modifiers(event_id, config)

    modifier_values = [m["modifier_value"] for m in modifiers]
    p_global_result = calculate_p_global(prior_data["prior"], modifier_values)

    return _build_output(event_id, config, prior_data, modifiers, p_global_result)


def _get_method_b_prior(event_id: str, config: dict) -> dict:
    """Get the Method B prior — config-driven routing."""

    prior_source = config.get("prior_source", "")

    # ── Dragos ICS/OT data (special case for DIG-CIC-002) ────────────
    if prior_source == "dragos":
        prior_data = method_b_dragos_prior(
            incidents=config.get("dragos_incidents_2025", 3300),
            sector_pct=config.get("dragos_manufacturing_pct", 0.67),
            total_orgs=config.get("dragos_total_mfg_orgs", 300000),
            dark_figure=config.get("dark_figure", 3.0),
        )
        prior_data.update({
            "data_source": "Dragos Year-in-Review 2026",
            "source_id": "B04",
            "observation_window": "Annual (Dragos 2025 data)",
            "n_observations": 1,
            "calculation_steps": (
                f"Dragos: {config.get('dragos_incidents_2025', 3300)} ransomware incidents x "
                f"{config.get('dragos_manufacturing_pct', 0.67)} manufacturing share / "
                f"{config.get('dragos_total_mfg_orgs', 300000)} total mfg orgs x "
                f"{config.get('dark_figure', 3.0)} dark figure"
            ),
        })
        return prior_data

    # ── DBIR decomposition (all DIG-RDE, DIG-FSD, DIG-SCC, DIG-CIC events) ──
    if prior_source == "dbir" or event_id in DBIR_EVENT_MAPPING:
        dark_figure = config.get("dark_figure", 1.0)
        prior_data = method_b_prior(event_id, dark_figure=dark_figure)
        prior_data.update({
            "data_source": "Verizon DBIR 2025",
            "source_id": "B01",
            "observation_window": "Annual (DBIR 2025)",
            "n_observations": 1,
        })
        return prior_data

    # Fallback for unmapped events
    fallback = get_fallback_rate(event_id)
    return {
        "prior": fallback, "method": "B", "formula": f"Fallback: {fallback}",
        "data_source": "hardcoded_seed", "source_id": "fallback",
        "confidence": "Low", "data_status": "FALLBACK_HARDCODED",
    }


# ---------------------------------------------------------------------------
#  Method C: Structural calibration
# ---------------------------------------------------------------------------

def _compute_method_c(event_id: str, config: dict) -> dict:
    """Compute using Method C (structural calibration)."""

    prior_data = _get_method_c_prior(event_id, config)
    modifiers = _get_modifiers(event_id, config)

    modifier_values = [m["modifier_value"] for m in modifiers]
    p_global_result = calculate_p_global(prior_data["prior"], modifier_values)

    return _build_output(event_id, config, prior_data, modifiers, p_global_result)


def _get_method_c_prior(event_id: str, config: dict) -> dict:
    """Get the Method C prior using structural calibration."""

    # Phase 1 hand-crafted events with manual_c source
    prior_source = config.get("prior_source", "")
    if prior_source == "manual_c":
        if event_id == "STR-TRD-001":
            return _prior_tariff(config)
        if event_id == "OPS-CMP-001":
            return _prior_chip_shortage(config)

    # Family-level calibrated defaults
    prefix = _get_family_prefix(event_id)
    family_defaults = METHOD_C_FAMILY_DEFAULTS.get(prefix)

    if family_defaults:
        prior_data = method_c_prior(
            p_preconditions=family_defaults["p_pre"],
            p_trigger=family_defaults["p_trig"],
            p_implementation=family_defaults["p_impl"],
            evidence=family_defaults["evidence"],
        )
        prior_data.update({
            "data_source": f"Structural calibration ({prefix} family defaults)",
            "source_id": "manual",
            "confidence": "Medium",
        })
        return prior_data

    # Last resort: generic 0.50 defaults
    prior_data = method_c_prior(
        p_preconditions=0.50,
        p_trigger=0.50,
        p_implementation=0.50,
        evidence={
            "p_preconditions": "DEFAULT_0.50_NO_DATA",
            "p_trigger": "DEFAULT_0.50_NO_DATA",
            "p_implementation": "DEFAULT_0.50_NO_DATA",
        },
    )
    prior_data.update({
        "data_source": "Structural calibration (generic defaults)",
        "source_id": "manual",
    })
    return prior_data


def _get_family_prefix(event_id: str) -> str:
    """Extract family prefix like 'STR-GEO' from 'STR-GEO-001'."""
    parts = event_id.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return event_id


def _prior_tariff(config: dict) -> dict:
    """STR-TRD-001: Major tariff increases via Method C."""
    p_pre = 0.80
    p_trig = 0.60
    p_impl = 0.70

    current_year = datetime.utcnow().year
    us_election_years = {2016, 2020, 2024, 2028}
    if current_year in us_election_years:
        p_trig = 0.80

    prior_data = method_c_prior(
        p_preconditions=p_pre,
        p_trigger=p_trig,
        p_implementation=p_impl,
        evidence={
            "p_preconditions": (
                "WTO TMR: cumulative trade-restrictive stockpile at record $2.4T. "
                "GPR trade tension sub-index elevated since 2018."
            ),
            "p_trigger": (
                f"{'US election year — historically +25% trade measures' if current_year in us_election_years else 'Non-election year but persistent US-China/EU-China tensions'}. "
                "WTO: 70% of election years 2016-2024 had new major measures."
            ),
            "p_implementation": (
                "Historical: ~70% of announced >25% tariffs implemented within 12 months. "
                "Based on WTO stockpile data 2016-2024."
            ),
        },
    )
    prior_data.update({
        "data_source": "WTO Trade Monitoring Report + structural analysis",
        "source_id": "B08+manual",
        "observation_window": "2016-2024 (structural analysis)",
    })
    return prior_data


def _prior_chip_shortage(config: dict) -> dict:
    """OPS-CMP-001: Semiconductor chip shortage via Method C."""
    p_pre = 0.70
    p_trig = 0.40
    p_impl = 0.60

    prior_data = method_c_prior(
        p_preconditions=p_pre,
        p_trigger=p_trig,
        p_implementation=p_impl,
        evidence={
            "p_preconditions": (
                "TSMC holds 60%+ of advanced semiconductor manufacturing. "
                "Geographic concentration in Taiwan creates structural fragility. "
                "WEF Global Risks 2025: supply chain concentration ranked #7."
            ),
            "p_trigger": (
                "Moderate: no active demand surge signal (PMI new orders near neutral). "
                "But geopolitical risk to Taiwan and potential trade restrictions remain."
            ),
            "p_implementation": (
                "2020-2022 shortage lasted 18+ months. Lead times expanded from 12 to 26 weeks. "
                "Once triggered, semiconductor shortages persist due to 6-month fab cycle times."
            ),
        },
    )
    prior_data.update({
        "data_source": "Structural analysis + FRED durable goods proxy",
        "source_id": "B04+C01",
        "observation_window": "Structural (ongoing)",
    })
    return prior_data


# ---------------------------------------------------------------------------
#  Modifiers
# ---------------------------------------------------------------------------

def _get_modifiers(event_id: str, config: dict) -> list[dict]:
    """Get all modifiers for an event based on its configuration."""
    modifiers = []
    modifier_sources = config.get("modifier_sources", [])

    for source in modifier_sources:
        mod = _fetch_modifier(event_id, source, config)
        if mod:
            modifiers.append(mod)

    return modifiers


def _fetch_modifier(event_id: str, source: str, config: dict) -> dict | None:
    """Fetch a single modifier from its source."""

    if source == "A01":
        from .connectors.copernicus import get_temperature_modifier
        mod = get_temperature_modifier()
        return {
            "name": mod.get("name", "ERA5 temperature"),
            "source_id": "A01",
            "indicator_value": mod.get("indicator_value"),
            "indicator_unit": mod.get("indicator_unit", ""),
            "modifier_value": mod.get("modifier", 1.0),
            "calibration": mod.get("calibration", {}),
            "status": mod.get("status", "FALLBACK"),
        }

    if source == "A02":
        from .connectors.usgs import get_recent_seismicity_modifier
        result = get_recent_seismicity_modifier()
        if result.success:
            return {
                "name": "Recent seismicity rate",
                "source_id": "A02",
                "indicator_value": result.data.get("ratio", 1.0),
                "indicator_unit": "ratio (recent vs long-term annual rate)",
                "modifier_value": result.data.get("modifier", 1.0),
                "calibration": {
                    "method": "ratio",
                    "recent_days": result.data.get("recent_days", 90),
                    "recent_annualized": result.data.get("recent_annualized"),
                    "longterm_annual": result.data.get("longterm_annual"),
                },
                "status": "COMPUTED",
            }

    if source == "A03":
        # Varies by event
        if event_id == "STR-ECO-001" or _get_family_prefix(event_id) in ("STR-ECO", "STR-FIN"):
            from .connectors.fred import get_yield_curve_modifier
            yc = get_yield_curve_modifier()
            return {
                "name": yc.get("name", "Yield curve"),
                "source_id": "A03",
                "indicator_value": yc.get("indicator_value"),
                "indicator_unit": yc.get("indicator_unit", ""),
                "modifier_value": yc.get("modifier", 1.0),
                "calibration": {"series_id": "T10Y2Y"},
                "status": yc.get("status", "FALLBACK"),
            }
        if _get_family_prefix(event_id) in ("OPS-CMP", "DIG-HWS", "PHY-MAT"):
            from .connectors.fred import get_durable_goods_modifier
            dg = get_durable_goods_modifier()
            return {
                "name": dg.get("name", "Durable goods demand"),
                "source_id": "A03",
                "indicator_value": dg.get("indicator_value"),
                "indicator_unit": dg.get("indicator_unit", ""),
                "modifier_value": dg.get("modifier", 1.0),
                "calibration": {"series_id": "ACDGNO", "proxy": "C01"},
                "status": dg.get("status", "FALLBACK"),
            }
        if _get_family_prefix(event_id) == "PHY-ENE":
            from .connectors.fred import get_pmi_modifier
            pmi = get_pmi_modifier()
            return {
                "name": pmi.get("name", "PMI demand"),
                "source_id": "A03",
                "indicator_value": pmi.get("indicator_value"),
                "indicator_unit": pmi.get("indicator_unit", ""),
                "modifier_value": pmi.get("modifier", 1.0),
                "calibration": {"series_id": "NAPMNOI", "proxy": "C05"},
                "status": pmi.get("status", "FALLBACK"),
            }

    if source == "A04":
        from .connectors.nvd import get_ics_cve_modifier
        mod = get_ics_cve_modifier()
        return {
            "name": mod.get("name", "ICS CVE growth"),
            "source_id": "A04",
            "indicator_value": mod.get("indicator_value"),
            "indicator_unit": mod.get("indicator_unit", ""),
            "modifier_value": mod.get("modifier", 1.0),
            "calibration": {"proxy": "C02"},
            "status": mod.get("status", "FALLBACK"),
        }

    if source == "A05":
        from .connectors.cisa import get_kev_modifier
        mod = get_kev_modifier()
        return {
            "name": "CISA KEV growth rate",
            "source_id": "A05",
            "indicator_value": mod.get("total_kev"),
            "indicator_unit": "total known exploited vulnerabilities",
            "modifier_value": mod.get("modifier", 1.0),
            "calibration": {"yoy_growth": mod.get("yoy_growth")},
            "status": mod.get("status", "FALLBACK"),
        }

    if source == "A06":
        from .connectors.gpr import get_gpr_modifier
        mod = get_gpr_modifier()
        return {
            "name": mod.get("name", "GPR Index"),
            "source_id": "A06",
            "indicator_value": mod.get("indicator_value"),
            "indicator_unit": mod.get("indicator_unit", ""),
            "modifier_value": mod.get("modifier", 1.0),
            "calibration": mod.get("calibration", {}),
            "status": mod.get("status", "FALLBACK"),
        }

    if source == "categorical":
        cat_mods = config.get("categorical_modifiers", [])
        if cat_mods:
            cat = cat_mods[0]
            condition_met = _evaluate_categorical_condition(cat.get("condition_check", ""))
            from .computation.modifiers import categorical_modifier
            mod = categorical_modifier(cat.get("condition_check", ""), condition_met)
            return {
                "name": cat.get("name", "Categorical modifier"),
                "source_id": "categorical",
                "indicator_value": condition_met,
                "indicator_unit": "boolean",
                "modifier_value": mod["modifier"],
                "calibration": {
                    "method": "categorical",
                    "justification": cat.get("justification", mod.get("justification", "")),
                },
                "status": "COMPUTED",
            }

        # Election year modifier for trade events
        if event_id.startswith("STR-TRD"):
            from .computation.modifiers import categorical_modifier
            current_year = datetime.utcnow().year
            is_election = current_year in {2016, 2020, 2024, 2028}
            mod = categorical_modifier("us_election_year", is_election)
            return {
                "name": "US election year",
                "source_id": "categorical",
                "indicator_value": is_election,
                "indicator_unit": "boolean",
                "modifier_value": mod["modifier"],
                "calibration": {"method": "categorical", "justification": mod["justification"]},
                "status": "COMPUTED",
            }

    # Unknown source — return neutral modifier
    logger.warning(f"Unknown modifier source '{source}' for {event_id}")
    return None


def _evaluate_categorical_condition(condition_name: str) -> bool:
    """Evaluate a categorical condition. Conservative defaults."""
    if condition_name == "is_pheic_active":
        return False
    return False


# ---------------------------------------------------------------------------
#  Divergence documentation
# ---------------------------------------------------------------------------

DIVERGENCE_REASONS = {
    "PHY-GEO-001": (
        "EXPECTED: Old rate (3.5%) included P(material impact on specific company). "
        "New Layer 1 prior (95%) is P(M6.0+ earthquake occurs anywhere in major seismic zones) "
        "which is nearly certain annually. The impact probability belongs in Layer 2."
    ),
    "STR-TRD-001": (
        "Method C structural analysis with 2024-2026 evidence: preconditions=0.80 "
        "(WTO stockpile at record), trigger=0.60, implementation=0.70. Higher than "
        "old estimate (22%) due to elevated trade tensions since 2018."
    ),
    "DIG-RDE-001": (
        "DBIR 2025 decomposition: 18% base breach rate x 44% ransomware share x "
        "50% ERP subsplit = 3.96%. Old rate (12%) was likely overall ransomware rate "
        "without the ERP-specific subsplit. New methodology is more precise."
    ),
    "DIG-CIC-002": (
        "Dragos 2026 methodology: 3300 incidents x 67% manufacturing / 300K orgs x "
        "3.0 dark figure = 2.21%. Old rate (5%) may have used a smaller denominator "
        "or higher incident count. Dragos data is the most authoritative source."
    ),
    "STR-ECO-001": (
        "World Bank data: counted G7 years with negative GDP growth. Any G7 country "
        "having a recession is frequent (44% of years since 2000). Old rate (15%) "
        "may have been for synchronized global recession, which is rarer."
    ),
    "PHY-CLI-003": (
        "EM-DAT data: 15 out of 25 years (2000-2024) had significant heat wave events "
        "in the EEA region = 60% annual probability. Old rate (6.5%) was clearly a "
        "Layer 2 estimate (chance of impact on a specific facility). Layer 1 measures "
        "whether a heat wave occurred anywhere in Europe — which is frequent."
    ),
}


def _get_divergence_reason(event_id: str, divergence: float) -> str | None:
    """Get documented reason for divergence, if applicable."""
    if divergence <= 0.50:
        return None
    return DIVERGENCE_REASONS.get(event_id, f"Divergence of {divergence:.0%} — needs documentation")


# ---------------------------------------------------------------------------
#  Output builder
# ---------------------------------------------------------------------------

def _build_output(event_id: str, config: dict, prior_data: dict,
                  modifiers: list[dict], p_global_result: dict) -> dict:
    """Build the full Section 9.2 output JSON."""

    # Compare with fallback rate
    fallback_rate = get_fallback_rate(event_id)
    computed_prior = prior_data["prior"]
    divergence = abs(computed_prior - fallback_rate) / fallback_rate if fallback_rate > 0 else 0

    return {
        "event_id": event_id,
        "event_name": config["name"],
        "domain": config["domain"],
        "family": config["family"],

        "layer1": {
            "prior": computed_prior,
            "method": prior_data.get("method", "?"),
            "derivation": {
                "formula": prior_data.get("formula", ""),
                "data_source": prior_data.get("data_source", ""),
                "source_id": prior_data.get("source_id", ""),
                "observation_window": prior_data.get("observation_window", ""),
                "n_observations": prior_data.get("n_observations"),
                "calculation_steps": prior_data.get("calculation_steps", ""),
                "confidence": prior_data.get("confidence", "Medium"),
                "sub_probabilities": prior_data.get("sub_probabilities"),
            },
            "modifiers": modifiers,
            "p_global": p_global_result["p_global"],
            "p_global_raw": p_global_result["p_global_raw"],
            "p_global_capped_at": p_global_result.get("capped_at"),
        },

        "layer2": {
            "note": "Layer 2 (client-specific) not computed in Phase 1",
        },

        "metadata": {
            "last_computed": datetime.utcnow().isoformat() + "Z",
            "spec_version": "2.3",
            "engine_version": "2.0.0",
            "data_freshness": {},
            "fallback_rate": fallback_rate,
            "divergence_from_fallback": round(divergence, 2),
            "divergence_acceptable": divergence <= 0.50,
            "divergence_reason": _get_divergence_reason(event_id, divergence),
        },
    }


def _fallback_result(event_id: str, reason: str = "") -> dict:
    """Generate a fallback result when computation fails entirely."""
    fallback_rate = get_fallback_rate(event_id)

    config = get_event_config(event_id) or {
        "name": event_id,
        "domain": "Unknown",
        "family": "Unknown",
    }

    return {
        "event_id": event_id,
        "event_name": config.get("name", event_id),
        "domain": config.get("domain", "Unknown"),
        "family": config.get("family", "Unknown"),

        "layer1": {
            "prior": fallback_rate,
            "method": "FALLBACK",
            "derivation": {
                "formula": f"Hardcoded fallback: {fallback_rate}",
                "data_source": "hardcoded_seed",
                "source_id": "fallback",
                "observation_window": "N/A",
                "n_observations": None,
                "calculation_steps": f"Fallback used because: {reason}",
                "confidence": "Low",
            },
            "modifiers": [],
            "p_global": fallback_rate,
            "p_global_raw": fallback_rate,
            "p_global_capped_at": None,
            "data_status": "FALLBACK_HARDCODED",
        },

        "layer2": {"note": "Layer 2 not computed (fallback mode)"},

        "metadata": {
            "last_computed": datetime.utcnow().isoformat() + "Z",
            "spec_version": "2.3",
            "engine_version": "2.0.0",
            "fallback_reason": reason,
            "data_status": "FALLBACK_HARDCODED",
        },
    }
