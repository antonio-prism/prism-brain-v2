"""
PRISM Brain V2 API - Taxonomy-Based Risk Event Endpoints

New V2 API provides a cleaner taxonomy structure with domain → family → event hierarchy.
Frontend calls these endpoints from api_client.py.

Replaces fragmented v1 endpoints with unified V2 structure that groups events by:
- Domain (PHYSICAL, STRUCTURAL, DIGITAL, OPERATIONAL)
- Family (grouped by 2-level family code like "1.1", "1.2", etc.)
- Events within each family
"""

from fastapi import HTTPException, Query
from typing import Optional, List, Dict, Any
from sqlalchemy import and_
from database.models import RiskEvent, RiskProbability
from datetime import datetime, timedelta

# Family code mapping from event_id prefix to taxonomy codes
FAMILY_CODE_MAP = {
    "PHY-CLI": "1.1",  # Climate Extremes & Weather Events
    "PHY-ENE": "1.2",  # Energy Supply & Grid Stability
    "PHY-MAT": "1.3",  # Natural Resources & Raw Materials
    "PHY-WAT": "1.4",  # Water Resources & Quality
    "PHY-GEO": "1.5",  # Geophysical Disasters
    "PHY-POL": "1.6",  # Contamination & Pollution
    "PHY-BIO": "1.7",  # Biological & Pandemic Risks
    "STR-GEO": "2.1",  # Geopolitical Conflict & Instability
    "STR-TRD": "2.2",  # Trade & Economic Policy Shifts
    "STR-REG": "2.3",  # Regulatory & Compliance Changes
    "STR-ECO": "2.4",  # Macroeconomic Shocks
    "STR-ENP": "2.5",  # Energy Transition & Climate Policy
    "STR-TEC": "2.6",  # Technology Policy & Regulatory Restrictions
    "STR-FIN": "2.7",  # Financial Market Disruptions
    "DIG-CIC": "3.1",  # Critical Infrastructure Cyberattacks
    "DIG-RDE": "3.2",  # Ransomware, Data Breaches & Exfiltration
    "DIG-SCC": "3.3",  # Supply Chain Cyberattacks
    "DIG-FSD": "3.4",  # Fraud, Social Engineering & Denial-of-Service
    "DIG-CLS": "3.5",  # Cloud & Platform Sovereignty
    "DIG-HWS": "3.6",  # Hardware, Industrial Equipment & Semiconductor Sovereignty
    "DIG-SWS": "3.7",  # Software & AI Sovereignty
    "OPS-MAR": "4.1",  # Port & Maritime Logistics
    "OPS-AIR": "4.2",  # Air Freight & Aviation
    "OPS-RLD": "4.3",  # Road & Rail Transport
    "OPS-CMP": "4.4",  # Component & Materials Shortages
    "OPS-SUP": "4.5",  # Supplier & Vendor Disruptions
    "OPS-MFG": "4.6",  # Manufacturing & Production Disruptions
    "OPS-WHS": "4.7",  # Warehouse & Inventory Management
}

# Domain mapping from layer1_primary to domain names
# NOTE: Database stores "STRUCTURAL" (not "STRATEGIC") per seed data
DOMAIN_MAP = {
    "PHYSICAL": "Physical",
    "STRUCTURAL": "Structural",
    "DIGITAL": "Digital",
    "OPERATIONAL": "Operational",
}


def get_domain_from_event(event: RiskEvent) -> str:
    """Extract domain from a RiskEvent."""
    if event.layer1_primary:
        return DOMAIN_MAP.get(event.layer1_primary, event.layer1_primary)
    return "Unknown"


def get_family_code_from_event_id(event_id: str) -> str:
    """Extract family code from event_id (first 7 chars like 'PHY-CLI')."""
    if len(event_id) >= 7:
        prefix = event_id[:7]
        return FAMILY_CODE_MAP.get(prefix, "0.0")
    return "0.0"


def get_family_name_from_event(event: RiskEvent) -> str:
    """Extract family name from RiskEvent layer2_primary or event_id."""
    if event.layer2_primary:
        return event.layer2_primary
    # Fallback to deriving from event_id
    if len(event.event_id) >= 7:
        prefix = event.event_id[:7]
        # Map back to full family names (must match seed data exactly)
        family_names = {
            "PHY-CLI": "Climate Extremes & Weather Events",
            "PHY-ENE": "Energy Supply & Grid Stability",
            "PHY-MAT": "Natural Resources & Raw Materials",
            "PHY-WAT": "Water Resources & Quality",
            "PHY-GEO": "Geophysical Disasters",
            "PHY-POL": "Contamination & Pollution",
            "PHY-BIO": "Biological & Pandemic Risks",
            "STR-GEO": "Geopolitical Conflict & Instability",
            "STR-TRD": "Trade & Economic Policy Shifts",
            "STR-REG": "Regulatory & Compliance Changes",
            "STR-ECO": "Macroeconomic Shocks",
            "STR-ENP": "Energy Transition & Climate Policy",
            "STR-TEC": "Technology Policy & Regulatory Restrictions",
            "STR-FIN": "Financial Market Disruptions",
            "DIG-CIC": "CRITICAL INFRASTRUCTURE CYBERATTACKS",
            "DIG-RDE": "RANSOMWARE, DATA BREACHES & EXFILTRATION",
            "DIG-SCC": "SUPPLY CHAIN CYBERATTACKS",
            "DIG-FSD": "FRAUD, SOCIAL ENGINEERING & DENIAL-OF-SERVICE",
            "DIG-CLS": "CLOUD & PLATFORM SOVEREIGNTY",
            "DIG-HWS": "HARDWARE, INDUSTRIAL EQUIPMENT & SEMICONDUCTOR SOVEREIGNTY",
            "DIG-SWS": "SOFTWARE & AI SOVEREIGNTY",
            "OPS-MAR": "Port & Maritime Logistics",
            "OPS-AIR": "Air Freight & Aviation",
            "OPS-RLD": "Road & Rail Transport",
            "OPS-CMP": "Component & Materials Shortages",
            "OPS-SUP": "Supplier & Vendor Disruptions",
            "OPS-MFG": "Manufacturing & Production Disruptions",
            "OPS-WHS": "Warehouse & Inventory Management",
        }
        return family_names.get(prefix, "Unknown")
    return "Unknown"


def serialize_v2_event(event: RiskEvent, latest_probability: Optional[RiskProbability] = None) -> Dict:
    """Convert a RiskEvent ORM object to V2 API format."""
    domain = get_domain_from_event(event)
    family_code = get_family_code_from_event_id(event.event_id)
    family_name = get_family_name_from_event(event)

    # Extract baseline probability as 0-100 scale
    base_rate_pct = 0
    if event.baseline_probability:
        # If stored as 0-1, convert to 0-100
        if event.baseline_probability <= 1.0:
            base_rate_pct = event.baseline_probability * 100
        else:
            base_rate_pct = event.baseline_probability

    # Extract confidence level from latest probability or default
    confidence_level = "MEDIUM"
    if latest_probability and latest_probability.confidence_score:
        conf = latest_probability.confidence_score
        if conf >= 0.8:
            confidence_level = "HIGH"
        elif conf >= 0.5:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

    return {
        "event_id": event.event_id,
        "event_name": event.event_name,
        "domain": domain,
        "family_code": family_code,
        "family_name": family_name,
        "description": event.description or "",
        "confidence_level": confidence_level,
        "base_rate_pct": base_rate_pct,
        "baseline_impact": event.baseline_impact or 0,
        "super_risk": event.super_risk or False,
        "geographic_scope": event.geographic_scope or "",
        "time_horizon": event.time_horizon or "",
        "methodology_tier": event.methodology_tier or "",
    }


def register_v2_routes(app, get_session_context):
    """Register all V2 taxonomy-based endpoints."""

    # ============== V2 Health Check ==============

    @app.get("/api/v2/health")
    async def v2_health():
        """Get V2 API health status with event counts."""
        with get_session_context() as session:
            total_events = session.query(RiskEvent).count()
            total_probabilities = session.query(RiskProbability).count()

            # Count events by domain
            domains_count = {}
            for domain_key in DOMAIN_MAP.keys():
                count = session.query(RiskEvent).filter(
                    RiskEvent.layer1_primary == domain_key
                ).count()
                if count > 0:
                    domains_count[DOMAIN_MAP[domain_key]] = count

            return {
                "status": "healthy",
                "v2_events": total_events,
                "v2_probabilities": total_probabilities,
                "domains": domains_count,
                "timestamp": datetime.utcnow().isoformat(),
            }

    # ============== V2 Taxonomy ==============

    @app.get("/api/v2/taxonomy")
    async def v2_taxonomy():
        """Get full V2 taxonomy: domains → families → event counts."""
        with get_session_context() as session:
            taxonomy = {}

            for domain_key, domain_name in DOMAIN_MAP.items():
                # Get all events in this domain
                domain_events = session.query(RiskEvent).filter(
                    RiskEvent.layer1_primary == domain_key
                ).all()

                if not domain_events:
                    continue

                # Group by family
                families = {}
                for event in domain_events:
                    family_code = get_family_code_from_event_id(event.event_id)
                    family_name = get_family_name_from_event(event)

                    if family_code not in families:
                        families[family_code] = {
                            "family_code": family_code,
                            "family_name": family_name,
                            "event_count": 0,
                        }
                    families[family_code]["event_count"] += 1

                taxonomy[domain_name] = {
                    "domain_name": domain_name,
                    "event_count": len(domain_events),
                    "families": list(families.values()),
                }

            return taxonomy

    # ============== V2 Events ==============

    @app.get("/api/v2/events")
    async def v2_events(
        domain: Optional[str] = Query(None, description="Filter by domain name"),
        family_code: Optional[str] = Query(None, description="Filter by family code"),
        search: Optional[str] = Query(None, description="Search in event_name or description"),
    ):
        """Get V2 events with optional filters."""
        with get_session_context() as session:
            query = session.query(RiskEvent)

            # Filter by domain
            if domain:
                # Find the layer1_primary value for this domain
                domain_key = None
                for key, name in DOMAIN_MAP.items():
                    if name.lower() == domain.lower():
                        domain_key = key
                        break
                if domain_key:
                    query = query.filter(RiskEvent.layer1_primary == domain_key)

            # Filter by family code
            if family_code:
                # Find all events with this family code
                matching_events = []
                for event in query.all():
                    if get_family_code_from_event_id(event.event_id) == family_code:
                        matching_events.append(event.id)
                if matching_events:
                    query = session.query(RiskEvent).filter(RiskEvent.id.in_(matching_events))
                else:
                    query = session.query(RiskEvent).filter(False)  # Empty result

            # Search
            if search:
                search_term = f"%{search}%"
                query = query.filter(
                    (RiskEvent.event_name.ilike(search_term)) |
                    (RiskEvent.description.ilike(search_term))
                )

            events = query.all()

            # Get latest probabilities for each event
            probability_map = {}
            for event_id in [e.event_id for e in events]:
                latest_prob = session.query(RiskProbability).filter(
                    RiskProbability.event_id == event_id
                ).order_by(RiskProbability.calculation_date.desc()).first()
                probability_map[event_id] = latest_prob

            return [
                serialize_v2_event(event, probability_map.get(event.event_id))
                for event in events
            ]

    # ============== V2 Single Event ==============

    @app.get("/api/v2/events/{event_id}")
    async def v2_event_detail(event_id: str):
        """Get full detail for a single V2 event."""
        with get_session_context() as session:
            event = session.query(RiskEvent).filter(
                RiskEvent.event_id == event_id
            ).first()

            if not event:
                raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

            latest_prob = session.query(RiskProbability).filter(
                RiskProbability.event_id == event_id
            ).order_by(RiskProbability.calculation_date.desc()).first()

            result = serialize_v2_event(event, latest_prob)

            # Add probability details if available
            if latest_prob:
                result["current_probability_pct"] = latest_prob.probability_pct or 0
                result["confidence_score"] = latest_prob.confidence_score or 0
                result["calculation_date"] = latest_prob.calculation_date.isoformat() if latest_prob.calculation_date else None
                result["ci_lower_pct"] = latest_prob.ci_lower_pct or 0
                result["ci_upper_pct"] = latest_prob.ci_upper_pct or 0

            return result

    # ============== V2 Domain ==============

    @app.get("/api/v2/domains/{domain}")
    async def v2_domain(domain: str):
        """Get all families and events within a domain."""
        with get_session_context() as session:
            # Normalize domain name to layer1_primary key
            domain_key = None
            domain_name = domain
            for key, name in DOMAIN_MAP.items():
                if name.lower() == domain.lower() or key.lower() == domain.lower():
                    domain_key = key
                    domain_name = name
                    break

            if not domain_key:
                raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")

            events = session.query(RiskEvent).filter(
                RiskEvent.layer1_primary == domain_key
            ).all()

            if not events:
                raise HTTPException(status_code=404, detail=f"No events found for domain '{domain}'")

            # Group by family
            families = {}
            for event in events:
                family_code = get_family_code_from_event_id(event.event_id)
                family_name = get_family_name_from_event(event)

                if family_code not in families:
                    families[family_code] = {
                        "family_code": family_code,
                        "family_name": family_name,
                        "events": [],
                    }

                # Get latest probability
                latest_prob = session.query(RiskProbability).filter(
                    RiskProbability.event_id == event.event_id
                ).order_by(RiskProbability.calculation_date.desc()).first()

                families[family_code]["events"].append(
                    serialize_v2_event(event, latest_prob)
                )

            return {
                "domain": domain_name,
                "event_count": len(events),
                "families": list(families.values()),
            }

    # ============== V2 Family ==============

    @app.get("/api/v2/families/{family_code}")
    async def v2_family(family_code: str):
        """Get all events within a specific family."""
        with get_session_context() as session:
            # Find all events matching this family code
            all_events = session.query(RiskEvent).all()
            matching_events = [
                e for e in all_events
                if get_family_code_from_event_id(e.event_id) == family_code
            ]

            if not matching_events:
                raise HTTPException(status_code=404, detail=f"Family '{family_code}' not found")

            # Get family name from first event
            family_name = get_family_name_from_event(matching_events[0])

            # Get latest probabilities
            probability_map = {}
            for event_id in [e.event_id for e in matching_events]:
                latest_prob = session.query(RiskProbability).filter(
                    RiskProbability.event_id == event_id
                ).order_by(RiskProbability.calculation_date.desc()).first()
                probability_map[event_id] = latest_prob

            return {
                "family_code": family_code,
                "family_name": family_name,
                "event_count": len(matching_events),
                "events": [
                    serialize_v2_event(event, probability_map.get(event.event_id))
                    for event in matching_events
                ],
            }

    # ============== V2 Probabilities ==============

    @app.get("/api/v2/probabilities")
    async def v2_probabilities(
        domain: Optional[str] = Query(None, description="Filter by domain"),
        family_code: Optional[str] = Query(None, description="Filter by family code"),
    ):
        """Get latest V2 probabilities for all active events with optional filters."""
        with get_session_context() as session:
            # Get all events
            event_query = session.query(RiskEvent)

            # Filter by domain
            if domain:
                domain_key = None
                for key, name in DOMAIN_MAP.items():
                    if name.lower() == domain.lower():
                        domain_key = key
                        break
                if domain_key:
                    event_query = event_query.filter(RiskEvent.layer1_primary == domain_key)

            events = event_query.all()

            # Filter by family code if specified
            if family_code:
                events = [
                    e for e in events
                    if get_family_code_from_event_id(e.event_id) == family_code
                ]

            # Get latest probability for each event
            results = []
            for event in events:
                latest_prob = session.query(RiskProbability).filter(
                    RiskProbability.event_id == event.event_id
                ).order_by(RiskProbability.calculation_date.desc()).first()

                if latest_prob:
                    prob_entry = serialize_v2_event(event, latest_prob)
                    prob_entry["current_probability_pct"] = latest_prob.probability_pct or 0
                    prob_entry["confidence_score"] = latest_prob.confidence_score or 0
                    prob_entry["calculation_date"] = latest_prob.calculation_date.isoformat() if latest_prob.calculation_date else None
                    results.append(prob_entry)

            return sorted(results, key=lambda x: x.get("current_probability_pct", 0), reverse=True)

    # ============== V2 Statistics ==============

    @app.get("/api/v2/stats")
    async def v2_stats():
        """Get V2 system statistics."""
        with get_session_context() as session:
            total_events = session.query(RiskEvent).count()
            total_probabilities = session.query(RiskProbability).count()

            # Count events by domain
            domain_stats = {}
            for domain_key, domain_name in DOMAIN_MAP.items():
                count = session.query(RiskEvent).filter(
                    RiskEvent.layer1_primary == domain_key
                ).count()
                if count > 0:
                    domain_stats[domain_name] = count

            # Calculate average probabilities
            avg_prob = 0
            latest_probs = session.query(RiskProbability).order_by(
                RiskProbability.calculation_date.desc()
            ).limit(total_events).all()

            if latest_probs:
                probs = [p.probability_pct for p in latest_probs if p.probability_pct is not None]
                if probs:
                    avg_prob = sum(probs) / len(probs)

            # Latest update time
            latest_calc = session.query(RiskProbability).order_by(
                RiskProbability.calculation_date.desc()
            ).first()

            return {
                "total_events": total_events,
                "total_probability_records": total_probabilities,
                "domains": domain_stats,
                "average_probability_pct": round(avg_prob, 2),
                "last_update": latest_calc.calculation_date.isoformat() if latest_calc else None,
                "timestamp": datetime.utcnow().isoformat(),
            }
