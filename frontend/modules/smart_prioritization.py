"""
PRISM Brain - Smart Prioritization Engine
==========================================
AI-assisted risk/process matching and prioritization.

Features:
1. Auto-match risks to processes based on relevance
2. Composite risk scoring algorithm
3. Process vulnerability mapping
4. Prioritized recommendations

The engine uses keyword matching, industry analysis, and
risk characteristics to intelligently prioritize assessments.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict

# App directory for loading data
APP_DIR = Path(__file__).parent.parent
RISK_DB_PATH = APP_DIR / "data" / "risk_database.json"
PROCESS_DB_PATH = APP_DIR / "data" / "process_framework.json"


# ============================================================================
# KEYWORD MAPPINGS FOR RISK-PROCESS MATCHING
# ============================================================================

# Process keywords that indicate vulnerability to specific risk types
PROCESS_RISK_KEYWORDS = {
    # Physical Risks
    'fire': ['manufacturing', 'warehouse', 'storage', 'production', 'facility', 'building', 'chemical', 'electrical'],
    'flood': ['facility', 'warehouse', 'storage', 'logistics', 'transportation', 'coastal', 'low-lying'],
    'earthquake': ['facility', 'building', 'infrastructure', 'manufacturing', 'construction'],
    'weather': ['logistics', 'transportation', 'supply', 'delivery', 'outdoor', 'agriculture'],
    'pandemic': ['workforce', 'employee', 'staff', 'human', 'labor', 'operations', 'customer'],

    # Structural Risks
    'supply': ['procurement', 'sourcing', 'vendor', 'supplier', 'purchasing', 'inventory', 'logistics'],
    'market': ['sales', 'revenue', 'pricing', 'customer', 'demand', 'financial', 'investment'],
    'regulatory': ['compliance', 'legal', 'governance', 'policy', 'audit', 'reporting', 'tax'],
    'economic': ['financial', 'treasury', 'investment', 'capital', 'budget', 'cost'],
    'currency': ['international', 'export', 'import', 'foreign', 'treasury', 'financial'],

    # Operational Risks
    'equipment': ['manufacturing', 'production', 'maintenance', 'machinery', 'plant', 'facility'],
    'quality': ['manufacturing', 'production', 'inspection', 'testing', 'compliance', 'customer'],
    'safety': ['manufacturing', 'operations', 'employee', 'workplace', 'health', 'environment'],
    'labor': ['human', 'employee', 'workforce', 'staffing', 'recruitment', 'training'],
    'process': ['operations', 'manufacturing', 'service', 'delivery', 'production'],

    # Digital Risks
    'cyber': ['it', 'technology', 'data', 'system', 'network', 'digital', 'software', 'cloud'],
    'data': ['information', 'customer', 'privacy', 'personal', 'database', 'analytics'],
    'system': ['it', 'technology', 'software', 'application', 'platform', 'infrastructure'],
    'ransomware': ['it', 'technology', 'data', 'system', 'backup', 'security'],
    'phishing': ['employee', 'email', 'communication', 'security', 'access'],
}

# Industry-specific risk amplifiers
# NOTE: Domain keys use UPPERCASE to match risk database Layer_1_Primary values
INDUSTRY_RISK_FACTORS = {
    'manufacturing': {
        'PHYSICAL': 1.3, 'OPERATIONAL': 1.4, 'STRUCTURAL': 1.1, 'DIGITAL': 0.9
    },
    'technology': {
        'PHYSICAL': 0.7, 'OPERATIONAL': 0.9, 'STRUCTURAL': 1.0, 'DIGITAL': 1.5
    },
    'finance': {
        'PHYSICAL': 0.6, 'OPERATIONAL': 0.8, 'STRUCTURAL': 1.4, 'DIGITAL': 1.4
    },
    'healthcare': {
        'PHYSICAL': 0.9, 'OPERATIONAL': 1.3, 'STRUCTURAL': 1.1, 'DIGITAL': 1.3
    },
    'retail': {
        'PHYSICAL': 1.0, 'OPERATIONAL': 1.1, 'STRUCTURAL': 1.2, 'DIGITAL': 1.1
    },
    'energy': {
        'PHYSICAL': 1.4, 'OPERATIONAL': 1.3, 'STRUCTURAL': 1.1, 'DIGITAL': 1.0
    },
    'logistics': {
        'PHYSICAL': 1.3, 'OPERATIONAL': 1.2, 'STRUCTURAL': 1.2, 'DIGITAL': 0.9
    },
    'general': {
        'PHYSICAL': 1.0, 'OPERATIONAL': 1.0, 'STRUCTURAL': 1.0, 'DIGITAL': 1.0
    }
}


# ============================================================================
# RISK-PROCESS MATCHING ENGINE
# ============================================================================

def calculate_risk_process_relevance(risk: Dict, process: Dict,
                                      industry: str = 'general') -> Dict:
    """
    Calculate how relevant a risk is to a specific process.

    Returns a relevance score (0-100) and matching reasons.
    """
    risk_name = risk.get('Event_Name', risk.get('risk_name', '')).lower()
    risk_desc = risk.get('Event_Description', '').lower()
    # Normalize domain to uppercase to match database values
    risk_domain = risk.get('Layer_1_Primary', risk.get('domain', 'OPERATIONAL')).upper()
    risk_category = risk.get('Layer_2_Primary', risk.get('category', '')).lower()

    process_name = process.get('process_name', process.get('Activity_Name', '')).lower()
    process_category = process.get('category', process.get('Process_Group_Name', '')).lower()

    score = 0
    reasons = []

    # 1. Keyword matching (0-40 points)
    keyword_matches = 0
    matched_keywords = []

    for risk_keyword, process_keywords in PROCESS_RISK_KEYWORDS.items():
        # Check if risk contains this keyword
        if risk_keyword in risk_name or risk_keyword in risk_desc:
            # Check if process matches any associated keywords
            for proc_kw in process_keywords:
                if proc_kw in process_name or proc_kw in process_category:
                    keyword_matches += 1
                    if risk_keyword not in matched_keywords:
                        matched_keywords.append(risk_keyword)
                    break

    keyword_score = min(keyword_matches * 10, 40)
    score += keyword_score
    if matched_keywords:
        reasons.append(f"Keyword matches: {', '.join(matched_keywords)}")

    # 2. Domain-process alignment (0-25 points)
    # NOTE: Keys use UPPERCASE to match database values
    domain_alignment = {
        'PHYSICAL': ['facility', 'warehouse', 'manufacturing', 'logistics', 'operations'],
        'STRUCTURAL': ['financial', 'procurement', 'sales', 'legal', 'strategy'],
        'OPERATIONAL': ['production', 'service', 'quality', 'human', 'maintenance'],
        'DIGITAL': ['it', 'technology', 'data', 'system', 'digital', 'cyber']
    }

    if risk_domain in domain_alignment:
        for keyword in domain_alignment[risk_domain]:
            if keyword in process_name or keyword in process_category:
                score += 25
                reasons.append(f"Process aligns with {risk_domain} domain")
                break

    # 3. Industry factor (0-20 points)
    industry_lower = industry.lower() if industry else 'general'
    industry_factors = INDUSTRY_RISK_FACTORS.get(industry_lower, INDUSTRY_RISK_FACTORS['general'])
    industry_multiplier = industry_factors.get(risk_domain, 1.0)

    if industry_multiplier > 1.1:
        industry_bonus = min((industry_multiplier - 1.0) * 50, 20)
        score += industry_bonus
        reasons.append(f"High {risk_domain} risk for {industry} industry")

    # 4. Process criticality boost (0-15 points)
    criticality = process.get('criticality_per_day', 0)
    if criticality > 50000:
        score += 15
        reasons.append("Critical process (high daily value)")
    elif criticality > 10000:
        score += 10
        reasons.append("Important process (moderate daily value)")
    elif criticality > 1000:
        score += 5

    # Ensure score is within 0-100
    score = min(max(score, 0), 100)

    return {
        'relevance_score': score,
        'relevance_level': 'High' if score >= 60 else 'Medium' if score >= 30 else 'Low',
        'reasons': reasons,
        'risk_id': risk.get('Event_ID', risk.get('risk_id', '')),
        'risk_name': risk.get('Event_Name', risk.get('risk_name', '')),
        'process_id': process.get('process_id', process.get('id', '')),
        'process_name': process.get('process_name', process.get('Activity_Name', ''))
    }


def auto_match_risks_to_process(process: Dict, risks: List[Dict],
                                 industry: str = 'general',
                                 top_n: int = 10) -> List[Dict]:
    """
    Automatically match and rank the most relevant risks for a process.

    Returns top N risks sorted by relevance.
    """
    matches = []

    for risk in risks:
        relevance = calculate_risk_process_relevance(risk, process, industry)
        if relevance['relevance_score'] > 0:
            matches.append({
                **relevance,
                'risk': risk
            })

    # Sort by relevance score descending
    matches.sort(key=lambda x: x['relevance_score'], reverse=True)

    return matches[:top_n]


def auto_match_processes_to_risk(risk: Dict, processes: List[Dict],
                                  industry: str = 'general',
                                  top_n: int = 10) -> List[Dict]:
    """
    Automatically match and rank the most relevant processes for a risk.

    Returns top N processes sorted by relevance.
    """
    matches = []

    for process in processes:
        relevance = calculate_risk_process_relevance(risk, process, industry)
        if relevance['relevance_score'] > 0:
            matches.append({
                **relevance,
                'process': process
            })

    # Sort by relevance score descending
    matches.sort(key=lambda x: x['relevance_score'], reverse=True)

    return matches[:top_n]


def generate_matching_matrix(processes: List[Dict], risks: List[Dict],
                              industry: str = 'general') -> Dict:
    """
    Generate a complete matching matrix between processes and risks.

    Returns a dictionary with relevance scores for each process-risk pair.
    """
    matrix = {}
    high_priority_pairs = []
    medium_priority_pairs = []

    for process in processes:
        proc_id = process.get('id', process.get('process_id', ''))
        matrix[proc_id] = {}

        for risk in risks:
            risk_id = risk.get('Event_ID', risk.get('risk_id', ''))
            relevance = calculate_risk_process_relevance(risk, process, industry)
            matrix[proc_id][risk_id] = relevance

            # Track priority pairs
            if relevance['relevance_score'] >= 60:
                high_priority_pairs.append(relevance)
            elif relevance['relevance_score'] >= 30:
                medium_priority_pairs.append(relevance)

    return {
        'matrix': matrix,
        'high_priority_count': len(high_priority_pairs),
        'medium_priority_count': len(medium_priority_pairs),
        'high_priority_pairs': sorted(high_priority_pairs,
                                       key=lambda x: x['relevance_score'],
                                       reverse=True)[:20],
        'total_processes': len(processes),
        'total_risks': len(risks)
    }


# ============================================================================
# COMPOSITE RISK SCORING
# ============================================================================

def calculate_composite_risk_score(assessment: Dict) -> Dict:
    """
    Calculate a composite risk score combining multiple factors.

    Factors:
    - Probability (25%)
    - Vulnerability (25%)
    - Criticality impact (25%)
    - Resilience gap (25%)

    Returns a score from 0-100 with breakdown.
    """
    probability = assessment.get('probability', 0.5)
    vulnerability = assessment.get('vulnerability', 0.5)
    resilience = assessment.get('resilience', 0.3)
    criticality = assessment.get('criticality_per_day', 0)
    downtime = assessment.get('expected_downtime', 5)

    # Normalize criticality to 0-1 (assuming max 100k per day)
    criticality_normalized = min(criticality / 100000, 1.0)

    # Calculate impact potential (criticality * downtime)
    max_potential_impact = 100000 * 365  # Max possible annual impact
    potential_impact = criticality * downtime * probability
    impact_normalized = min(potential_impact / max_potential_impact, 1.0)

    # Resilience gap (how much resilience is lacking)
    resilience_gap = 1 - resilience

    # Calculate component scores (0-25 each)
    prob_score = probability * 25
    vuln_score = vulnerability * 25
    impact_score = impact_normalized * 25
    resilience_score = resilience_gap * 25

    # Total composite score
    composite_score = prob_score + vuln_score + impact_score + resilience_score

    # Determine priority level
    if composite_score >= 70:
        priority = 'Critical'
        priority_color = '#dc3545'  # Red
    elif composite_score >= 50:
        priority = 'High'
        priority_color = '#fd7e14'  # Orange
    elif composite_score >= 30:
        priority = 'Medium'
        priority_color = '#ffc107'  # Yellow
    else:
        priority = 'Low'
        priority_color = '#28a745'  # Green

    return {
        'composite_score': round(composite_score, 1),
        'priority': priority,
        'priority_color': priority_color,
        'breakdown': {
            'probability': round(prob_score, 1),
            'vulnerability': round(vuln_score, 1),
            'impact_potential': round(impact_score, 1),
            'resilience_gap': round(resilience_score, 1)
        },
        'raw_values': {
            'probability': probability,
            'vulnerability': vulnerability,
            'resilience': resilience,
            'criticality': criticality,
            'downtime': downtime
        }
    }


def rank_assessments_by_priority(assessments: List[Dict]) -> List[Dict]:
    """
    Rank all assessments by composite risk score.

    Returns assessments sorted by priority with scores.
    """
    scored_assessments = []

    for assessment in assessments:
        score_data = calculate_composite_risk_score(assessment)
        scored_assessments.append({
            **assessment,
            **score_data
        })

    # Sort by composite score descending
    scored_assessments.sort(key=lambda x: x['composite_score'], reverse=True)

    # Add rank
    for i, assessment in enumerate(scored_assessments, 1):
        assessment['rank'] = i

    return scored_assessments


# ============================================================================
# PROCESS VULNERABILITY MAPPING
# ============================================================================

def calculate_process_vulnerability(process: Dict,
                                     related_assessments: List[Dict]) -> Dict:
    """
    Calculate overall vulnerability for a process across all its risks.

    Identifies concentration risk and vulnerability hotspots.
    """
    if not related_assessments:
        return {
            'process_id': process.get('id', ''),
            'process_name': process.get('process_name', ''),
            'total_risks': 0,
            'vulnerability_score': 0,
            'concentration_risk': 'None',
            'top_risks': []
        }

    # Calculate aggregate metrics
    total_exposure = sum(a.get('exposure', 0) for a in related_assessments)
    avg_vulnerability = sum(a.get('vulnerability', 0.5) for a in related_assessments) / len(related_assessments)
    avg_resilience = sum(a.get('resilience', 0.3) for a in related_assessments) / len(related_assessments)

    # Count high-probability risks
    high_prob_risks = len([a for a in related_assessments if a.get('probability', 0) >= 0.5])

    # Domain concentration
    domain_counts = defaultdict(int)
    for a in related_assessments:
        domain = a.get('domain', 'Unknown')
        domain_counts[domain] += 1

    # Check for concentration risk
    total_risks = len(related_assessments)
    max_domain_concentration = max(domain_counts.values()) / total_risks if total_risks > 0 else 0

    if max_domain_concentration >= 0.6:
        concentration_risk = 'High'
        dominant_domain = max(domain_counts, key=domain_counts.get)
        concentration_note = f"Over-exposed to {dominant_domain} risks ({max_domain_concentration:.0%})"
    elif max_domain_concentration >= 0.4:
        concentration_risk = 'Medium'
        dominant_domain = max(domain_counts, key=domain_counts.get)
        concentration_note = f"Moderate concentration in {dominant_domain} risks"
    else:
        concentration_risk = 'Low'
        concentration_note = "Well-diversified risk exposure"

    # Calculate vulnerability score (0-100)
    vulnerability_score = (
        avg_vulnerability * 30 +
        (1 - avg_resilience) * 30 +
        (high_prob_risks / max(total_risks, 1)) * 20 +
        max_domain_concentration * 20
    ) * 100 / 100

    # Get top risks for this process
    sorted_risks = sorted(related_assessments,
                          key=lambda x: x.get('exposure', 0),
                          reverse=True)

    return {
        'process_id': process.get('id', ''),
        'process_name': process.get('process_name', ''),
        'criticality': process.get('criticality_per_day', 0),
        'total_risks': total_risks,
        'total_exposure': total_exposure,
        'avg_vulnerability': avg_vulnerability,
        'avg_resilience': avg_resilience,
        'vulnerability_score': round(vulnerability_score, 1),
        'concentration_risk': concentration_risk,
        'concentration_note': concentration_note,
        'domain_distribution': dict(domain_counts),
        'high_probability_risks': high_prob_risks,
        'top_risks': sorted_risks[:5]
    }


def generate_vulnerability_map(processes: List[Dict],
                                assessments: List[Dict]) -> Dict:
    """
    Generate a complete vulnerability map for all processes.

    Identifies which processes are most at risk.
    """
    # Group assessments by process
    assessments_by_process = defaultdict(list)
    for a in assessments:
        proc_id = a.get('process_id')
        assessments_by_process[proc_id].append(a)

    # Calculate vulnerability for each process
    process_vulnerabilities = []
    for process in processes:
        proc_id = process.get('id', '')
        related_assessments = assessments_by_process.get(proc_id, [])
        vuln_data = calculate_process_vulnerability(process, related_assessments)
        process_vulnerabilities.append(vuln_data)

    # Sort by vulnerability score
    process_vulnerabilities.sort(key=lambda x: x['vulnerability_score'], reverse=True)

    # Calculate summary statistics
    scores = [p['vulnerability_score'] for p in process_vulnerabilities]
    critical_processes = len([s for s in scores if s >= 70])
    high_risk_processes = len([s for s in scores if 50 <= s < 70])

    return {
        'processes': process_vulnerabilities,
        'summary': {
            'total_processes': len(processes),
            'critical_count': critical_processes,
            'high_risk_count': high_risk_processes,
            'avg_vulnerability': sum(scores) / len(scores) if scores else 0,
            'max_vulnerability': max(scores) if scores else 0
        },
        'most_vulnerable': process_vulnerabilities[:5],
        'generated_at': datetime.now().isoformat()
    }


# ============================================================================
# PRIORITIZATION RECOMMENDATIONS
# ============================================================================

def generate_prioritization_recommendations(assessments: List[Dict],
                                             processes: List[Dict],
                                             risks: List[Dict]) -> Dict:
    """
    Generate AI-assisted prioritization recommendations.

    Returns actionable insights for risk management.
    """
    if not assessments:
        return {
            'recommendations': [],
            'summary': 'No assessments available for analysis'
        }

    # Score and rank all assessments
    ranked_assessments = rank_assessments_by_priority(assessments)

    # Get vulnerability map
    vuln_map = generate_vulnerability_map(processes, assessments)

    recommendations = []

    # 1. Critical priority items
    critical_items = [a for a in ranked_assessments if a['priority'] == 'Critical']
    if critical_items:
        recommendations.append({
            'type': 'critical_alert',
            'title': 'Critical Priority Items',
            'message': f'{len(critical_items)} process-risk combinations require immediate attention',
            'items': [
                {
                    'process': a.get('process_name', 'Unknown'),
                    'risk': a.get('risk_name', 'Unknown'),
                    'score': a['composite_score']
                }
                for a in critical_items[:5]
            ],
            'action': 'Review and implement mitigation strategies immediately'
        })

    # 2. Process concentration warnings
    concentrated_processes = [
        p for p in vuln_map['processes']
        if p['concentration_risk'] == 'High'
    ]
    if concentrated_processes:
        recommendations.append({
            'type': 'concentration_warning',
            'title': 'Risk Concentration Alert',
            'message': f'{len(concentrated_processes)} processes have concentrated risk exposure',
            'items': [
                {
                    'process': p['process_name'],
                    'note': p['concentration_note']
                }
                for p in concentrated_processes[:3]
            ],
            'action': 'Diversify risk mitigation strategies across domains'
        })

    # 3. High-value process protection
    high_value_vulnerable = [
        p for p in vuln_map['processes']
        if p['criticality'] > 10000 and p['vulnerability_score'] > 50
    ]
    if high_value_vulnerable:
        recommendations.append({
            'type': 'high_value_warning',
            'title': 'High-Value Process Protection',
            'message': f'{len(high_value_vulnerable)} critical business processes have elevated risk',
            'items': [
                {
                    'process': p['process_name'],
                    'daily_value': p['criticality'],
                    'vulnerability': p['vulnerability_score']
                }
                for p in high_value_vulnerable[:3]
            ],
            'action': 'Prioritize resilience improvements for these processes'
        })

    # 4. Quick wins (high impact, low effort)
    quick_wins = [
        a for a in ranked_assessments
        if a['breakdown']['resilience_gap'] > 15  # Low resilience
        and a['breakdown']['vulnerability'] > 15  # High vulnerability
        and a.get('resilience', 0.3) < 0.4  # Room for improvement
    ]
    if quick_wins:
        recommendations.append({
            'type': 'quick_wins',
            'title': 'Quick Win Opportunities',
            'message': f'{len(quick_wins)} items could benefit from resilience improvements',
            'items': [
                {
                    'process': a.get('process_name', 'Unknown'),
                    'risk': a.get('risk_name', 'Unknown'),
                    'current_resilience': f"{a.get('resilience', 0.3):.0%}"
                }
                for a in quick_wins[:5]
            ],
            'action': 'Increase resilience measures (backup systems, redundancy, training)'
        })

    # 5. Summary statistics
    total_exposure = sum(a.get('exposure', 0) for a in assessments)
    avg_score = sum(a['composite_score'] for a in ranked_assessments) / len(ranked_assessments)

    return {
        'recommendations': recommendations,
        'statistics': {
            'total_assessments': len(assessments),
            'total_exposure': total_exposure,
            'average_priority_score': round(avg_score, 1),
            'critical_count': len([a for a in ranked_assessments if a['priority'] == 'Critical']),
            'high_count': len([a for a in ranked_assessments if a['priority'] == 'High']),
            'medium_count': len([a for a in ranked_assessments if a['priority'] == 'Medium']),
            'low_count': len([a for a in ranked_assessments if a['priority'] == 'Low'])
        },
        'top_10_priorities': ranked_assessments[:10],
        'generated_at': datetime.now().isoformat()
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_priority_color(priority: str) -> str:
    """Get color for priority level."""
    colors = {
        'Critical': '#dc3545',
        'High': '#fd7e14',
        'Medium': '#ffc107',
        'Low': '#28a745'
    }
    return colors.get(priority, '#6c757d')


def get_priority_icon(priority: str) -> str:
    """Get icon for priority level."""
    icons = {
        'Critical': '🔴',
        'High': '🟠',
        'Medium': '🟡',
        'Low': '🟢'
    }
    return icons.get(priority, '⚪')
