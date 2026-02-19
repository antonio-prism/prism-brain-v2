"""
PRISM Brain - Probability Calculation Engine
=============================================
Converts external data into probability scores (0-1) for each risk.

The engine uses a weighted multi-factor approach:
- Historical Frequency (30%): How often has this risk occurred?
- Trend Direction (25%): Is the risk increasing or decreasing?
- Current Conditions (25%): What do current indicators show?
- Geographic/Industry Exposure (20%): Client-specific factors

Each factor produces a score from 0-1, which are then weighted
and combined to produce a final probability.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Import external data module
from modules.external_data import (
    fetch_all_external_data,
    fetch_news_data,
    fetch_weather_data,
    fetch_economic_data,
    fetch_cyber_threat_data,
    fetch_operational_data,
    get_db_connection
)

# Load risk database for reference
APP_DIR = Path(__file__).parent.parent
RISK_DB_PATH = APP_DIR / "data" / "risk_database.json"


def load_risk_database() -> List[Dict]:
    """Load the risk database."""
    try:
        with open(RISK_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading risk database: {e}")
        return []


# ============================================================================
# PROBABILITY FACTOR WEIGHTS
# ============================================================================

FACTOR_WEIGHTS = {
    'historical_frequency': 0.30,
    'trend_direction': 0.25,
    'current_conditions': 0.25,
    'exposure_factor': 0.20
}


# ============================================================================
# RISK CATEGORY MAPPINGS
# ============================================================================

# Map risk keywords to external data sources
# NOTE: Keys use UPPERCASE to match risk database Layer_1_Primary values
RISK_DATA_MAPPINGS = {
    'PHYSICAL': {
        'data_sources': ['weather', 'news'],
        'keywords': {
            'fire': ['wildfire_risk', 'extreme_heat'],
            'flood': ['flood_risk'],
            'earthquake': ['seismic'],
            'storm': ['storm_risk'],
            'weather': ['flood_risk', 'storm_risk', 'extreme_heat'],
            'natural': ['flood_risk', 'storm_risk', 'wildfire_risk', 'drought_risk'],
            'pandemic': ['health_emergency'],
            'accident': ['safety_incidents']
        }
    },
    'STRUCTURAL': {
        'data_sources': ['economic', 'news'],
        'keywords': {
            'market': ['market_volatility', 'recession_probability'],
            'economic': ['gdp_growth', 'inflation_rate', 'recession_probability'],
            'supply': ['supply_chain_stress', 'trade_disruption_risk'],
            'regulatory': ['compliance_issues'],
            'financial': ['market_volatility', 'currency_stability'],
            'currency': ['currency_stability'],
            'inflation': ['inflation_rate'],
            'trade': ['trade_disruption_risk']
        }
    },
    'OPERATIONAL': {
        'data_sources': ['operational', 'news'],
        'keywords': {
            'equipment': ['equipment_failure_rate'],
            'staff': ['staff_turnover'],
            'labor': ['staff_turnover', 'unemployment'],
            'process': ['process_error_rate'],
            'quality': ['quality_defects_ppm'],
            'safety': ['safety_incidents'],
            'compliance': ['compliance_issues'],
            'human': ['process_error_rate', 'staff_turnover']
        }
    },
    'DIGITAL': {
        'data_sources': ['cyber', 'news'],
        'keywords': {
            'cyber': ['ransomware', 'phishing', 'ddos', 'data_exfiltration'],
            'ransomware': ['ransomware'],
            'phishing': ['phishing'],
            'malware': ['ransomware', 'data_exfiltration'],
            'breach': ['data_exfiltration'],
            'ddos': ['ddos'],
            'hacking': ['ransomware', 'data_exfiltration'],
            'data': ['data_exfiltration'],
            'insider': ['insider_threat'],
            'system': ['ddos', 'critical_vulnerabilities']
        }
    }
}


# ============================================================================
# PROBABILITY CALCULATION FUNCTIONS
# ============================================================================

def calculate_historical_frequency_score(risk: Dict, external_data: Dict) -> float:
    """
    Calculate probability based on historical incident frequency.
    Returns a score from 0 to 1.
    """
    # Normalize domain to uppercase to match database values
    domain = risk.get('domain', 'OPERATIONAL').upper()
    risk_name = risk.get('risk_name', '').lower()

    # Get news data for the domain
    news_data = external_data.get('news', {}).get(domain.lower(), {})
    total_incidents = news_data.get('total_incidents', 50)

    # Base frequency score (normalized)
    # Assume max 500 incidents per year as upper bound
    base_score = min(total_incidents / 500, 1.0)

    # Adjust based on specific risk keywords
    keyword_boost = 0
    incidents_by_type = news_data.get('incidents_by_type', {})

    for keyword, incident_type in [
        ('fire', 'fire'), ('flood', 'flood'), ('cyber', 'cyber_attack'),
        ('equipment', 'equipment_failure'), ('market', 'market_crash')
    ]:
        if keyword in risk_name:
            type_incidents = incidents_by_type.get(incident_type, 0)
            keyword_boost = max(keyword_boost, type_incidents / 100)

    # Combine base score with keyword boost
    final_score = min(base_score * 0.7 + keyword_boost * 0.3, 1.0)

    return round(final_score, 3)


def calculate_trend_score(risk: Dict, external_data: Dict) -> float:
    """
    Calculate probability adjustment based on trend direction.
    Returns a score from 0 to 1.
    """
    # Normalize domain to uppercase to match database values
    domain = risk.get('domain', 'OPERATIONAL').upper()
    news_data = external_data.get('news', {}).get(domain.lower(), {})

    trend = news_data.get('trend', 'stable')
    trend_percentage = news_data.get('trend_percentage', 0)

    # Base score for each trend
    trend_scores = {
        'increasing': 0.7,
        'stable': 0.5,
        'decreasing': 0.3
    }

    base_score = trend_scores.get(trend, 0.5)

    # Adjust based on percentage change
    if trend == 'increasing':
        adjustment = min(trend_percentage / 50, 0.3)  # Up to +0.3 for 50%+ increase
        final_score = base_score + adjustment
    elif trend == 'decreasing':
        adjustment = min(abs(trend_percentage) / 50, 0.2)  # Up to -0.2 for 50%+ decrease
        final_score = base_score - adjustment
    else:
        final_score = base_score

    return round(max(0, min(final_score, 1.0)), 3)


def calculate_current_conditions_score(risk: Dict, external_data: Dict) -> float:
    """
    Calculate probability based on current environmental conditions.
    Returns a score from 0 to 1.
    """
    # Normalize domain to uppercase to match database values
    domain = risk.get('domain', 'OPERATIONAL').upper()
    risk_name = risk.get('risk_name', '').lower()

    score = 0.5  # Default neutral score

    if domain == 'PHYSICAL':
        weather = external_data.get('weather', {})
        indicators = weather.get('indicators', {})

        # Check relevant weather indicators
        relevant_scores = []
        if 'fire' in risk_name or 'wildfire' in risk_name:
            relevant_scores.append(indicators.get('wildfire_risk', 0.2))
            relevant_scores.append(indicators.get('extreme_heat', 0.2))
        if 'flood' in risk_name:
            relevant_scores.append(indicators.get('flood_risk', 0.2))
        if 'storm' in risk_name or 'hurricane' in risk_name or 'tornado' in risk_name:
            relevant_scores.append(indicators.get('storm_risk', 0.2))
        if 'drought' in risk_name:
            relevant_scores.append(indicators.get('drought_risk', 0.2))

        if relevant_scores:
            score = sum(relevant_scores) / len(relevant_scores)
        else:
            # Generic physical risk - average all indicators
            if indicators:
                score = sum(indicators.values()) / len(indicators)

        # Apply seasonal factor
        seasonal_factor = weather.get('seasonal_factor', 1.0)
        score = score * seasonal_factor

    elif domain == 'STRUCTURAL':
        economic = external_data.get('economic', {})
        indicators = economic.get('indicators', {})

        if 'market' in risk_name or 'financial' in risk_name:
            volatility = indicators.get('market_volatility', 20) / 100
            recession = economic.get('recession_probability', 0.15)
            score = (volatility + recession) / 2

        elif 'supply' in risk_name:
            score = indicators.get('supply_chain_stress', 0.3)

        elif 'inflation' in risk_name or 'economic' in risk_name:
            inflation = indicators.get('inflation_rate', 3) / 20  # Normalize to 0-1
            gdp = (5 - indicators.get('gdp_growth', 2)) / 10  # Lower GDP = higher risk
            score = (inflation + max(0, gdp)) / 2

        elif 'currency' in risk_name:
            score = 1 - indicators.get('currency_stability', 0.85)

        elif 'trade' in risk_name:
            score = economic.get('trade_disruption_risk', 0.2)

        else:
            # Generic structural risk
            score = economic.get('recession_probability', 0.15)

    elif domain == 'DIGITAL':
        cyber = external_data.get('cyber', {})
        threat_levels = cyber.get('threat_levels', {})

        if 'ransomware' in risk_name:
            score = threat_levels.get('ransomware', 0.3)
        elif 'phishing' in risk_name:
            score = threat_levels.get('phishing', 0.4)
        elif 'ddos' in risk_name:
            score = threat_levels.get('ddos', 0.2)
        elif 'breach' in risk_name or 'data' in risk_name:
            score = threat_levels.get('data_exfiltration', 0.25)
        elif 'insider' in risk_name:
            score = threat_levels.get('insider_threat', 0.15)
        else:
            # Generic cyber risk - use overall threat level
            score = cyber.get('overall_threat_level', 0.3)

    elif domain == 'OPERATIONAL':
        operational = external_data.get('operational', {})
        indicators = operational.get('indicators', {})

        if 'equipment' in risk_name or 'machine' in risk_name:
            score = indicators.get('equipment_failure_rate', 0.05) * 5  # Scale up
        elif 'staff' in risk_name or 'labor' in risk_name or 'employee' in risk_name:
            score = indicators.get('staff_turnover', 0.12)
        elif 'process' in risk_name or 'error' in risk_name:
            score = indicators.get('process_error_rate', 0.03) * 10  # Scale up
        elif 'compliance' in risk_name:
            score = min(indicators.get('compliance_issues', 2) / 10, 1.0)
        elif 'safety' in risk_name:
            score = min(indicators.get('safety_incidents', 5) / 20, 1.0)
        else:
            # Generic operational risk
            if indicators:
                score = sum([
                    indicators.get('equipment_failure_rate', 0.05) * 3,
                    indicators.get('staff_turnover', 0.12),
                    indicators.get('process_error_rate', 0.03) * 5
                ]) / 3

    return round(max(0, min(score, 1.0)), 3)


def calculate_exposure_factor(risk: Dict, client_data: Dict) -> float:
    """
    Calculate client-specific exposure factor based on industry and region.
    Returns a score from 0 to 1.
    """
    industry = client_data.get('industry', 'general').lower()
    region = client_data.get('region', 'global').lower()
    # Normalize domain to uppercase to match database values
    domain = risk.get('domain', 'OPERATIONAL').upper()
    risk_name = risk.get('risk_name', '').lower()

    # Industry exposure multipliers (keys use UPPERCASE to match database)
    industry_exposure = {
        'technology': {'DIGITAL': 1.3, 'OPERATIONAL': 0.9, 'PHYSICAL': 0.8, 'STRUCTURAL': 1.0},
        'manufacturing': {'DIGITAL': 0.9, 'OPERATIONAL': 1.3, 'PHYSICAL': 1.2, 'STRUCTURAL': 1.0},
        'finance': {'DIGITAL': 1.4, 'OPERATIONAL': 0.8, 'PHYSICAL': 0.7, 'STRUCTURAL': 1.3},
        'healthcare': {'DIGITAL': 1.2, 'OPERATIONAL': 1.2, 'PHYSICAL': 0.9, 'STRUCTURAL': 1.0},
        'retail': {'DIGITAL': 1.1, 'OPERATIONAL': 1.1, 'PHYSICAL': 1.0, 'STRUCTURAL': 1.1},
        'energy': {'DIGITAL': 1.0, 'OPERATIONAL': 1.2, 'PHYSICAL': 1.3, 'STRUCTURAL': 1.1},
        'logistics': {'DIGITAL': 0.9, 'OPERATIONAL': 1.3, 'PHYSICAL': 1.2, 'STRUCTURAL': 1.1},
        'general': {'DIGITAL': 1.0, 'OPERATIONAL': 1.0, 'PHYSICAL': 1.0, 'STRUCTURAL': 1.0}
    }

    # Regional exposure multipliers (for physical/structural risks)
    region_exposure = {
        'north america': {'earthquake': 1.2, 'hurricane': 1.3, 'tornado': 1.4, 'flood': 1.1},
        'europe': {'earthquake': 0.8, 'hurricane': 0.6, 'flood': 1.2, 'regulatory': 1.3},
        'asia': {'earthquake': 1.4, 'typhoon': 1.3, 'flood': 1.3, 'tsunami': 1.2},
        'global': {'earthquake': 1.0, 'flood': 1.0, 'storm': 1.0}
    }

    # Base exposure from industry
    industry_multipliers = industry_exposure.get(industry, industry_exposure['general'])
    base_exposure = industry_multipliers.get(domain, 1.0)

    # Regional adjustment for physical risks
    regional_adjustment = 1.0
    if domain == 'PHYSICAL':
        region_factors = region_exposure.get(region, region_exposure['global'])
        for keyword, factor in region_factors.items():
            if keyword in risk_name:
                regional_adjustment = factor
                break

    # Calculate final exposure (normalize to 0-1 range)
    raw_exposure = base_exposure * regional_adjustment
    normalized_exposure = (raw_exposure - 0.5) / 1.5  # Map 0.5-2.0 to 0-1

    return round(max(0, min(normalized_exposure + 0.5, 1.0)), 3)


def calculate_risk_probability(risk: Dict, external_data: Dict,
                                client_data: Dict = None) -> Dict:
    """
    Calculate the overall probability for a single risk.

    Returns a dictionary with:
    - probability: Final probability score (0-1)
    - factors: Individual factor scores
    - confidence: Data confidence level
    """
    if client_data is None:
        client_data = {'industry': 'general', 'region': 'global'}

    # Calculate each factor
    historical = calculate_historical_frequency_score(risk, external_data)
    trend = calculate_trend_score(risk, external_data)
    conditions = calculate_current_conditions_score(risk, external_data)
    exposure = calculate_exposure_factor(risk, client_data)

    # Apply weights
    weighted_score = (
        historical * FACTOR_WEIGHTS['historical_frequency'] +
        trend * FACTOR_WEIGHTS['trend_direction'] +
        conditions * FACTOR_WEIGHTS['current_conditions'] +
        exposure * FACTOR_WEIGHTS['exposure_factor']
    )

    # Determine confidence based on data quality
    data_quality = external_data.get('metadata', {}).get('data_quality', 'simulated')
    confidence = 0.7 if data_quality == 'simulated' else 0.9

    return {
        'probability': round(weighted_score, 3),
        'factors': {
            'historical_frequency': historical,
            'trend_direction': trend,
            'current_conditions': conditions,
            'exposure_factor': exposure
        },
        'weights': FACTOR_WEIGHTS,
        'confidence': confidence,
        'calculated_at': datetime.now().isoformat()
    }


def calculate_all_probabilities(risks: List[Dict], client_data: Dict = None,
                                 force_refresh: bool = False) -> Dict:
    """
    Calculate probabilities for all risks.

    Returns a dictionary mapping risk IDs to probability data.
    """
    if client_data is None:
        client_data = {'industry': 'general', 'region': 'global'}

    # Fetch external data
    external_data = fetch_all_external_data(
        client_industry=client_data.get('industry', 'general'),
        client_region=client_data.get('region', 'global')
    )

    results = {}
    for risk in risks:
        risk_id = risk.get('id', risk.get('risk_name', 'unknown'))
        results[risk_id] = calculate_risk_probability(risk, external_data, client_data)

    return {
        'probabilities': results,
        'external_data_summary': {
            'fetched_at': external_data.get('metadata', {}).get('fetched_at'),
            'data_quality': external_data.get('metadata', {}).get('data_quality', 'simulated')
        },
        'total_risks_calculated': len(results)
    }


def update_risk_probabilities_in_db(client_id: int, probabilities: Dict):
    """
    Update risk probabilities in the database for a client.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        for risk_id, prob_data in probabilities.get('probabilities', {}).items():
            cursor.execute('''
                UPDATE client_risks
                SET probability = ?,
                    probability_factors = ?,
                    probability_updated = datetime('now')
                WHERE client_id = ? AND risk_id = ?
            ''', (
                prob_data['probability'],
                json.dumps(prob_data['factors']),
                client_id,
                risk_id
            ))

        conn.commit()


def get_probability_summary(probabilities: Dict) -> Dict:
    """
    Get summary statistics for calculated probabilities.
    """
    probs = [p['probability'] for p in probabilities.get('probabilities', {}).values()]

    if not probs:
        return {'error': 'No probabilities calculated'}

    return {
        'total_risks': len(probs),
        'average_probability': round(sum(probs) / len(probs), 3),
        'max_probability': round(max(probs), 3),
        'min_probability': round(min(probs), 3),
        'high_risk_count': len([p for p in probs if p >= 0.7]),
        'medium_risk_count': len([p for p in probs if 0.3 <= p < 0.7]),
        'low_risk_count': len([p for p in probs if p < 0.3])
    }


# ============================================================================
# PROBABILITY EXPLANATION
# ============================================================================

def explain_probability(risk: Dict, prob_data: Dict) -> str:
    """
    Generate a human-readable explanation of the probability calculation.
    """
    factors = prob_data.get('factors', {})
    weights = prob_data.get('weights', FACTOR_WEIGHTS)

    explanation = f"""
**Probability Calculation for: {risk.get('risk_name', 'Unknown Risk')}**

Final Probability: **{prob_data.get('probability', 0):.1%}**

**Factor Breakdown:**

1. **Historical Frequency** (Weight: {weights['historical_frequency']:.0%})
   - Score: {factors.get('historical_frequency', 0):.1%}
   - Based on: Past incident frequency in this risk category

2. **Trend Direction** (Weight: {weights['trend_direction']:.0%})
   - Score: {factors.get('trend_direction', 0):.1%}
   - Based on: Whether incidents are increasing or decreasing

3. **Current Conditions** (Weight: {weights['current_conditions']:.0%})
   - Score: {factors.get('current_conditions', 0):.1%}
   - Based on: Real-time environmental and market indicators

4. **Exposure Factor** (Weight: {weights['exposure_factor']:.0%})
   - Score: {factors.get('exposure_factor', 0):.1%}
   - Based on: Client industry and regional risk exposure

**Data Confidence:** {prob_data.get('confidence', 0):.0%}
**Last Updated:** {prob_data.get('calculated_at', 'Unknown')}
"""
    return explanation
