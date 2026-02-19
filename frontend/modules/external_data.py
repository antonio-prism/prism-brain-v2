"""
PRISM Brain - External Data Fetching Module (Phase 4 Enhanced)
===============================================================
Fetches and caches external data for probability calculations.

Data Sources (Free APIs):
- OpenWeatherMap API (weather data) - Free tier: 1000 calls/day
- NewsAPI.org (news/incident data) - Free tier: 100 requests/day
- World Bank API (economic indicators) - Free, no key needed
- Simulated cyber/operational data (free alternatives limited)

Fallback: All sources gracefully fall back to simulated data if APIs fail.
"""

import json
import random
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

# Database imports
import sqlite3
from contextlib import contextmanager

# Get database path
APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "prism_brain.db"

# API Configuration - Users can set these via the UI or environment variables
API_CONFIG = {
    'openweathermap': {
        'base_url': 'https://api.openweathermap.org/data/2.5',
        'key_env': 'OPENWEATHERMAP_API_KEY',
        'free_tier_limit': 1000,  # calls per day
    },
    'newsapi': {
        'base_url': 'https://newsapi.org/v2',
        'key_env': 'NEWSAPI_API_KEY',
        'free_tier_limit': 100,  # requests per day
    },
    'worldbank': {
        'base_url': 'https://api.worldbank.org/v2',
        'key_env': None,  # No key needed
        'free_tier_limit': None,
    }
}

# Request timeout for API calls (seconds)
API_TIMEOUT = 10


@contextmanager
def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_external_data_tables():
    """Initialize external data tables if they don't exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Table for cached external data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS external_data_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                data_key TEXT NOT NULL,
                data_value TEXT NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                UNIQUE(source_type, data_key)
            )
        ''')

        # Table for data source configurations (enhanced for Phase 4)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT UNIQUE NOT NULL,
                source_type TEXT NOT NULL,
                api_endpoint TEXT,
                api_key TEXT,
                is_active BOOLEAN DEFAULT 1,
                refresh_interval_hours INTEGER DEFAULT 168,
                last_refresh TIMESTAMP,
                last_status TEXT DEFAULT 'pending',
                error_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table for API keys (secure storage)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service_name TEXT UNIQUE NOT NULL,
                api_key TEXT NOT NULL,
                is_valid BOOLEAN DEFAULT 1,
                last_validated TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table for historical data trends
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                risk_category TEXT NOT NULL,
                region TEXT DEFAULT 'global',
                period_start DATE NOT NULL,
                period_end DATE NOT NULL,
                incident_count INTEGER DEFAULT 0,
                severity_avg REAL DEFAULT 0,
                trend_direction TEXT DEFAULT 'stable',
                data_source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table for refresh schedule tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS refresh_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT UNIQUE NOT NULL,
                last_refresh TIMESTAMP,
                next_refresh TIMESTAMP,
                refresh_interval_hours INTEGER DEFAULT 168,
                auto_refresh BOOLEAN DEFAULT 1
            )
        ''')

        conn.commit()

        # Initialize default refresh schedules
        _init_default_schedules(cursor, conn)


def _init_default_schedules(cursor, conn):
    """Initialize default refresh schedules for each data source type."""
    default_schedules = [
        ('weather', 24),      # Daily
        ('news', 168),        # Weekly
        ('economic', 168),    # Weekly
        ('cyber', 24),        # Daily
        ('operational', 168)  # Weekly
    ]

    for source_type, interval in default_schedules:
        cursor.execute('''
            INSERT OR IGNORE INTO refresh_schedule (source_type, refresh_interval_hours)
            VALUES (?, ?)
        ''', (source_type, interval))

    conn.commit()


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def save_api_key(service_name: str, api_key: str) -> bool:
    """Save an API key for a service."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO api_keys (service_name, api_key, last_validated)
                VALUES (?, ?, datetime('now'))
            ''', (service_name, api_key))
            conn.commit()
            return True
    except Exception:
        return False


def get_api_key(service_name: str) -> Optional[str]:
    """
    Get API key for a service.
    Priority: 1) Streamlit secrets, 2) Database storage

    To use Streamlit secrets, add to your Streamlit Cloud app settings:
    [api_keys]
    openweathermap = "your-key-here"
    newsapi = "your-key-here"
    """
    # First, try Streamlit secrets (recommended for persistence)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'api_keys' in st.secrets:
            key = st.secrets['api_keys'].get(service_name)
            if key:
                return key
    except Exception:
        pass

    # Fall back to database storage
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT api_key FROM api_keys WHERE service_name = ? AND is_valid = 1',
                (service_name,)
            )
            row = cursor.fetchone()
            return row['api_key'] if row else None
    except Exception:
        return None


def validate_api_key(service_name: str, api_key: str) -> Dict:
    """Validate an API key by making a test request."""
    result = {'valid': False, 'message': '', 'service': service_name}

    try:
        if service_name == 'openweathermap':
            # Test OpenWeatherMap API
            url = f"{API_CONFIG['openweathermap']['base_url']}/weather"
            params = {'q': 'London', 'appid': api_key}
            response = requests.get(url, params=params, timeout=API_TIMEOUT)
            if response.status_code == 200:
                result['valid'] = True
                result['message'] = 'API key is valid'
            elif response.status_code == 401:
                result['message'] = 'Invalid API key'
            else:
                result['message'] = f'API error: {response.status_code}'

        elif service_name == 'newsapi':
            # Test NewsAPI
            url = f"{API_CONFIG['newsapi']['base_url']}/top-headlines"
            params = {'country': 'us', 'pageSize': 1, 'apiKey': api_key}
            response = requests.get(url, params=params, timeout=API_TIMEOUT)
            if response.status_code == 200:
                result['valid'] = True
                result['message'] = 'API key is valid'
            elif response.status_code == 401:
                result['message'] = 'Invalid API key'
            else:
                result['message'] = f'API error: {response.status_code}'

        else:
            result['message'] = 'Unknown service'

    except requests.exceptions.Timeout:
        result['message'] = 'Connection timeout'
    except requests.exceptions.RequestException as e:
        result['message'] = f'Connection error: {str(e)[:50]}'
    except Exception as e:
        result['message'] = f'Error: {str(e)[:50]}'

    # Update validation status in database
    if result['valid']:
        save_api_key(service_name, api_key)
    else:
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE api_keys SET is_valid = 0 WHERE service_name = ?',
                    (service_name,)
                )
                conn.commit()
        except Exception:
            pass

    return result


def get_api_status() -> Dict:
    """Get status of all configured APIs."""
    status = {
        'openweathermap': {'configured': False, 'status': 'not_configured', 'source': None},
        'newsapi': {'configured': False, 'status': 'not_configured', 'source': None},
        'worldbank': {'configured': True, 'status': 'available', 'source': 'free'}
    }

    for service in ['openweathermap', 'newsapi']:
        # Check Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'api_keys' in st.secrets:
                if st.secrets['api_keys'].get(service):
                    status[service]['configured'] = True
                    status[service]['status'] = 'configured'
                    status[service]['source'] = 'streamlit_secrets'
                    continue
        except Exception:
            pass

        # Check database
        key = get_api_key(service)
        if key:
            status[service]['configured'] = True
            status[service]['status'] = 'configured'
            status[service]['source'] = 'database'

    return status


# ============================================================================
# DATA SOURCE MANAGEMENT
# ============================================================================

def get_data_sources() -> List[Dict]:
    """Get all configured data sources."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM data_sources ORDER BY source_name')
            return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return []


def add_data_source(source_name: str, source_type: str,
                    api_endpoint: str = None, api_key: str = None,
                    refresh_interval_hours: int = 168) -> int:
    """Add a new data source configuration."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO data_sources
                (source_name, source_type, api_endpoint, api_key, refresh_interval_hours)
                VALUES (?, ?, ?, ?, ?)
            ''', (source_name, source_type, api_endpoint, api_key, refresh_interval_hours))
            conn.commit()
            return cursor.lastrowid
    except Exception:
        return -1


def toggle_data_source(source_id: int, is_active: bool):
    """Enable or disable a data source."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE data_sources SET is_active = ? WHERE id = ?',
                (is_active, source_id)
            )
            conn.commit()
    except Exception:
        pass


# ============================================================================
# CACHED DATA OPERATIONS
# ============================================================================

def get_cached_data(source_type: str, data_key: str) -> Optional[Dict]:
    """Get cached data if not expired."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    SELECT * FROM external_data_cache
                    WHERE source_type = ? AND data_key = ?
                    AND (expires_at IS NULL OR expires_at > datetime('now'))
                ''', (source_type, data_key))
            except Exception:
                cursor.execute('''
                    SELECT * FROM external_data_cache
                    WHERE source_name = ? AND data_key = ?
                    AND (expires_at IS NULL OR expires_at > datetime('now'))
                ''', (source_type, data_key))
            row = cursor.fetchone()
            if row:
                return {
                    'data': json.loads(row['data_value']),
                    'fetched_at': row['fetched_at'],
                    'expires_at': row['expires_at']
                }
    except Exception:
        pass
    return None


def save_cached_data(source_type: str, data_key: str, data_value: Any,
                     expires_hours: int = 168):
    """Save data to cache with expiration."""
    try:
        expires_at = datetime.now() + timedelta(hours=expires_hours)
        with get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO external_data_cache
                    (source_type, data_key, data_value, fetched_at, expires_at)
                    VALUES (?, ?, ?, datetime('now'), ?)
                ''', (source_type, data_key, json.dumps(data_value), expires_at.isoformat()))
            except Exception:
                cursor.execute('''
                    INSERT OR REPLACE INTO external_data_cache
                    (source_name, data_key, data_value, fetched_at, expires_at)
                    VALUES (?, ?, ?, datetime('now'), ?)
                ''', (source_type, data_key, json.dumps(data_value), expires_at.isoformat()))
            conn.commit()
    except Exception:
        pass


def clear_expired_cache():
    """Remove expired cache entries."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM external_data_cache
                WHERE expires_at IS NOT NULL AND expires_at < datetime('now')
            ''')
            conn.commit()
            return cursor.rowcount
    except Exception:
        return 0


# ============================================================================
# REAL API FETCHERS (with fallback to simulated)
# ============================================================================

def fetch_weather_data_real(region: str = "global") -> Optional[Dict]:
    """
    Fetch real weather data from OpenWeatherMap API.
    Returns None if API call fails (will fall back to simulated).
    """
    api_key = get_api_key('openweathermap')
    if not api_key:
        return None

    try:
        # Map region to city for API query
        region_cities = {
            'global': 'London',
            'europe': 'London',
            'north_america': 'New York',
            'asia': 'Tokyo',
            'south_america': 'Sao Paulo',
            'africa': 'Lagos',
            'oceania': 'Sydney'
        }
        city = region_cities.get(region.lower(), 'London')

        url = f"{API_CONFIG['openweathermap']['base_url']}/weather"
        params = {'q': city, 'appid': api_key, 'units': 'metric'}

        response = requests.get(url, params=params, timeout=API_TIMEOUT)

        if response.status_code == 200:
            weather_data = response.json()

            # Convert weather conditions to risk indicators
            temp = weather_data.get('main', {}).get('temp', 20)
            humidity = weather_data.get('main', {}).get('humidity', 50)
            wind_speed = weather_data.get('wind', {}).get('speed', 5)
            weather_id = weather_data.get('weather', [{}])[0].get('id', 800)

            # Calculate risk indicators from weather data
            flood_risk = min(0.7, humidity / 100 * 0.8) if humidity > 70 else 0.1
            storm_risk = min(0.6, wind_speed / 20) if wind_speed > 10 else 0.1
            extreme_heat = min(0.5, (temp - 30) / 20) if temp > 30 else 0.05
            wildfire_risk = min(0.4, (35 - humidity) / 50 * 0.6) if humidity < 35 and temp > 25 else 0.05

            return {
                'region': region,
                'city': city,
                'period': 'current',
                'indicators': {
                    'flood_risk': round(max(0.05, flood_risk), 2),
                    'storm_risk': round(max(0.05, storm_risk), 2),
                    'extreme_heat': round(max(0.05, extreme_heat), 2),
                    'drought_risk': round(max(0.05, 1 - humidity / 100 * 0.5), 2),
                    'wildfire_risk': round(max(0.05, wildfire_risk), 2)
                },
                'raw_data': {
                    'temperature': temp,
                    'humidity': humidity,
                    'wind_speed': wind_speed,
                    'weather_condition': weather_data.get('weather', [{}])[0].get('description', 'unknown')
                },
                'alerts_active': 1 if weather_id < 700 else 0,
                'seasonal_factor': 1.1 if temp > 25 or temp < 5 else 1.0,
                'data_quality': 'live_api',
                'source': 'OpenWeatherMap',
                'fetched_at': datetime.now().isoformat()
            }

    except Exception:
        pass

    return None


def fetch_news_data_real(risk_category: str, region: str = "global") -> Optional[Dict]:
    """
    Fetch real news data from NewsAPI.
    Returns None if API call fails (will fall back to simulated).
    """
    api_key = get_api_key('newsapi')
    if not api_key:
        return None

    try:
        # Map risk categories to search keywords
        category_keywords = {
            'Physical': 'natural disaster OR flood OR earthquake OR fire OR storm',
            'Structural': 'market crash OR supply chain OR bankruptcy OR regulation',
            'Operational': 'factory accident OR equipment failure OR labor strike OR recall',
            'Digital': 'cyber attack OR data breach OR ransomware OR hacking'
        }

        domain = risk_category.split('_')[0] if '_' in risk_category else risk_category
        keywords = category_keywords.get(domain, 'business risk')

        url = f"{API_CONFIG['newsapi']['base_url']}/everything"
        params = {
            'q': keywords,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 50,
            'apiKey': api_key
        }

        response = requests.get(url, params=params, timeout=API_TIMEOUT)

        if response.status_code == 200:
            news_data = response.json()
            total_results = news_data.get('totalResults', 0)
            articles = news_data.get('articles', [])

            # Analyze article sentiment/frequency for trend
            recent_count = len(articles)

            # Determine trend based on article count
            if recent_count > 40:
                trend = 'increasing'
                trend_pct = random.uniform(10, 25)
            elif recent_count > 20:
                trend = 'stable'
                trend_pct = random.uniform(-5, 10)
            else:
                trend = 'decreasing'
                trend_pct = random.uniform(-15, 0)

            return {
                'category': risk_category,
                'region': region,
                'period': 'last_30_days',
                'total_incidents': min(total_results, 500),
                'recent_articles': recent_count,
                'trend': trend,
                'trend_percentage': round(trend_pct, 1),
                'sample_headlines': [a.get('title', '')[:100] for a in articles[:3]],
                'data_quality': 'live_api',
                'source': 'NewsAPI',
                'fetched_at': datetime.now().isoformat()
            }

    except Exception:
        pass

    return None


def fetch_economic_data_real(region: str = "global") -> Optional[Dict]:
    """
    Fetch real economic data from World Bank API (free, no key needed).
    Returns None if API call fails (will fall back to simulated).
    """
    try:
        # Map region to World Bank country code
        region_codes = {
            'global': 'WLD',
            'europe': 'EUU',
            'north_america': 'NAC',
            'asia': 'EAS',
            'south_america': 'LCN',
            'africa': 'SSF'
        }
        country_code = region_codes.get(region.lower(), 'WLD')

        # Fetch GDP growth indicator
        url = f"{API_CONFIG['worldbank']['base_url']}/country/{country_code}/indicator/NY.GDP.MKTP.KD.ZG"
        params = {'format': 'json', 'per_page': 5, 'date': '2020:2024'}

        response = requests.get(url, params=params, timeout=API_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and data[1]:
                # Get most recent GDP growth value
                gdp_values = [d for d in data[1] if d.get('value') is not None]
                gdp_growth = gdp_values[0]['value'] if gdp_values else 2.5

                # Calculate other indicators based on GDP (simplified model)
                return {
                    'region': region,
                    'period': 'current_year',
                    'indicators': {
                        'gdp_growth': round(gdp_growth, 1),
                        'inflation_rate': round(abs(gdp_growth) * 1.2 + random.uniform(1, 3), 1),
                        'unemployment': round(max(3, 8 - gdp_growth + random.uniform(-1, 2)), 1),
                        'market_volatility': round(max(10, 25 - gdp_growth * 2 + random.uniform(-5, 5)), 1),
                        'supply_chain_stress': round(max(0.1, 0.3 - gdp_growth / 20), 2),
                        'currency_stability': round(min(1.0, 0.8 + gdp_growth / 20), 2)
                    },
                    'recession_probability': round(max(0.05, 0.3 - gdp_growth / 10), 2),
                    'market_sentiment': 'bullish' if gdp_growth > 2 else 'neutral' if gdp_growth > 0 else 'bearish',
                    'trade_disruption_risk': round(max(0.1, 0.25 - gdp_growth / 20), 2),
                    'data_quality': 'live_api',
                    'source': 'World Bank',
                    'fetched_at': datetime.now().isoformat()
                }

    except Exception:
        pass

    return None


# ============================================================================
# SIMULATED DATA FETCHERS (fallback)
# ============================================================================

def fetch_news_data_simulated(risk_category: str, region: str = "global") -> Dict:
    """Simulated news data (fallback when API unavailable)."""
    base_incidents = {
        'Physical': {'fire': 45, 'flood': 30, 'earthquake': 5, 'storm': 60},
        'Structural': {'market_crash': 8, 'supply_disruption': 25, 'regulatory': 40},
        'Operational': {'equipment_failure': 80, 'staff_shortage': 35, 'process_error': 50},
        'Digital': {'cyber_attack': 120, 'data_breach': 45, 'system_outage': 65}
    }

    domain = risk_category.split('_')[0] if '_' in risk_category else risk_category
    incidents = base_incidents.get(domain, {'general': 30})

    for key in incidents:
        incidents[key] = int(incidents[key] * (0.8 + random.random() * 0.4))

    trends = ['increasing', 'stable', 'decreasing']
    trend_weights = [0.35, 0.45, 0.20]

    return {
        'category': risk_category,
        'region': region,
        'period': 'last_12_months',
        'total_incidents': sum(incidents.values()),
        'incidents_by_type': incidents,
        'trend': random.choices(trends, weights=trend_weights)[0],
        'trend_percentage': round(random.uniform(-15, 25), 1),
        'data_quality': 'simulated',
        'source': 'PRISM Simulated',
        'fetched_at': datetime.now().isoformat()
    }


def fetch_weather_data_simulated(region: str = "global") -> Dict:
    """Simulated weather data (fallback when API unavailable)."""
    return {
        'region': region,
        'period': 'current_season',
        'indicators': {
            'flood_risk': round(random.uniform(0.1, 0.5), 2),
            'storm_risk': round(random.uniform(0.15, 0.45), 2),
            'extreme_heat': round(random.uniform(0.1, 0.3), 2),
            'drought_risk': round(random.uniform(0.05, 0.25), 2),
            'wildfire_risk': round(random.uniform(0.05, 0.35), 2)
        },
        'alerts_active': random.randint(0, 5),
        'seasonal_factor': round(random.uniform(0.8, 1.3), 2),
        'climate_trend': random.choice(['warming', 'variable', 'stable']),
        'data_quality': 'simulated',
        'source': 'PRISM Simulated',
        'fetched_at': datetime.now().isoformat()
    }


def fetch_economic_data_simulated(region: str = "global") -> Dict:
    """Simulated economic data (fallback when API unavailable)."""
    return {
        'region': region,
        'period': 'current_quarter',
        'indicators': {
            'gdp_growth': round(random.uniform(-2, 4), 1),
            'inflation_rate': round(random.uniform(1, 8), 1),
            'unemployment': round(random.uniform(3, 12), 1),
            'market_volatility': round(random.uniform(10, 35), 1),
            'supply_chain_stress': round(random.uniform(0.1, 0.6), 2),
            'currency_stability': round(random.uniform(0.7, 1.0), 2)
        },
        'recession_probability': round(random.uniform(0.05, 0.35), 2),
        'market_sentiment': random.choice(['bullish', 'neutral', 'bearish']),
        'trade_disruption_risk': round(random.uniform(0.1, 0.4), 2),
        'data_quality': 'simulated',
        'source': 'PRISM Simulated',
        'fetched_at': datetime.now().isoformat()
    }


def fetch_cyber_threat_data(industry: str = "general") -> Dict:
    """
    Fetch cybersecurity threat intelligence.
    Currently simulated (free cyber APIs are limited).
    """
    cache_key = f"cyber_{industry}"
    cached = get_cached_data("cyber", cache_key)
    if cached:
        return cached['data']

    threat_levels = {
        'ransomware': round(random.uniform(0.2, 0.6), 2),
        'phishing': round(random.uniform(0.3, 0.7), 2),
        'ddos': round(random.uniform(0.1, 0.4), 2),
        'data_exfiltration': round(random.uniform(0.15, 0.45), 2),
        'insider_threat': round(random.uniform(0.1, 0.3), 2),
        'supply_chain_attack': round(random.uniform(0.1, 0.35), 2)
    }

    data = {
        'industry': industry,
        'period': 'last_30_days',
        'threat_levels': threat_levels,
        'overall_threat_level': round(sum(threat_levels.values()) / len(threat_levels), 2),
        'active_campaigns': random.randint(2, 15),
        'new_vulnerabilities': random.randint(50, 200),
        'critical_vulnerabilities': random.randint(5, 25),
        'trend': random.choice(['increasing', 'stable', 'decreasing']),
        'top_threat_actors': ['APT Groups', 'Ransomware Gangs', 'Script Kiddies'][:random.randint(1, 3)],
        'data_quality': 'simulated',
        'source': 'PRISM Simulated',
        'fetched_at': datetime.now().isoformat()
    }

    save_cached_data("cyber", cache_key, data, expires_hours=24)
    return data


def fetch_operational_data(industry: str = "general") -> Dict:
    """
    Fetch operational risk indicators.
    Currently simulated.
    """
    cache_key = f"operational_{industry}"
    cached = get_cached_data("operational", cache_key)
    if cached:
        return cached['data']

    data = {
        'industry': industry,
        'period': 'current_year',
        'indicators': {
            'equipment_failure_rate': round(random.uniform(0.02, 0.15), 3),
            'staff_turnover': round(random.uniform(0.05, 0.25), 2),
            'process_error_rate': round(random.uniform(0.01, 0.08), 3),
            'compliance_issues': random.randint(0, 10),
            'safety_incidents': random.randint(0, 20),
            'quality_defects_ppm': random.randint(100, 5000)
        },
        'industry_benchmark': {
            'equipment_failure_rate': 0.05,
            'staff_turnover': 0.12,
            'process_error_rate': 0.03
        },
        'trend': random.choice(['improving', 'stable', 'deteriorating']),
        'data_quality': 'simulated',
        'source': 'PRISM Simulated',
        'fetched_at': datetime.now().isoformat()
    }

    save_cached_data("operational", cache_key, data, expires_hours=168)
    return data


# ============================================================================
# MAIN DATA FETCHERS (with automatic fallback)
# ============================================================================

def fetch_news_data(risk_category: str, region: str = "global") -> Dict:
    """
    Fetch news/incident data - tries real API first, falls back to simulated.
    """
    cache_key = f"{risk_category}_{region}"
    cached = get_cached_data("news", cache_key)
    if cached:
        return cached['data']

    # Try real API first
    data = fetch_news_data_real(risk_category, region)

    # Fall back to simulated if API fails
    if data is None:
        data = fetch_news_data_simulated(risk_category, region)

    # Cache the data
    save_cached_data("news", cache_key, data, expires_hours=168)
    return data


def fetch_weather_data(region: str = "global") -> Dict:
    """
    Fetch weather data - tries real API first, falls back to simulated.
    """
    cache_key = f"weather_{region}"
    cached = get_cached_data("weather", cache_key)
    if cached:
        return cached['data']

    # Try real API first
    data = fetch_weather_data_real(region)

    # Fall back to simulated if API fails
    if data is None:
        data = fetch_weather_data_simulated(region)

    save_cached_data("weather", cache_key, data, expires_hours=24)
    return data


def fetch_economic_data(region: str = "global") -> Dict:
    """
    Fetch economic data - tries real API first, falls back to simulated.
    """
    cache_key = f"economic_{region}"
    cached = get_cached_data("economic", cache_key)
    if cached:
        return cached['data']

    # Try real API first
    data = fetch_economic_data_real(region)

    # Fall back to simulated if API fails
    if data is None:
        data = fetch_economic_data_simulated(region)

    save_cached_data("economic", cache_key, data, expires_hours=168)
    return data


# ============================================================================
# AGGREGATED DATA FETCHING
# ============================================================================

def fetch_all_external_data(client_industry: str = "general",
                            client_region: str = "global") -> Dict:
    """
    Fetch all external data sources for a client.
    Returns aggregated data for probability calculations.
    """
    return {
        'news': {
            'physical': fetch_news_data('Physical', client_region),
            'structural': fetch_news_data('Structural', client_region),
            'operational': fetch_news_data('Operational', client_region),
            'digital': fetch_news_data('Digital', client_region)
        },
        'weather': fetch_weather_data(client_region),
        'economic': fetch_economic_data(client_region),
        'cyber': fetch_cyber_threat_data(client_industry),
        'operational': fetch_operational_data(client_industry),
        'metadata': {
            'fetched_at': datetime.now().isoformat(),
            'client_industry': client_industry,
            'client_region': client_region,
            'api_status': get_api_status()
        }
    }


def get_data_freshness() -> Dict:
    """Get information about how fresh the cached data is."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute('''
                    SELECT source_type as source,
                           COUNT(*) as entries,
                           MIN(fetched_at) as oldest,
                           MAX(fetched_at) as newest
                    FROM external_data_cache
                    WHERE expires_at IS NULL OR expires_at > datetime('now')
                    GROUP BY source_type
                ''')
            except Exception:
                cursor.execute('''
                    SELECT source_name as source,
                           COUNT(*) as entries,
                           MIN(fetched_at) as oldest,
                           MAX(fetched_at) as newest
                    FROM external_data_cache
                    WHERE expires_at IS NULL OR expires_at > datetime('now')
                    GROUP BY source_name
                ''')

            results = {}
            for row in cursor.fetchall():
                results[row['source']] = {
                    'entries': row['entries'],
                    'oldest': row['oldest'],
                    'newest': row['newest']
                }
            return results
    except Exception:
        return {}


def get_refresh_schedule() -> List[Dict]:
    """Get the refresh schedule for all data source types."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM refresh_schedule ORDER BY source_type')
            return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return []


def update_refresh_schedule(source_type: str, interval_hours: int, auto_refresh: bool = True):
    """Update the refresh schedule for a data source type."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE refresh_schedule
                SET refresh_interval_hours = ?, auto_refresh = ?
                WHERE source_type = ?
            ''', (interval_hours, auto_refresh, source_type))
            conn.commit()
    except Exception:
        pass


def refresh_all_data(client_industry: str = "general",
                     client_region: str = "global",
                     force: bool = False) -> Dict:
    """
    Refresh all external data sources.
    If force=True, ignores cache and fetches fresh data.
    """
    if force:
        # Clear cache to force refresh
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM external_data_cache')
                conn.commit()
        except Exception:
            pass

    # Fetch fresh data
    data = fetch_all_external_data(client_industry, client_region)

    # Update refresh timestamps
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for source_type in ['weather', 'news', 'economic', 'cyber', 'operational']:
                cursor.execute('''
                    UPDATE refresh_schedule
                    SET last_refresh = datetime('now'),
                        next_refresh = datetime('now', '+' || refresh_interval_hours || ' hours')
                    WHERE source_type = ?
                ''', (source_type,))
            conn.commit()
    except Exception:
        pass

    return {
        'success': True,
        'refreshed_at': datetime.now().isoformat(),
        'sources_refreshed': ['news', 'weather', 'economic', 'cyber', 'operational'],
        'api_status': get_api_status(),
        'data': data
    }


# NOTE: init_external_data_tables() is NOT called at import time.
# It is called lazily on first use by the Data Sources page.
# This avoids re-creating/checking tables on every Streamlit rerun.
_external_tables_initialized = False


def ensure_external_tables():
    """Initialize external data tables once, lazily."""
    global _external_tables_initialized
    if not _external_tables_initialized:
        try:
            init_external_data_tables()
            _external_tables_initialized = True
        except Exception as e:
            print(f"Warning: Could not initialize external data tables: {e}")
