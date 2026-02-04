"""
World Bank API Client

World Bank Open Data API
https://data.worldbank.org/

Provides economic indicators for 200+ countries.
No API key required.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List

from ..base_client import BaseAPIClient, RateLimiter


class WorldBankClient(BaseAPIClient):
    """
    Client for World Bank API.

    Rate Limit: No strict limit
    Authentication: None required
    """

    # Common indicators
    INDICATORS = {
        'gdp_growth': 'NY.GDP.MKTP.KD.ZG',          # GDP growth (annual %)
        'inflation': 'FP.CPI.TOTL.ZG',               # Inflation, consumer prices
        'unemployment': 'SL.UEM.TOTL.ZS',            # Unemployment rate
        'trade_balance': 'NE.RSB.GNFS.ZS',           # External balance
        'debt_gdp': 'GC.DOD.TOTL.GD.ZS',             # Central government debt
        'political_stability': 'PV.EST',              # Political stability index
        'governance': 'GE.EST',                       # Government effectiveness
        'rule_of_law': 'RL.EST',                      # Rule of law index
    }

    def __init__(self, rate_limiter: Optional[RateLimiter] = None, **kwargs):
        super().__init__(
            base_url="https://api.worldbank.org/v2",
            rate_limiter=rate_limiter,
            **kwargs
        )

    @property
    def source_name(self) -> str:
        return "WORLD_BANK"

    def is_configured(self) -> bool:
        return True  # No API key needed

    async def fetch_data(
        self,
        event_id: str,
        start_date: datetime,
        end_date: datetime,
        country_codes: Optional[List[str]] = None,
        indicators: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch economic indicators from World Bank.

        Args:
            event_id: PRISM event ID
            start_date: Start year
            end_date: End year
            country_codes: ISO country codes (default: world)
            indicators: List of indicator codes

        Returns:
            Standardized data dict
        """
        countries = country_codes or ['WLD']  # World aggregate
        indicator_codes = indicators or list(self.INDICATORS.values())[:3]

        cache_key = self._generate_cache_key(
            event_id,
            start_date.year,
            end_date.year,
            tuple(countries),
            tuple(indicator_codes)
        )

        async def _fetch():
            all_data = {}

            for indicator in indicator_codes:
                try:
                    url = f"/country/{';'.join(countries)}/indicator/{indicator}"
                    params = {
                        'format': 'json',
                        'date': f"{start_date.year}:{end_date.year}",
                        'per_page': 1000
                    }

                    response = await self._make_request(url, params=params)

                    # World Bank returns [metadata, data]
                    if isinstance(response, list) and len(response) > 1:
                        all_data[indicator] = response[1] or []
                    else:
                        all_data[indicator] = []

                except Exception as e:
                    self.logger.warning(f"Failed to fetch {indicator}: {e}")
                    all_data[indicator] = []

            return {
                'source': self.source_name,
                'event_id': event_id,
                'data': all_data,
                'metadata': {
                    'fetch_time': datetime.utcnow().isoformat(),
                    'indicators_fetched': len(all_data),
                    'countries': countries,
                    'date_range': [start_date.year, end_date.year]
                }
            }

        return await self.get_cached_or_fetch(cache_key, _fetch, ttl=86400)

    def extract_indicators(
        self,
        raw_data: Dict[str, Any],
        indicator_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract economic indicators from World Bank data.

        Returns most recent value for each indicator.
        """
        data = raw_data.get('data', {})
        results = {}

        for indicator_code, values in data.items():
            if not values:
                continue

            # Find indicator name
            indicator_name = None
            for name, code in self.INDICATORS.items():
                if code == indicator_code:
                    indicator_name = name
                    break
            indicator_name = indicator_name or indicator_code

            # Get most recent non-null value
            for entry in sorted(values, key=lambda x: x.get('date', '0'), reverse=True):
                value = entry.get('value')
                if value is not None:
                    results[indicator_name] = float(value)
                    break

        return results

    async def fetch_governance_indicators(
        self,
        country_code: str,
        year: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Fetch World Governance Indicators for a country.

        Returns:
            Dict with governance scores (-2.5 to 2.5 scale)
        """
        year = year or datetime.now().year - 1
        governance_indicators = [
            'CC.EST',  # Control of Corruption
            'GE.EST',  # Government Effectiveness
            'PV.EST',  # Political Stability
            'RQ.EST',  # Regulatory Quality
            'RL.EST',  # Rule of Law
            'VA.EST',  # Voice and Accountability
        ]

        result = await self.fetch_data(
            event_id='GOVERNANCE',
            start_date=datetime(year, 1, 1),
            end_date=datetime(year, 12, 31),
            country_codes=[country_code],
            indicators=governance_indicators
        )

        return self.extract_indicators(result, {})

    async def health_check(self) -> Dict[str, Any]:
        """Check World Bank API health."""
        import time
        start = time.time()

        try:
            response = await self._make_request(
                "/country/WLD/indicator/NY.GDP.MKTP.KD.ZG",
                params={'format': 'json', 'per_page': 1}
            )

            return {
                'source': self.source_name,
                'status': 'OPERATIONAL',
                'configured': True,
                'response_time_ms': int((time.time() - start) * 1000)
            }
        except Exception as e:
            return {
                'source': self.source_name,
                'status': 'ERROR',
                'error': str(e),
                'response_time_ms': int((time.time() - start) * 1000)
            }
