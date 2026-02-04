"""
IMF Client - International Monetary Fund Data API
https://data.imf.org/

International financial statistics and economic data.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from ..base_client import BaseAPIClient, RateLimiter


class IMFClient(BaseAPIClient):
    """No rate limit. No authentication required."""

    INDICATORS = {
        'gdp_current': 'NGDP_RPCH',  # Real GDP growth
        'inflation': 'PCPIPCH',       # Inflation rate
        'unemployment': 'LUR',         # Unemployment rate
        'current_account': 'BCA_NGDPD', # Current account balance
        'government_debt': 'GGXWDG_NGDP', # Government gross debt
    }

    def __init__(self, rate_limiter: Optional[RateLimiter] = None, **kwargs):
        super().__init__(
            base_url="https://www.imf.org/external/datamapper/api/v1",
            rate_limiter=rate_limiter,
            **kwargs
        )

    @property
    def source_name(self) -> str:
        return "IMF"

    async def fetch_data(self, event_id: str, start_date: datetime, end_date: datetime,
                         countries: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        countries = countries or ['USA', 'CHN', 'DEU', 'JPN', 'GBR']
        cache_key = self._generate_cache_key(event_id, start_date.year, end_date.year, tuple(countries))

        async def _fetch():
            all_data = {}
            for indicator_code in list(self.INDICATORS.values())[:3]:
                try:
                    response = await self._make_request(f"/{indicator_code}")
                    values = response.get('values', {}).get(indicator_code, {})
                    # Filter by countries
                    filtered = {c: values.get(c, {}) for c in countries if c in values}
                    all_data[indicator_code] = filtered
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {indicator_code}: {e}")

            return {
                'source': self.source_name,
                'event_id': event_id,
                'data': all_data,
                'metadata': {
                    'fetch_time': datetime.utcnow().isoformat(),
                    'indicators_fetched': len(all_data),
                    'countries': countries
                }
            }

        return await self.get_cached_or_fetch(cache_key, _fetch, ttl=86400)

    def extract_indicators(self, raw_data: Dict[str, Any], indicator_config: Dict[str, Any]) -> Dict[str, float]:
        data = raw_data.get('data', {})
        results = {}

        for indicator_code, country_data in data.items():
            name = next((k for k, v in self.INDICATORS.items() if v == indicator_code), indicator_code.lower())
            values = []
            for country, years in country_data.items():
                if years:
                    latest = max(years.keys())
                    val = years.get(latest)
                    if val is not None:
                        values.append(float(val))

            if values:
                results[f"{name}_avg"] = sum(values) / len(values)
                results[f"{name}_max"] = max(values)

        return results

    async def health_check(self) -> Dict[str, Any]:
        import time
        start = time.time()
        try:
            await self._make_request("/NGDP_RPCH")
            return {'source': self.source_name, 'status': 'OPERATIONAL', 'configured': True,
                    'response_time_ms': int((time.time() - start) * 1000)}
        except Exception as e:
            return {'source': self.source_name, 'status': 'ERROR', 'error': str(e)}
