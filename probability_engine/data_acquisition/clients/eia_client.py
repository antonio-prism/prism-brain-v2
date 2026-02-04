"""
EIA API Client - U.S. Energy Information Administration
https://www.eia.gov/opendata/

Energy prices, production, and consumption data.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from ..base_client import BaseAPIClient, RateLimiter


class EIAClient(BaseAPIClient):
    """Rate Limit: 5000 requests/hour. API key required (free)."""

    SERIES = {
        'oil_wti': 'PET.RWTC.D',
        'oil_brent': 'PET.RBRTE.D',
        'natural_gas': 'NG.RNGWHHD.D',
        'gasoline': 'PET.EMM_EPMR_PTE_NUS_DPG.W',
        'coal': 'COAL.PRODUCTION.TOT-US-TOT.Q',
    }

    def __init__(self, api_key: str, rate_limiter: Optional[RateLimiter] = None, **kwargs):
        super().__init__(
            api_key=api_key,
            base_url="https://api.eia.gov/v2",
            rate_limiter=rate_limiter or RateLimiter(5000, 3600),
            **kwargs
        )

    @property
    def source_name(self) -> str:
        return "EIA"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def fetch_data(self, event_id: str, start_date: datetime, end_date: datetime,
                         series_ids: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        if not self.is_configured():
            return {'source': self.source_name, 'event_id': event_id, 'data': {}, 'metadata': {'error': 'Not configured'}}

        series = series_ids or ['PET.RWTC.D', 'NG.RNGWHHD.D']
        cache_key = self._generate_cache_key(event_id, start_date.date(), end_date.date(), tuple(series))

        async def _fetch():
            all_data = {}
            for series_id in series:
                try:
                    # EIA v2 API structure
                    route = series_id.replace('.', '/')
                    params = {
                        'api_key': self.api_key,
                        'start': start_date.strftime('%Y-%m-%d'),
                        'end': end_date.strftime('%Y-%m-%d'),
                        'frequency': 'daily'
                    }
                    response = await self._make_request(f"/seriesid/{series_id}", params=params)
                    all_data[series_id] = response.get('response', {}).get('data', [])
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {series_id}: {e}")
                    all_data[series_id] = []

            return {
                'source': self.source_name,
                'event_id': event_id,
                'data': all_data,
                'metadata': {'fetch_time': datetime.utcnow().isoformat(), 'series_fetched': len(all_data)}
            }

        return await self.get_cached_or_fetch(cache_key, _fetch, ttl=3600)

    def extract_indicators(self, raw_data: Dict[str, Any], indicator_config: Dict[str, Any]) -> Dict[str, float]:
        data = raw_data.get('data', {})
        results = {}
        for series_id, values in data.items():
            name = series_id.split('.')[-2].lower() if '.' in series_id else series_id.lower()
            if values:
                latest = values[0].get('value') if isinstance(values[0], dict) else values[0]
                if latest is not None:
                    results[f"{name}_price"] = float(latest)
        return results

    async def health_check(self) -> Dict[str, Any]:
        import time
        start = time.time()
        if not self.is_configured():
            return {'source': self.source_name, 'status': 'NOT_CONFIGURED', 'configured': False}
        try:
            await self._make_request("/seriesid/PET.RWTC.D", params={'api_key': self.api_key, 'length': 1})
            return {'source': self.source_name, 'status': 'OPERATIONAL', 'configured': True,
                    'response_time_ms': int((time.time() - start) * 1000)}
        except Exception as e:
            return {'source': self.source_name, 'status': 'ERROR', 'error': str(e)}
