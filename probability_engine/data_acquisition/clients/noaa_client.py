"""
NOAA Climate Data Online Client

National Oceanic and Atmospheric Administration
https://www.ncdc.noaa.gov/cdo-web/

Historical weather and climate data.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from ..base_client import BaseAPIClient, RateLimiter


class NOAAClient(BaseAPIClient):
    """
    Client for NOAA Climate Data Online API.

    Rate Limit: 1000 requests/day
    Authentication: API token required (free)
    """

    DATASETS = {
        'GHCND': 'Daily summaries',
        'GSOM': 'Monthly summaries',
        'GSOY': 'Annual summaries',
        'NORMAL_DLY': 'Climate normals',
    }

    def __init__(self, api_token: str, rate_limiter: Optional[RateLimiter] = None, **kwargs):
        super().__init__(
            api_key=api_token,
            base_url="https://www.ncdc.noaa.gov/cdo-web/api/v2",
            rate_limiter=rate_limiter or RateLimiter(1000, 86400),
            **kwargs
        )

    @property
    def source_name(self) -> str:
        return "NOAA"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def fetch_data(
        self,
        event_id: str,
        start_date: datetime,
        end_date: datetime,
        location_id: str = "FIPS:US",
        datatypes: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if not self.is_configured():
            return {'source': self.source_name, 'event_id': event_id, 'data': [], 'metadata': {'error': 'Not configured'}}

        datatypes = datatypes or ['TMAX', 'TMIN', 'PRCP', 'SNOW']
        cache_key = self._generate_cache_key(event_id, start_date.date(), end_date.date(), location_id)

        async def _fetch():
            headers = {'token': self.api_key}
            params = {
                'datasetid': 'GHCND',
                'locationid': location_id,
                'startdate': start_date.strftime('%Y-%m-%d'),
                'enddate': end_date.strftime('%Y-%m-%d'),
                'datatypeid': ','.join(datatypes),
                'limit': 1000,
                'units': 'metric'
            }

            try:
                response = await self._make_request("/data", params=params, headers=headers)
                return {
                    'source': self.source_name,
                    'event_id': event_id,
                    'data': response.get('results', []),
                    'metadata': {
                        'fetch_time': datetime.utcnow().isoformat(),
                        'record_count': len(response.get('results', [])),
                        'location': location_id
                    }
                }
            except Exception as e:
                return {'source': self.source_name, 'event_id': event_id, 'data': [], 'metadata': {'error': str(e)}}

        return await self.get_cached_or_fetch(cache_key, _fetch, ttl=86400)

    def extract_indicators(self, raw_data: Dict[str, Any], indicator_config: Dict[str, Any]) -> Dict[str, float]:
        results = raw_data.get('data', [])
        if not results:
            return {'data_points': 0.0}

        # Aggregate by datatype
        by_type = {}
        for r in results:
            dtype = r.get('datatype')
            value = r.get('value')
            if dtype and value is not None:
                by_type.setdefault(dtype, []).append(value)

        indicators = {'data_points': float(len(results))}
        for dtype, values in by_type.items():
            indicators[f"{dtype.lower()}_avg"] = sum(values) / len(values)
            indicators[f"{dtype.lower()}_max"] = max(values)
            indicators[f"{dtype.lower()}_min"] = min(values)

        return indicators

    async def health_check(self) -> Dict[str, Any]:
        import time
        start = time.time()
        if not self.is_configured():
            return {'source': self.source_name, 'status': 'NOT_CONFIGURED', 'configured': False}
        try:
            headers = {'token': self.api_key}
            await self._make_request("/datasets", params={'limit': 1}, headers=headers)
            return {'source': self.source_name, 'status': 'OPERATIONAL', 'configured': True,
                    'response_time_ms': int((time.time() - start) * 1000)}
        except Exception as e:
            return {'source': self.source_name, 'status': 'ERROR', 'error': str(e)}
