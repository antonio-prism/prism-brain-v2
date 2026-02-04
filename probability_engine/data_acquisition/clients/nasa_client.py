"""
NASA Earthdata Client
https://www.earthdata.nasa.gov/

Satellite observations and Earth science data.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from ..base_client import BaseAPIClient, RateLimiter


class NASAClient(BaseAPIClient):
    """NASA Earthdata APIs. Authentication via Earthdata token."""

    def __init__(self, token: Optional[str] = None, rate_limiter: Optional[RateLimiter] = None, **kwargs):
        super().__init__(
            api_key=token,
            base_url="https://cmr.earthdata.nasa.gov/search",
            rate_limiter=rate_limiter,
            **kwargs
        )

    @property
    def source_name(self) -> str:
        return "NASA_EARTHDATA"

    def is_configured(self) -> bool:
        return True  # Can query without token for basic searches

    async def fetch_data(self, event_id: str, start_date: datetime, end_date: datetime,
                         keywords: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Fetch NASA dataset information (not actual satellite data)."""
        keywords = keywords or "climate disaster"
        cache_key = self._generate_cache_key(event_id, start_date.date(), end_date.date(), keywords)

        async def _fetch():
            params = {
                'keyword': keywords,
                'temporal': f"{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}",
                'page_size': 50,
                'format': 'json'
            }
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"

            try:
                response = await self._make_request("/collections.json", params=params, headers=headers)
                entries = response.get('feed', {}).get('entry', [])
                return {
                    'source': self.source_name,
                    'event_id': event_id,
                    'data': entries,
                    'metadata': {
                        'fetch_time': datetime.utcnow().isoformat(),
                        'dataset_count': len(entries),
                        'keywords': keywords
                    }
                }
            except Exception as e:
                return {'source': self.source_name, 'event_id': event_id, 'data': [], 'metadata': {'error': str(e)}}

        return await self.get_cached_or_fetch(cache_key, _fetch, ttl=86400)

    def extract_indicators(self, raw_data: Dict[str, Any], indicator_config: Dict[str, Any]) -> Dict[str, float]:
        """NASA data typically needs specialized processing. Returns basic metrics."""
        entries = raw_data.get('data', [])
        return {
            'dataset_count': float(len(entries)),
            'data_available': 1.0 if entries else 0.0
        }

    async def health_check(self) -> Dict[str, Any]:
        import time
        start = time.time()
        try:
            await self._make_request("/collections.json", params={'page_size': 1})
            return {'source': self.source_name, 'status': 'OPERATIONAL', 'configured': True,
                    'response_time_ms': int((time.time() - start) * 1000)}
        except Exception as e:
            return {'source': self.source_name, 'status': 'ERROR', 'error': str(e)}
