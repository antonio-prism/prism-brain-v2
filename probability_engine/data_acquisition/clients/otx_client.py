"""
AlienVault OTX Client - Open Threat Exchange
https://otx.alienvault.com/

Cyber threat intelligence and indicators of compromise.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from ..base_client import BaseAPIClient, RateLimiter


class OTXClient(BaseAPIClient):
    """Rate Limit: 10000 requests/day. API key required (free)."""

    def __init__(self, api_key: str, rate_limiter: Optional[RateLimiter] = None, **kwargs):
        super().__init__(
            api_key=api_key,
            base_url="https://otx.alienvault.com/api/v1",
            rate_limiter=rate_limiter or RateLimiter(10000, 86400),
            **kwargs
        )

    @property
    def source_name(self) -> str:
        return "OTX"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def fetch_data(self, event_id: str, start_date: datetime, end_date: datetime,
                         pulse_tags: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        if not self.is_configured():
            return {'source': self.source_name, 'event_id': event_id, 'data': [], 'metadata': {'error': 'Not configured'}}

        tags = pulse_tags or ['malware', 'ransomware', 'apt']
        cache_key = self._generate_cache_key(event_id, start_date.date(), tuple(tags))

        async def _fetch():
            headers = {'X-OTX-API-KEY': self.api_key}
            # Get recent pulses (threat reports)
            params = {'limit': 50, 'modified_since': start_date.strftime('%Y-%m-%dT%H:%M:%S')}

            try:
                response = await self._make_request("/pulses/subscribed", params=params, headers=headers)
                pulses = response.get('results', [])

                # Filter by tags if specified
                if tags:
                    pulses = [p for p in pulses if any(t in p.get('tags', []) for t in tags)]

                return {
                    'source': self.source_name,
                    'event_id': event_id,
                    'data': pulses,
                    'metadata': {
                        'fetch_time': datetime.utcnow().isoformat(),
                        'pulse_count': len(pulses),
                        'tags_filtered': tags
                    }
                }
            except Exception as e:
                return {'source': self.source_name, 'event_id': event_id, 'data': [], 'metadata': {'error': str(e)}}

        return await self.get_cached_or_fetch(cache_key, _fetch, ttl=3600)

    def extract_indicators(self, raw_data: Dict[str, Any], indicator_config: Dict[str, Any]) -> Dict[str, float]:
        pulses = raw_data.get('data', [])
        if not pulses:
            return {'pulse_count': 0.0, 'indicator_count': 0.0, 'threat_level': 0.0}

        total_indicators = sum(len(p.get('indicators', [])) for p in pulses)

        # Estimate threat level based on activity
        threat_level = min(1.0, len(pulses) / 20)  # Normalize to 0-1

        return {
            'pulse_count': float(len(pulses)),
            'indicator_count': float(total_indicators),
            'threat_level': threat_level,
            'avg_indicators_per_pulse': total_indicators / len(pulses) if pulses else 0.0
        }

    async def health_check(self) -> Dict[str, Any]:
        import time
        start = time.time()
        if not self.is_configured():
            return {'source': self.source_name, 'status': 'NOT_CONFIGURED', 'configured': False}
        try:
            headers = {'X-OTX-API-KEY': self.api_key}
            await self._make_request("/user/me", headers=headers)
            return {'source': self.source_name, 'status': 'OPERATIONAL', 'configured': True,
                    'response_time_ms': int((time.time() - start) * 1000)}
        except Exception as e:
            return {'source': self.source_name, 'status': 'ERROR', 'error': str(e)}
