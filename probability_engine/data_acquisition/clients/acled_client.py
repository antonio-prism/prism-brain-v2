"""
ACLED API Client

Armed Conflict Location & Event Data Project
https://acleddata.com/

Provides real-time conflict data for 200+ countries.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
import aiohttp

from ..base_client import BaseAPIClient, RateLimiter


class ACLEDClient(BaseAPIClient):
    """
    Client for ACLED API.

    Rate Limit: 500 requests/hour (free tier)
    Authentication: API key + email required
    """

    def __init__(
        self,
        api_key: str,
        email: str,
        rate_limiter: Optional[RateLimiter] = None,
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            base_url="https://api.acleddata.com/acled/read",
            rate_limiter=rate_limiter or RateLimiter(500, 3600),
            **kwargs
        )
        self.email = email

    @property
    def source_name(self) -> str:
        return "ACLED"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.email)

    async def fetch_data(
        self,
        event_id: str,
        start_date: datetime,
        end_date: datetime,
        countries: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch conflict events from ACLED.

        Args:
            event_id: PRISM event ID
            start_date: Start of date range
            end_date: End of date range
            countries: List of countries to filter
            event_types: List of ACLED event types

        Returns:
            Standardized data dict
        """
        if not self.is_configured():
            return self._empty_response(event_id, "API not configured")

        # Default event types for conflict monitoring
        if event_types is None:
            event_types = [
                'Battles',
                'Violence against civilians',
                'Explosions/Remote violence',
                'Protests',
                'Riots',
                'Strategic developments'
            ]

        cache_key = self._generate_cache_key(
            event_id,
            start_date.date(),
            end_date.date(),
            tuple(countries or []),
            tuple(event_types)
        )

        async def _fetch():
            params = {
                'key': self.api_key,
                'email': self.email,
                'event_date': f"{start_date.strftime('%Y-%m-%d')}|{end_date.strftime('%Y-%m-%d')}",
                'event_date_where': 'BETWEEN',
                'limit': 5000,
                'fields': 'event_id_cnty|event_date|event_type|sub_event_type|actor1|actor2|country|region|latitude|longitude|fatalities|notes'
            }

            if countries:
                params['country'] = '|'.join(countries)
            if event_types:
                params['event_type'] = '|'.join(event_types)

            data = await self._make_request(self.base_url, params=params)

            return {
                'source': self.source_name,
                'event_id': event_id,
                'data': data.get('data', []),
                'metadata': {
                    'fetch_time': datetime.utcnow().isoformat(),
                    'record_count': len(data.get('data', [])),
                    'date_range': [start_date.date().isoformat(), end_date.date().isoformat()],
                    'success': data.get('success', False)
                }
            }

        return await self.get_cached_or_fetch(cache_key, _fetch, ttl=3600)

    def extract_indicators(
        self,
        raw_data: Dict[str, Any],
        indicator_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract conflict indicators from ACLED data.

        Returns:
            - conflict_event_count: Total events
            - fatalities_total: Sum of fatalities
            - violence_intensity: Violence events per day
            - protest_frequency: Protests per day
        """
        events = raw_data.get('data', [])
        if not events:
            return self._default_indicators()

        # Count by type
        event_types = {}
        fatalities = 0
        for event in events:
            etype = event.get('event_type', 'Unknown')
            event_types[etype] = event_types.get(etype, 0) + 1
            fatalities += int(event.get('fatalities', 0) or 0)

        # Calculate time span
        date_range = raw_data.get('metadata', {}).get('date_range', [])
        if len(date_range) == 2:
            start = datetime.fromisoformat(date_range[0])
            end = datetime.fromisoformat(date_range[1])
            days = max((end - start).days, 1)
        else:
            days = 90  # Default

        # Violence events (battles + violence against civilians + explosions)
        violence_events = (
            event_types.get('Battles', 0) +
            event_types.get('Violence against civilians', 0) +
            event_types.get('Explosions/Remote violence', 0)
        )

        # Protest events
        protest_events = event_types.get('Protests', 0) + event_types.get('Riots', 0)

        return {
            'conflict_event_count': float(len(events)),
            'fatalities_total': float(fatalities),
            'fatalities_per_day': float(fatalities) / days,
            'violence_intensity': float(violence_events) / days,
            'protest_frequency': float(protest_events) / days,
            'events_per_day': float(len(events)) / days
        }

    def _empty_response(self, event_id: str, reason: str) -> Dict[str, Any]:
        return {
            'source': self.source_name,
            'event_id': event_id,
            'data': [],
            'metadata': {
                'fetch_time': datetime.utcnow().isoformat(),
                'record_count': 0,
                'error': reason
            }
        }

    def _default_indicators(self) -> Dict[str, float]:
        return {
            'conflict_event_count': 0.0,
            'fatalities_total': 0.0,
            'fatalities_per_day': 0.0,
            'violence_intensity': 0.0,
            'protest_frequency': 0.0,
            'events_per_day': 0.0
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check ACLED API health."""
        import time
        start = time.time()

        if not self.is_configured():
            return {
                'source': self.source_name,
                'status': 'NOT_CONFIGURED',
                'configured': False
            }

        try:
            # Simple query to test connectivity
            params = {
                'key': self.api_key,
                'email': self.email,
                'limit': 1
            }
            response = await self._make_request(self.base_url, params=params)

            return {
                'source': self.source_name,
                'status': 'OPERATIONAL' if response.get('success') else 'DEGRADED',
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
