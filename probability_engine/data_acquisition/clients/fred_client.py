"""
FRED API Client

Federal Reserve Economic Data
https://fred.stlouisfed.org/

800,000+ economic time series from the Federal Reserve.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List

from ..base_client import BaseAPIClient, RateLimiter


class FREDClient(BaseAPIClient):
    """
    Client for FRED API.

    Rate Limit: 120 requests/minute
    Authentication: API key required (free)
    """

    # Common series IDs
    SERIES = {
        'fed_funds_rate': 'FEDFUNDS',
        'treasury_10y': 'DGS10',
        'treasury_2y': 'DGS2',
        'unemployment': 'UNRATE',
        'cpi_inflation': 'CPIAUCSL',
        'gdp': 'GDP',
        'sp500': 'SP500',
        'vix': 'VIXCLS',
        'oil_wti': 'DCOILWTICO',
        'dollar_index': 'DTWEXBGS',
    }

    def __init__(
        self,
        api_key: str,
        rate_limiter: Optional[RateLimiter] = None,
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            base_url="https://api.stlouisfed.org/fred",
            rate_limiter=rate_limiter or RateLimiter(120, 60),
            **kwargs
        )

    @property
    def source_name(self) -> str:
        return "FRED"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def fetch_data(
        self,
        event_id: str,
        start_date: datetime,
        end_date: datetime,
        series_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch economic series from FRED.

        Args:
            event_id: PRISM event ID
            start_date: Start date
            end_date: End date
            series_ids: List of FRED series IDs

        Returns:
            Standardized data dict
        """
        if not self.is_configured():
            return self._empty_response(event_id, "API not configured")

        series = series_ids or list(self.SERIES.values())[:5]

        cache_key = self._generate_cache_key(
            event_id,
            start_date.date(),
            end_date.date(),
            tuple(series)
        )

        async def _fetch():
            all_data = {}

            for series_id in series:
                try:
                    params = {
                        'series_id': series_id,
                        'api_key': self.api_key,
                        'file_type': 'json',
                        'observation_start': start_date.strftime('%Y-%m-%d'),
                        'observation_end': end_date.strftime('%Y-%m-%d')
                    }

                    response = await self._make_request(
                        "/series/observations",
                        params=params
                    )

                    observations = response.get('observations', [])
                    all_data[series_id] = [
                        {
                            'date': obs.get('date'),
                            'value': float(obs.get('value')) if obs.get('value') != '.' else None
                        }
                        for obs in observations
                        if obs.get('value') != '.'
                    ]

                except Exception as e:
                    self.logger.warning(f"Failed to fetch {series_id}: {e}")
                    all_data[series_id] = []

            return {
                'source': self.source_name,
                'event_id': event_id,
                'data': all_data,
                'metadata': {
                    'fetch_time': datetime.utcnow().isoformat(),
                    'series_fetched': len(all_data),
                    'date_range': [start_date.date().isoformat(), end_date.date().isoformat()]
                }
            }

        return await self.get_cached_or_fetch(cache_key, _fetch, ttl=3600)

    def extract_indicators(
        self,
        raw_data: Dict[str, Any],
        indicator_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract financial indicators from FRED data.

        Returns latest value and changes for each series.
        """
        data = raw_data.get('data', {})
        results = {}

        for series_id, observations in data.items():
            if not observations:
                continue

            # Find friendly name
            series_name = None
            for name, sid in self.SERIES.items():
                if sid == series_id:
                    series_name = name
                    break
            series_name = series_name or series_id.lower()

            # Get latest value
            valid_obs = [o for o in observations if o.get('value') is not None]
            if valid_obs:
                latest = valid_obs[-1]['value']
                results[f"{series_name}_current"] = latest

                # Calculate change if enough data
                if len(valid_obs) >= 2:
                    previous = valid_obs[-2]['value']
                    if previous != 0:
                        pct_change = (latest - previous) / abs(previous) * 100
                        results[f"{series_name}_change_pct"] = pct_change

        return results

    def _empty_response(self, event_id: str, reason: str) -> Dict[str, Any]:
        return {
            'source': self.source_name,
            'event_id': event_id,
            'data': {},
            'metadata': {'error': reason}
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check FRED API health."""
        import time
        start = time.time()

        if not self.is_configured():
            return {
                'source': self.source_name,
                'status': 'NOT_CONFIGURED',
                'configured': False
            }

        try:
            params = {
                'series_id': 'GNPCA',
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': 1
            }
            await self._make_request("/series/observations", params=params)

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
