"""
GDELT API Client

Global Database of Events, Language, and Tone
https://www.gdeltproject.org/

Real-time global news and event monitoring with sentiment analysis.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import re

from ..base_client import BaseAPIClient, RateLimiter


class GDELTFilter:
    """Filter to reduce noise in GDELT data."""

    def __init__(
        self,
        min_source_rank: int = 3,
        tone_bounds: tuple = (-50, 50),
        min_sources: int = 2
    ):
        self.min_source_rank = min_source_rank
        self.tone_bounds = tone_bounds
        self.min_sources = min_sources
        self.seen_urls = set()

    def filter_articles(self, articles: List[Dict]) -> List[Dict]:
        """Filter articles based on quality criteria."""
        filtered = []
        for article in articles:
            if self._passes_quality(article):
                filtered.append(article)
        return filtered

    def _passes_quality(self, article: Dict) -> bool:
        # Check tone bounds
        tone = article.get('tone', 0)
        if not self.tone_bounds[0] <= tone <= self.tone_bounds[1]:
            return False

        # Check for duplicates
        url = article.get('url', '')
        if url in self.seen_urls:
            return False
        self.seen_urls.add(url)

        return True

    def aggregate_tone(self, articles: List[Dict]) -> tuple:
        """Calculate aggregate tone with confidence."""
        if not articles:
            return 0.0, 0.0

        tones = [a.get('tone', 0) for a in articles if 'tone' in a]
        if not tones:
            return 0.0, 0.0

        avg_tone = sum(tones) / len(tones)
        # Confidence based on number of sources
        confidence = min(1.0, len(tones) / 10)

        return avg_tone, confidence


class GDELTClient(BaseAPIClient):
    """
    Client for GDELT API.

    Rate Limit: No strict limit
    Authentication: None required
    Note: Data can be noisy - use GDELTFilter
    """

    def __init__(self, rate_limiter: Optional[RateLimiter] = None, **kwargs):
        super().__init__(
            base_url="https://api.gdeltproject.org/api/v2",
            rate_limiter=rate_limiter,
            **kwargs
        )
        self.filter = GDELTFilter()

    @property
    def source_name(self) -> str:
        return "GDELT"

    async def fetch_data(
        self,
        event_id: str,
        start_date: datetime,
        end_date: datetime,
        keywords: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch news/events from GDELT.

        Args:
            event_id: PRISM event ID
            keywords: Search keywords
        """
        search_query = ' OR '.join(keywords or ['risk', 'crisis'])

        cache_key = self._generate_cache_key(
            event_id,
            start_date.date(),
            end_date.date(),
            search_query[:50]
        )

        async def _fetch():
            # GDELT DOC API for article search
            params = {
                'query': search_query,
                'mode': 'artlist',
                'maxrecords': 250,
                'format': 'json',
                'startdatetime': start_date.strftime('%Y%m%d%H%M%S'),
                'enddatetime': end_date.strftime('%Y%m%d%H%M%S')
            }

            try:
                response = await self._make_request("/doc/doc", params=params)
                articles = response.get('articles', [])

                # Apply filtering
                filtered = self.filter.filter_articles(articles)

                return {
                    'source': self.source_name,
                    'event_id': event_id,
                    'data': filtered,
                    'metadata': {
                        'fetch_time': datetime.utcnow().isoformat(),
                        'record_count': len(filtered),
                        'raw_count': len(articles),
                        'query': search_query
                    }
                }
            except Exception as e:
                self.logger.error(f"GDELT fetch error: {e}")
                return {
                    'source': self.source_name,
                    'event_id': event_id,
                    'data': [],
                    'metadata': {'error': str(e)}
                }

        return await self.get_cached_or_fetch(cache_key, _fetch, ttl=900)

    def extract_indicators(
        self,
        raw_data: Dict[str, Any],
        indicator_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract sentiment and event indicators."""
        articles = raw_data.get('data', [])

        if not articles:
            return {
                'article_count': 0.0,
                'avg_tone': 0.0,
                'tone_confidence': 0.0,
                'negative_ratio': 0.0
            }

        # Calculate tone metrics
        avg_tone, confidence = self.filter.aggregate_tone(articles)

        # Count negative articles
        negative_count = sum(1 for a in articles if a.get('tone', 0) < -5)
        negative_ratio = negative_count / len(articles) if articles else 0

        return {
            'article_count': float(len(articles)),
            'avg_tone': avg_tone,
            'tone_confidence': confidence,
            'negative_ratio': negative_ratio
        }

    async def health_check(self) -> Dict[str, Any]:
        import time
        start = time.time()
        try:
            params = {'query': 'test', 'mode': 'artlist', 'maxrecords': 1, 'format': 'json'}
            await self._make_request("/doc/doc", params=params)
            return {
                'source': self.source_name,
                'status': 'OPERATIONAL',
                'configured': True,
                'response_time_ms': int((time.time() - start) * 1000)
            }
        except Exception as e:
            return {'source': self.source_name, 'status': 'ERROR', 'error': str(e)}
