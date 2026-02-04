"""
Base API Client

Abstract base class for all data source API clients.
Provides common functionality for rate limiting, caching, retry logic, and error handling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio
import aiohttp
import hashlib
import json
import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, max_requests: int, time_window: int = 3600):
        """
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds (default: 1 hour)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.tokens = max_requests
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request token is available."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens based on elapsed time
            self.tokens = min(
                self.max_requests,
                self.tokens + (elapsed * self.max_requests / self.time_window)
            )
            self.last_update = now

            if self.tokens < 1:
                # Calculate wait time
                wait_time = (1 - self.tokens) * self.time_window / self.max_requests
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 1

            self.tokens -= 1


class CacheManager:
    """Simple in-memory cache with TTL support."""

    def __init__(self, default_ttl: int = 3600):
        self.cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self.default_ttl = default_ttl

    async def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self.cache:
            value, expiry = self.cache[key]
            if time.time() < expiry:
                return value
            else:
                del self.cache[key]
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cached value with TTL."""
        ttl = ttl or self.default_ttl
        self.cache[key] = (value, time.time() + ttl)

    def clear(self):
        """Clear all cached values."""
        self.cache.clear()


class BaseAPIClient(ABC):
    """
    Abstract base class for all data source API clients.

    Provides:
    - Rate limiting
    - Caching
    - Exponential backoff retry
    - Error handling
    - Data validation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "",
        rate_limiter: Optional[RateLimiter] = None,
        cache_manager: Optional[CacheManager] = None,
        timeout: int = 30
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limiter = rate_limiter
        self.cache_manager = cache_manager or CacheManager()
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the data source name (e.g., 'ACLED', 'GDELT')."""
        pass

    @abstractmethod
    async def fetch_data(
        self,
        event_id: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch data for a specific risk event within a date range.

        Args:
            event_id: PRISM event ID (e.g., "GEO-001")
            start_date: Start of time window
            end_date: End of time window
            **kwargs: Additional source-specific parameters

        Returns:
            Dict with standardized structure:
            {
                'source': 'SOURCE_NAME',
                'event_id': 'EVENT_ID',
                'data': { ... raw data ... },
                'metadata': {
                    'fetch_time': 'ISO timestamp',
                    'record_count': int,
                    'date_range': [start, end]
                }
            }
        """
        pass

    @abstractmethod
    def extract_indicators(
        self,
        raw_data: Dict[str, Any],
        indicator_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract numerical indicators from raw API data.

        Args:
            raw_data: Output from fetch_data()
            indicator_config: Configuration for this indicator

        Returns:
            Dict of indicator_name -> value
        """
        pass

    def _generate_cache_key(self, *args) -> str:
        """Generate a unique cache key from arguments."""
        key_str = f"{self.source_name}:" + ":".join(str(a) for a in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def get_cached_or_fetch(
        self,
        cache_key: str,
        fetch_func,
        ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Cache-then-fetch pattern.

        Args:
            cache_key: Unique cache key
            fetch_func: Async function to fetch data
            ttl: Cache TTL in seconds

        Returns:
            Cached or freshly fetched data
        """
        # Check cache
        cached = await self.cache_manager.get(cache_key)
        if cached:
            self.logger.debug(f"Cache hit: {cache_key[:20]}...")
            return cached

        # Wait for rate limiter
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        # Fetch with retry
        data = await self._fetch_with_retry(fetch_func)

        # Cache result
        if data:
            await self.cache_manager.set(cache_key, data, ttl)

        return data

    async def _fetch_with_retry(
        self,
        fetch_func,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ) -> Dict[str, Any]:
        """
        Fetch data with exponential backoff retry.

        Args:
            fetch_func: Async function to call
            max_retries: Maximum retry attempts
            backoff_factor: Exponential backoff multiplier

        Returns:
            Fetched data or empty dict on failure
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return await fetch_func()

            except aiohttp.ClientResponseError as e:
                last_error = e
                if e.status == 429:  # Rate limited
                    wait_time = backoff_factor ** (attempt + 2)
                    self.logger.warning(f"Rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                elif e.status >= 500:  # Server error
                    wait_time = backoff_factor ** attempt
                    self.logger.warning(f"Server error {e.status}, retry in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Client error {e.status}: {e.message}")
                    raise

            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Request timed out after {self.timeout}s")
                wait_time = backoff_factor ** attempt
                self.logger.warning(f"Timeout, retry in {wait_time}s")
                await asyncio.sleep(wait_time)

            except Exception as e:
                last_error = e
                self.logger.error(f"Unexpected error: {e}")
                if attempt == max_retries - 1:
                    raise

        self.logger.error(f"Failed after {max_retries} attempts: {last_error}")
        return {'error': str(last_error), 'data': None}

    async def _make_request(
        self,
        url: str,
        method: str = 'GET',
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request.

        Args:
            url: Full URL or endpoint (will be joined with base_url)
            method: HTTP method
            params: Query parameters
            headers: HTTP headers
            json_data: JSON body for POST requests

        Returns:
            Parsed JSON response
        """
        if not url.startswith('http'):
            url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"

        default_headers = {'Accept': 'application/json'}
        if headers:
            default_headers.update(headers)

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method,
                url,
                params=params,
                headers=default_headers,
                json=json_data
            ) as response:
                response.raise_for_status()
                return await response.json()

    def is_configured(self) -> bool:
        """Check if the client has necessary credentials."""
        return True  # Override in subclasses that require API keys

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the data source.

        Returns:
            Dict with status, response_time, etc.
        """
        start_time = time.time()
        try:
            # Subclasses should override with actual health check
            return {
                'source': self.source_name,
                'status': 'UNKNOWN',
                'configured': self.is_configured(),
                'response_time_ms': 0
            }
        except Exception as e:
            return {
                'source': self.source_name,
                'status': 'ERROR',
                'error': str(e),
                'response_time_ms': int((time.time() - start_time) * 1000)
            }
