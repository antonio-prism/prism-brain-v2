"""
PRISM Engine — Base connector with shared HTTP logic, caching, and retry.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [1, 5, 15]  # Exponential backoff in seconds


class ConnectorResult:
    """Standardized result from any connector."""

    def __init__(self, source_id: str, success: bool, data: dict | None = None,
                 error: str | None = None, cached: bool = False):
        self.source_id = source_id
        self.success = success
        self.data = data or {}
        self.error = error
        self.cached = cached
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "cached": self.cached,
            "timestamp": self.timestamp,
        }


def _cache_key(source_id: str, params: dict) -> str:
    """Generate a unique cache key from source + parameters."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(f"{source_id}:{param_str}".encode()).hexdigest()


def get_cached(source_id: str, params: dict, max_age_hours: int = 24) -> dict | None:
    """Load cached response if fresh enough."""
    key = _cache_key(source_id, params)
    cache_file = CACHE_DIR / f"{source_id}_{key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_time = datetime.fromisoformat(cached.get("timestamp", "2000-01-01"))
        if datetime.utcnow() - cached_time > timedelta(hours=max_age_hours):
            return None
        logger.debug(f"Cache hit for {source_id}")
        return cached.get("data")
    except Exception:
        return None


def save_cache(source_id: str, params: dict, data: dict) -> None:
    """Save response to cache."""
    key = _cache_key(source_id, params)
    cache_file = CACHE_DIR / f"{source_id}_{key}.json"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "source_id": source_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data,
            }, f)
    except Exception as e:
        logger.warning(f"Failed to cache {source_id}: {e}")


def fetch_with_retry(url: str, params: dict | None = None, headers: dict | None = None,
                     timeout: int = 30) -> requests.Response | None:
    """HTTP GET with exponential backoff retry."""
    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                logger.warning(f"Rate limited (429) on attempt {attempt+1}, waiting {delay}s")
                time.sleep(delay)
                continue
            logger.warning(f"HTTP {resp.status_code} on attempt {attempt+1} for {url}")
            if resp.status_code >= 500:
                time.sleep(delay)
                continue
            return resp  # 4xx other than 429 — don't retry
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt+1} for {url}")
            time.sleep(delay)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error on attempt {attempt+1}: {e}")
            time.sleep(delay)
    logger.error(f"All {MAX_RETRIES} retries failed for {url}")
    return None
