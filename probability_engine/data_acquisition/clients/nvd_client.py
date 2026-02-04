"""
NVD Client - National Vulnerability Database
https://nvd.nist.gov/

CVE vulnerability data with CVSS scores.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from ..base_client import BaseAPIClient, RateLimiter


class NVDClient(BaseAPIClient):
    """Rate Limit: 50/30s without key, 5000 with key. API key optional."""

    def __init__(self, api_key: Optional[str] = None, rate_limiter: Optional[RateLimiter] = None, **kwargs):
        rate_limit = 5000 if api_key else 50
        super().__init__(
            api_key=api_key,
            base_url="https://services.nvd.nist.gov/rest/json/cves/2.0",
            rate_limiter=rate_limiter or RateLimiter(rate_limit, 30 if not api_key else 3600),
            **kwargs
        )

    @property
    def source_name(self) -> str:
        return "NVD"

    async def fetch_data(self, event_id: str, start_date: datetime, end_date: datetime,
                         cvss_min: float = 7.0, **kwargs) -> Dict[str, Any]:
        cache_key = self._generate_cache_key(event_id, start_date.date(), end_date.date(), cvss_min)

        async def _fetch():
            headers = {'apiKey': self.api_key} if self.api_key else {}
            params = {
                'pubStartDate': start_date.strftime('%Y-%m-%dT00:00:00.000'),
                'pubEndDate': end_date.strftime('%Y-%m-%dT23:59:59.999'),
                'cvssV3Severity': 'HIGH',  # HIGH or CRITICAL
                'resultsPerPage': 200
            }

            try:
                response = await self._make_request("", params=params, headers=headers)
                vulnerabilities = response.get('vulnerabilities', [])
                return {
                    'source': self.source_name,
                    'event_id': event_id,
                    'data': vulnerabilities,
                    'metadata': {
                        'fetch_time': datetime.utcnow().isoformat(),
                        'cve_count': len(vulnerabilities),
                        'cvss_filter': cvss_min
                    }
                }
            except Exception as e:
                return {'source': self.source_name, 'event_id': event_id, 'data': [], 'metadata': {'error': str(e)}}

        return await self.get_cached_or_fetch(cache_key, _fetch, ttl=3600)

    def extract_indicators(self, raw_data: Dict[str, Any], indicator_config: Dict[str, Any]) -> Dict[str, float]:
        cves = raw_data.get('data', [])
        if not cves:
            return {'cve_count': 0.0, 'avg_cvss': 0.0, 'critical_count': 0.0}

        cvss_scores = []
        critical_count = 0
        for cve in cves:
            metrics = cve.get('cve', {}).get('metrics', {})
            cvss_v3 = metrics.get('cvssMetricV31', [{}])[0] if metrics.get('cvssMetricV31') else {}
            score = cvss_v3.get('cvssData', {}).get('baseScore', 0)
            if score:
                cvss_scores.append(score)
                if score >= 9.0:
                    critical_count += 1

        return {
            'cve_count': float(len(cves)),
            'avg_cvss': sum(cvss_scores) / len(cvss_scores) if cvss_scores else 0.0,
            'critical_count': float(critical_count),
            'high_severity_ratio': len([s for s in cvss_scores if s >= 7.0]) / len(cves) if cves else 0.0
        }

    async def health_check(self) -> Dict[str, Any]:
        import time
        start = time.time()
        try:
            await self._make_request("", params={'resultsPerPage': 1})
            return {'source': self.source_name, 'status': 'OPERATIONAL', 'configured': True,
                    'response_time_ms': int((time.time() - start) * 1000)}
        except Exception as e:
            return {'source': self.source_name, 'status': 'ERROR', 'error': str(e)}
