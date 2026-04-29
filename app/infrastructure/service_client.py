"""
Production Service Client with Circuit Breaker
================================================

Extracted from resonant_chat.py — provides resilient HTTP calls
to internal microservices (memory, billing, auth) with automatic
circuit breaking on repeated failures.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = ServiceState.HEALTHY
    
    def call_allowed(self) -> bool:
        current_time = asyncio.get_event_loop().time()
        if self.state == ServiceState.FAILED:
            # Auto-reset after timeout
            if current_time - self.last_failure_time > self.timeout:
                self.state = ServiceState.DEGRADED
                self.failure_count = 0
                logger.info(f"🔄 Circuit breaker auto-reset for service")
                return True
            return False
        return True
    
    def record_success(self):
        self.failure_count = 0
        if self.state != ServiceState.HEALTHY:
            self.state = ServiceState.HEALTHY
            logger.info(f"✅ Circuit breaker restored to healthy")
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()
        if self.failure_count >= self.failure_threshold:
            self.state = ServiceState.FAILED
            logger.warning(f"🚨 Circuit breaker OPEN after {self.failure_count} failures")

class ServiceClient:
    def __init__(self):
        self.circuit_breakers = {
            "memory_service": CircuitBreaker(failure_threshold=3, timeout=60),
            "billing_service": CircuitBreaker(failure_threshold=2, timeout=30),
            "auth_service": CircuitBreaker(failure_threshold=2, timeout=30),
        }
        self.session = None
    
    async def get_session(self):
        if self.session is None or self.session.is_closed:
            self.session = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=3.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        return self.session
    
    async def call_service(
        self, 
        service_name: str, 
        method: str, 
        url: str, 
        **kwargs
    ) -> Optional[Dict]:
        circuit_breaker = self.circuit_breakers.get(service_name)
        if not circuit_breaker or not circuit_breaker.call_allowed():
            print(f"[SVC] Circuit breaker OPEN for {service_name}", flush=True)
            return None
        
        session = await self.get_session()
        max_retries = 2
        base_delay = 0.1
        
        for attempt in range(max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = await session.get(url, **kwargs)
                elif method.upper() == "POST":
                    response = await session.post(url, **kwargs)
                else:
                    return None
                
                # Handle different response statuses
                if response.status_code < 400:
                    circuit_breaker.record_success()
                    return response.json() if response.content else None
                elif response.status_code == 404:
                    logger.warning(f"❌ Endpoint not found for {service_name}: {url}")
                    circuit_breaker.record_failure()
                    return None
                elif response.status_code >= 500:
                    logger.warning(f"🔥 Server error for {service_name}: {response.status_code}")
                    if attempt < max_retries:
                        await asyncio.sleep(base_delay * (2 ** attempt))
                        continue
                    else:
                        circuit_breaker.record_failure()
                        return None
                else:
                    logger.warning(f"⚠️ Client error for {service_name}: {response.status_code}")
                    return None
                    
            except (httpx.RequestError, httpx.TimeoutException) as e:
                print(f"[SVC] Network error for {service_name} (attempt {attempt + 1}): {e}", flush=True)
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    continue
                break
        
        circuit_breaker.record_failure()
        return None

# Global service client instance
service_client = ServiceClient()
