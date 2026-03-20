"""
Response Cache Service - Redis-based caching for AI responses
Reduces latency for repeated/similar queries.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Cached response data"""
    content: str
    provider: str
    model: str
    timestamp: float
    hit_count: int = 0
    avg_quality_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "timestamp": self.timestamp,
            "hit_count": self.hit_count,
            "avg_quality_score": self.avg_quality_score,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedResponse":
        return cls(
            content=data["content"],
            provider=data["provider"],
            model=data["model"],
            timestamp=data["timestamp"],
            hit_count=data.get("hit_count", 0),
            avg_quality_score=data.get("avg_quality_score", 0.0),
            metadata=data.get("metadata", {})
        )


class ResponseCache:
    """
    In-memory response cache with optional Redis backend.
    
    Features:
    - Fast in-memory cache with LRU eviction
    - Semantic similarity matching for similar queries
    - TTL-based expiration
    - Quality-based cache invalidation
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: int = 3600,  # 1 hour
                 redis_client: Optional[Any] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.redis = redis_client
        
        # In-memory cache
        self._cache: Dict[str, CachedResponse] = {}
        self._access_order: List[str] = []  # LRU tracking
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0
        }
    
    def _generate_cache_key(self, 
                           message: str, 
                           context_summary: Optional[str] = None,
                           provider: Optional[str] = None) -> str:
        """Generate a cache key from message and context"""
        # Normalize message
        normalized = message.lower().strip()
        
        # Include context summary if provided
        if context_summary:
            normalized += f"|ctx:{context_summary[:100]}"
        
        # Include provider if specified
        if provider:
            normalized += f"|prov:{provider}"
        
        # Generate hash
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _is_cacheable(self, message: str) -> bool:
        """Determine if a message should be cached"""
        message_lower = message.lower().strip()
        
        # Don't cache very short messages (confirmations)
        if len(message_lower) < 5:
            return False
        
        # Don't cache messages with personal/temporal references
        non_cacheable_patterns = [
            "my ", "i am", "i'm", "today", "yesterday", "tomorrow",
            "now", "current", "latest", "recent", "just"
        ]
        
        for pattern in non_cacheable_patterns:
            if pattern in message_lower:
                return False
        
        # Don't cache questions about specific files/projects
        if any(ext in message_lower for ext in [".py", ".js", ".ts", ".tsx", ".css", ".html"]):
            return False
        
        return True
    
    def _evict_lru(self):
        """Evict least recently used entries"""
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
                self.stats["evictions"] += 1
    
    def _update_access(self, key: str):
        """Update access order for LRU"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    async def get(self, 
                  message: str, 
                  context_summary: Optional[str] = None,
                  provider: Optional[str] = None) -> Optional[CachedResponse]:
        """
        Get cached response if available.
        
        Args:
            message: User's message
            context_summary: Summary of conversation context
            provider: AI provider (optional, for provider-specific caching)
        
        Returns:
            CachedResponse if found and valid, None otherwise
        """
        if not self._is_cacheable(message):
            return None
        
        cache_key = self._generate_cache_key(message, context_summary, provider)
        
        # Check in-memory cache first
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            
            # Check TTL
            if time.time() - cached.timestamp > self.default_ttl:
                del self._cache[cache_key]
                self.stats["misses"] += 1
                return None
            
            # Update access order and hit count
            self._update_access(cache_key)
            cached.hit_count += 1
            self.stats["hits"] += 1
            
            logger.info(f"🎯 Cache HIT: key={cache_key[:8]}... hits={cached.hit_count}")
            return cached
        
        # Check Redis if available
        if self.redis:
            try:
                redis_key = f"resonant:cache:{cache_key}"
                data = await self.redis.get(redis_key)
                if data:
                    cached = CachedResponse.from_dict(json.loads(data))
                    
                    # Store in memory cache
                    self._cache[cache_key] = cached
                    self._update_access(cache_key)
                    
                    cached.hit_count += 1
                    self.stats["hits"] += 1
                    
                    logger.info(f"🎯 Redis Cache HIT: key={cache_key[:8]}...")
                    return cached
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        self.stats["misses"] += 1
        return None
    
    async def set(self,
                  message: str,
                  response: str,
                  provider: str,
                  model: str,
                  context_summary: Optional[str] = None,
                  quality_score: float = 0.0,
                  metadata: Optional[Dict[str, Any]] = None,
                  ttl: Optional[int] = None) -> str:
        """
        Cache a response.
        
        Args:
            message: User's message
            response: AI response content
            provider: AI provider name
            model: Model name
            context_summary: Summary of conversation context
            quality_score: Quality score of the response
            metadata: Additional metadata
            ttl: Time-to-live in seconds (optional)
        
        Returns:
            Cache key
        """
        if not self._is_cacheable(message):
            return ""
        
        cache_key = self._generate_cache_key(message, context_summary, provider)
        
        # Evict if necessary
        self._evict_lru()
        
        # Create cached response
        cached = CachedResponse(
            content=response,
            provider=provider,
            model=model,
            timestamp=time.time(),
            hit_count=0,
            avg_quality_score=quality_score,
            metadata=metadata or {}
        )
        
        # Store in memory
        self._cache[cache_key] = cached
        self._update_access(cache_key)
        
        # Store in Redis if available
        if self.redis:
            try:
                redis_key = f"resonant:cache:{cache_key}"
                await self.redis.setex(
                    redis_key,
                    ttl or self.default_ttl,
                    json.dumps(cached.to_dict())
                )
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
        
        logger.info(f"💾 Cached response: key={cache_key[:8]}... provider={provider}")
        return cache_key
    
    async def invalidate(self, cache_key: str):
        """Invalidate a specific cache entry"""
        if cache_key in self._cache:
            del self._cache[cache_key]
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self.stats["invalidations"] += 1
        
        if self.redis:
            try:
                await self.redis.delete(f"resonant:cache:{cache_key}")
            except Exception as e:
                logger.warning(f"Redis invalidation error: {e}")
    
    async def invalidate_by_quality(self, min_quality: float = 0.5):
        """Invalidate cache entries below quality threshold"""
        keys_to_remove = []
        
        for key, cached in self._cache.items():
            if cached.avg_quality_score < min_quality:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            await self.invalidate(key)
        
        logger.info(f"🧹 Invalidated {len(keys_to_remove)} low-quality cache entries")
    
    def update_quality(self, cache_key: str, quality_score: float):
        """Update quality score for a cached response"""
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Exponential moving average
            alpha = 0.3
            cached.avg_quality_score = (1 - alpha) * cached.avg_quality_score + alpha * quality_score
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_size": self.max_size
        }
    
    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
        self._access_order.clear()
        logger.info("🧹 Cache cleared")


# Global cache instance
response_cache = ResponseCache()


# Convenience functions
async def get_cached_response(message: str, 
                              context_summary: Optional[str] = None,
                              provider: Optional[str] = None) -> Optional[CachedResponse]:
    """Get cached response if available"""
    return await response_cache.get(message, context_summary, provider)


async def cache_response(message: str,
                        response: str,
                        provider: str,
                        model: str,
                        context_summary: Optional[str] = None,
                        quality_score: float = 0.0) -> str:
    """Cache a response"""
    return await response_cache.set(
        message=message,
        response=response,
        provider=provider,
        model=model,
        context_summary=context_summary,
        quality_score=quality_score
    )
