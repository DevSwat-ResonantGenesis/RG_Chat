"""Web Search Service for Resonant Chat.

Provides real-time web search capabilities using multiple providers:
- Tavily (primary - best for AI applications)
- SerpAPI (fallback)
- DuckDuckGo (free fallback)
"""
import os
import logging
import httpx
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSearchResult:
    """Represents a single search result."""
    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        source: str = "unknown",
        published_date: Optional[str] = None,
    ):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.published_date = published_date
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "published_date": self.published_date,
        }


class WebSearchService:
    """Web search service with multiple provider support."""
    
    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.serp_api_key = os.getenv("SERPAPI_KEY")
        self.timeout = 10.0

    @staticmethod
    def _split_keys(raw: Optional[str]) -> List[str]:
        if not raw:
            return []
        return [k.strip() for k in raw.split(",") if k.strip()]
    
    def set_api_keys(self, tavily_key: Optional[str] = None, serp_key: Optional[str] = None):
        """Set API keys dynamically (for user-provided keys)."""
        if tavily_key:
            self.tavily_api_key = tavily_key
        if serp_key:
            self.serp_api_key = serp_key
    
    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_type: str = "general",  # general, news, images
    ) -> List[WebSearchResult]:
        """
        Perform web search using available providers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            search_type: Type of search (general, news, images)
        
        Returns:
            List of WebSearchResult objects
        """
        results = []
        
        tavily_keys = self._split_keys(self.tavily_api_key)
        if tavily_keys:
            for idx, key in enumerate(tavily_keys):
                try:
                    results = await self._search_tavily(query, max_results, search_type, api_key=key)
                    if results:
                        logger.info(
                            f"🔍 Tavily search returned {len(results)} results for: {query[:50]} (key_index={idx})"
                        )
                        return results
                except httpx.HTTPStatusError as e:
                    status = getattr(e.response, "status_code", None)
                    logger.warning(f"Tavily search failed (status={status}, key_index={idx}): {e}")
                except Exception as e:
                    logger.warning(f"Tavily search failed (key_index={idx}): {e}")
        
        serp_keys = self._split_keys(self.serp_api_key)
        if serp_keys:
            for idx, key in enumerate(serp_keys):
                try:
                    results = await self._search_serpapi(query, max_results, api_key=key)
                    if results:
                        logger.info(
                            f"🔍 SerpAPI search returned {len(results)} results for: {query[:50]} (key_index={idx})"
                        )
                        return results
                except httpx.HTTPStatusError as e:
                    status = getattr(e.response, "status_code", None)
                    logger.warning(f"SerpAPI search failed (status={status}, key_index={idx}): {e}")
                except Exception as e:
                    logger.warning(f"SerpAPI search failed (key_index={idx}): {e}")
        
        # Try DuckDuckGo as free fallback
        try:
            results = await self._search_duckduckgo(query, max_results)
            if results:
                logger.info(f"🔍 DuckDuckGo search returned {len(results)} results for: {query[:50]}")
                return results
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
        
        logger.warning(f"All search providers failed for query: {query[:50]}")
        return []
    
    async def _search_tavily(
        self,
        query: str,
        max_results: int,
        search_type: str,
        api_key: str,
    ) -> List[WebSearchResult]:
        """Search using Tavily API."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "advanced" if search_type == "news" else "basic",
                    "include_answer": True,
                    "include_raw_content": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", []):
                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source="tavily",
                    published_date=item.get("published_date"),
                ))
            
            # Include Tavily's AI-generated answer if available
            if data.get("answer"):
                results.insert(0, WebSearchResult(
                    title="AI Summary",
                    url="",
                    snippet=data["answer"],
                    source="tavily_answer",
                ))
            
            return results
    
    async def _search_serpapi(self, query: str, max_results: int, api_key: str) -> List[WebSearchResult]:
        """Search using SerpAPI (Google results)."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                "https://serpapi.com/search",
                params={
                    "api_key": api_key,
                    "q": query,
                    "num": max_results,
                    "engine": "google",
                },
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("organic_results", []):
                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="serpapi",
                ))
            
            return results
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Search using DuckDuckGo (free, no API key needed)."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                # Try HTML version with proper headers
                response = await client.get(
                    "https://duckduckgo.com/html/",
                    params={
                        "q": query,
                    },
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                )
                response.raise_for_status()
                
                results = []
                html_content = response.text
                
                # Simple HTML parsing for results
                if 'result__a' in html_content:
                    import re
                    # Extract basic result information
                    result_links = re.findall(r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>', html_content)
                    result_snippets = re.findall(r'<a[^>]*class="result__snippet"[^>]*>([^<]*)</a>', html_content)
                    
                    for i, (url, title) in enumerate(result_links[:max_results]):
                        snippet = result_snippets[i] if i < len(result_snippets) else ""
                        results.append(WebSearchResult(
                            title=title.strip() if title else "Result",
                            url=url.strip() if url else "",
                            snippet=snippet.strip() if snippet else "No snippet available",
                            source="duckduckgo",
                        ))
                
                if results:
                    logger.info(f"🔍 DuckDuckGo HTML search returned {len(results)} results for: {query[:50]}")
                else:
                    # Fallback to basic answer generation
                    results.append(WebSearchResult(
                        title="Search Result",
                        url="",
                        snippet=f"I performed a web search for '{query}' but couldn't retrieve specific results. The search service is working but may need API keys for detailed results.",
                        source="fallback",
                    ))
                
                return results
                
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return [WebSearchResult(
                title="Search Unavailable",
                url="",
                snippet=f"Web search is currently unavailable. The query '{query}' was processed but external search services are not responding.",
                source="fallback",
            )]
    
    def should_search(self, message: str) -> bool:
        """Determine if a message requires web search.

        Uses keyword triggers plus regex patterns to catch conversational
        phrasing like 'check online', 'shops in SF', 'open early', etc.
        """
        import re as _re
        message_lower = message.lower()

        # Time-sensitive queries
        time_triggers = [
            "today", "yesterday", "this week", "this month", "this year",
            "current", "latest", "recent", "now", "2024", "2025", "2026",
            "what time", "what date", "what day",
        ]

        # Information queries
        info_triggers = [
            "what is", "who is", "where is", "when is", "how to",
            "news about", "events in", "weather in", "price of",
            "stock price", "crypto price", "bitcoin", "ethereum",
            "search for", "look up", "find information", "find out",
            "what happened", "what's happening",
            "tell me about", "information about", "info about",
            "can you find", "can u find", "can you check", "can u check",
            "check online", "search online", "look online", "browse",
            "google", "look it up", "search it up", "web search",
            "search the web", "search web", "check the web",
            "find it online", "find that online", "look this up",
        ]

        # Location-based queries
        location_triggers = [
            "near me", "in my area",
            "restaurants", "hotels", "stores", "shops", "cafes", "bars",
            "coffee shops", "bakeries", "pharmacies", "grocery",
            "events in", "things to do in", "places in",
            "open now", "open early", "open late", "open 24",
            "hours of operation", "opening hours", "business hours",
        ]

        # Real-world factual queries
        factual_triggers = [
            "how much does", "how much is", "cost of", "price of",
            "compare", "vs ", "versus", "review of", "reviews",
            "best ", "top ", "recommended", "rating",
            "schedule", "timetable", "flight", "train",
            "address of", "phone number", "contact",
            "directions to", "how to get to", "route to",
        ]

        all_triggers = time_triggers + info_triggers + location_triggers + factual_triggers

        matched = [trigger for trigger in all_triggers if trigger in message_lower]
        if matched:
            logger.info(f"🔍 Web search trigger matched: {matched[:5]} (message={message[:80]!r})")
            return True

        # Regex patterns for conversational search intent (case-insensitive)
        search_patterns = [
            # "shops/restaurants/places in <City>"
            r"\b(?:shops?|stores?|restaurants?|cafes?|places?|hotels?|bars?)\s+(?:in|near|around)\s+",
            # Explicit "check/search/look" + "online/web/internet"
            r"\b(?:check|search|look|browse|find)\b.{0,20}\b(?:online|web|internet|it up)\b",
            r"\b(?:do|run|make)\b.{0,20}\b(?:web\s+search|search\s+(?:the\s+)?web|search\s+online)\b",
            r"\b(?:web\s+search|search\s+(?:the\s+)?web|search\s+online)\b",
            # "what are the ... in <place>"
            r"\bwhat\s+(?:are|is)\s+(?:the\s+)?(?:best|good|popular|top|cheap)",
            # "is there a ... near/in"
            r"\bis\s+there\s+(?:a|an|any)\s+\w+.{0,30}\b(?:near|in|around)\b",
            # "open at/before/after <time>"
            r"\bopen\s+(?:at|before|after|by|from|until)\b",
        ]
        for pattern in search_patterns:
            if _re.search(pattern, message_lower):
                logger.info(f"🔍 Web search regex matched: {pattern[:50]} (message={message[:80]!r})")
                return True

        # Case-sensitive: "in <Capitalized City>" (e.g. "in San Francisco")
        if _re.search(r"\b(?:in|around|near)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", message):
            logger.info(f"🔍 Web search city name matched (message={message[:80]!r})")
            return True

        # If message is short (<60 chars) and contains a city/location name with
        # a question word or verb, it's likely a location query
        if len(message_lower) < 60:
            location_regex = _re.search(
                r"\b(?:in|near|around)\s+(?:san\s+francisco|new\s+york|los\s+angeles|chicago|"
                r"seattle|austin|boston|miami|denver|portland|london|paris|tokyo|berlin|"
                r"toronto|vancouver|sydney|sf|nyc|la|soma|downtown|midtown)\b",
                message_lower,
            )
            if location_regex:
                logger.info(f"🔍 Web search location matched: {location_regex.group()} (message={message[:80]!r})")
                return True

        return False
    
    def format_results_for_context(self, results: List[WebSearchResult]) -> str:
        """Format search results as context for the LLM."""
        if not results:
            return ""
        
        formatted = "🔍 **Web Search Results:**\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"**{i}. {result.title}**\n"
            if result.url:
                formatted += f"   URL: {result.url}\n"
            formatted += f"   {result.snippet}\n\n"
        
        formatted += "---\n*Use the above information to answer the user's question.*\n"
        return formatted


# Global instance
web_search = WebSearchService()
