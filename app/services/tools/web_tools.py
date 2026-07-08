"""
Web & Search Tools
===================

Real executors for all search/web-related tools.
Each tool either delegates to the chat pipeline (web_search, image_search)
or makes direct API calls (fetch_url, weather, stock, time).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict

import httpx

from .base import BaseIntegrationSkill

logger = logging.getLogger(__name__)


# ── fetch_url / read_webpage ──

class FetchUrlTool(BaseIntegrationSkill):
    skill_id = "fetch_url"
    skill_name = "Fetch URL"
    api_key_names = []
    intent_keywords = ["fetch url", "read url", "get page", "open url"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        url = self._extract_url(message)
        if not url:
            return {"success": False, "action": "fetch_url", "error": "No URL found in message."}
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "ResonantChat/1.0"})
                resp.raise_for_status()
                text = resp.text[:8000]
                return {
                    "success": True,
                    "action": "fetch_url",
                    "url": url,
                    "status_code": resp.status_code,
                    "content_length": len(resp.text),
                    "summary": f"**Fetched** [{url}]({url}) ({resp.status_code})\n\n```\n{text}\n```",
                }
        except Exception as e:
            return {"success": False, "action": "fetch_url", "error": f"Failed to fetch {url}: {e}"}

    def _extract_url(self, message: str) -> str | None:
        urls = re.findall(r'https?://[^\s<>"\']+', message)
        return urls[0] if urls else None


class ReadWebpageTool(FetchUrlTool):
    skill_id = "read_webpage"
    skill_name = "Read Webpage"
    intent_keywords = ["read webpage", "read page", "read website"]


class ReadManyPagesTool(BaseIntegrationSkill):
    skill_id = "read_many_pages"
    skill_name = "Read Many Pages"
    api_key_names = []
    intent_keywords = ["read many pages", "read multiple pages"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        urls = re.findall(r'https?://[^\s<>"\']+', message)
        if not urls:
            return {"success": False, "action": "read_many_pages", "error": "No URLs found in message."}
        results = []
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for url in urls[:5]:
                try:
                    resp = await client.get(url, headers={"User-Agent": "ResonantChat/1.0"})
                    results.append({"url": url, "status": resp.status_code, "snippet": resp.text[:2000]})
                except Exception as e:
                    results.append({"url": url, "status": "error", "snippet": str(e)[:200]})
        summary = f"**Fetched {len(results)} pages:**\n\n"
        for r in results:
            summary += f"- [{r['url']}]({r['url']}) — {r['status']}\n"
        return {"success": True, "action": "read_many_pages", "pages": results, "summary": summary}


# ── Search variants (delegate to pipeline with context prefix) ──

class _PipelineSearchTool(BaseIntegrationSkill):
    """Base for tools that delegate to the web search pipeline with a query prefix."""
    _prefix: str = ""
    api_key_names = []

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "action": self.skill_id,
            "delegate_to_pipeline": True,
            "search_prefix": self._prefix,
            "summary": f"{self.skill_name} will be executed through the chat pipeline.",
        }


class RedditSearchTool(_PipelineSearchTool):
    skill_id = "reddit_search"
    skill_name = "Reddit Search"
    _prefix = "site:reddit.com"
    intent_keywords = ["reddit", "search reddit"]


class ImageSearchTool(_PipelineSearchTool):
    skill_id = "image_search"
    skill_name = "Image Search"
    _prefix = ""
    intent_keywords = ["search images", "find images", "image search"]


class NewsSearchTool(_PipelineSearchTool):
    skill_id = "news_search"
    skill_name = "News Search"
    _prefix = "latest news"
    intent_keywords = ["news", "latest news", "headlines"]


class PlacesSearchTool(_PipelineSearchTool):
    skill_id = "places_search"
    skill_name = "Places Search"
    _prefix = "near me"
    intent_keywords = ["places near", "find nearby", "places search"]


class YoutubeSearchTool(_PipelineSearchTool):
    skill_id = "youtube_search"
    skill_name = "YouTube Search"
    _prefix = "site:youtube.com"
    intent_keywords = ["youtube", "search youtube", "find video"]


class DeepResearchTool(_PipelineSearchTool):
    skill_id = "deep_research"
    skill_name = "Deep Research"
    _prefix = ""
    intent_keywords = ["deep research", "research thoroughly", "in-depth research"]


class WikipediaTool(BaseIntegrationSkill):
    skill_id = "wikipedia"
    skill_name = "Wikipedia"
    api_key_names = []
    intent_keywords = ["wikipedia", "wiki"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        query = re.sub(r'(?i)(wikipedia|wiki|search|for|about|look up)\s*', '', message).strip()
        if not query:
            query = message
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_"),
                    headers={"User-Agent": "ResonantChat/1.0"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    title = data.get("title", query)
                    extract = data.get("extract", "No summary available.")
                    url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                    summary = f"**{title}** ([Wikipedia]({url}))\n\n{extract}"
                    return {"success": True, "action": "wikipedia", "summary": summary, "title": title, "url": url}
                else:
                    return {
                        "success": True,
                        "action": "wikipedia",
                        "delegate_to_pipeline": True,
                        "summary": f"Wikipedia article not found for '{query}'. Falling back to web search.",
                    }
        except Exception as e:
            return {"success": False, "action": "wikipedia", "error": str(e)[:300]}


# ── Weather ──

class WeatherTool(_PipelineSearchTool):
    skill_id = "weather"
    skill_name = "Weather"
    _prefix = "current weather forecast"
    intent_keywords = ["weather", "forecast", "temperature", "rain", "snow"]


# ── Stock / Crypto ──

class StockCryptoTool(_PipelineSearchTool):
    skill_id = "stock_crypto"
    skill_name = "Stock & Crypto"
    _prefix = "current price"
    intent_keywords = ["stock", "crypto", "bitcoin", "ethereum", "stock price"]


class StockMarketDataTool(_PipelineSearchTool):
    skill_id = "stock_market_data"
    skill_name = "Stock Market Data"
    _prefix = "stock market data"
    intent_keywords = ["stock market", "market data", "s&p 500", "nasdaq"]


# ── Scraping ──

class ScrapePageTool(FetchUrlTool):
    skill_id = "scrape_page"
    skill_name = "Scrape Page"
    intent_keywords = ["scrape", "scrape page", "extract from page"]


class ScrapePlatformsTool(_PipelineSearchTool):
    skill_id = "scrape_platforms"
    skill_name = "Scrape Platforms"
    _prefix = ""
    intent_keywords = ["scrape platforms", "scrape multiple"]


# ── Utilities ──

class GetCurrentTimeTool(BaseIntegrationSkill):
    skill_id = "get_current_time"
    skill_name = "Current Time"
    api_key_names = []
    intent_keywords = ["current time", "what time", "time now"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "success": True,
            "action": "get_current_time",
            "time_utc": now.isoformat(),
            "summary": f"**Current UTC time:** {now.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        }


class GetSystemInfoTool(BaseIntegrationSkill):
    skill_id = "get_system_info"
    skill_name = "System Info"
    api_key_names = []
    intent_keywords = ["system info", "system status"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "get_system_info",
            "summary": (
                "**Resonant Genesis System**\n\n"
                "- Platform: ResonantGraph AI v2.0\n"
                "- Tools: 208 across 15+ categories\n"
                "- Services: 27 microservices running\n"
                "- Agent Architect: 29 real tools\n"
                "- Memory: Hash Sphere with 3D semantic coordinates\n"
            ),
        }


# ── Platform API (generic proxy) ──

class PlatformApiSearchTool(_PipelineSearchTool):
    skill_id = "platform_api_search"
    skill_name = "Platform API Search"
    _prefix = ""
    intent_keywords = ["platform api", "api search"]


class PlatformApiCallTool(_PipelineSearchTool):
    skill_id = "platform_api_call"
    skill_name = "Platform API Call"
    _prefix = ""
    intent_keywords = ["call api", "api call"]


# ── Registry of all web tools ──

WEB_TOOLS = {
    "fetch_url": FetchUrlTool(),
    "read_webpage": ReadWebpageTool(),
    "read_many_pages": ReadManyPagesTool(),
    "reddit_search": RedditSearchTool(),
    "image_search": ImageSearchTool(),
    "news_search": NewsSearchTool(),
    "places_search": PlacesSearchTool(),
    "youtube_search": YoutubeSearchTool(),
    "deep_research": DeepResearchTool(),
    "wikipedia": WikipediaTool(),
    "weather": WeatherTool(),
    "stock_crypto": StockCryptoTool(),
    "stock_market_data": StockMarketDataTool(),
    "scrape_page": ScrapePageTool(),
    "scrape_platforms": ScrapePlatformsTool(),
    "get_current_time": GetCurrentTimeTool(),
    "get_system_info": GetSystemInfoTool(),
    "platform_api_search": PlatformApiSearchTool(),
    "platform_api_call": PlatformApiCallTool(),
}
