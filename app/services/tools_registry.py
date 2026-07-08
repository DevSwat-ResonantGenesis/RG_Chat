"""
Tools Registry Service
========================

Multi-Tool system for Resonant Chat. Tools are modular capabilities
that can be connected/disconnected to the chat pipeline via API.

Each tool:
- Has a unique ID, name, description, icon
- Can be enabled/disabled per user
- Can execute actions on behalf of the user
- Can be routed to specific agents/teams
- Returns structured results to the chat

Built-in tools:
- code_visualizer: Analyze codebases, trace pipelines, navigate code
- web_search: Search the web for information
- image_generation: Generate images with DALL-E
- memory_search: Search user's memory/knowledge base
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    ANALYSIS = "analysis"
    SEARCH = "search"
    GENERATION = "generation"
    MEMORY = "memory"
    UTILITY = "utility"


@dataclass
class ToolDefinition:
    """Definition of an available tool."""
    id: str
    name: str
    description: str
    icon: str  # SVG icon name or emoji
    category: ToolCategory
    agent_type: Optional[str] = None  # Maps to agent type for routing
    team_id: Optional[str] = None  # Maps to team for routing
    service_url: Optional[str] = None  # Internal service URL
    capabilities: List[str] = field(default_factory=list)
    trigger_keywords: List[str] = field(default_factory=list)
    credit_cost: int = 0
    requires_api_key: Optional[str] = None  # Provider key needed
    is_default: bool = False  # Enabled by default for new users


# ============================================
# BUILT-IN TOOL DEFINITIONS
# ============================================

BUILTIN_TOOLS: Dict[str, ToolDefinition] = {
    "code_visualizer": ToolDefinition(
        id="code_visualizer",
        name="Code Visualizer",
        description="Analyze codebases, trace execution pipelines, navigate code structure, generate reports, run governance checks. Upload or connect a GitHub repo to analyze.",
        icon="code",
        category=ToolCategory.ANALYSIS,
        agent_type="code",
        service_url=os.getenv("AST_ANALYSIS_SERVICE_URL") or os.getenv("CODE_VISUALIZER_URL", "http://rg_ast_analysis:8000"),
        capabilities=[
            "analyze_codebase",
            "trace_pipeline",
            "scan_github",
            "upload_scan",
            "governance_check",
            "list_functions",
            "list_endpoints",
            "filter_pipeline",
            "compare_projects",
            "full_report",
        ],
        credit_cost=200,
        is_default=True,
    ),
    "web_search": ToolDefinition(
        id="web_search",
        name="Web Search",
        description="Search the web for real-time information, news, documentation, and answers.",
        icon="search",
        category=ToolCategory.SEARCH,
        agent_type="research",
        capabilities=["web_search", "news_search"],
        credit_cost=50,
        requires_api_key="tavily",
        is_default=True,
    ),
    "image_generation": ToolDefinition(
        id="image_generation",
        name="Image Generation",
        description="Generate images using DALL-E 3. Describe what you want and get AI-generated images.",
        icon="image",
        category=ToolCategory.GENERATION,
        capabilities=["generate_image", "edit_image"],
        credit_cost=100,
        requires_api_key="openai",
        is_default=True,
    ),
    "memory_search": ToolDefinition(
        id="memory_search",
        name="Memory Search",
        description="Deep search through your conversation history, memories, and knowledge base.",
        icon="brain",
        category=ToolCategory.MEMORY,
        agent_type="research",
        service_url=os.getenv("MEMORY_SERVICE_URL", "http://memory_service:8000"),
        capabilities=["search_memories", "search_conversations"],
        credit_cost=20,
        is_default=True,
    ),
    "memory_library": ToolDefinition(
        id="memory_library",
        name="Memory Library",
        description="Open your unified memory library with long-term memory, anchors, and recent context.",
        icon="memory",
        category=ToolCategory.MEMORY,
        agent_type="memory",
        capabilities=["open_memory_panel", "browse_memory_library", "memory_timeline"],
        credit_cost=10,
        is_default=True,
    ),
    "agent_architect": ToolDefinition(
        id="agent_architect",
        name="Agent Architect",
        description="Autonomous agent builder & orchestrator. Designs, builds, configures, runs, diagnoses, and manages agents using a ReAct loop with real tools, persistent memory, and SSE streaming.",
        icon="agents",
        category=ToolCategory.UTILITY,
        agent_type="orchestration",
        service_url=os.getenv("AGENT_ARCHITECT_URL", "http://agent_architect:8000"),
        capabilities=["build_agent", "run_agent", "modify_agent", "delete_agent",
                       "diagnose_agent", "list_agents", "agent_operations",
                       "set_trigger", "review_runs", "workspace_management"],
        credit_cost=25,
        is_default=True,
    ),
    "state_physics": ToolDefinition(
        id="state_physics",
        name="State Physics",
        description="Open the State Physics visualization for real-time state-space and universe analytics.",
        icon="state_physics",
        category=ToolCategory.ANALYSIS,
        agent_type="analysis",
        capabilities=["open_state_physics_panel", "state_metrics", "state_visualization"],
        credit_cost=20,
        is_default=True,
    ),
    "ide_workspace": ToolDefinition(
        id="ide_workspace",
        name="IDE Workspace",
        description="Open IDE workspace tools for coding, terminal execution, and live preview.",
        icon="ide",
        category=ToolCategory.UTILITY,
        agent_type="code",
        capabilities=["open_ide_panel", "workspace_terminal", "workspace_preview"],
        credit_cost=20,
        is_default=True,
    ),
    "rabbit_post": ToolDefinition(
        id="rabbit_post",
        name="Rabbit Post",
        description="Create a post on Rabbit (Reddit-like community). Specify title, body, and community.",
        icon="rabbit",
        category=ToolCategory.UTILITY,
        service_url=os.getenv("RABBIT_API_URL", "http://rabbit_api_service:8000"),
        capabilities=["create_rabbit_post", "list_rabbit_communities"],
        credit_cost=10,
        is_default=True,
    ),
    "google_drive": ToolDefinition(
        id="google_drive",
        name="Google Drive",
        description="Access your Google Drive: list files, search documents, read file contents, and create new files.",
        icon="folder",
        category=ToolCategory.UTILITY,
        capabilities=["list_files", "search_files", "read_file", "create_file"],
        credit_cost=15,
        requires_api_key="google-drive",
        is_default=True,
    ),
    "google_calendar": ToolDefinition(
        id="google_calendar",
        name="Google Calendar",
        description="Access your Google Calendar: list upcoming events, create events, check schedule, and manage meetings.",
        icon="calendar",
        category=ToolCategory.UTILITY,
        capabilities=["list_events", "create_event", "check_availability"],
        credit_cost=15,
        requires_api_key="google-calendar",
        is_default=True,
    ),
    "figma": ToolDefinition(
        id="figma",
        name="Figma",
        description="Access your Figma projects: list files, get design details, inspect components, and export assets.",
        icon="design",
        category=ToolCategory.UTILITY,
        capabilities=["list_files", "get_file", "list_components", "get_styles"],
        credit_cost=15,
        requires_api_key="figma",
        is_default=True,
    ),
    "sigma": ToolDefinition(
        id="sigma",
        name="Sigma Computing",
        description="Access your Sigma Computing dashboards and workbooks: list reports, view analytics, query data.",
        icon="chart",
        category=ToolCategory.ANALYSIS,
        capabilities=["list_workbooks", "get_workbook", "query_data"],
        credit_cost=15,
        requires_api_key="sigma",
        is_default=True,
    ),
    # ── Granular web/search tools ──
    # The neural tool classifier (tool_classifier.py) predicts these fine-grained
    # IDs directly (not just "web_search"), and tool_executor.py / web_tools.py
    # already implement real handlers for all of them. They must be registered
    # here too, or tools_registry.get_tool() returns None and resonant_chat.py
    # silently drops the tool call before it ever reaches the executor.
    "news_search": ToolDefinition(
        id="news_search", name="News Search",
        description="Search for the latest news and headlines on a topic.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["news_search"], credit_cost=50, requires_api_key="tavily", is_default=True,
    ),
    "weather": ToolDefinition(
        id="weather", name="Weather",
        description="Get the current weather and forecast for a location.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["weather"], credit_cost=50, requires_api_key="tavily", is_default=True,
    ),
    "stock_crypto": ToolDefinition(
        id="stock_crypto", name="Stock & Crypto",
        description="Look up current stock and cryptocurrency prices.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["stock_crypto"], credit_cost=50, requires_api_key="tavily", is_default=True,
    ),
    "stock_market_data": ToolDefinition(
        id="stock_market_data", name="Stock Market Data",
        description="Look up broader stock market data and indices.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["stock_market_data"], credit_cost=50, requires_api_key="tavily", is_default=True,
    ),
    "reddit_search": ToolDefinition(
        id="reddit_search", name="Reddit Search",
        description="Search Reddit discussions and posts on a topic.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["reddit_search"], credit_cost=50, requires_api_key="tavily", is_default=True,
    ),
    "places_search": ToolDefinition(
        id="places_search", name="Places Search",
        description="Find places and businesses near a location.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["places_search"], credit_cost=50, requires_api_key="tavily", is_default=True,
    ),
    "youtube_search": ToolDefinition(
        id="youtube_search", name="YouTube Search",
        description="Search YouTube for videos on a topic.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["youtube_search"], credit_cost=50, requires_api_key="tavily", is_default=True,
    ),
    "image_search": ToolDefinition(
        id="image_search", name="Image Search",
        description="Search the web for images.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["image_search"], credit_cost=50, requires_api_key="tavily", is_default=True,
    ),
    "deep_research": ToolDefinition(
        id="deep_research", name="Deep Research",
        description="Perform in-depth, multi-source research on a topic.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["deep_research"], credit_cost=50, requires_api_key="tavily", is_default=True,
    ),
    "wikipedia": ToolDefinition(
        id="wikipedia", name="Wikipedia",
        description="Look up a Wikipedia article summary.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["wikipedia"], credit_cost=20, is_default=True,
    ),
    "fetch_url": ToolDefinition(
        id="fetch_url", name="Fetch URL",
        description="Fetch and read the raw content of a specific URL.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["fetch_url"], credit_cost=20, is_default=True,
    ),
    "read_webpage": ToolDefinition(
        id="read_webpage", name="Read Webpage",
        description="Read and summarize the content of a webpage.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["read_webpage"], credit_cost=20, is_default=True,
    ),
    "read_many_pages": ToolDefinition(
        id="read_many_pages", name="Read Many Pages",
        description="Fetch and read multiple webpages at once.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["read_many_pages"], credit_cost=30, is_default=True,
    ),
    "scrape_page": ToolDefinition(
        id="scrape_page", name="Scrape Page",
        description="Scrape structured content from a webpage.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["scrape_page"], credit_cost=20, is_default=True,
    ),
    "scrape_platforms": ToolDefinition(
        id="scrape_platforms", name="Scrape Platforms",
        description="Scrape content across multiple platforms for a query.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["scrape_platforms"], credit_cost=50, requires_api_key="tavily", is_default=True,
    ),
    "platform_api_search": ToolDefinition(
        id="platform_api_search", name="Platform API Search",
        description="Search via a connected platform API.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["platform_api_search"], credit_cost=50, requires_api_key="tavily", is_default=True,
    ),
    "platform_api_call": ToolDefinition(
        id="platform_api_call", name="Platform API Call",
        description="Call a connected platform API directly.",
        icon="search", category=ToolCategory.SEARCH, agent_type="research",
        capabilities=["platform_api_call"], credit_cost=50, requires_api_key="tavily", is_default=True,
    ),
    "get_current_time": ToolDefinition(
        id="get_current_time", name="Current Time",
        description="Get the current UTC time.",
        icon="search", category=ToolCategory.UTILITY,
        capabilities=["get_current_time"], credit_cost=5, is_default=True,
    ),
    "get_system_info": ToolDefinition(
        id="get_system_info", name="System Info",
        description="Get Resonant Genesis platform/system status info.",
        icon="search", category=ToolCategory.UTILITY,
        capabilities=["get_system_info"], credit_cost=5, is_default=True,
    ),
}


class ToolsRegistry:
    """
    Registry for managing available tools and user tool preferences.

    Tools can be enabled/disabled per user. The registry tracks which
    tools are active and routes requests to the appropriate tool handler.
    """

    _MAX_USER_CACHE = 500  # evict oldest half when exceeded

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = dict(BUILTIN_TOOLS)
        # Per-user enabled skills: {user_id: {skill_id: True/False}}
        # Uses OrderedDict for LRU eviction to prevent unbounded memory growth
        self._user_tools: OrderedDict[str, Dict[str, bool]] = OrderedDict()

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools with their definitions."""
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "icon": s.icon,
                "category": s.category.value,
                "capabilities": s.capabilities,
                "credit_cost": s.credit_cost,
                "requires_api_key": s.requires_api_key,
                "is_default": s.is_default,
            }
            for s in self.tools.values()
        ]

    def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        """Get a tool definition by ID."""
        return self.tools.get(tool_id)

    def get_user_tools(self, user_id: str) -> Dict[str, bool]:
        """Get enabled/disabled status of all tools for a user."""
        if user_id not in self._user_tools:
            # Evict oldest entries if cache is too large
            if len(self._user_tools) >= self._MAX_USER_CACHE:
                evict_count = self._MAX_USER_CACHE // 2
                for _ in range(evict_count):
                    self._user_tools.popitem(last=False)
                logger.info(f"Evicted {evict_count} stale user tool caches")
            # Initialize with defaults
            self._user_tools[user_id] = {
                sid: s.is_default for sid, s in self.tools.items()
            }
        else:
            # Move to end (most recently used)
            self._user_tools.move_to_end(user_id)
        return self._user_tools[user_id]

    def get_enabled_tools(self, user_id: str) -> List[ToolDefinition]:
        """Get list of enabled tools for a user."""
        user_prefs = self.get_user_tools(user_id)
        return [
            self.tools[sid]
            for sid, enabled in user_prefs.items()
            if enabled and sid in self.tools
        ]

    def enable_tool(self, user_id: str, tool_id: str) -> bool:
        """Enable a tool for a user."""
        if tool_id not in self.tools:
            return False
        if user_id not in self._user_tools:
            self._user_tools[user_id] = {
                sid: s.is_default for sid, s in self.tools.items()
            }
        self._user_tools[user_id][tool_id] = True
        logger.info(f"Tool {tool_id} enabled for user {user_id}")
        return True

    def disable_tool(self, user_id: str, tool_id: str) -> bool:
        """Disable a tool for a user."""
        if tool_id not in self.tools:
            return False
        if user_id not in self._user_tools:
            self._user_tools[user_id] = {
                sid: s.is_default for sid, s in self.tools.items()
            }
        self._user_tools[user_id][tool_id] = False
        logger.info(f"Tool {tool_id} disabled for user {user_id}")
        return True


    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a new tool (for plugins/extensions)."""
        self.tools[tool.id] = tool
        logger.info(f"Registered tool: {tool.id}")

    def unregister_tool(self, tool_id: str) -> bool:
        """Unregister a tool."""
        if tool_id in self.tools:
            del self.tools[tool_id]
            logger.info(f"Unregistered tool: {tool_id}")
            return True
        return False


# Global singleton
tools_registry = ToolsRegistry()
