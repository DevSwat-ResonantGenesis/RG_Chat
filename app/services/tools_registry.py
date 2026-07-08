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
    "memory_rag_ask": ToolDefinition(
        id="memory_rag_ask", name="Memory RAG Ask",
        description="Ask an open-ended question directly against the user's stored memory (RAG).",
        icon="memory", category=ToolCategory.MEMORY,
        capabilities=["memory_rag_ask"], credit_cost=25, is_default=True,
    ),
    "memory_universe": ToolDefinition(
        id="memory_universe", name="Memory Universe",
        description="Show a layered/clustered view of the user's memory (short-term/active vs long-term/archived).",
        icon="memory", category=ToolCategory.MEMORY,
        capabilities=["memory_universe"], credit_cost=15, is_default=True,
    ),
    # ── Agent Architect Extra (backfill) ──
    "agents_list": ToolDefinition(
        id="agents_list", name="Agents List",
        description="Agents List (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_list"], credit_cost=10, is_default=True,
    ),
    "agents_create": ToolDefinition(
        id="agents_create", name="Agents Create",
        description="Agents Create (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_create"], credit_cost=10, is_default=True,
    ),
    "agents_start": ToolDefinition(
        id="agents_start", name="Agents Start",
        description="Agents Start (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_start"], credit_cost=10, is_default=True,
    ),
    "agents_stop": ToolDefinition(
        id="agents_stop", name="Agents Stop",
        description="Agents Stop (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_stop"], credit_cost=10, is_default=True,
    ),
    "agents_status": ToolDefinition(
        id="agents_status", name="Agents Status",
        description="Agents Status (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_status"], credit_cost=10, is_default=True,
    ),
    "agents_delete": ToolDefinition(
        id="agents_delete", name="Agents Delete",
        description="Agents Delete (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_delete"], credit_cost=10, is_default=True,
    ),
    "agents_update": ToolDefinition(
        id="agents_update", name="Agents Update",
        description="Agents Update (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_update"], credit_cost=10, is_default=True,
    ),
    "agents_sessions": ToolDefinition(
        id="agents_sessions", name="Agents Sessions",
        description="Agents Sessions (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_sessions"], credit_cost=10, is_default=True,
    ),
    "agents_session_steps": ToolDefinition(
        id="agents_session_steps", name="Agents Session Steps",
        description="Agents Session Steps (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_session_steps"], credit_cost=10, is_default=True,
    ),
    "agents_session_trace": ToolDefinition(
        id="agents_session_trace", name="Agents Session Trace",
        description="Agents Session Trace (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_session_trace"], credit_cost=10, is_default=True,
    ),
    "agents_metrics": ToolDefinition(
        id="agents_metrics", name="Agents Metrics",
        description="Agents Metrics (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_metrics"], credit_cost=10, is_default=True,
    ),
    "agents_session_detail": ToolDefinition(
        id="agents_session_detail", name="Agents Session Detail",
        description="Agents Session Detail (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_session_detail"], credit_cost=10, is_default=True,
    ),
    "agents_session_cancel": ToolDefinition(
        id="agents_session_cancel", name="Agents Session Cancel",
        description="Agents Session Cancel (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_session_cancel"], credit_cost=10, is_default=True,
    ),
    "agents_available_tools": ToolDefinition(
        id="agents_available_tools", name="Agents Available Tools",
        description="Agents Available Tools (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_available_tools"], credit_cost=10, is_default=True,
    ),
    "agents_templates": ToolDefinition(
        id="agents_templates", name="Agents Templates",
        description="Agents Templates (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_templates"], credit_cost=10, is_default=True,
    ),
    "agents_versions": ToolDefinition(
        id="agents_versions", name="Agents Versions",
        description="Agents Versions (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agents_versions"], credit_cost=10, is_default=True,
    ),
    "schedule_agent": ToolDefinition(
        id="schedule_agent", name="Schedule Agent",
        description="Schedule Agent (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["schedule_agent"], credit_cost=10, is_default=True,
    ),
    "run_snapshot": ToolDefinition(
        id="run_snapshot", name="Run Snapshot",
        description="Run Snapshot (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["run_snapshot"], credit_cost=10, is_default=True,
    ),
    "list_workspace_tools": ToolDefinition(
        id="list_workspace_tools", name="List Workspace Tools",
        description="List Workspace Tools (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["list_workspace_tools"], credit_cost=10, is_default=True,
    ),
    "agent_snapshot": ToolDefinition(
        id="agent_snapshot", name="Agent Snapshot",
        description="Agent Snapshot (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["agent_snapshot"], credit_cost=10, is_default=True,
    ),
    "session_log": ToolDefinition(
        id="session_log", name="Session Log",
        description="Session Log (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["session_log"], credit_cost=10, is_default=True,
    ),
    "workspace_snapshot": ToolDefinition(
        id="workspace_snapshot", name="Workspace Snapshot",
        description="Workspace Snapshot (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["workspace_snapshot"], credit_cost=10, is_default=True,
    ),
    "run_agent": ToolDefinition(
        id="run_agent", name="Run Agent",
        description="Run Agent (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["run_agent"], credit_cost=10, is_default=True,
    ),
    "present_options": ToolDefinition(
        id="present_options", name="Present Options",
        description="Present Options (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["present_options"], credit_cost=10, is_default=True,
    ),
    "build_agent": ToolDefinition(
        id="build_agent", name="Build Agent",
        description="Build Agent (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["build_agent"], credit_cost=10, is_default=True,
    ),
    "continue_build": ToolDefinition(
        id="continue_build", name="Continue Build",
        description="Continue Build (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["continue_build"], credit_cost=10, is_default=True,
    ),
    "message_build": ToolDefinition(
        id="message_build", name="Message Build",
        description="Message Build (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["message_build"], credit_cost=10, is_default=True,
    ),
    "stop_run": ToolDefinition(
        id="stop_run", name="Stop Run",
        description="Stop Run (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["stop_run"], credit_cost=10, is_default=True,
    ),
    "set_trigger": ToolDefinition(
        id="set_trigger", name="Set Trigger",
        description="Set Trigger (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["set_trigger"], credit_cost=10, is_default=True,
    ),
    "set_workspace_name": ToolDefinition(
        id="set_workspace_name", name="Set Workspace Name",
        description="Set Workspace Name (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["set_workspace_name"], credit_cost=10, is_default=True,
    ),
    "open_interface_editor": ToolDefinition(
        id="open_interface_editor", name="Open Interface Editor",
        description="Open Interface Editor (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["open_interface_editor"], credit_cost=10, is_default=True,
    ),
    "get_user_memory": ToolDefinition(
        id="get_user_memory", name="Get User Memory",
        description="Get User Memory (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["get_user_memory"], credit_cost=10, is_default=True,
    ),
    "update_user_memory": ToolDefinition(
        id="update_user_memory", name="Update User Memory",
        description="Update User Memory (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["update_user_memory"], credit_cost=10, is_default=True,
    ),
    "list_workspace_databases": ToolDefinition(
        id="list_workspace_databases", name="List Workspace Databases",
        description="List Workspace Databases (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["list_workspace_databases"], credit_cost=10, is_default=True,
    ),
    "query_cross_agent_database": ToolDefinition(
        id="query_cross_agent_database", name="Query Cross Agent Database",
        description="Query Cross Agent Database (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["query_cross_agent_database"], credit_cost=10, is_default=True,
    ),
    "get_credits_info": ToolDefinition(
        id="get_credits_info", name="Get Credits Info",
        description="Get Credits Info (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["get_credits_info"], credit_cost=10, is_default=True,
    ),
    "present_billing_offer": ToolDefinition(
        id="present_billing_offer", name="Present Billing Offer",
        description="Present Billing Offer (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["present_billing_offer"], credit_cost=10, is_default=True,
    ),
    "create_tool": ToolDefinition(
        id="create_tool", name="Create Tool",
        description="Create Tool (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["create_tool"], credit_cost=10, is_default=True,
    ),
    "list_tools": ToolDefinition(
        id="list_tools", name="List Tools",
        description="List Tools (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["list_tools"], credit_cost=10, is_default=True,
    ),
    "delete_tool": ToolDefinition(
        id="delete_tool", name="Delete Tool",
        description="Delete Tool (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["delete_tool"], credit_cost=10, is_default=True,
    ),
    "update_tool": ToolDefinition(
        id="update_tool", name="Update Tool",
        description="Update Tool (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["update_tool"], credit_cost=10, is_default=True,
    ),
    "auto_build_tool": ToolDefinition(
        id="auto_build_tool", name="Auto Build Tool",
        description="Auto Build Tool (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["auto_build_tool"], credit_cost=10, is_default=True,
    ),
    "list_built_tools": ToolDefinition(
        id="list_built_tools", name="List Built Tools",
        description="List Built Tools (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["list_built_tools"], credit_cost=10, is_default=True,
    ),
    "execute_built_tool": ToolDefinition(
        id="execute_built_tool", name="Execute Built Tool",
        description="Execute Built Tool (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["execute_built_tool"], credit_cost=10, is_default=True,
    ),
    "check_tool_exists": ToolDefinition(
        id="check_tool_exists", name="Check Tool Exists",
        description="Check Tool Exists (see tool_executor.py for details).",
        icon="agent", category=ToolCategory.UTILITY, agent_type="architect",
        capabilities=["check_tool_exists"], credit_cost=10, is_default=True,
    ),
    # ── Memory Hash Sphere (backfill) ──
    "memory_read": ToolDefinition(
        id="memory_read", name="Memory Read",
        description="Memory Read (see tool_executor.py for details).",
        icon="memory", category=ToolCategory.MEMORY,
        capabilities=["memory_read"], credit_cost=15, is_default=True,
    ),
    "memory_write": ToolDefinition(
        id="memory_write", name="Memory Write",
        description="Memory Write (see tool_executor.py for details).",
        icon="memory", category=ToolCategory.MEMORY,
        capabilities=["memory_write"], credit_cost=15, is_default=True,
    ),
    "memory_stats": ToolDefinition(
        id="memory_stats", name="Memory Stats",
        description="Memory Stats (see tool_executor.py for details).",
        icon="memory", category=ToolCategory.MEMORY,
        capabilities=["memory_stats"], credit_cost=15, is_default=True,
    ),
    "memory_facts": ToolDefinition(
        id="memory_facts", name="Memory Facts",
        description="Memory Facts (see tool_executor.py for details).",
        icon="memory", category=ToolCategory.MEMORY,
        capabilities=["memory_facts"], credit_cost=15, is_default=True,
    ),
    "hash_sphere_search": ToolDefinition(
        id="hash_sphere_search", name="Hash Sphere Search",
        description="Hash Sphere Search (see tool_executor.py for details).",
        icon="memory", category=ToolCategory.MEMORY,
        capabilities=["hash_sphere_search"], credit_cost=15, is_default=True,
    ),
    "hash_sphere_anchor": ToolDefinition(
        id="hash_sphere_anchor", name="Hash Sphere Anchor",
        description="Hash Sphere Anchor (see tool_executor.py for details).",
        icon="memory", category=ToolCategory.MEMORY,
        capabilities=["hash_sphere_anchor"], credit_cost=15, is_default=True,
    ),
    "hash_sphere_list_anchors": ToolDefinition(
        id="hash_sphere_list_anchors", name="Hash Sphere List Anchors",
        description="Hash Sphere List Anchors (see tool_executor.py for details).",
        icon="memory", category=ToolCategory.MEMORY,
        capabilities=["hash_sphere_list_anchors"], credit_cost=15, is_default=True,
    ),
    "hash_sphere_hash": ToolDefinition(
        id="hash_sphere_hash", name="Hash Sphere Hash",
        description="Hash Sphere Hash (see tool_executor.py for details).",
        icon="memory", category=ToolCategory.MEMORY,
        capabilities=["hash_sphere_hash"], credit_cost=15, is_default=True,
    ),
    "hash_sphere_resonance": ToolDefinition(
        id="hash_sphere_resonance", name="Hash Sphere Resonance",
        description="Hash Sphere Resonance (see tool_executor.py for details).",
        icon="memory", category=ToolCategory.MEMORY,
        capabilities=["hash_sphere_resonance"], credit_cost=15, is_default=True,
    ),
    # ── Code Visualizer Granular (backfill) ──
    "code_visualizer_scan": ToolDefinition(
        id="code_visualizer_scan", name="Code Visualizer Scan",
        description="Code Visualizer Scan (see tool_executor.py for details).",
        icon="code", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["code_visualizer_scan"], credit_cost=20, is_default=True,
    ),
    "code_visualizer_functions": ToolDefinition(
        id="code_visualizer_functions", name="Code Visualizer Functions",
        description="Code Visualizer Functions (see tool_executor.py for details).",
        icon="code", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["code_visualizer_functions"], credit_cost=20, is_default=True,
    ),
    "code_visualizer_trace": ToolDefinition(
        id="code_visualizer_trace", name="Code Visualizer Trace",
        description="Code Visualizer Trace (see tool_executor.py for details).",
        icon="code", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["code_visualizer_trace"], credit_cost=20, is_default=True,
    ),
    "code_visualizer_governance": ToolDefinition(
        id="code_visualizer_governance", name="Code Visualizer Governance",
        description="Code Visualizer Governance (see tool_executor.py for details).",
        icon="code", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["code_visualizer_governance"], credit_cost=20, is_default=True,
    ),
    "code_visualizer_graph": ToolDefinition(
        id="code_visualizer_graph", name="Code Visualizer Graph",
        description="Code Visualizer Graph (see tool_executor.py for details).",
        icon="code", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["code_visualizer_graph"], credit_cost=20, is_default=True,
    ),
    "code_visualizer_pipeline": ToolDefinition(
        id="code_visualizer_pipeline", name="Code Visualizer Pipeline",
        description="Code Visualizer Pipeline (see tool_executor.py for details).",
        icon="code", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["code_visualizer_pipeline"], credit_cost=20, is_default=True,
    ),
    "code_visualizer_filter": ToolDefinition(
        id="code_visualizer_filter", name="Code Visualizer Filter",
        description="Code Visualizer Filter (see tool_executor.py for details).",
        icon="code", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["code_visualizer_filter"], credit_cost=20, is_default=True,
    ),
    "code_visualizer_by_type": ToolDefinition(
        id="code_visualizer_by_type", name="Code Visualizer By Type",
        description="Code Visualizer By Type (see tool_executor.py for details).",
        icon="code", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["code_visualizer_by_type"], credit_cost=20, is_default=True,
    ),
    # ── State Physics (backfill) ──
    "sp_state": ToolDefinition(
        id="sp_state", name="Sp State",
        description="Sp State (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_state"], credit_cost=10, is_default=True,
    ),
    "sp_reset": ToolDefinition(
        id="sp_reset", name="Sp Reset",
        description="Sp Reset (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_reset"], credit_cost=10, is_default=True,
    ),
    "sp_nodes": ToolDefinition(
        id="sp_nodes", name="Sp Nodes",
        description="Sp Nodes (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_nodes"], credit_cost=10, is_default=True,
    ),
    "sp_metrics": ToolDefinition(
        id="sp_metrics", name="Sp Metrics",
        description="Sp Metrics (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_metrics"], credit_cost=10, is_default=True,
    ),
    "sp_identity": ToolDefinition(
        id="sp_identity", name="Sp Identity",
        description="Sp Identity (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_identity"], credit_cost=10, is_default=True,
    ),
    "sp_simulate": ToolDefinition(
        id="sp_simulate", name="Sp Simulate",
        description="Sp Simulate (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_simulate"], credit_cost=10, is_default=True,
    ),
    "sp_galaxy": ToolDefinition(
        id="sp_galaxy", name="Sp Galaxy",
        description="Sp Galaxy (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_galaxy"], credit_cost=10, is_default=True,
    ),
    "sp_demo": ToolDefinition(
        id="sp_demo", name="Sp Demo",
        description="Sp Demo (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_demo"], credit_cost=10, is_default=True,
    ),
    "sp_asymmetry": ToolDefinition(
        id="sp_asymmetry", name="Sp Asymmetry",
        description="Sp Asymmetry (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_asymmetry"], credit_cost=10, is_default=True,
    ),
    "sp_physics_config": ToolDefinition(
        id="sp_physics_config", name="Sp Physics Config",
        description="Sp Physics Config (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_physics_config"], credit_cost=10, is_default=True,
    ),
    "sp_entropy_config": ToolDefinition(
        id="sp_entropy_config", name="Sp Entropy Config",
        description="Sp Entropy Config (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_entropy_config"], credit_cost=10, is_default=True,
    ),
    "sp_entropy_toggle": ToolDefinition(
        id="sp_entropy_toggle", name="Sp Entropy Toggle",
        description="Sp Entropy Toggle (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_entropy_toggle"], credit_cost=10, is_default=True,
    ),
    "sp_entropy_perturbation": ToolDefinition(
        id="sp_entropy_perturbation", name="Sp Entropy Perturbation",
        description="Sp Entropy Perturbation (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_entropy_perturbation"], credit_cost=10, is_default=True,
    ),
    "sp_agent_spawn": ToolDefinition(
        id="sp_agent_spawn", name="Sp Agent Spawn",
        description="Sp Agent Spawn (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_agent_spawn"], credit_cost=10, is_default=True,
    ),
    "sp_agent_step": ToolDefinition(
        id="sp_agent_step", name="Sp Agent Step",
        description="Sp Agent Step (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_agent_step"], credit_cost=10, is_default=True,
    ),
    "sp_agent_kill": ToolDefinition(
        id="sp_agent_kill", name="Sp Agent Kill",
        description="Sp Agent Kill (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_agent_kill"], credit_cost=10, is_default=True,
    ),
    "sp_agents_spawn": ToolDefinition(
        id="sp_agents_spawn", name="Sp Agents Spawn",
        description="Sp Agents Spawn (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_agents_spawn"], credit_cost=10, is_default=True,
    ),
    "sp_agents_kill_all": ToolDefinition(
        id="sp_agents_kill_all", name="Sp Agents Kill All",
        description="Sp Agents Kill All (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_agents_kill_all"], credit_cost=10, is_default=True,
    ),
    "sp_experiment": ToolDefinition(
        id="sp_experiment", name="Sp Experiment",
        description="Sp Experiment (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_experiment"], credit_cost=10, is_default=True,
    ),
    "sp_memory_cost": ToolDefinition(
        id="sp_memory_cost", name="Sp Memory Cost",
        description="Sp Memory Cost (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_memory_cost"], credit_cost=10, is_default=True,
    ),
    "sp_metrics_record": ToolDefinition(
        id="sp_metrics_record", name="Sp Metrics Record",
        description="Sp Metrics Record (see tool_executor.py for details).",
        icon="physics", category=ToolCategory.ANALYSIS,
        capabilities=["sp_metrics_record"], credit_cost=10, is_default=True,
    ),
    # ── Rabbit (backfill) ──
    "create_rabbit_post": ToolDefinition(
        id="create_rabbit_post", name="Create Rabbit Post",
        description="Create Rabbit Post (see tool_executor.py for details).",
        icon="community", category=ToolCategory.UTILITY,
        capabilities=["create_rabbit_post"], credit_cost=5, is_default=True,
    ),
    "list_rabbit_communities": ToolDefinition(
        id="list_rabbit_communities", name="List Rabbit Communities",
        description="List Rabbit Communities (see tool_executor.py for details).",
        icon="community", category=ToolCategory.UTILITY,
        capabilities=["list_rabbit_communities"], credit_cost=5, is_default=True,
    ),
    "list_rabbit_posts": ToolDefinition(
        id="list_rabbit_posts", name="List Rabbit Posts",
        description="List Rabbit Posts (see tool_executor.py for details).",
        icon="community", category=ToolCategory.UTILITY,
        capabilities=["list_rabbit_posts"], credit_cost=5, is_default=True,
    ),
    "rabbit_vote": ToolDefinition(
        id="rabbit_vote", name="Rabbit Vote",
        description="Rabbit Vote (see tool_executor.py for details).",
        icon="community", category=ToolCategory.UTILITY,
        capabilities=["rabbit_vote"], credit_cost=5, is_default=True,
    ),
    "create_rabbit_community": ToolDefinition(
        id="create_rabbit_community", name="Create Rabbit Community",
        description="Create Rabbit Community (see tool_executor.py for details).",
        icon="community", category=ToolCategory.UTILITY,
        capabilities=["create_rabbit_community"], credit_cost=5, is_default=True,
    ),
    "get_rabbit_community": ToolDefinition(
        id="get_rabbit_community", name="Get Rabbit Community",
        description="Get Rabbit Community (see tool_executor.py for details).",
        icon="community", category=ToolCategory.UTILITY,
        capabilities=["get_rabbit_community"], credit_cost=5, is_default=True,
    ),
    "search_rabbit_posts": ToolDefinition(
        id="search_rabbit_posts", name="Search Rabbit Posts",
        description="Search Rabbit Posts (see tool_executor.py for details).",
        icon="community", category=ToolCategory.UTILITY,
        capabilities=["search_rabbit_posts"], credit_cost=5, is_default=True,
    ),
    "get_rabbit_post": ToolDefinition(
        id="get_rabbit_post", name="Get Rabbit Post",
        description="Get Rabbit Post (see tool_executor.py for details).",
        icon="community", category=ToolCategory.UTILITY,
        capabilities=["get_rabbit_post"], credit_cost=5, is_default=True,
    ),
    "delete_rabbit_post": ToolDefinition(
        id="delete_rabbit_post", name="Delete Rabbit Post",
        description="Delete Rabbit Post (see tool_executor.py for details).",
        icon="community", category=ToolCategory.UTILITY,
        capabilities=["delete_rabbit_post"], credit_cost=5, is_default=True,
    ),
    "create_rabbit_comment": ToolDefinition(
        id="create_rabbit_comment", name="Create Rabbit Comment",
        description="Create Rabbit Comment (see tool_executor.py for details).",
        icon="community", category=ToolCategory.UTILITY,
        capabilities=["create_rabbit_comment"], credit_cost=5, is_default=True,
    ),
    "list_rabbit_comments": ToolDefinition(
        id="list_rabbit_comments", name="List Rabbit Comments",
        description="List Rabbit Comments (see tool_executor.py for details).",
        icon="community", category=ToolCategory.UTILITY,
        capabilities=["list_rabbit_comments"], credit_cost=5, is_default=True,
    ),
    "delete_rabbit_comment": ToolDefinition(
        id="delete_rabbit_comment", name="Delete Rabbit Comment",
        description="Delete Rabbit Comment (see tool_executor.py for details).",
        icon="community", category=ToolCategory.UTILITY,
        capabilities=["delete_rabbit_comment"], credit_cost=5, is_default=True,
    ),
    # ── Dev Tools (backfill) ──
    "execute_code": ToolDefinition(
        id="execute_code", name="Execute Code",
        description="Execute Code (see tool_executor.py for details).",
        icon="terminal", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["execute_code"], credit_cost=15, is_default=True,
    ),
    "http_request": ToolDefinition(
        id="http_request", name="Http Request",
        description="Http Request (see tool_executor.py for details).",
        icon="terminal", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["http_request"], credit_cost=15, is_default=True,
    ),
    "external_http_request": ToolDefinition(
        id="external_http_request", name="External Http Request",
        description="External Http Request (see tool_executor.py for details).",
        icon="terminal", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["external_http_request"], credit_cost=15, is_default=True,
    ),
    "dev_tool": ToolDefinition(
        id="dev_tool", name="Dev Tool",
        description="Dev Tool (see tool_executor.py for details).",
        icon="terminal", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["dev_tool"], credit_cost=15, is_default=True,
    ),
    # ── Github Git (backfill) ──
    "github_create_repo": ToolDefinition(
        id="github_create_repo", name="Github Create Repo",
        description="Github Create Repo (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["github_create_repo"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    "github_list_repos": ToolDefinition(
        id="github_list_repos", name="Github List Repos",
        description="Github List Repos (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["github_list_repos"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    "github_list_files": ToolDefinition(
        id="github_list_files", name="Github List Files",
        description="Github List Files (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["github_list_files"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    "github_download_file": ToolDefinition(
        id="github_download_file", name="Github Download File",
        description="Github Download File (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["github_download_file"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    "github_upload_file": ToolDefinition(
        id="github_upload_file", name="Github Upload File",
        description="Github Upload File (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["github_upload_file"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    "github_pull_request": ToolDefinition(
        id="github_pull_request", name="Github Pull Request",
        description="Github Pull Request (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["github_pull_request"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    "github_issue": ToolDefinition(
        id="github_issue", name="Github Issue",
        description="Github Issue (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["github_issue"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    "github_commit": ToolDefinition(
        id="github_commit", name="Github Commit",
        description="Github Commit (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["github_commit"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    "github_comment": ToolDefinition(
        id="github_comment", name="Github Comment",
        description="Github Comment (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["github_comment"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    "git_clone": ToolDefinition(
        id="git_clone", name="Git Clone",
        description="Git Clone (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["git_clone"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    "git_branch": ToolDefinition(
        id="git_branch", name="Git Branch",
        description="Git Branch (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["git_branch"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    "git_merge": ToolDefinition(
        id="git_merge", name="Git Merge",
        description="Git Merge (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["git_merge"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    "git_push": ToolDefinition(
        id="git_push", name="Git Push",
        description="Git Push (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["git_push"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    "git_pull": ToolDefinition(
        id="git_pull", name="Git Pull",
        description="Git Pull (see tool_executor.py for details).",
        icon="github", category=ToolCategory.ANALYSIS, agent_type="code",
        capabilities=["git_pull"], credit_cost=20, requires_api_key="github", is_default=False,
    ),
    # ── Filesystem (backfill) ──
    "file_read": ToolDefinition(
        id="file_read", name="File Read",
        description="File Read (see tool_executor.py for details).",
        icon="file", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["file_read"], credit_cost=10, is_default=True,
    ),
    "file_write": ToolDefinition(
        id="file_write", name="File Write",
        description="File Write (see tool_executor.py for details).",
        icon="file", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["file_write"], credit_cost=10, is_default=True,
    ),
    "file_edit": ToolDefinition(
        id="file_edit", name="File Edit",
        description="File Edit (see tool_executor.py for details).",
        icon="file", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["file_edit"], credit_cost=10, is_default=True,
    ),
    "multi_edit": ToolDefinition(
        id="multi_edit", name="Multi Edit",
        description="Multi Edit (see tool_executor.py for details).",
        icon="file", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["multi_edit"], credit_cost=10, is_default=True,
    ),
    "file_list": ToolDefinition(
        id="file_list", name="File List",
        description="File List (see tool_executor.py for details).",
        icon="file", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["file_list"], credit_cost=10, is_default=True,
    ),
    "file_delete": ToolDefinition(
        id="file_delete", name="File Delete",
        description="File Delete (see tool_executor.py for details).",
        icon="file", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["file_delete"], credit_cost=10, is_default=True,
    ),
    "grep_search": ToolDefinition(
        id="grep_search", name="Grep Search",
        description="Grep Search (see tool_executor.py for details).",
        icon="file", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["grep_search"], credit_cost=10, is_default=True,
    ),
    "find_by_name": ToolDefinition(
        id="find_by_name", name="Find By Name",
        description="Find By Name (see tool_executor.py for details).",
        icon="file", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["find_by_name"], credit_cost=10, is_default=True,
    ),
    "run_command": ToolDefinition(
        id="run_command", name="Run Command",
        description="Run Command (see tool_executor.py for details).",
        icon="file", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["run_command"], credit_cost=10, is_default=True,
    ),
    "command_status": ToolDefinition(
        id="command_status", name="Command Status",
        description="Command Status (see tool_executor.py for details).",
        icon="file", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["command_status"], credit_cost=10, is_default=True,
    ),
    "file_download_curl": ToolDefinition(
        id="file_download_curl", name="File Download Curl",
        description="File Download Curl (see tool_executor.py for details).",
        icon="file", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["file_download_curl"], credit_cost=10, is_default=True,
    ),
    "file_upload_curl": ToolDefinition(
        id="file_upload_curl", name="File Upload Curl",
        description="File Upload Curl (see tool_executor.py for details).",
        icon="file", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["file_upload_curl"], credit_cost=10, is_default=True,
    ),
    "file_extract_zip": ToolDefinition(
        id="file_extract_zip", name="File Extract Zip",
        description="File Extract Zip (see tool_executor.py for details).",
        icon="file", category=ToolCategory.UTILITY, agent_type="code",
        capabilities=["file_extract_zip"], credit_cost=10, is_default=True,
    ),
    # ── Media (backfill) ──
    "generate_image": ToolDefinition(
        id="generate_image", name="Generate Image",
        description="Generate Image (see tool_executor.py for details).",
        icon="media", category=ToolCategory.GENERATION,
        capabilities=["generate_image"], credit_cost=40, is_default=True,
    ),
    "generate_audio": ToolDefinition(
        id="generate_audio", name="Generate Audio",
        description="Generate Audio (see tool_executor.py for details).",
        icon="media", category=ToolCategory.GENERATION,
        capabilities=["generate_audio"], credit_cost=40, is_default=True,
    ),
    "generate_music": ToolDefinition(
        id="generate_music", name="Generate Music",
        description="Generate Music (see tool_executor.py for details).",
        icon="media", category=ToolCategory.GENERATION,
        capabilities=["generate_music"], credit_cost=40, is_default=True,
    ),
    "generate_video": ToolDefinition(
        id="generate_video", name="Generate Video",
        description="Generate Video (see tool_executor.py for details).",
        icon="media", category=ToolCategory.GENERATION,
        capabilities=["generate_video"], credit_cost=40, is_default=True,
    ),
    "generate_chart": ToolDefinition(
        id="generate_chart", name="Generate Chart",
        description="Generate Chart (see tool_executor.py for details).",
        icon="media", category=ToolCategory.GENERATION,
        capabilities=["generate_chart"], credit_cost=40, is_default=True,
    ),
    "visualize": ToolDefinition(
        id="visualize", name="Visualize",
        description="Visualize (see tool_executor.py for details).",
        icon="media", category=ToolCategory.GENERATION,
        capabilities=["visualize"], credit_cost=40, is_default=True,
    ),
    # ── Email Messaging (backfill) ──
    "gmail_send": ToolDefinition(
        id="gmail_send", name="Gmail Send",
        description="Gmail Send (see tool_executor.py for details).",
        icon="mail", category=ToolCategory.UTILITY,
        capabilities=["gmail_send"], credit_cost=15, requires_api_key="google", is_default=False,
    ),
    "gmail_read": ToolDefinition(
        id="gmail_read", name="Gmail Read",
        description="Gmail Read (see tool_executor.py for details).",
        icon="mail", category=ToolCategory.UTILITY,
        capabilities=["gmail_read"], credit_cost=15, requires_api_key="google", is_default=False,
    ),
    "slack_send": ToolDefinition(
        id="slack_send", name="Slack Send",
        description="Slack Send (see tool_executor.py for details).",
        icon="mail", category=ToolCategory.UTILITY,
        capabilities=["slack_send"], credit_cost=15, requires_api_key="slack", is_default=False,
    ),
    "slack_read": ToolDefinition(
        id="slack_read", name="Slack Read",
        description="Slack Read (see tool_executor.py for details).",
        icon="mail", category=ToolCategory.UTILITY,
        capabilities=["slack_read"], credit_cost=15, requires_api_key="slack", is_default=False,
    ),
    "send_email": ToolDefinition(
        id="send_email", name="Send Email",
        description="Send Email (see tool_executor.py for details).",
        icon="mail", category=ToolCategory.UTILITY,
        capabilities=["send_email"], credit_cost=15, is_default=False,
    ),
    "configure_smtp": ToolDefinition(
        id="configure_smtp", name="Configure Smtp",
        description="Configure Smtp (see tool_executor.py for details).",
        icon="mail", category=ToolCategory.UTILITY,
        capabilities=["configure_smtp"], credit_cost=15, is_default=False,
    ),
    "delete_smtp": ToolDefinition(
        id="delete_smtp", name="Delete Smtp",
        description="Delete Smtp (see tool_executor.py for details).",
        icon="mail", category=ToolCategory.UTILITY,
        capabilities=["delete_smtp"], credit_cost=15, is_default=False,
    ),
    # ── Documents (backfill) ──
    "google_sheets": ToolDefinition(
        id="google_sheets", name="Google Sheets",
        description="Google Sheets (see tool_executor.py for details).",
        icon="document", category=ToolCategory.GENERATION,
        capabilities=["google_sheets"], credit_cost=25, requires_api_key="google", is_default=False,
    ),
    "google_docs": ToolDefinition(
        id="google_docs", name="Google Docs",
        description="Google Docs (see tool_executor.py for details).",
        icon="document", category=ToolCategory.GENERATION,
        capabilities=["google_docs"], credit_cost=25, requires_api_key="google", is_default=False,
    ),
    "create_presentation": ToolDefinition(
        id="create_presentation", name="Create Presentation",
        description="Create Presentation (see tool_executor.py for details).",
        icon="document", category=ToolCategory.GENERATION,
        capabilities=["create_presentation"], credit_cost=25, is_default=False,
    ),
    # ── Oauth Integrations (backfill) ──
    "notion": ToolDefinition(
        id="notion", name="Notion",
        description="Notion (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["notion"], credit_cost=15, requires_api_key="notion", is_default=False,
    ),
    "discord": ToolDefinition(
        id="discord", name="Discord",
        description="Discord (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["discord"], credit_cost=15, requires_api_key="discord", is_default=False,
    ),
    "asana": ToolDefinition(
        id="asana", name="Asana",
        description="Asana (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["asana"], credit_cost=15, requires_api_key="asana", is_default=False,
    ),
    "clickup": ToolDefinition(
        id="clickup", name="Clickup",
        description="Clickup (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["clickup"], credit_cost=15, requires_api_key="clickup", is_default=False,
    ),
    "linear": ToolDefinition(
        id="linear", name="Linear",
        description="Linear (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["linear"], credit_cost=15, requires_api_key="linear", is_default=False,
    ),
    "monday": ToolDefinition(
        id="monday", name="Monday",
        description="Monday (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["monday"], credit_cost=15, requires_api_key="monday", is_default=False,
    ),
    "miro": ToolDefinition(
        id="miro", name="Miro",
        description="Miro (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["miro"], credit_cost=15, requires_api_key="miro", is_default=False,
    ),
    "atlassian": ToolDefinition(
        id="atlassian", name="Atlassian",
        description="Atlassian (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["atlassian"], credit_cost=15, requires_api_key="atlassian", is_default=False,
    ),
    "zoom": ToolDefinition(
        id="zoom", name="Zoom",
        description="Zoom (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["zoom"], credit_cost=15, requires_api_key="zoom", is_default=False,
    ),
    "calendly": ToolDefinition(
        id="calendly", name="Calendly",
        description="Calendly (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["calendly"], credit_cost=15, requires_api_key="calendly", is_default=False,
    ),
    "dropbox": ToolDefinition(
        id="dropbox", name="Dropbox",
        description="Dropbox (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["dropbox"], credit_cost=15, requires_api_key="dropbox", is_default=False,
    ),
    "dribbble": ToolDefinition(
        id="dribbble", name="Dribbble",
        description="Dribbble (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["dribbble"], credit_cost=15, requires_api_key="dribbble", is_default=False,
    ),
    "typeform": ToolDefinition(
        id="typeform", name="Typeform",
        description="Typeform (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["typeform"], credit_cost=15, requires_api_key="typeform", is_default=False,
    ),
    "hubspot": ToolDefinition(
        id="hubspot", name="Hubspot",
        description="Hubspot (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["hubspot"], credit_cost=15, requires_api_key="hubspot", is_default=False,
    ),
    "salesforce": ToolDefinition(
        id="salesforce", name="Salesforce",
        description="Salesforce (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["salesforce"], credit_cost=15, requires_api_key="salesforce", is_default=False,
    ),
    "pipedrive": ToolDefinition(
        id="pipedrive", name="Pipedrive",
        description="Pipedrive (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["pipedrive"], credit_cost=15, requires_api_key="pipedrive", is_default=False,
    ),
    "attio": ToolDefinition(
        id="attio", name="Attio",
        description="Attio (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["attio"], credit_cost=15, requires_api_key="attio", is_default=False,
    ),
    "zoho_crm": ToolDefinition(
        id="zoho_crm", name="Zoho Crm",
        description="Zoho Crm (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["zoho_crm"], credit_cost=15, requires_api_key="zoho", is_default=False,
    ),
    "mailchimp": ToolDefinition(
        id="mailchimp", name="Mailchimp",
        description="Mailchimp (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["mailchimp"], credit_cost=15, requires_api_key="mailchimp", is_default=False,
    ),
    "airtable": ToolDefinition(
        id="airtable", name="Airtable",
        description="Airtable (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["airtable"], credit_cost=15, requires_api_key="airtable", is_default=False,
    ),
    "gitlab": ToolDefinition(
        id="gitlab", name="Gitlab",
        description="Gitlab (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["gitlab"], credit_cost=15, requires_api_key="gitlab", is_default=False,
    ),
    "linkedin": ToolDefinition(
        id="linkedin", name="Linkedin",
        description="Linkedin (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["linkedin"], credit_cost=15, requires_api_key="linkedin", is_default=False,
    ),
    "twitter_x": ToolDefinition(
        id="twitter_x", name="Twitter X",
        description="Twitter X (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["twitter_x"], credit_cost=15, requires_api_key="twitter", is_default=False,
    ),
    "xero": ToolDefinition(
        id="xero", name="Xero",
        description="Xero (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["xero"], credit_cost=15, requires_api_key="xero", is_default=False,
    ),
    "microsoft": ToolDefinition(
        id="microsoft", name="Microsoft",
        description="Microsoft (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["microsoft"], credit_cost=15, requires_api_key="microsoft", is_default=False,
    ),
    "youtube": ToolDefinition(
        id="youtube", name="Youtube",
        description="Youtube (see tool_executor.py for details).",
        icon="integration", category=ToolCategory.UTILITY,
        capabilities=["youtube"], credit_cost=15, requires_api_key="youtube", is_default=False,
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
