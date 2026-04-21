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

    # Maps granular classifier tool IDs → parent executor tool IDs.
    # When the neural classifier predicts e.g. "agents_create", resolve to "agent_architect".
    TOOL_RESOLUTION = {
        # agents_* → agent_architect
        "agents_list": "agent_architect",
        "agents_create": "agent_architect",
        "agents_start": "agent_architect",
        "agents_stop": "agent_architect",
        "agents_status": "agent_architect",
        "agents_delete": "agent_architect",
        "agents_sessions": "agent_architect",
        "agents_session_steps": "agent_architect",
        "agents_session_trace": "agent_architect",
        "agents_metrics": "agent_architect",
        "agents_session_detail": "agent_architect",
        "agents_session_cancel": "agent_architect",
        "agents_update": "agent_architect",
        "agents_available_tools": "agent_architect",
        "agents_templates": "agent_architect",
        "agents_versions": "agent_architect",
        "schedule_agent": "agent_architect",
        "run_snapshot": "agent_architect",
        "list_workspace_tools": "agent_architect",
        "agent_snapshot": "agent_architect",
        "session_log": "agent_architect",
        "workspace_snapshot": "agent_architect",
        "run_agent": "agent_architect",
        "present_options": "agent_architect",
        "build_agent": "agent_architect",
        "continue_build": "agent_architect",
        "message_build": "agent_architect",
        "stop_run": "agent_architect",
        "set_trigger": "agent_architect",
        "set_workspace_name": "agent_architect",
        "open_interface_editor": "agent_architect",
        "get_user_memory": "agent_architect",
        "update_user_memory": "agent_architect",
        "list_workspace_databases": "agent_architect",
        "query_cross_agent_database": "agent_architect",
        "get_credits_info": "agent_architect",
        "present_billing_offer": "agent_architect",
        # code_visualizer_* → code_visualizer
        "code_visualizer_scan": "code_visualizer",
        "code_visualizer_functions": "code_visualizer",
        "code_visualizer_trace": "code_visualizer",
        "code_visualizer_governance": "code_visualizer",
        "code_visualizer_graph": "code_visualizer",
        "code_visualizer_pipeline": "code_visualizer",
        "code_visualizer_filter": "code_visualizer",
        "code_visualizer_by_type": "code_visualizer",
        # sp_* → state_physics
        "sp_state": "state_physics",
        "sp_reset": "state_physics",
        "sp_nodes": "state_physics",
        "sp_metrics": "state_physics",
        "sp_identity": "state_physics",
        "sp_simulate": "state_physics",
        "sp_galaxy": "state_physics",
        "sp_demo": "state_physics",
        "sp_asymmetry": "state_physics",
        "sp_physics_config": "state_physics",
        "sp_entropy_config": "state_physics",
        "sp_entropy_toggle": "state_physics",
        "sp_entropy_perturbation": "state_physics",
        "sp_agent_spawn": "state_physics",
        "sp_agent_step": "state_physics",
        "sp_agent_kill": "state_physics",
        "sp_agents_spawn": "state_physics",
        "sp_agents_kill_all": "state_physics",
        "sp_experiment": "state_physics",
        "sp_memory_cost": "state_physics",
        "sp_metrics_record": "state_physics",
        # rabbit_* → rabbit_post
        "create_rabbit_post": "rabbit_post",
        "list_rabbit_communities": "rabbit_post",
        "list_rabbit_posts": "rabbit_post",
        "rabbit_vote": "rabbit_post",
        "create_rabbit_community": "rabbit_post",
        "get_rabbit_community": "rabbit_post",
        "search_rabbit_posts": "rabbit_post",
        "get_rabbit_post": "rabbit_post",
        "delete_rabbit_post": "rabbit_post",
        "create_rabbit_comment": "rabbit_post",
        "list_rabbit_comments": "rabbit_post",
        "delete_rabbit_comment": "rabbit_post",
        # memory_* → memory_search
        "memory_read": "memory_search",
        "memory_write": "memory_search",
        "memory_stats": "memory_search",
        "hash_sphere_search": "memory_search",
        "hash_sphere_anchor": "memory_search",
        "hash_sphere_list_anchors": "memory_search",
        "hash_sphere_hash": "memory_search",
        "hash_sphere_resonance": "memory_search",
        # media → image_generation
        "generate_image": "image_generation",
        "generate_audio": "image_generation",
        "generate_music": "image_generation",
        "generate_video": "image_generation",
        # integrations → google_drive / google_calendar / figma
        "gmail_send": "google_drive",
        "gmail_read": "google_drive",
        "google_sheets": "google_drive",
        "google_docs": "google_drive",
        "slack_send": "google_drive",
        "slack_read": "google_drive",
        "send_email": "google_drive",
        # search variants → web_search
        "fetch_url": "web_search",
        "read_webpage": "web_search",
        "read_many_pages": "web_search",
        "reddit_search": "web_search",
        "image_search": "web_search",
        "news_search": "web_search",
        "places_search": "web_search",
        "youtube_search": "web_search",
        "deep_research": "web_search",
        "wikipedia": "web_search",
        "scrape_page": "web_search",
        "scrape_platforms": "web_search",
        # utilities → web_search (these need live data)
        "weather": "web_search",
        "stock_crypto": "web_search",
        "stock_market_data": "web_search",
        "get_current_time": "web_search",
        # developer tools → code_visualizer (nearest executor)
        "execute_code": "code_visualizer",
        "http_request": "code_visualizer",
        "external_http_request": "code_visualizer",
        "dev_tool": "code_visualizer",
        # filesystem / IDE → ide_workspace
        "file_read": "ide_workspace",
        "file_write": "ide_workspace",
        "file_edit": "ide_workspace",
        "multi_edit": "ide_workspace",
        "file_list": "ide_workspace",
        "file_delete": "ide_workspace",
        "grep_search": "ide_workspace",
        "find_by_name": "ide_workspace",
        "run_command": "ide_workspace",
        "command_status": "ide_workspace",
        "file_download_curl": "ide_workspace",
        "file_upload_curl": "ide_workspace",
        "file_extract_zip": "ide_workspace",
        # git → ide_workspace
        "git_clone": "ide_workspace",
        "git_branch": "ide_workspace",
        "git_merge": "ide_workspace",
        "git_push": "ide_workspace",
        "git_pull": "ide_workspace",
        # github → ide_workspace
        "github_create_repo": "ide_workspace",
        "github_list_repos": "ide_workspace",
        "github_list_files": "ide_workspace",
        "github_download_file": "ide_workspace",
        "github_upload_file": "ide_workspace",
        "github_pull_request": "ide_workspace",
        "github_issue": "ide_workspace",
        "github_commit": "ide_workspace",
        "github_comment": "ide_workspace",
        # visualize/chart → code_visualizer
        "visualize": "code_visualizer",
        "generate_chart": "code_visualizer",
        # documents → google_drive
        "create_presentation": "google_drive",
        # platform → web_search (generic)
        "platform_api_search": "web_search",
        "platform_api_call": "web_search",
        # tool management → agent_architect
        "create_tool": "agent_architect",
        "list_tools": "agent_architect",
        "delete_tool": "agent_architect",
        "update_tool": "agent_architect",
        # auto builder → agent_architect
        "auto_build_tool": "agent_architect",
        "list_built_tools": "agent_architect",
        "execute_built_tool": "agent_architect",
        "check_tool_exists": "agent_architect",
        # smtp → google_drive
        "configure_smtp": "google_drive",
        "delete_smtp": "google_drive",
        # oauth integrations → google_drive (generic integration executor)
        "notion": "google_drive",
        "discord": "google_drive",
        "asana": "google_drive",
        "clickup": "google_drive",
        "linear": "google_drive",
        "monday": "google_drive",
        "miro": "google_drive",
        "atlassian": "google_drive",
        "zoom": "google_drive",
        "calendly": "google_drive",
        "dropbox": "google_drive",
        "dribbble": "google_drive",
        "typeform": "google_drive",
        "hubspot": "google_drive",
        "salesforce": "google_drive",
        "pipedrive": "google_drive",
        "attio": "google_drive",
        "zoho_crm": "google_drive",
        "mailchimp": "google_drive",
        "airtable": "google_drive",
        "gitlab": "google_drive",
        "linkedin": "google_drive",
        "twitter_x": "google_drive",
        "xero": "google_drive",
        "microsoft": "google_drive",
        "youtube": "google_drive",
    }

    def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        """Get a tool definition by ID, with resolution fallback for granular tools."""
        tool = self.tools.get(tool_id)
        if tool is not None:
            return tool
        # Resolve granular tool → parent executor
        parent_id = self.TOOL_RESOLUTION.get(tool_id)
        if parent_id:
            return self.tools.get(parent_id)
        return None

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
