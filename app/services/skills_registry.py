"""
Skills Registry Service
========================

Multi-Skill system for Resonant Chat. Skills are modular capabilities
that can be connected/disconnected to the chat pipeline via API.

Each skill:
- Has a unique ID, name, description, icon
- Can be enabled/disabled per user
- Can execute actions on behalf of the user
- Can be routed to specific agents/teams
- Returns structured results to the chat

Built-in skills:
- code_visualizer: Analyze codebases, trace pipelines, navigate code
- web_search: Search the web for information
- image_generation: Generate images with DALL-E
- memory_search: Search user's memory/knowledge base
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SkillCategory(str, Enum):
    ANALYSIS = "analysis"
    SEARCH = "search"
    GENERATION = "generation"
    MEMORY = "memory"
    UTILITY = "utility"


@dataclass
class SkillDefinition:
    """Definition of an available skill."""
    id: str
    name: str
    description: str
    icon: str  # SVG icon name or emoji
    category: SkillCategory
    agent_type: Optional[str] = None  # Maps to agent type for routing
    team_id: Optional[str] = None  # Maps to team for routing
    service_url: Optional[str] = None  # Internal service URL
    capabilities: List[str] = field(default_factory=list)
    credit_cost: int = 0
    requires_api_key: Optional[str] = None  # Provider key needed
    is_default: bool = False  # Enabled by default for new users


# ============================================
# BUILT-IN SKILL DEFINITIONS
# ============================================

BUILTIN_SKILLS: Dict[str, SkillDefinition] = {
    "code_visualizer": SkillDefinition(
        id="code_visualizer",
        name="Code Visualizer",
        description="Analyze codebases, trace execution pipelines, navigate code structure, generate reports, run governance checks. Upload or connect a GitHub repo to analyze.",
        icon="code",
        category=SkillCategory.ANALYSIS,
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
    "web_search": SkillDefinition(
        id="web_search",
        name="Web Search",
        description="Search the web for real-time information, news, documentation, and answers.",
        icon="search",
        category=SkillCategory.SEARCH,
        agent_type="research",
        capabilities=["web_search", "news_search"],
        credit_cost=50,
        requires_api_key="tavily",
        is_default=True,
    ),
    "image_generation": SkillDefinition(
        id="image_generation",
        name="Image Generation",
        description="Generate images using DALL-E 3. Describe what you want and get AI-generated images.",
        icon="image",
        category=SkillCategory.GENERATION,
        capabilities=["generate_image", "edit_image"],
        credit_cost=100,
        requires_api_key="openai",
        is_default=True,
    ),
    "memory_search": SkillDefinition(
        id="memory_search",
        name="Memory Search",
        description="Deep search through your conversation history, memories, and knowledge base.",
        icon="brain",
        category=SkillCategory.MEMORY,
        agent_type="research",
        service_url=os.getenv("MEMORY_SERVICE_URL", "http://memory_service:8000"),
        capabilities=["search_memories", "search_conversations"],
        credit_cost=20,
        is_default=True,
    ),
    "memory_library": SkillDefinition(
        id="memory_library",
        name="Memory Library",
        description="Open your unified memory library with long-term memory, anchors, and recent context.",
        icon="memory",
        category=SkillCategory.MEMORY,
        agent_type="memory",
        capabilities=["open_memory_panel", "browse_memory_library", "memory_timeline"],
        credit_cost=10,
        is_default=True,
    ),
    "agents_os": SkillDefinition(
        id="agents_os",
        name="Agents OS",
        description="Open and query Agents OS for agent orchestration, lifecycle, and execution controls.",
        icon="agents",
        category=SkillCategory.UTILITY,
        agent_type="orchestration",
        capabilities=["open_agents_panel", "list_agents", "agent_operations",
                       "rename_agent", "delete_agent", "update_agent",
                       "start_agent", "stop_agent"],
        credit_cost=25,
        is_default=True,
    ),
    "state_physics": SkillDefinition(
        id="state_physics",
        name="State Physics",
        description="Open the State Physics visualization for real-time state-space and universe analytics.",
        icon="state_physics",
        category=SkillCategory.ANALYSIS,
        agent_type="analysis",
        capabilities=["open_state_physics_panel", "state_metrics", "state_visualization"],
        credit_cost=20,
        is_default=True,
    ),
    "ide_workspace": SkillDefinition(
        id="ide_workspace",
        name="IDE Workspace",
        description="Open IDE workspace tools for coding, terminal execution, and live preview.",
        icon="ide",
        category=SkillCategory.UTILITY,
        agent_type="code",
        capabilities=["open_ide_panel", "workspace_terminal", "workspace_preview"],
        credit_cost=20,
        is_default=True,
    ),
    "rabbit_post": SkillDefinition(
        id="rabbit_post",
        name="Rabbit Post",
        description="Create a post on Rabbit (Reddit-like community). Specify title, body, and community.",
        icon="rabbit",
        category=SkillCategory.UTILITY,
        service_url=os.getenv("RABBIT_API_URL", "http://rabbit_api_service:8000"),
        capabilities=["create_rabbit_post", "list_rabbit_communities"],
        credit_cost=10,
        is_default=True,
    ),
    "google_drive": SkillDefinition(
        id="google_drive",
        name="Google Drive",
        description="Access your Google Drive: list files, search documents, read file contents, and create new files.",
        icon="folder",
        category=SkillCategory.UTILITY,
        capabilities=["list_files", "search_files", "read_file", "create_file"],
        credit_cost=15,
        requires_api_key="google-drive",
        is_default=True,
    ),
    "google_calendar": SkillDefinition(
        id="google_calendar",
        name="Google Calendar",
        description="Access your Google Calendar: list upcoming events, create events, check schedule, and manage meetings.",
        icon="calendar",
        category=SkillCategory.UTILITY,
        capabilities=["list_events", "create_event", "check_availability"],
        credit_cost=15,
        requires_api_key="google-calendar",
        is_default=True,
    ),
    "figma": SkillDefinition(
        id="figma",
        name="Figma",
        description="Access your Figma projects: list files, get design details, inspect components, and export assets.",
        icon="design",
        category=SkillCategory.UTILITY,
        capabilities=["list_files", "get_file", "list_components", "get_styles"],
        credit_cost=15,
        requires_api_key="figma",
        is_default=True,
    ),
    "sigma": SkillDefinition(
        id="sigma",
        name="Sigma Computing",
        description="Access your Sigma Computing dashboards and workbooks: list reports, view analytics, query data.",
        icon="chart",
        category=SkillCategory.ANALYSIS,
        capabilities=["list_workbooks", "get_workbook", "query_data"],
        credit_cost=15,
        requires_api_key="sigma",
        is_default=True,
    ),
}


class SkillsRegistry:
    """
    Registry for managing available skills and user skill preferences.

    Skills can be enabled/disabled per user. The registry tracks which
    skills are active and routes requests to the appropriate skill handler.
    """

    def __init__(self):
        self.skills: Dict[str, SkillDefinition] = dict(BUILTIN_SKILLS)
        # Per-user enabled skills: {user_id: {skill_id: True/False}}
        self._user_skills: Dict[str, Dict[str, bool]] = {}

    def list_skills(self) -> List[Dict[str, Any]]:
        """List all available skills with their definitions."""
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
            for s in self.skills.values()
        ]

    def get_skill(self, skill_id: str) -> Optional[SkillDefinition]:
        """Get a skill definition by ID."""
        return self.skills.get(skill_id)

    def get_user_skills(self, user_id: str) -> Dict[str, bool]:
        """Get enabled/disabled status of all skills for a user."""
        if user_id not in self._user_skills:
            # Initialize with defaults
            self._user_skills[user_id] = {
                sid: s.is_default for sid, s in self.skills.items()
            }
        return self._user_skills[user_id]

    def get_enabled_skills(self, user_id: str) -> List[SkillDefinition]:
        """Get list of enabled skills for a user."""
        user_prefs = self.get_user_skills(user_id)
        return [
            self.skills[sid]
            for sid, enabled in user_prefs.items()
            if enabled and sid in self.skills
        ]

    def enable_skill(self, user_id: str, skill_id: str) -> bool:
        """Enable a skill for a user."""
        if skill_id not in self.skills:
            return False
        if user_id not in self._user_skills:
            self._user_skills[user_id] = {
                sid: s.is_default for sid, s in self.skills.items()
            }
        self._user_skills[user_id][skill_id] = True
        logger.info(f"Skill {skill_id} enabled for user {user_id}")
        return True

    def disable_skill(self, user_id: str, skill_id: str) -> bool:
        """Disable a skill for a user."""
        if skill_id not in self.skills:
            return False
        if user_id not in self._user_skills:
            self._user_skills[user_id] = {
                sid: s.is_default for sid, s in self.skills.items()
            }
        self._user_skills[user_id][skill_id] = False
        logger.info(f"Skill {skill_id} disabled for user {user_id}")
        return True


    def register_skill(self, skill: SkillDefinition) -> None:
        """Register a new skill (for plugins/extensions)."""
        self.skills[skill.id] = skill
        logger.info(f"Registered skill: {skill.id}")

    def unregister_skill(self, skill_id: str) -> bool:
        """Unregister a skill."""
        if skill_id in self.skills:
            del self.skills[skill_id]
            logger.info(f"Unregistered skill: {skill_id}")
            return True
        return False


# Global singleton
skills_registry = SkillsRegistry()
