"""
Modular Integration Tools for Resonant Chat
=============================================

Each tool is a separate file that can be easily connected/disconnected
without affecting Resonant Chat's core intelligence.

Pattern (same as code_visualizer):
1. Detect intent from user message
2. Check if tool is enabled
3. Fetch user's API key / webhook / credentials
4. Execute the real API call
5. Return structured result to chat pipeline
"""

from .base import BaseIntegrationSkill
from .figma import FigmaSkill
from .google_drive import GoogleDriveSkill
from .google_calendar import GoogleCalendarSkill
from .sigma import SigmaSkill
from .web_tools import WEB_TOOLS
from .memory_tools import MEMORY_TOOLS
from .code_visualizer_tools import CODE_VISUALIZER_TOOLS
from .state_physics_tools import STATE_PHYSICS_TOOLS
from .rabbit_tools import RABBIT_TOOLS
from .dev_tools import DEV_TOOLS
from .github_tools import GITHUB_TOOLS
from .filesystem_tools import FILESYSTEM_TOOLS
from .media_tools import MEDIA_TOOLS
from .email_tools import EMAIL_TOOLS
from .google_docs_tools import GOOGLE_DOCS_TOOLS
from .oauth_integrations import OAUTH_TOOLS

# Registry of all modular integration skills
INTEGRATION_SKILLS = {
    "figma": FigmaSkill(),
    "google_drive": GoogleDriveSkill(),
    "google_calendar": GoogleCalendarSkill(),
    "sigma": SigmaSkill(),
    **WEB_TOOLS,
    **MEMORY_TOOLS,
    **CODE_VISUALIZER_TOOLS,
    **STATE_PHYSICS_TOOLS,
    **RABBIT_TOOLS,
    **DEV_TOOLS,
    **GITHUB_TOOLS,
    **FILESYSTEM_TOOLS,
    **MEDIA_TOOLS,
    **EMAIL_TOOLS,
    **GOOGLE_DOCS_TOOLS,
    **OAUTH_TOOLS,
}


def get_integration_skill(skill_id: str):
    """Get a modular integration skill by ID."""
    return INTEGRATION_SKILLS.get(skill_id)


def is_integration_intent(message: str) -> str | None:
    """Check if a message targets any integration skill. Returns skill_id or None."""
    for skill_id, skill in INTEGRATION_SKILLS.items():
        if skill.detect_intent(message):
            return skill_id
    return None
