"""
Modular Integration Tools for Resonant Chat
=============================================

Each tool is a separate file that can be easily connected/disconnected
without affecting Resonant Chat's core intelligence.

Pattern (same as code_visualizer / agents_os):
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

# Registry of all modular integration skills
# NOTE: INTEGRATION_SKILLS name kept for now — internal detail, not public API
INTEGRATION_SKILLS = {
    "figma": FigmaSkill(),
    "google_drive": GoogleDriveSkill(),
    "google_calendar": GoogleCalendarSkill(),
    "sigma": SigmaSkill(),
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
