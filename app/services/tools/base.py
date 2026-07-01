"""
Base Integration Skill
======================

All modular integration skills inherit from this base class.
Each skill implements:
- detect_intent(): Does the user message target this skill?
- get_credentials(): Fetch API key / token from user context
- execute(): Run the actual API call and return structured results
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class BaseIntegrationSkill(ABC):
    """Base class for all modular integration skills."""

    skill_id: str = ""
    skill_name: str = ""
    api_key_names: List[str] = []  # Keys to look for in user_api_keys
    intent_keywords: List[str] = []  # Keywords that signal this skill

    def detect_intent(self, message: str) -> bool:
        """Check if the user message targets this skill."""
        msg = (message or "").strip().lower()
        if not msg:
            return False
        for kw in self.intent_keywords:
            if len(kw) < 10:
                if re.search(r'\b' + re.escape(kw) + r'\b', msg):
                    return True
            else:
                if kw in msg:
                    return True
        return False

    def get_credentials(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract API key / token from the execution context."""
        user_keys = context.get("user_api_keys") or {}
        for key_name in self.api_key_names:
            val = user_keys.get(key_name)
            if val:
                return val
        return None

    async def get_google_access_token(self, context: Dict[str, Any]) -> Optional[str]:
        """Fetch stored Google credentials, exchanging an OAuth refresh
        token (Google's are prefixed '1//') for a short-lived access token.

        Returns None if no credentials are stored at all. Raises if
        credentials are stored but the refresh exchange fails.
        """
        token = self.get_credentials(context)
        if not token:
            return None
        if token.startswith("1//"):
            token = await self._refresh_google_token(token)
        return token

    @staticmethod
    async def _refresh_google_token(refresh_token: str) -> str:
        """Exchange a Google OAuth refresh token for a fresh access token."""
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise ValueError("GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not configured")

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            resp.raise_for_status()
            return resp.json()["access_token"]

    @abstractmethod
    async def execute(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the skill action. Must be implemented by each skill."""
        ...

    def _no_credentials_error(self) -> Dict[str, Any]:
        """Standard error when credentials are missing."""
        return {
            "success": False,
            "action": self.skill_id,
            "error": (
                f"**{self.skill_name}** is not connected. "
                f"Go to **Settings → Connect Profiles** and add your "
                f"{self.skill_name} API key/token to use this skill."
            ),
        }
