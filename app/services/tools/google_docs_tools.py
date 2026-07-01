"""
Google Docs / Sheets / Slides Tools
======================================

Real executors for google_sheets, google_docs, create_presentation.
Uses Google APIs with user's OAuth token.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from .base import BaseIntegrationSkill

logger = logging.getLogger(__name__)


class GoogleSheetsTool(BaseIntegrationSkill):
    skill_id = "google_sheets"
    skill_name = "Google Sheets"
    api_key_names = ["google-drive", "google_drive", "gdrive", "google-sheets"]
    intent_keywords = ["google sheets", "spreadsheet", "create sheet"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            token = await self.get_google_access_token(context)
        except Exception as e:
            return {"success": False, "action": "google_sheets", "error": f"Failed to refresh Google Sheets token: {e}. Please reconnect in **Settings → Connect Profiles**."}
        if not token:
            return self._no_credentials_error()
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    "https://sheets.googleapis.com/v4/spreadsheets",
                    headers={"Authorization": f"Bearer {token}"},
                )
                return {"success": True, "action": "google_sheets", "summary": "Google Sheets accessed.", "data": resp.json() if resp.status_code == 200 else {}}
        except Exception as e:
            return {"success": False, "action": "google_sheets", "error": str(e)[:300]}


class GoogleDocsTool(BaseIntegrationSkill):
    skill_id = "google_docs"
    skill_name = "Google Docs"
    api_key_names = ["google-drive", "google_drive", "gdrive", "google-docs"]
    intent_keywords = ["google docs", "create document", "open document"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            token = await self.get_google_access_token(context)
        except Exception as e:
            return {"success": False, "action": "google_docs", "error": f"Failed to refresh Google Docs token: {e}. Please reconnect in **Settings → Connect Profiles**."}
        if not token:
            return self._no_credentials_error()
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    "https://docs.googleapis.com/v1/documents",
                    headers={"Authorization": f"Bearer {token}"},
                )
                return {"success": True, "action": "google_docs", "summary": "Google Docs accessed."}
        except Exception as e:
            return {"success": False, "action": "google_docs", "error": str(e)[:300]}


class CreatePresentationTool(BaseIntegrationSkill):
    skill_id = "create_presentation"
    skill_name = "Create Presentation"
    api_key_names = ["google-drive", "google_drive", "gdrive"]
    intent_keywords = ["create presentation", "google slides", "make slides"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            token = await self.get_google_access_token(context)
        except Exception as e:
            return {"success": False, "action": "create_presentation", "error": f"Failed to refresh Google Slides token: {e}. Please reconnect in **Settings → Connect Profiles**."}
        if not token:
            return self._no_credentials_error()
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    "https://slides.googleapis.com/v1/presentations",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={"title": "New Presentation"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {"success": True, "action": "create_presentation", "summary": f"**Presentation created:** {data.get('presentationId', 'unknown')}"}
                return {"success": False, "action": "create_presentation", "error": f"Google Slides API returned {resp.status_code}"}
        except Exception as e:
            return {"success": False, "action": "create_presentation", "error": str(e)[:300]}


GOOGLE_DOCS_TOOLS = {
    "google_sheets": GoogleSheetsTool(),
    "google_docs": GoogleDocsTool(),
    "create_presentation": CreatePresentationTool(),
}
