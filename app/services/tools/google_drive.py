"""
Google Drive Integration Skill
===============================

Access Google Drive: list files, search documents,
read file contents, and create new files.

Requires: Google Drive OAuth2 access token
API Docs: https://developers.google.com/drive/api/v3/reference
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

import httpx

from .base import BaseIntegrationSkill

logger = logging.getLogger(__name__)

DRIVE_API = "https://www.googleapis.com/drive/v3"


class GoogleDriveSkill(BaseIntegrationSkill):
    skill_id = "google_drive"
    skill_name = "Google Drive"
    api_key_names = ["google-drive", "google_drive", "gdrive", "google-drive-token"]
    intent_keywords = [
        "google drive", "my drive", "my google drive",
        "list files", "list my files", "show my files",
        "search drive", "search my drive", "find in drive",
        "drive files", "drive documents", "drive folders",
        "read document", "open document", "get document",
        "create document", "create file in drive",
        "upload to drive", "save to drive",
        "shared drive", "shared with me",
        "access drive", "connect drive", "open drive",
    ]

    def _detect_action(self, message: str) -> str:
        msg = message.lower()
        if any(k in msg for k in ["search", "find", "look for"]):
            return "search"
        if any(k in msg for k in ["create", "new doc", "new file"]):
            return "create"
        return "list"

    async def execute(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        api_key = self.get_credentials(context)
        if not api_key:
            return self._no_credentials_error()

        action = self._detect_action(message)
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if action == "search":
                    query = re.sub(
                        r"(?i)(search|find|look for|in drive|on drive|google drive|my drive)\s*",
                        "",
                        message,
                    ).strip() or message
                    resp = await client.get(
                        f"{DRIVE_API}/files",
                        headers=headers,
                        params={
                            "q": f"name contains '{query}' and trashed = false",
                            "fields": "files(id,name,mimeType,modifiedTime,webViewLink)",
                            "pageSize": 20,
                            "orderBy": "modifiedTime desc",
                        },
                    )
                else:
                    resp = await client.get(
                        f"{DRIVE_API}/files",
                        headers=headers,
                        params={
                            "q": "trashed = false",
                            "fields": "files(id,name,mimeType,modifiedTime,webViewLink)",
                            "pageSize": 25,
                            "orderBy": "modifiedTime desc",
                        },
                    )

                resp.raise_for_status()
                files = resp.json().get("files", [])

                if not files:
                    summary = "**No files found** in your Google Drive for this query."
                else:
                    summary = f"**Google Drive** ({len(files)} files)\n\n"
                    for f in files[:25]:
                        mime = f.get("mimeType", "")
                        icon = (
                            "📁" if "folder" in mime
                            else "📊" if "spreadsheet" in mime
                            else "📽️" if "presentation" in mime
                            else "📄"
                        )
                        link = f.get("webViewLink", "")
                        modified = (f.get("modifiedTime") or "")[:10]
                        name = f.get("name", "Untitled")
                        summary += f"- {icon} **{name}** — {modified}"
                        if link:
                            summary += f" — [Open]({link})"
                        summary += "\n"

                return {
                    "success": True,
                    "action": action,
                    "summary": summary,
                    "files": files,
                }

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                return {
                    "success": False,
                    "action": "google_drive",
                    "error": "Google Drive access denied — your API key may be expired. Reconnect in **Settings → Connect Profiles**.",
                }
            return {
                "success": False,
                "action": "google_drive",
                "error": f"Google Drive API error: {e.response.text[:200]}",
            }
        except Exception as e:
            return {"success": False, "action": "google_drive", "error": str(e)[:300]}
