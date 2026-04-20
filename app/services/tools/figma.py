"""
Figma Integration Skill
========================

Access Figma projects: list files, get design details,
inspect components, get styles, and export assets.

Requires: Figma Personal Access Token
API Docs: https://www.figma.com/developers/api
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

import httpx

from .base import BaseIntegrationSkill

logger = logging.getLogger(__name__)

FIGMA_API = "https://api.figma.com/v1"


class FigmaSkill(BaseIntegrationSkill):
    skill_id = "figma"
    skill_name = "Figma"
    api_key_names = ["figma", "figma_token", "figma-token"]
    intent_keywords = [
        "figma", "my figma", "figma files", "figma project",
        "figma design", "figma component", "figma styles",
        "design file", "my designs", "figma team",
        "figma pages", "figma frames", "inspect design",
        "export from figma", "access figma", "connect figma",
        "open figma", "show figma",
    ]

    def _detect_action(self, message: str) -> str:
        msg = message.lower()
        # Detect unsupported write actions FIRST
        if any(k in msg for k in [
            "create file", "create new file", "new file",
            "create project", "new project", "make file",
            "create design", "new design", "make design",
            "create a file", "create a new",
            "delete file", "remove file", "rename file",
            "edit file", "modify file", "update file",
        ]):
            return "unsupported_write"
        if any(k in msg for k in ["component", "components", "inspect"]):
            return "components"
        if any(k in msg for k in ["style", "styles", "color", "colors", "typography"]):
            return "styles"
        if any(k in msg for k in ["export", "download", "image"]):
            return "export"
        return "list_files"

    def _extract_file_key(self, message: str) -> str | None:
        match = re.search(
            r"(?:figma\.com/(?:file|design)/|file[_\s]?key[:\s=]+)([a-zA-Z0-9]{20,})",
            message,
        )
        return match.group(1) if match else None

    async def execute(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        api_key = self.get_credentials(context)
        if not api_key:
            return self._no_credentials_error()

        headers = {"X-Figma-Token": api_key}
        action = self._detect_action(message)
        file_key = self._extract_file_key(message)

        # ── Unsupported write actions ─────────────────────────────────
        if action == "unsupported_write":
            return {
                "success": False,
                "action": "unsupported_write",
                "error": (
                    "The Figma API does **not** support creating, deleting, or editing files. "
                    "These actions must be done directly in the Figma app at "
                    "[figma.com](https://www.figma.com). "
                    "I **can** help you with: listing your files, inspecting components, "
                    "viewing styles, and exporting assets. "
                    "Try: `show my figma files` or paste a Figma URL to inspect it."
                ),
            }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if file_key and action == "components":
                    resp = await client.get(
                        f"{FIGMA_API}/files/{file_key}/components", headers=headers
                    )
                    resp.raise_for_status()
                    components = resp.json().get("meta", {}).get("components", [])
                    summary = f"**Figma Components** ({len(components)} found)\n\n"
                    for c in components[:30]:
                        summary += f"- \U0001f9e9 **{c.get('name', 'Unnamed')}** — {c.get('description', '')[:80]}\n"
                    return {
                        "success": True,
                        "action": "components",
                        "summary": summary,
                        "components": components,
                    }

                elif file_key and action == "styles":
                    resp = await client.get(
                        f"{FIGMA_API}/files/{file_key}/styles", headers=headers
                    )
                    resp.raise_for_status()
                    styles = resp.json().get("meta", {}).get("styles", [])
                    summary = f"**Figma Styles** ({len(styles)} found)\n\n"
                    for s in styles[:30]:
                        stype = s.get("style_type", "")
                        summary += f"- \U0001f3a8 **{s.get('name', 'Unnamed')}** ({stype}) — {s.get('description', '')[:80]}\n"
                    return {
                        "success": True,
                        "action": "styles",
                        "summary": summary,
                        "styles": styles,
                    }

                elif file_key:
                    resp = await client.get(
                        f"{FIGMA_API}/files/{file_key}?depth=1", headers=headers
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    doc = data.get("document", {})
                    pages = doc.get("children", [])
                    summary = f"**Figma File: {data.get('name', 'Untitled')}**\n\n"
                    summary += f"- **Last modified**: {data.get('lastModified', 'N/A')[:10]}\n"
                    summary += f"- **Pages**: {len(pages)}\n\n"
                    for p in pages[:20]:
                        children = p.get("children", [])
                        summary += f"- \U0001f4c4 **{p.get('name', 'Untitled')}** — {len(children)} frames\n"
                    return {
                        "success": True,
                        "action": "get_file",
                        "summary": summary,
                    }

                else:
                    # List user info + recent files
                    resp = await client.get(f"{FIGMA_API}/me", headers=headers)
                    resp.raise_for_status()
                    user_data = resp.json()
                    user_name = user_data.get("handle", user_data.get("email", "Unknown"))

                    teams_resp = await client.get(
                        f"{FIGMA_API}/me/files", headers=headers
                    )
                    files = (
                        teams_resp.json().get("files", [])
                        if teams_resp.status_code == 200
                        else []
                    )

                    summary = f"**Figma Account: {user_name}**\n\n"
                    if files:
                        summary += f"**Recent Files** ({len(files)})\n\n"
                        for f in files[:20]:
                            summary += f"- \U0001f3a8 **{f.get('name', 'Untitled')}** — key: `{f.get('key', '')}`\n"
                        summary += "\nTo inspect a file, say: `show figma file <key>` or paste a Figma URL."
                    else:
                        summary += (
                            "No recent files found. Paste a Figma file URL to inspect it:\n"
                            "Example: `show figma https://www.figma.com/file/abc123/My-Design`"
                        )

                    return {
                        "success": True,
                        "action": "list_files",
                        "summary": summary,
                    }

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                return {
                    "success": False,
                    "action": "figma",
                    "error": "Figma access denied — your token may be expired. Reconnect in **Settings → Connect Profiles**.",
                }
            return {
                "success": False,
                "action": "figma",
                "error": f"Figma API error: {e.response.text[:200]}",
            }
        except Exception as e:
            return {"success": False, "action": "figma", "error": str(e)[:300]}
