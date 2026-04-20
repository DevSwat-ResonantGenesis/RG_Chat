"""
Sigma Integration Skill
========================

Access Sigma Computing dashboards and workbooks:
list workbooks, view dashboards, query data, export reports.

Requires: Sigma API token (client_id + client_secret or bearer token)
API Docs: https://help.sigmacomputing.com/reference/
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from .base import BaseIntegrationSkill

logger = logging.getLogger(__name__)

SIGMA_API = "https://aws-api.sigmacomputing.com/v2"


class SigmaSkill(BaseIntegrationSkill):
    skill_id = "sigma"
    skill_name = "Sigma Computing"
    api_key_names = ["sigma", "sigma_token", "sigma-token", "sigma_api_key"]
    intent_keywords = [
        "sigma", "my sigma", "sigma account",
        "sigma dashboard", "sigma dashboards",
        "sigma workbook", "sigma workbooks",
        "sigma report", "sigma reports",
        "sigma data", "sigma analytics",
        "access sigma", "connect sigma", "open sigma",
        "show sigma", "list sigma",
    ]

    async def execute(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        api_key = self.get_credentials(context)
        if not api_key:
            return self._no_credentials_error()

        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # List workbooks
                resp = await client.get(
                    f"{SIGMA_API}/workbooks",
                    headers=headers,
                    params={"limit": 25},
                )
                resp.raise_for_status()
                data = resp.json()
                workbooks = data.get("entries", data.get("workbooks", []))

                if not workbooks:
                    summary = (
                        "**Sigma Computing** — Connected ✅\n\n"
                        "No workbooks found. Create one in your Sigma dashboard."
                    )
                else:
                    summary = f"**Sigma Computing** ({len(workbooks)} workbooks)\n\n"
                    for wb in workbooks[:25]:
                        name = wb.get("name", wb.get("title", "Untitled"))
                        wid = wb.get("workbookId", wb.get("id", ""))
                        updated = (wb.get("updatedAt") or wb.get("updated_at") or "")[:10]
                        summary += f"- 📊 **{name}** — ID: `{wid[:12]}...` — {updated}\n"
                    summary += "\nTo view a workbook, say: `show sigma workbook <name>`"

                return {
                    "success": True,
                    "action": "list_workbooks",
                    "summary": summary,
                    "workbooks": workbooks,
                }

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                return {
                    "success": False,
                    "action": "sigma",
                    "error": "Sigma access denied — your token may be expired. Reconnect in **Settings → Connect Profiles**.",
                }
            return {
                "success": False,
                "action": "sigma",
                "error": f"Sigma API error: {e.response.text[:200]}",
            }
        except Exception as e:
            return {"success": False, "action": "sigma", "error": str(e)[:300]}
