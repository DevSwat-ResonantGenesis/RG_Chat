"""
Developer Tools
================

Real executors for execute_code, http_request, dev_tool.
Calls code_execution_service and IDE service.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import httpx

from .base import BaseIntegrationSkill

logger = logging.getLogger(__name__)

CODE_EXEC_URL = os.getenv("CODE_EXECUTION_URL", "http://code_execution_service:8000")
IDE_SERVICE_URL = os.getenv("IDE_SERVICE_URL", "http://ide_platform_service:8080")


class ExecuteCodeTool(BaseIntegrationSkill):
    skill_id = "execute_code"
    skill_name = "Execute Code"
    api_key_names = []
    intent_keywords = ["execute code", "run code", "run python", "run javascript"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{CODE_EXEC_URL}/execute",
                    json={"code": message, "user_id": user_id, "language": "python"},
                    headers={"x-user-id": user_id},
                )
                resp.raise_for_status()
                data = resp.json()
                output = data.get("output", data.get("stdout", ""))
                error = data.get("error", data.get("stderr", ""))
                summary = f"**Code Execution Result:**\n\n"
                if output:
                    summary += f"```\n{output[:3000]}\n```\n"
                if error:
                    summary += f"\n**Errors:**\n```\n{error[:1000]}\n```"
                return {"success": True, "action": "execute_code", "summary": summary, "output": output, "error": error}
        except Exception as e:
            return {"success": False, "action": "execute_code", "error": str(e)[:300]}


class HttpRequestTool(BaseIntegrationSkill):
    skill_id = "http_request"
    skill_name = "HTTP Request"
    api_key_names = []
    intent_keywords = ["http request", "api request", "curl"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        import re
        urls = re.findall(r'https?://[^\s<>"\']+', message)
        if not urls:
            return {"success": False, "action": "http_request", "error": "No URL found in message."}
        url = urls[0]
        method = "GET"
        for m in ["POST", "PUT", "DELETE", "PATCH"]:
            if m.lower() in message.lower():
                method = m
                break
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                if method == "GET":
                    resp = await client.get(url)
                else:
                    resp = await client.request(method, url)
                return {
                    "success": True,
                    "action": "http_request",
                    "status_code": resp.status_code,
                    "summary": f"**{method} {url}** → {resp.status_code}\n\n```\n{resp.text[:3000]}\n```",
                }
        except Exception as e:
            return {"success": False, "action": "http_request", "error": str(e)[:300]}


class ExternalHttpRequestTool(HttpRequestTool):
    skill_id = "external_http_request"
    skill_name = "External HTTP Request"
    intent_keywords = ["external request", "fetch external"]


class DevToolTool(BaseIntegrationSkill):
    skill_id = "dev_tool"
    skill_name = "Dev Tool"
    api_key_names = []
    intent_keywords = ["dev tool", "developer tool"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "dev_tool",
            "delegate_to_pipeline": True,
            "summary": "Developer tool request routed to IDE service.",
        }


DEV_TOOLS = {
    "execute_code": ExecuteCodeTool(),
    "http_request": HttpRequestTool(),
    "external_http_request": ExternalHttpRequestTool(),
    "dev_tool": DevToolTool(),
}
