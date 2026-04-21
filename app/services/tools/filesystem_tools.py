"""
Filesystem & IDE Tools
========================

Real executors for file_*, grep_search, find_by_name, run_command, command_status.
Calls IDE service.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import httpx

from .base import BaseIntegrationSkill

logger = logging.getLogger(__name__)

IDE_SERVICE_URL = os.getenv("IDE_SERVICE_URL", "http://ide_platform_service:8080")


class _IDETool(BaseIntegrationSkill):
    """Base for IDE/filesystem tools."""
    api_key_names = []
    _endpoint: str = "/ide/file/read"
    _method: str = "POST"

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {"message": message, "user_id": user_id, "action": self.skill_id}
                if self._method == "POST":
                    resp = await client.post(f"{IDE_SERVICE_URL}{self._endpoint}", json=payload, headers={"x-user-id": user_id})
                else:
                    resp = await client.get(f"{IDE_SERVICE_URL}{self._endpoint}", params={"user_id": user_id}, headers={"x-user-id": user_id})
                resp.raise_for_status()
                data = resp.json()
                return {"success": True, "action": self.skill_id, "summary": str(data)[:3000], "data": data}
        except Exception as e:
            return {"success": False, "action": self.skill_id, "error": str(e)[:300]}


class FileReadTool(_IDETool):
    skill_id = "file_read"; skill_name = "File Read"; _endpoint = "/ide/file/read"; intent_keywords = ["read file", "open file"]

class FileWriteTool(_IDETool):
    skill_id = "file_write"; skill_name = "File Write"; _endpoint = "/ide/file/write"; intent_keywords = ["write file", "create file"]

class FileEditTool(_IDETool):
    skill_id = "file_edit"; skill_name = "File Edit"; _endpoint = "/ide/file/edit"; intent_keywords = ["edit file"]

class MultiEditTool(_IDETool):
    skill_id = "multi_edit"; skill_name = "Multi Edit"; _endpoint = "/ide/file/multi-edit"; intent_keywords = ["multi edit"]

class FileListTool(_IDETool):
    skill_id = "file_list"; skill_name = "File List"; _endpoint = "/ide/file/list"; _method = "GET"; intent_keywords = ["list files", "directory listing"]

class FileDeleteTool(_IDETool):
    skill_id = "file_delete"; skill_name = "File Delete"; _endpoint = "/ide/file/delete"; intent_keywords = ["delete file"]

class GrepSearchTool(_IDETool):
    skill_id = "grep_search"; skill_name = "Grep Search"; _endpoint = "/ide/grep"; intent_keywords = ["grep", "search in files"]

class FindByNameTool(_IDETool):
    skill_id = "find_by_name"; skill_name = "Find By Name"; _endpoint = "/ide/find"; intent_keywords = ["find file", "find by name"]

class RunCommandTool(_IDETool):
    skill_id = "run_command"; skill_name = "Run Command"; _endpoint = "/ide/command"; intent_keywords = ["run command", "execute command"]

class CommandStatusTool(_IDETool):
    skill_id = "command_status"; skill_name = "Command Status"; _endpoint = "/ide/command/status"; _method = "GET"; intent_keywords = ["command status"]


class FileDownloadCurlTool(_IDETool):
    skill_id = "file_download_curl"; skill_name = "Download File (curl)"; _endpoint = "/ide/file/download"; intent_keywords = ["download file", "curl download"]

class FileUploadCurlTool(_IDETool):
    skill_id = "file_upload_curl"; skill_name = "Upload File (curl)"; _endpoint = "/ide/file/upload"; intent_keywords = ["upload file", "curl upload"]

class FileExtractZipTool(_IDETool):
    skill_id = "file_extract_zip"; skill_name = "Extract ZIP"; _endpoint = "/ide/file/extract"; intent_keywords = ["extract zip", "unzip"]


FILESYSTEM_TOOLS = {
    "file_read": FileReadTool(),
    "file_write": FileWriteTool(),
    "file_edit": FileEditTool(),
    "multi_edit": MultiEditTool(),
    "file_list": FileListTool(),
    "file_delete": FileDeleteTool(),
    "grep_search": GrepSearchTool(),
    "find_by_name": FindByNameTool(),
    "run_command": RunCommandTool(),
    "command_status": CommandStatusTool(),
    "file_download_curl": FileDownloadCurlTool(),
    "file_upload_curl": FileUploadCurlTool(),
    "file_extract_zip": FileExtractZipTool(),
}
