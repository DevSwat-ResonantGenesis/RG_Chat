"""
GitHub & Git Tools
===================

Real executors for github_* and git_* tools.
GitHub tools use the GitHub REST API with user's BYOK token.
Git tools delegate to code_execution_service.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import httpx

from .base import BaseIntegrationSkill

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
CODE_EXEC_URL = os.getenv("CODE_EXECUTION_URL", "http://code_execution_service:8000")
# code_execution_service requires this on every route (see its app/security.py) —
# without it, any container reachable on app-network could run arbitrary shell
# commands there (it has /var/run/docker.sock mounted for its own sandboxing).
CODE_EXEC_INTERNAL_KEY = os.getenv("CODE_EXECUTION_INTERNAL_SERVICE_KEY", "")


class _GitHubTool(BaseIntegrationSkill):
    """Base for GitHub API tools. Requires user's GitHub token via BYOK."""
    api_key_names = ["github", "github-token", "github_token"]
    _method: str = "GET"
    _path: str = "/user/repos"

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        token = self.get_credentials(context)
        if not token:
            return self._no_credentials_error()
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                if self._method == "GET":
                    resp = await client.get(f"{GITHUB_API}{self._path}", headers=headers)
                else:
                    resp = await client.post(f"{GITHUB_API}{self._path}", headers=headers, json=self._build_body(message))
                resp.raise_for_status()
                data = resp.json()
                return {"success": True, "action": self.skill_id, "summary": self._format(data), "data": data}
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                return {"success": False, "action": self.skill_id, "error": "GitHub token expired or invalid. Reconnect in **Settings → Connect Profiles**."}
            return {"success": False, "action": self.skill_id, "error": f"GitHub API error: {e.response.text[:300]}"}
        except Exception as e:
            return {"success": False, "action": self.skill_id, "error": str(e)[:300]}

    def _build_body(self, message: str) -> dict:
        return {}

    def _format(self, data: Any) -> str:
        if isinstance(data, list):
            return f"**{len(data)} items returned.**\n\n" + "\n".join(f"- {item.get('name', item.get('title', str(item)[:100]))}" for item in data[:20])
        return str(data)[:3000]


class GitHubCreateRepoTool(_GitHubTool):
    skill_id = "github_create_repo"; skill_name = "GitHub Create Repo"; _method = "POST"; _path = "/user/repos"
    intent_keywords = ["create repo", "new repository", "github create"]
    def _build_body(self, message: str) -> dict:
        import re
        name = re.sub(r'(?i)(create|new|repo|repository|github|for|me)\s*', '', message).strip().replace(" ", "-")[:40] or "new-repo"
        return {"name": name, "private": False, "auto_init": True}

class GitHubListReposTool(_GitHubTool):
    skill_id = "github_list_repos"; skill_name = "GitHub List Repos"; _path = "/user/repos?sort=updated&per_page=20"
    intent_keywords = ["list repos", "my repos", "github repos"]
    def _format(self, data):
        if not data: return "No repositories found."
        summary = f"**{len(data)} repositories:**\n\n"
        for r in data[:20]:
            summary += f"- **{r.get('full_name', '?')}** — {r.get('description', 'No description')[:80]} ⭐{r.get('stargazers_count', 0)}\n"
        return summary

class GitHubListFilesTool(_GitHubTool):
    skill_id = "github_list_files"; skill_name = "GitHub List Files"; _path = "/repos"
    intent_keywords = ["list files github", "repo files", "github files"]

class GitHubDownloadFileTool(_GitHubTool):
    skill_id = "github_download_file"; skill_name = "GitHub Download File"
    intent_keywords = ["download file github", "get file github"]

class GitHubUploadFileTool(_GitHubTool):
    skill_id = "github_upload_file"; skill_name = "GitHub Upload File"; _method = "POST"
    intent_keywords = ["upload file github", "push file github"]

class GitHubPullRequestTool(_GitHubTool):
    skill_id = "github_pull_request"; skill_name = "GitHub Pull Request"; _method = "POST"
    intent_keywords = ["pull request", "create pr", "open pr"]

class GitHubIssueTool(_GitHubTool):
    skill_id = "github_issue"; skill_name = "GitHub Issue"; _method = "POST"
    intent_keywords = ["github issue", "create issue", "open issue"]

class GitHubCommitTool(_GitHubTool):
    skill_id = "github_commit"; skill_name = "GitHub Commit"
    intent_keywords = ["github commit", "latest commit"]

class GitHubCommentTool(_GitHubTool):
    skill_id = "github_comment"; skill_name = "GitHub Comment"; _method = "POST"
    intent_keywords = ["github comment", "comment on issue"]


# ── Git tools (shell commands via code_execution_service) ──

class _GitTool(BaseIntegrationSkill):
    api_key_names = []
    _cmd: str = "git status"

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{CODE_EXEC_URL}/terminal/execute",
                    json={"command": self._cmd, "timeout": 25},
                    headers={"x-user-id": user_id, "x-internal-service-key": CODE_EXEC_INTERNAL_KEY},
                )
                resp.raise_for_status()
                data = resp.json()
                output = data.get("stdout") or data.get("output") or ""
                stderr = data.get("stderr") or ""
                combined = (output + ("\n" + stderr if stderr else "")).strip()
                return {"success": data.get("exit_code", 0) == 0, "action": self.skill_id, "summary": f"```\n{combined[:3000]}\n```"}
        except Exception as e:
            return {"success": False, "action": self.skill_id, "error": str(e)[:300]}


class GitCloneTool(_GitTool):
    skill_id = "git_clone"; skill_name = "Git Clone"; _cmd = "git clone"; intent_keywords = ["git clone"]

class GitBranchTool(_GitTool):
    skill_id = "git_branch"; skill_name = "Git Branch"; _cmd = "git branch"; intent_keywords = ["git branch"]

class GitMergeTool(_GitTool):
    skill_id = "git_merge"; skill_name = "Git Merge"; _cmd = "git merge"; intent_keywords = ["git merge"]

class GitPushTool(_GitTool):
    skill_id = "git_push"; skill_name = "Git Push"; _cmd = "git push"; intent_keywords = ["git push"]

class GitPullTool(_GitTool):
    skill_id = "git_pull"; skill_name = "Git Pull"; _cmd = "git pull"; intent_keywords = ["git pull"]


GITHUB_TOOLS = {
    "github_create_repo": GitHubCreateRepoTool(),
    "github_list_repos": GitHubListReposTool(),
    "github_list_files": GitHubListFilesTool(),
    "github_download_file": GitHubDownloadFileTool(),
    "github_upload_file": GitHubUploadFileTool(),
    "github_pull_request": GitHubPullRequestTool(),
    "github_issue": GitHubIssueTool(),
    "github_commit": GitHubCommitTool(),
    "github_comment": GitHubCommentTool(),
    "git_clone": GitCloneTool(),
    "git_branch": GitBranchTool(),
    "git_merge": GitMergeTool(),
    "git_push": GitPushTool(),
    "git_pull": GitPullTool(),
}
