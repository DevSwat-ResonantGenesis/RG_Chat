"""
Tool Executor Service
=======================

Executes tool actions on behalf of users within Resonant Chat.
Each tool has its own executor that handles the specific API calls
and returns structured results for the chat response.
"""

from __future__ import annotations

import logging
import re
import os
from typing import Any, Dict, List, Optional

import httpx

from .tools_registry import ToolDefinition, tools_registry
from .tools import INTEGRATION_SKILLS




logger = logging.getLogger(__name__)

CODE_VISUALIZER_URL = os.getenv("AST_ANALYSIS_SERVICE_URL") or os.getenv("CODE_VISUALIZER_URL", "http://rg_ast_analysis:8000")
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://memory_service:8000")
STATE_PHYSICS_URL = os.getenv("STATE_PHYSICS_URL", "http://rg_users_invarients_sim:8091")
IDE_SERVICE_URL = os.getenv("IDE_SERVICE_URL", "http://ide_platform_service:8080")
AGENT_ARCHITECT_URL = os.getenv("AGENT_ARCHITECT_URL", "http://agent_architect:8000")


class ToolExecutor:
    """Executes tool actions and returns structured results."""

    def __init__(self):
        self._executors = {
            # ── Core tools ──
            "code_visualizer": self._execute_code_visualizer,
            "web_search": self._execute_web_search,
            "image_generation": self._execute_image_generation,
            "memory_search": self._execute_memory_search,
            "memory_library": self._execute_memory_library,
            "state_physics": self._execute_state_physics,
            "ide_workspace": self._execute_ide_workspace,
            "rabbit_post": self._execute_rabbit_post,
            # ── Modular integrations ──
            "google_drive": self._execute_integration,
            "google_calendar": self._execute_integration,
            "figma": self._execute_integration,
            "sigma": self._execute_integration,
            # ── Agent Architect (RG_agent_architect — 29 real tools) ──
            "agent_architect": self._execute_agent_architect,
            "agents_list": self._execute_agent_architect,
            "agents_create": self._execute_agent_architect,
            "agents_start": self._execute_agent_architect,
            "agents_stop": self._execute_agent_architect,
            "agents_status": self._execute_agent_architect,
            "agents_delete": self._execute_agent_architect,
            "agents_update": self._execute_agent_architect,
            "agents_sessions": self._execute_agent_architect,
            "agents_session_steps": self._execute_agent_architect,
            "agents_session_trace": self._execute_agent_architect,
            "agents_metrics": self._execute_agent_architect,
            "agents_session_detail": self._execute_agent_architect,
            "agents_session_cancel": self._execute_agent_architect,
            "agents_available_tools": self._execute_agent_architect,
            "agents_templates": self._execute_agent_architect,
            "agents_versions": self._execute_agent_architect,
            "schedule_agent": self._execute_agent_architect,
            "run_snapshot": self._execute_agent_architect,
            "list_workspace_tools": self._execute_agent_architect,
            "agent_snapshot": self._execute_agent_architect,
            "session_log": self._execute_agent_architect,
            "workspace_snapshot": self._execute_agent_architect,
            "run_agent": self._execute_agent_architect,
            "present_options": self._execute_agent_architect,
            "build_agent": self._execute_agent_architect,
            "continue_build": self._execute_agent_architect,
            "message_build": self._execute_agent_architect,
            "stop_run": self._execute_agent_architect,
            "set_trigger": self._execute_agent_architect,
            "set_workspace_name": self._execute_agent_architect,
            "open_interface_editor": self._execute_agent_architect,
            "get_user_memory": self._execute_agent_architect,
            "update_user_memory": self._execute_agent_architect,
            "list_workspace_databases": self._execute_agent_architect,
            "query_cross_agent_database": self._execute_agent_architect,
            "get_credits_info": self._execute_agent_architect,
            "present_billing_offer": self._execute_agent_architect,
            "create_tool": self._execute_agent_architect,
            "list_tools": self._execute_agent_architect,
            "delete_tool": self._execute_agent_architect,
            "update_tool": self._execute_agent_architect,
            "auto_build_tool": self._execute_agent_architect,
            "list_built_tools": self._execute_agent_architect,
            "execute_built_tool": self._execute_agent_architect,
            "check_tool_exists": self._execute_agent_architect,
            # ── Web & Search tools (modular: web_tools.py) ──
            "fetch_url": self._execute_integration,
            "read_webpage": self._execute_integration,
            "read_many_pages": self._execute_integration,
            "reddit_search": self._execute_integration,
            "image_search": self._execute_integration,
            "news_search": self._execute_integration,
            "places_search": self._execute_integration,
            "youtube_search": self._execute_integration,
            "deep_research": self._execute_integration,
            "wikipedia": self._execute_integration,
            "weather": self._execute_integration,
            "stock_crypto": self._execute_integration,
            "stock_market_data": self._execute_integration,
            "scrape_page": self._execute_integration,
            "scrape_platforms": self._execute_integration,
            "get_current_time": self._execute_integration,
            "get_system_info": self._execute_integration,
            "platform_api_search": self._execute_integration,
            "platform_api_call": self._execute_integration,
            # ── Memory / Hash Sphere tools (modular: memory_tools.py) ──
            "memory_read": self._execute_integration,
            "memory_write": self._execute_integration,
            "memory_stats": self._execute_integration,
            "hash_sphere_search": self._execute_integration,
            "hash_sphere_anchor": self._execute_integration,
            "hash_sphere_list_anchors": self._execute_integration,
            "hash_sphere_hash": self._execute_integration,
            "hash_sphere_resonance": self._execute_integration,
            # ── Code Visualizer granular (modular: code_visualizer_tools.py) ──
            "code_visualizer_scan": self._execute_integration,
            "code_visualizer_functions": self._execute_integration,
            "code_visualizer_trace": self._execute_integration,
            "code_visualizer_governance": self._execute_integration,
            "code_visualizer_graph": self._execute_integration,
            "code_visualizer_pipeline": self._execute_integration,
            "code_visualizer_filter": self._execute_integration,
            "code_visualizer_by_type": self._execute_integration,
            # ── State Physics granular (modular: state_physics_tools.py) ──
            "sp_state": self._execute_integration,
            "sp_reset": self._execute_integration,
            "sp_nodes": self._execute_integration,
            "sp_metrics": self._execute_integration,
            "sp_identity": self._execute_integration,
            "sp_simulate": self._execute_integration,
            "sp_galaxy": self._execute_integration,
            "sp_demo": self._execute_integration,
            "sp_asymmetry": self._execute_integration,
            "sp_physics_config": self._execute_integration,
            "sp_entropy_config": self._execute_integration,
            "sp_entropy_toggle": self._execute_integration,
            "sp_entropy_perturbation": self._execute_integration,
            "sp_agent_spawn": self._execute_integration,
            "sp_agent_step": self._execute_integration,
            "sp_agent_kill": self._execute_integration,
            "sp_agents_spawn": self._execute_integration,
            "sp_agents_kill_all": self._execute_integration,
            "sp_experiment": self._execute_integration,
            "sp_memory_cost": self._execute_integration,
            "sp_metrics_record": self._execute_integration,
            # ── Rabbit Community (modular: rabbit_tools.py) ──
            "create_rabbit_post": self._execute_integration,
            "list_rabbit_communities": self._execute_integration,
            "list_rabbit_posts": self._execute_integration,
            "rabbit_vote": self._execute_integration,
            "create_rabbit_community": self._execute_integration,
            "get_rabbit_community": self._execute_integration,
            "search_rabbit_posts": self._execute_integration,
            "get_rabbit_post": self._execute_integration,
            "delete_rabbit_post": self._execute_integration,
            "create_rabbit_comment": self._execute_integration,
            "list_rabbit_comments": self._execute_integration,
            "delete_rabbit_comment": self._execute_integration,
            # ── Developer tools (modular: dev_tools.py) ──
            "execute_code": self._execute_integration,
            "http_request": self._execute_integration,
            "external_http_request": self._execute_integration,
            "dev_tool": self._execute_integration,
            # ── GitHub + Git (modular: github_tools.py) ──
            "github_create_repo": self._execute_integration,
            "github_list_repos": self._execute_integration,
            "github_list_files": self._execute_integration,
            "github_download_file": self._execute_integration,
            "github_upload_file": self._execute_integration,
            "github_pull_request": self._execute_integration,
            "github_issue": self._execute_integration,
            "github_commit": self._execute_integration,
            "github_comment": self._execute_integration,
            "git_clone": self._execute_integration,
            "git_branch": self._execute_integration,
            "git_merge": self._execute_integration,
            "git_push": self._execute_integration,
            "git_pull": self._execute_integration,
            # ── Filesystem / IDE (modular: filesystem_tools.py) ──
            "file_read": self._execute_integration,
            "file_write": self._execute_integration,
            "file_edit": self._execute_integration,
            "multi_edit": self._execute_integration,
            "file_list": self._execute_integration,
            "file_delete": self._execute_integration,
            "grep_search": self._execute_integration,
            "find_by_name": self._execute_integration,
            "run_command": self._execute_integration,
            "command_status": self._execute_integration,
            "file_download_curl": self._execute_integration,
            "file_upload_curl": self._execute_integration,
            "file_extract_zip": self._execute_integration,
            # ── Media (modular: media_tools.py) ──
            "generate_image": self._execute_integration,
            "generate_audio": self._execute_integration,
            "generate_music": self._execute_integration,
            "generate_video": self._execute_integration,
            "generate_chart": self._execute_integration,
            "visualize": self._execute_integration,
            # ── Email / Messaging (modular: email_tools.py) ──
            "gmail_send": self._execute_integration,
            "gmail_read": self._execute_integration,
            "slack_send": self._execute_integration,
            "slack_read": self._execute_integration,
            "send_email": self._execute_integration,
            "configure_smtp": self._execute_integration,
            "delete_smtp": self._execute_integration,
            # ── Documents (modular: google_docs_tools.py) ──
            "google_sheets": self._execute_integration,
            "google_docs": self._execute_integration,
            "create_presentation": self._execute_integration,
            # ── OAuth Integrations (modular: oauth_integrations.py) ──
            "notion": self._execute_integration,
            "discord": self._execute_integration,
            "asana": self._execute_integration,
            "clickup": self._execute_integration,
            "linear": self._execute_integration,
            "monday": self._execute_integration,
            "miro": self._execute_integration,
            "atlassian": self._execute_integration,
            "zoom": self._execute_integration,
            "calendly": self._execute_integration,
            "dropbox": self._execute_integration,
            "dribbble": self._execute_integration,
            "typeform": self._execute_integration,
            "hubspot": self._execute_integration,
            "salesforce": self._execute_integration,
            "pipedrive": self._execute_integration,
            "attio": self._execute_integration,
            "zoho_crm": self._execute_integration,
            "mailchimp": self._execute_integration,
            "airtable": self._execute_integration,
            "gitlab": self._execute_integration,
            "linkedin": self._execute_integration,
            "twitter_x": self._execute_integration,
            "xero": self._execute_integration,
            "microsoft": self._execute_integration,
            "youtube": self._execute_integration,
        }

    async def execute(
        self,
        tool: ToolDefinition,
        message: str,
        user_id: str,
        user_role: str = "user",
        is_superuser: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a tool and return results."""
        executor = self._executors.get(tool.id)
        if not executor:
            return {
                "tool_id": tool.id,
                "success": False,
                "error": f"No executor for tool: {tool.id}",
            }

        try:
            exec_context = dict(context or {})
            exec_context.setdefault("user_role", user_role)
            exec_context.setdefault("is_superuser", is_superuser)
            exec_context["_integration_tool_id"] = tool.id
            result = await executor(message, user_id, exec_context)
            result["tool_id"] = tool.id
            result["tool_name"] = tool.name
            return result
        except Exception as e:
            logger.error(f"Tool execution failed ({tool.id}): {e}")
            return {
                "tool_id": tool.id,
                "tool_name": tool.name,
                "success": False,
                "error": str(e),
            }

    # ============================================
    # CODE VISUALIZER SKILL
    # ============================================

    def _build_auth(self, user_id: str, context: Dict[str, Any]) -> Dict[str, str]:
        """Build service headers using shared AuthContext when available.

        Internal service-to-service calls use x-user-* headers injected
        by the gateway — NOT JWT Bearer tokens.
        """
        if AuthContext and build_service_headers:
            auth = AuthContext(
                user_id=user_id,
                org_id=context.get("org_id"),
                github_token=(context.get("github_token") or "").strip() or None,
                user_role=str(context.get("user_role", "user")),
                is_superuser=bool(context.get("is_superuser", False)),
                unlimited_credits=bool(context.get("unlimited_credits", False)),
            )
            return build_service_headers(auth)
        # Fallback if shared module not available
        headers = {
            "x-user-id": user_id,
            "x-user-role": str(context.get("user_role", "user")),
            "x-is-superuser": "true" if bool(context.get("is_superuser", False)) else "false",
            "x-unlimited-credits": "true" if bool(context.get("unlimited_credits", False)) else "false",
        }
        github_token = (context.get("github_token") or "").strip()
        if github_token:
            headers["x-github-token"] = github_token
        return headers

    async def _execute_code_visualizer(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Code Visualizer actions based on user message."""
        action = self._detect_cv_action(message)
        headers = self._build_auth(user_id, context)

        async with httpx.AsyncClient(timeout=120.0) as client:
            if action == "scan_github":
                return await self._cv_scan_github(client, message, headers, context)
            elif action == "trace":
                return await self._cv_trace(client, message, headers, context)
            elif action == "list_functions":
                return await self._cv_list_functions(client, headers, context)
            elif action == "list_endpoints":
                return await self._cv_list_endpoints(client, headers, context)
            elif action == "governance":
                return await self._cv_governance(client, headers, context)
            elif action == "reachability":
                return await self._cv_reachability(client, headers, context)
            elif action == "full_pipeline":
                return await self._cv_full_pipeline(client, message, headers, context)
            elif action == "broken_connections":
                return await self._cv_broken_connections(client, headers, context)
            elif action == "list_pipelines":
                return await self._cv_list_pipelines(client, headers, context)
            elif action == "get_analysis":
                return await self._cv_get_analysis(client, headers, context)
            elif action == "list_analyses":
                return await self._cv_list_analyses(client, headers)
            elif action == "rescan":
                return await self._cv_rescan(client, message, headers, context)
            else:
                return await self._cv_analyze_or_help(client, message, headers, context)

    def _detect_cv_action(self, message: str) -> str:
        """Detect which Code Visualizer action to perform."""
        msg = message.lower()

        if any(k in msg for k in ["scan github", "scan repo", "clone repo", "github.com"]):
            return "scan_github"
        if any(k in msg for k in ["trace", "trace for me", "trace execution", "trace pipeline", "trace this"]):
            return "trace"
        if any(k in msg for k in ["list functions", "show functions", "all functions"]):
            return "list_functions"
        if any(k in msg for k in ["list endpoints", "show endpoints", "all endpoints", "api endpoints"]):
            return "list_endpoints"
        if any(k in msg for k in [
            "reachability",
            "reachability analysis",
            "graph janitor",
            "graph janitor scan",
            "graph union",
            "merged graph",
        ]):
            return "reachability"
        if any(k in msg for k in ["governance", "governance check", "compliance"]):
            return "governance"
        if any(k in msg for k in [
            "broken connection", "broken import", "broken dep",
            "list broken", "show broken", "analyze broken",
            "unresolved import", "missing import",
        ]):
            return "broken_connections"
        if any(k in msg for k in [
            "list pipeline", "show pipeline", "all pipeline",
            "list all pipeline", "detected pipeline",
            "pipeline data", "pipeline summary",
        ]):
            return "list_pipelines"
        if any(k in msg for k in ["full pipeline", "complete pipeline", "entire flow"]):
            return "full_pipeline"
        if any(k in msg for k in ["get analysis", "show analysis", "analysis result"]):
            return "get_analysis"
        if any(k in msg for k in ["list analyses", "my analyses", "previous analyses"]):
            return "list_analyses"
        # Re-analyze / rescan patterns: user wants to redo analysis on existing repos
        if any(k in msg for k in [
            "reanalyse", "reanalyze", "re-analyse", "re-analyze",
            "analyse again", "analyze again", "scan again", "rescan", "re-scan",
            "run analysis", "redo analysis", "redo scan",
        ]):
            return "rescan"

        return "analyze_or_help"

    async def _cv_scan_github(
        self, client: httpx.AsyncClient, message: str, headers: Dict, context: Dict
    ) -> Dict[str, Any]:
        """Scan a GitHub repository."""
        # Extract one or more GitHub URLs from message
        repo_urls = [u.rstrip("/.") for u in re.findall(r'https?://github\.com/[\w\-./]+', message)]
        if not repo_urls:
            return {
                "success": False,
                "action": "scan_github",
                "error": "No GitHub URL found in message. Please provide a GitHub repository URL.",
                "hint": "Example: scan github https://github.com/user/repo",
            }

        # Token hints in prompt:
        # 1) token=<PAT> / github_token=<PAT>
        # 2) Natural language: "access token <PAT>", "acces token <PAT>", "token <PAT>"
        # 3) bare GitHub PAT in text (ghp_..., github_pat_..., etc.)
        token_match = re.search(r'(?:github[_\s-]?token|acce?ss?\s+token|token)\s*[:=]?\s*([A-Za-z0-9_\-]{20,})', message, flags=re.IGNORECASE)
        github_token = token_match.group(1) if token_match else None
        if not github_token:
            pat_match = re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,255}|github_pat_[A-Za-z0-9_]{20,255})\b', message)
            github_token = pat_match.group(0) if pat_match else None
        if not github_token:
            context_token = (context.get("github_token") or "").strip()
            github_token = context_token or None
        logger.info(f"🔑 CV GitHub token extraction: found={'yes' if github_token else 'no'}, len={len(github_token) if github_token else 0}, source={'regex' if token_match else 'pat' if github_token else 'context' if context.get('github_token') else 'none'}")

        if len(repo_urls) > 1:
            repos_payload = []
            projects: List[str] = []
            for idx, repo_url in enumerate(repo_urls, start=1):
                parts = repo_url.rstrip("/").split("/")
                project_name = parts[-1] if parts else f"repo{idx}"
                repos_payload.append({
                    "repo_url": repo_url,
                    "label": project_name,
                    **({"token": github_token} if github_token else {}),
                })
                projects.append(project_name)

            try:
                resp = await client.post(
                    f"{CODE_VISUALIZER_URL}/api/v1/scan/github/multi",
                    json={"repos": repos_payload},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

                stats = (data.get("analysis") or {}).get("stats", data.get("stats", {}))
                analysis_id = data.get("analysis_id", "")
                summary = (
                    f"**Multi-Repo Code Analysis Complete ({len(projects)} repos)**\n\n"
                    f"- **Repos**: {', '.join(projects)}\n"
                    f"- **Files**: {stats.get('total_files', 0)}\n"
                    f"- **Services**: {stats.get('total_services', 0)}\n"
                    f"- **Functions**: {stats.get('total_functions', 0)}\n"
                    f"- **Endpoints**: {stats.get('total_endpoints', 0)}\n"
                    f"- **Connections**: {stats.get('total_connections', 0)}\n"
                    f"- **Broken Connections**: {stats.get('broken_connections', 0)}\n\n"
                    f"**Analysis ID**: `{analysis_id}`\n"
                    "You can now ask for cross-repo traces or governance checks on this merged graph."
                )

                return {
                    "success": True,
                    "action": "scan_github_multi",
                    "analysis_id": analysis_id,
                    "project_name": ", ".join(projects),
                    "projects": projects,
                    "stats": stats,
                    "summary": summary,
                    "credits_deducted": data.get("credits_deducted", 0),
                }
            except httpx.HTTPStatusError as e:
                return {
                    "success": False,
                    "action": "scan_github_multi",
                    "error": f"Multi-repo GitHub scan failed: {e.response.text[:300]}",
                }

        repo_url = repo_urls[0]
        parts = repo_url.rstrip("/").split("/")
        project_name = parts[-1] if parts else "repo"

        try:
            resp = await client.post(
                f"{CODE_VISUALIZER_URL}/api/v1/scan/github",
                json={
                    "repo_url": repo_url,
                    "project_name": project_name,
                    **({"token": github_token} if github_token else {}),
                },
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            analysis = data.get("analysis", {})
            stats = analysis.get("stats", {})
            nodes = analysis.get("nodes", [])
            analysis_id = data.get("analysis_id", "")

            # Build summary
            services = [n for n in nodes if n.get("type") == "service"]
            endpoints = [n for n in nodes if n.get("type") == "endpoint"]
            functions = [n for n in nodes if n.get("type") == "function"]

            summary = (
                f"**Code Analysis Complete: {project_name}**\n\n"
                f"- **Files**: {stats.get('total_files', 0)}\n"
                f"- **Services**: {stats.get('total_services', 0)}\n"
                f"- **Functions**: {stats.get('total_functions', 0)}\n"
                f"- **Endpoints**: {stats.get('total_endpoints', 0)}\n"
                f"- **Connections**: {stats.get('total_connections', 0)}\n"
                f"- **Broken Connections**: {stats.get('broken_connections', 0)}\n\n"
            )

            if services:
                summary += "**Services Found:**\n"
                for svc in services[:15]:
                    summary += f"- {svc.get('label', svc.get('id', 'unknown'))}\n"
                if len(services) > 15:
                    summary += f"- ...and {len(services) - 15} more\n"
                summary += "\n"

            if endpoints:
                summary += "**Top Endpoints:**\n"
                for ep in endpoints[:10]:
                    method = ep.get("method", "")
                    route = ep.get("route", ep.get("path", ""))
                    svc = ep.get("service", "")
                    summary += f"- `{method} {route}` ({svc})\n"
                if len(endpoints) > 10:
                    summary += f"- ...and {len(endpoints) - 10} more\n"

            summary += f"\n**Analysis ID**: `{analysis_id}`\n"
            summary += "You can now ask me to trace pipelines, list functions, or run governance checks on this analysis."

            return {
                "success": True,
                "action": "scan_github",
                "analysis_id": analysis_id,
                "project_name": project_name,
                "stats": stats,
                "summary": summary,
                "credits_deducted": data.get("credits_deducted", 0),
            }
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in (401, 403):
                return {
                    "success": False,
                    "action": "scan_github",
                    "error": (
                        "Access denied — this repository may be private. "
                        "Please add a GitHub access token in Settings → API Keys, "
                        "then try again."
                    ),
                }
            return {
                "success": False,
                "action": "scan_github",
                "error": f"GitHub scan failed: {e.response.text[:200]}",
            }

    async def _cv_trace(
        self, client: httpx.AsyncClient, message: str, headers: Dict, context: Dict
    ) -> Dict[str, Any]:
        """Trace execution flow from a starting node."""
        analysis_id = context.get("analysis_id", "")
        if not analysis_id:
            return {
                "success": False,
                "action": "trace",
                "error": "No analysis loaded. Please scan a codebase first (e.g., 'scan github https://github.com/user/repo').",
            }

        # Extract start node from message
        start_node = self._extract_node_name(message)
        max_depth = 10

        # Check for depth specification
        depth_match = re.search(r'depth\s*[:=]?\s*(\d+)', message)
        if depth_match:
            max_depth = min(int(depth_match.group(1)), 50)

        try:
            resp = await client.post(
                f"{CODE_VISUALIZER_URL}/api/analysis/{analysis_id}/trace",
                json={"start_node": start_node, "max_depth": max_depth},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            nodes = data.get("nodes", [])
            connections = data.get("connections", [])

            summary = f"**Trace from `{start_node}`** (depth: {max_depth})\n\n"
            summary += f"- **Nodes found**: {len(nodes)}\n"
            summary += f"- **Connections**: {len(connections)}\n\n"

            if nodes:
                summary += "**Execution Flow:**\n"
                for n in nodes[:20]:
                    ntype = n.get("type", "")
                    label = n.get("label", n.get("id", ""))
                    svc = n.get("service", "")
                    summary += f"- [{ntype}] `{label}` ({svc})\n"
                if len(nodes) > 20:
                    summary += f"- ...and {len(nodes) - 20} more nodes\n"

            if connections:
                summary += "\n**Connection Flow:**\n"
                for c in connections[:15]:
                    src = c.get("source_id", "")
                    tgt = c.get("target_id", "")
                    ctype = c.get("type", "")
                    summary += f"- `{src}` → `{tgt}` ({ctype})\n"
                if len(connections) > 15:
                    summary += f"- ...and {len(connections) - 15} more\n"

            return {
                "success": True,
                "action": "trace",
                "start_node": start_node,
                "node_count": len(nodes),
                "connection_count": len(connections),
                "summary": summary,
                "nodes": nodes[:30],
                "connections": connections[:30],
            }
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "action": "trace",
                "error": f"Trace failed: {e.response.text[:200]}",
            }

    async def _cv_list_functions(
        self, client: httpx.AsyncClient, headers: Dict, context: Dict
    ) -> Dict[str, Any]:
        """List all functions in the analysis."""
        analysis_id = context.get("analysis_id", "")
        if not analysis_id:
            return {"success": False, "action": "list_functions", "error": "No analysis loaded."}

        try:
            resp = await client.get(
                f"{CODE_VISUALIZER_URL}/api/analysis/{analysis_id}/functions",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            functions = data.get("functions", [])

            summary = f"**Functions Found: {len(functions)}**\n\n"
            for f in functions[:30]:
                label = f.get("label", f.get("id", ""))
                svc = f.get("service", "")
                summary += f"- `{label}` ({svc})\n"
            if len(functions) > 30:
                summary += f"- ...and {len(functions) - 30} more\n"

            return {
                "success": True,
                "action": "list_functions",
                "count": len(functions),
                "summary": summary,
            }
        except httpx.HTTPStatusError as e:
            return {"success": False, "action": "list_functions", "error": str(e)}

    async def _cv_list_endpoints(
        self, client: httpx.AsyncClient, headers: Dict, context: Dict
    ) -> Dict[str, Any]:
        """List all endpoints in the analysis."""
        analysis_id = context.get("analysis_id", "")
        if not analysis_id:
            return {"success": False, "action": "list_endpoints", "error": "No analysis loaded."}

        try:
            resp = await client.get(
                f"{CODE_VISUALIZER_URL}/api/analysis/{analysis_id}/by-type/endpoint",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            nodes = data.get("nodes", [])

            summary = f"**Endpoints Found: {len(nodes)}**\n\n"
            for ep in nodes[:30]:
                method = ep.get("method", "")
                route = ep.get("route", ep.get("path", ""))
                svc = ep.get("service", "")
                label = ep.get("label", "")
                summary += f"- `{method} {route}` — {label} ({svc})\n"
            if len(nodes) > 30:
                summary += f"- ...and {len(nodes) - 30} more\n"

            return {
                "success": True,
                "action": "list_endpoints",
                "count": len(nodes),
                "summary": summary,
            }
        except httpx.HTTPStatusError as e:
            return {"success": False, "action": "list_endpoints", "error": str(e)}

    async def _cv_governance(
        self, client: httpx.AsyncClient, headers: Dict, context: Dict
    ) -> Dict[str, Any]:
        """Run governance check on analysis."""
        analysis_id = context.get("analysis_id", "")
        if not analysis_id:
            return {"success": False, "action": "governance", "error": "No analysis loaded."}

        try:
            resp = await client.post(
                f"{CODE_VISUALIZER_URL}/api/analysis/{analysis_id}/governance",
                json={"drift_threshold": 20.0},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            gov = data.get("governance", {})

            summary = "**Governance Report**\n\n"
            summary += f"- **Score**: {gov.get('overall_score', 'N/A')}\n"
            summary += f"- **Live Nodes**: {data.get('live_count', 'N/A')}\n"
            summary += f"- **Invalid Nodes**: {data.get('invalid_count', 'N/A')}\n"
            summary += f"- **Credits Used**: {data.get('credits_deducted', 0)}\n"

            violations = gov.get("violations", [])
            if violations:
                summary += f"\n**Violations ({len(violations)}):**\n"
                for v in violations[:10]:
                    summary += f"- {v}\n"

            return {
                "success": True,
                "action": "governance",
                "summary": summary,
                "governance": gov,
            }
        except httpx.HTTPStatusError as e:
            return {"success": False, "action": "governance", "error": str(e)}

    async def _cv_reachability(
        self, client: httpx.AsyncClient, headers: Dict, context: Dict
    ) -> Dict[str, Any]:
        """Run Graph Janitor reachability scan on analysis."""
        analysis_id = context.get("analysis_id", "")
        if not analysis_id:
            return {"success": False, "action": "reachability", "error": "No analysis loaded."}

        try:
            resp = await client.post(
                f"{CODE_VISUALIZER_URL}/api/analysis/{analysis_id}/agent/scan",
                json={"drift_threshold": 20.0, "max_proposals": 15},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            indicators = data.get("health_indicators") or {}
            metrics = data.get("metrics") or {}
            proposals = data.get("proposals") or []

            summary = "**Reachability Analysis**\n\n"
            summary += f"- **Status**: {indicators.get('status_emoji', '')} {indicators.get('status', 'unknown')}\n"
            summary += f"- **Health Score**: {indicators.get('health_score', 'N/A')}\n"
            summary += f"- **Reachability Score**: {metrics.get('reachability_score', 'N/A')}%\n"
            summary += f"- **Unreachable Nodes**: {metrics.get('unreachable_nodes', 'N/A')}\n"
            summary += f"- **Isolated Nodes**: {metrics.get('isolated_nodes', 'N/A')}\n"
            summary += f"- **Orphan Endpoints**: {metrics.get('orphan_endpoints', 'N/A')}\n"

            recommendations = indicators.get("recommendations") or []
            if recommendations:
                summary += "\n**Recommendations:**\n"
                for rec in recommendations[:5]:
                    summary += f"- {rec}\n"

            if proposals:
                summary += f"\n**Top Proposals ({min(len(proposals), 5)} of {len(proposals)}):**\n"
                for proposal in proposals[:5]:
                    summary += (
                        f"- `{proposal.get('proposal', 'REVIEW')}`"
                        f" — {proposal.get('reason', 'Issue detected')}"
                        f" (risk {proposal.get('risk', 'N/A')})\n"
                    )

            return {
                "success": True,
                "action": "reachability",
                "summary": summary,
                "health_indicators": indicators,
                "metrics": metrics,
                "proposals": proposals,
            }
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "action": "reachability",
                "error": f"Reachability analysis failed: {e.response.text[:300]}",
            }

    async def _cv_full_pipeline(
        self, client: httpx.AsyncClient, message: str, headers: Dict, context: Dict
    ) -> Dict[str, Any]:
        """Trace full pipeline from a starting node."""
        analysis_id = context.get("analysis_id", "")
        if not analysis_id:
            return {"success": False, "action": "full_pipeline", "error": "No analysis loaded."}

        start_node = self._extract_node_name(message)

        try:
            resp = await client.post(
                f"{CODE_VISUALIZER_URL}/api/analysis/{analysis_id}/full-pipeline",
                json={"start_node": start_node, "max_depth": 50},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            nodes = data.get("nodes", [])
            connections = data.get("connections", [])

            summary = f"**Full Pipeline from `{start_node}`**\n\n"
            summary += f"- **Total Nodes**: {len(nodes)}\n"
            summary += f"- **Total Connections**: {len(connections)}\n\n"

            if nodes:
                summary += "**Pipeline Steps:**\n"
                for n in nodes[:25]:
                    summary += f"- [{n.get('type', '')}] `{n.get('label', n.get('id', ''))}`\n"

            return {
                "success": True,
                "action": "full_pipeline",
                "summary": summary,
                "node_count": len(nodes),
                "connection_count": len(connections),
            }
        except httpx.HTTPStatusError as e:
            return {"success": False, "action": "full_pipeline", "error": str(e)}

    async def _cv_broken_connections(
        self, client: httpx.AsyncClient, headers: Dict, context: Dict
    ) -> Dict[str, Any]:
        """Fetch REAL broken connections from the CV analysis and aggregate by service."""
        analysis_id = context.get("analysis_id", "")
        if not analysis_id:
            return {"success": False, "action": "broken_connections", "error": "No analysis loaded. Scan a repo first."}

        try:
            resp = await client.get(
                f"{CODE_VISUALIZER_URL}/api/analysis/{analysis_id}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            analysis = data.get("analysis", data)
            connections = analysis.get("connections", [])
            stats = analysis.get("stats", {})

            broken = [c for c in connections if c.get("status") == "broken"]
            total_broken = len(broken)
            total_connections = len(connections)
            error_rate = (total_broken / total_connections * 100) if total_connections else 0

            # Aggregate by connection type
            by_type: Dict[str, int] = {}
            for c in broken:
                t = c.get("type", "unknown")
                by_type[t] = by_type.get(t, 0) + 1

            # Aggregate by source service
            by_service: Dict[str, int] = {}
            for c in broken:
                src = c.get("source_id", "")
                svc = src.split(":")[0] if ":" in src else "unknown"
                by_service[svc] = by_service.get(svc, 0) + 1
            sorted_services = sorted(by_service.items(), key=lambda x: -x[1])

            # Top broken targets
            targets: Dict[str, int] = {}
            for c in broken:
                t = c.get("target_id", "?")
                targets[t] = targets.get(t, 0) + 1
            sorted_targets = sorted(targets.items(), key=lambda x: -x[1])

            summary = f"**Broken Connections Analysis** (from CV scan)\n\n"
            summary += f"- **Total connections**: {total_connections:,}\n"
            summary += f"- **Broken**: {total_broken:,} ({error_rate:.2f}% error rate)\n"
            summary += f"- **Unique broken targets**: {len(targets):,}\n\n"

            if by_type:
                summary += "**By type:**\n"
                for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
                    summary += f"- {t}: {count:,}\n"
                summary += "\n"

            summary += "**By source service (top 15):**\n"
            for svc, count in sorted_services[:15]:
                summary += f"- `{svc}`: {count:,}\n"
            summary += "\n"

            summary += "**Top 20 broken import targets:**\n"
            for target, count in sorted_targets[:20]:
                summary += f"- `{target}`: {count} references\n"

            return {
                "success": True,
                "action": "broken_connections",
                "total_connections": total_connections,
                "total_broken": total_broken,
                "error_rate": round(error_rate, 2),
                "unique_targets": len(targets),
                "by_type": by_type,
                "by_service": dict(sorted_services[:15]),
                "top_targets": dict(sorted_targets[:20]),
                "summary": summary,
            }
        except httpx.HTTPStatusError as e:
            return {"success": False, "action": "broken_connections", "error": f"Failed: {e.response.text[:300]}"}
        except Exception as e:
            return {"success": False, "action": "broken_connections", "error": str(e)[:300]}

    async def _cv_list_pipelines(
        self, client: httpx.AsyncClient, headers: Dict, context: Dict
    ) -> Dict[str, Any]:
        """Fetch REAL pipeline data from the CV analysis."""
        analysis_id = context.get("analysis_id", "")
        if not analysis_id:
            return {"success": False, "action": "list_pipelines", "error": "No analysis loaded. Scan a repo first."}

        try:
            resp = await client.get(
                f"{CODE_VISUALIZER_URL}/api/analysis/{analysis_id}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            analysis = data.get("analysis", data)
            pipelines = analysis.get("pipelines", {})

            if isinstance(pipelines, list):
                pipelines = {p.get("name", f"pipeline_{i}"): p for i, p in enumerate(pipelines)}

            summary = f"**Pipelines Detected: {len(pipelines)}**\n\n"

            if not pipelines:
                summary += "No pipelines detected in this analysis. Try running a scan first.\n"
            else:
                summary += "| Pipeline | Nodes | Connections | Description |\n"
                summary += "|----------|-------|-------------|-------------|\n"
                for name, pipeline in pipelines.items():
                    if isinstance(pipeline, dict):
                        nodes = pipeline.get("nodes", [])
                        conns = pipeline.get("connections", [])
                        desc = pipeline.get("description", "")
                        node_count = len(nodes) if isinstance(nodes, list) else "?"
                        conn_count = len(conns) if isinstance(conns, list) else "?"
                        summary += f"| `{name}` | {node_count} | {conn_count} | {desc} |\n"
                    else:
                        summary += f"| `{name}` | ? | ? | |\n"

            return {
                "success": True,
                "action": "list_pipelines",
                "pipeline_count": len(pipelines),
                "pipelines": {
                    name: {
                        "nodes": len(p.get("nodes", [])) if isinstance(p, dict) and isinstance(p.get("nodes"), list) else 0,
                        "connections": len(p.get("connections", [])) if isinstance(p, dict) and isinstance(p.get("connections"), list) else 0,
                        "description": p.get("description", "") if isinstance(p, dict) else "",
                    }
                    for name, p in pipelines.items()
                },
                "summary": summary,
            }
        except httpx.HTTPStatusError as e:
            return {"success": False, "action": "list_pipelines", "error": f"Failed: {e.response.text[:300]}"}
        except Exception as e:
            return {"success": False, "action": "list_pipelines", "error": str(e)[:300]}

    async def _cv_get_analysis(
        self, client: httpx.AsyncClient, headers: Dict, context: Dict
    ) -> Dict[str, Any]:
        """Get existing analysis details."""
        analysis_id = context.get("analysis_id", "")
        if not analysis_id:
            return {"success": False, "action": "get_analysis", "error": "No analysis loaded."}

        try:
            resp = await client.get(
                f"{CODE_VISUALIZER_URL}/api/analysis/{analysis_id}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            stats = data.get("stats", {})

            summary = f"**Analysis `{analysis_id[:12]}...`**\n\n"
            for k, v in stats.items():
                summary += f"- **{k}**: {v}\n"

            return {
                "success": True,
                "action": "get_analysis",
                "analysis_id": analysis_id,
                "stats": stats,
                "summary": summary,
            }
        except httpx.HTTPStatusError as e:
            return {"success": False, "action": "get_analysis", "error": str(e)}

    async def _cv_list_analyses(
        self, client: httpx.AsyncClient, headers: Dict
    ) -> Dict[str, Any]:
        """List user's saved analyses from the DB."""
        try:
            resp = await client.get(
                f"{CODE_VISUALIZER_URL}/api/v1/analyses",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            analyses = data.get("analyses") or []
            storage_limit = data.get("storage_limit")

            if not analyses:
                summary = (
                    "**Your Saved Analyses**\n\n"
                    "No saved analyses yet. Scan a GitHub repo or upload code to create one.\n\n"
                    "Example: `scan github https://github.com/user/repo`"
                )
            else:
                limit_info = f"{len(analyses)}" if storage_limit is None else f"{len(analyses)}/{storage_limit}"
                summary = f"**Your Saved Analyses** ({limit_info})\n\n"
                for a in analyses[:20]:
                    stats = a.get("stats") or {}
                    source_icon = "🐙" if a.get("source") == "github" else "📦"
                    created = (a.get("created_at") or "")[:10]
                    summary += (
                        f"{source_icon} **{a.get('project_name', 'Unnamed')}**"
                        f" — ID: `{a['analysis_id'][:12]}...`"
                        f" ({stats.get('total_functions', 0)} functions,"
                        f" {stats.get('total_endpoints', 0)} endpoints,"
                        f" {stats.get('total_files', 0)} files)"
                        f" — {created}\n"
                    )
                    if a.get("repo_url"):
                        summary += f"  └ {a['repo_url']}\n"
                if len(analyses) > 20:
                    summary += f"\n...and {len(analyses) - 20} more\n"
                summary += "\nTo continue on an analysis, say: `load analysis <id>`"

            return {
                "success": True,
                "action": "list_analyses",
                "analyses": analyses,
                "storage_limit": storage_limit,
                "summary": summary,
            }
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "action": "list_analyses",
                "error": f"Failed to list analyses: {e.response.text[:200]}",
            }
        except Exception as e:
            return {
                "success": False,
                "action": "list_analyses",
                "error": f"Failed to list analyses: {str(e)[:200]}",
            }

    async def _cv_rescan(self, client: httpx.AsyncClient, message: str, headers: Dict, context: Dict) -> Dict[str, Any]:
        """Re-scan: list saved analyses and rescan the most recent repos."""
        github_token = context.get("github_token", "")
        try:
            list_result = await self._cv_list_analyses(client, headers)
            if list_result.get("success") and list_result.get("analyses"):
                analyses = list_result["analyses"]
                rescan_repos = []
                for a in analyses[:5]:
                    url = a.get("repo_url") or a.get("source_url") or ""
                    name = a.get("project_name", "unknown")
                    if url and "github.com" in url:
                        rescan_repos.append({"url": url, "name": name, "id": a.get("id", "")})
                
                if rescan_repos:
                    results = []
                    last_analysis_id = ""
                    for repo in rescan_repos:
                        try:
                            scan_payload = {"repo_url": repo["url"], "project_name": repo["name"]}
                            if github_token:
                                scan_payload["token"] = github_token
                            resp = await client.post(
                                f"{CODE_VISUALIZER_URL}/api/v1/scan/github",
                                json=scan_payload,
                                headers=headers,
                                timeout=90.0,
                            )
                            if resp.status_code in (200, 201):
                                data = resp.json()
                                analysis = data.get("analysis", {})
                                stats = analysis.get("stats", data.get("stats", {}))
                                aid = data.get("analysis_id", "")
                                if aid:
                                    last_analysis_id = aid
                                results.append(
                                    f"**{repo['name']}** \u2014 "
                                    f"Files: {stats.get('total_files', '?')}, "
                                    f"Functions: {stats.get('total_functions', '?')}, "
                                    f"Endpoints: {stats.get('total_endpoints', '?')}, "
                                    f"Connections: {stats.get('total_connections', '?')}"
                                )
                            elif resp.status_code in (401, 403):
                                results.append(f"**{repo['name']}** \u2014 access denied (private repo? Add a GitHub access token in Settings \u2192 API Keys)")
                            else:
                                results.append(f"**{repo['name']}** \u2014 scan failed ({resp.status_code})")
                        except Exception as e:
                            results.append(f"**{repo['name']}** \u2014 error: {str(e)[:100]}")
                    
                    summary = "**Re-analysis complete!**\n\n" + "\n".join(f"- {r}" for r in results)
                    if last_analysis_id:
                        summary += f"\n\n**Analysis ID**: `{last_analysis_id}`"
                    result = {
                        "success": True,
                        "action": "rescan",
                        "summary": summary,
                        "panel_url": "/code-visualizer?embed=1",
                    }
                    if last_analysis_id:
                        result["analysis_id"] = last_analysis_id
                    return result
                else:
                    return list_result
            else:
                return {
                    "success": True,
                    "action": "help",
                    "summary": (
                        "No previous analyses found to re-scan. To analyze a repository:\n\n"
                        "`scan github https://github.com/user/repo`"
                    ),
                }
        except Exception as e:
            return {
                "success": False,
                "action": "rescan",
                "error": f"Re-scan failed: {str(e)[:200]}",
            }

    async def _cv_analyze_or_help(
        self, client: httpx.AsyncClient, message: str, headers: Dict, context: Dict
    ) -> Dict[str, Any]:
        """Default: provide help or try to analyze based on context."""
        analysis_id = context.get("analysis_id", "")

        if analysis_id:
            # User has an active analysis — show what they can do
            return {
                "success": True,
                "action": "help",
                "summary": (
                    f"**Code Visualizer is active** (Analysis: `{analysis_id[:12]}...`)\n\n"
                    "You can ask me to:\n"
                    "- **Trace** a function or endpoint: `trace send_message`\n"
                    "- **List functions**: `show all functions`\n"
                    "- **List endpoints**: `show all endpoints`\n"
                    "- **Reachability**: `run reachability analysis`\n"
                    "- **Governance check**: `run governance check`\n"
                    "- **Full pipeline**: `trace full pipeline from main`\n"
                    "- **Scan another repo**: `scan github https://github.com/user/repo`\n"
                ),
            }
        else:
            # No active analysis — try to list saved analyses to be helpful
            try:
                list_result = await self._cv_list_analyses(client, headers)
                if list_result.get("success") and list_result.get("analyses"):
                    return list_result
            except Exception:
                pass
            
            return {
                "success": True,
                "action": "help",
                "summary": (
                    "**Code Visualizer Skill**\n\n"
                    "I can analyze codebases for you! Here's how to get started:\n\n"
                    "1. **Scan a GitHub repo**: `scan github https://github.com/user/repo`\n"
                    "2. **Upload code**: Use the Code Visualizer page to upload a .zip\n\n"
                    "After scanning, you can:\n"
                    "- Trace execution pipelines\n"
                    "- List all functions and endpoints\n"
                    "- Run reachability analysis (Graph Janitor scan)\n"
                    "- Run governance checks\n"
                    "- Navigate the code structure\n"
                ),
            }

    async def _execute_state_physics(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Open State Physics panel and validate service reachability."""
        panel_url = "/state-physics?embed=1"
        service_ok = False
        status_hint = "unverified"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{STATE_PHYSICS_URL}/")
                service_ok = resp.status_code < 500
                status_hint = f"http {resp.status_code}"
        except Exception:
            service_ok = False
            status_hint = "service unavailable"

        summary = (
            "**State Physics panel ready.**\n\n"
            f"- Panel URL: {panel_url}\n"
            f"- Backend endpoint: `{STATE_PHYSICS_URL}`\n"
            f"- Reachability: {status_hint}\n"
        )

        return {
            "success": True,
            "action": "open_state_physics_panel",
            "panel_url": panel_url,
            "service_ok": service_ok,
            "summary": summary,
        }

    async def _execute_ide_workspace(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Open IDE panel and verify IDE service endpoint health."""
        panel_url = "/ide?embed=1"
        service_ok = False
        status_hint = "unverified"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{IDE_SERVICE_URL}/health")
                service_ok = resp.status_code < 500
                status_hint = f"http {resp.status_code}"
        except Exception:
            service_ok = False
            status_hint = "service unavailable"

        summary = (
            "**IDE workspace panel ready.**\n\n"
            f"- Panel URL: {panel_url}\n"
            f"- IDE endpoint: `{IDE_SERVICE_URL}`\n"
            f"- Reachability: {status_hint}\n"
        )

        return {
            "success": True,
            "action": "open_ide_panel",
            "panel_url": panel_url,
            "service_ok": service_ok,
            "summary": summary,
        }

    async def _execute_memory_library(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Open unified memory library panel and provide quick memory counts."""
        panel_url = "/resonant-memory?embed=1"
        memory_count = 0
        anchor_count = 0
        service_ok = False
        status_hint = "unverified"

        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                stats_resp = await client.get(
                    f"{MEMORY_SERVICE_URL}/api/v1/memory/stats",
                    params={"user_id": user_id},
                )
                stats_resp.raise_for_status()
                stats_data = stats_resp.json() if isinstance(stats_resp.json(), dict) else {}
                memory_count = int(stats_data.get("total_memories", 0) or 0)
                anchor_count = int(
                    stats_data.get("total_anchors", stats_data.get("total_clusters", 0)) or 0
                )
                service_ok = True
                status_hint = f"http {stats_resp.status_code}"
        except Exception:
            service_ok = False
            status_hint = "service unavailable"

        summary = (
            "**Memory Library is ready.**\n\n"
            f"- Open panel: {panel_url}\n"
            f"- Memory endpoint: `{MEMORY_SERVICE_URL}`\n"
            f"- Reachability: {status_hint}\n"
            f"- Total memories: {memory_count}\n"
            f"- Anchors / clusters: {anchor_count}\n"
        )

        return {
            "success": True,
            "action": "open_memory_panel",
            "panel_url": panel_url,
            "service_ok": service_ok,
            "memory_count": memory_count,
            "anchor_count": anchor_count,
            "summary": summary,
        }

    def _extract_node_name(self, message: str) -> str:
        """Extract a node/function name from a message."""
        # Try to find quoted names
        quoted = re.findall(r'[`"\']([^`"\']+)[`"\']', message)
        if quoted:
            return quoted[0]

        # Try to find after 'trace' or 'from'
        patterns = [
            r'trace\s+(?:for\s+me\s+)?(?:the\s+)?(\S+)',
            r'from\s+(\S+)',
            r'pipeline\s+(?:of\s+)?(\S+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                name = match.group(1).strip(".,;!?")
                if name and name not in {"the", "a", "an", "this", "that", "me"}:
                    return name

        return "main"

    # ============================================
    # WEB SEARCH SKILL (delegates to existing web_search service)
    # ============================================

    async def _execute_web_search(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute web search — delegates to existing web search in chat pipeline."""
        return {
            "success": True,
            "action": "web_search",
            "delegate_to_pipeline": True,
            "summary": "Web search will be executed through the chat pipeline.",
        }

    # ============================================
    # IMAGE GENERATION SKILL (delegates to existing image_generation)
    # ============================================

    async def _execute_image_generation(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute image generation — delegates to existing pipeline."""
        return {
            "success": True,
            "action": "image_generation",
            "delegate_to_pipeline": True,
            "summary": "Image generation will be executed through the chat pipeline.",
        }

    # ============================================
    # MEMORY SEARCH SKILL
    # ============================================

    async def _execute_memory_search(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Search user's memories."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{MEMORY_SERVICE_URL}/memory/hash-sphere/extract",
                    json={
                        "query": message,
                        "user_id": user_id,
                        "limit": 10,
                        "use_anchors": True,
                        "use_proximity": True,
                        "use_resonance": True,
                        "use_rag_fallback": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                memories = data.get("memories", [])

                if not memories:
                    return {
                        "success": True,
                        "action": "memory_search",
                        "summary": "No relevant memories found for your query.",
                        "count": 0,
                    }

                summary = f"**Found {len(memories)} Relevant Memories:**\n\n"
                for i, mem in enumerate(memories[:10], 1):
                    content = mem.get("content", "")[:200]
                    score = mem.get("hybrid_score", 0)
                    summary += f"{i}. {content}... (score: {score:.2f})\n\n"

                return {
                    "success": True,
                    "action": "memory_search",
                    "summary": summary,
                    "count": len(memories),
                    "memories": memories[:10],
                }
        except Exception as e:
            return {
                "success": False,
                "action": "memory_search",
                "error": f"Memory search failed: {e}",
            }

    # ============================================
    # RABBIT POST SKILL (uses shared/tools/rabbit.py)
    # ============================================

    async def _execute_rabbit_post(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a Rabbit post — Rabbit services currently disabled."""
        return {
            "success": False,
            "action": "rabbit_post",
            "error": "Rabbit community services are currently disabled",
        }

    # ============================================
    # MODULAR INTEGRATION SKILLS (figma, google_drive, google_calendar, sigma, etc.)
    # Each skill lives in its own file under skills/ directory.
    # Easy to connect/disconnect without breaking Resonant Chat.
    # ============================================

    async def _execute_integration(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delegate to the appropriate modular integration skill file.

        This method is called for all integration skills (figma, google_drive,
        google_calendar, sigma, etc.). It looks up the skill in the INTEGRATION_SKILLS
        registry and calls its execute() method.
        """
        # Determine which skill is being executed from the executor map
        # The skill_id is passed through context by the execute() wrapper
        skill_id = context.get("_integration_tool_id")
        if not skill_id:
            # Fallback: detect from message
            from .tools import is_integration_intent
            skill_id = is_integration_intent(message)

        if not skill_id or skill_id not in INTEGRATION_SKILLS:
            return {
                "success": False,
                "action": "integration",
                "error": f"Unknown integration skill: {skill_id}",
            }

        skill_module = INTEGRATION_SKILLS[skill_id]
        logger.info(f"🔌 Executing modular integration skill: {skill_id} ({skill_module.skill_name})")
        return await skill_module.execute(message, user_id, context)

    # ============================================
    # AGENT ARCHITECT SKILL (ReAct orchestrator)
    # ============================================

    async def _execute_agent_architect(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delegate to the standalone Agent Architect service.

        The architect runs a ReAct loop with 26 tools (build_agent, run_agent,
        modify_agent, set_trigger, check_integrations, memory, blockchain, etc.).
        We try SSE streaming first, then fall back to sync.
        """
        panel_url = "/agents?embed=1"
        headers = {
            "x-user-id": user_id,
            "x-user-role": str(context.get("user_role", "user")),
            "x-is-superuser": "true" if bool(context.get("is_superuser", False)) else "false",
            "x-unlimited-credits": "true" if bool(context.get("unlimited_credits", False)) else "false",
        }
        svc_payload = {
            "message": message,
            "workspace_id": user_id,
            "user_id": user_id,
            "context": context.get("prev_assistant_content", ""),
            "conversation_history": context.get("recent_messages", []),
            "user_api_keys": context.get("user_api_keys", {}),
        }

        # Try SSE streaming first for real-time progress
        try:
            result = await self._architect_delegate_to_services(svc_payload, headers, panel_url)
            return result
        except Exception as e:
            logger.error(f"Agent architect delegation failed: {e}")
            return {
                "success": False,
                "action": "open_agents_panel",
                "panel_url": panel_url,
                "error": f"Agent Architect unavailable: {e}",
                "summary": (
                    "**Agent Architect** is currently unavailable. "
                    "You can still manage agents directly from the **Agents** panel.\n\n"
                    f"- Open panel: {panel_url}"
                ),
            }

    async def _architect_delegate_to_services(
        self, svc_payload: Dict, headers: Dict[str, str], panel_url: str
    ) -> Dict[str, Any]:
        """Call the architect service via SSE streaming with sync fallback.

        Captures every tool_call + tool_result into a grounded action log
        so the LLM in resonant_chat can only report what actually happened.
        """
        result: Dict[str, Any] = {
            "success": True,
            "action": "open_agents_panel",
            "panel_url": panel_url,
            "summary": "",
        }

        accumulated_text = ""
        action_log: List[str] = []  # human-readable grounded log
        actions_taken = []
        _pending_tool: Optional[Dict] = None  # track tool_call → tool_result pairs

        # Try SSE streaming
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{AGENT_ARCHITECT_URL}/api/message/stream",
                    json=svc_payload,
                    headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        raise httpx.HTTPStatusError(
                            f"Architect returned {resp.status_code}",
                            request=resp.request,
                            response=resp,
                        )

                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            import json
                            event = json.loads(data_str)
                        except Exception:
                            continue

                        etype = event.get("type", "")
                        edata = event.get("data", event)

                        if etype == "text":
                            accumulated_text += edata.get("content", "")

                        elif etype == "tool_call":
                            tool_name = edata.get("tool", edata.get("name", "?"))
                            args_keys = edata.get("args_keys", [])
                            progress_msg = edata.get("message", f"Calling {tool_name}...")
                            action_log.append(f"▶ {progress_msg}")
                            _pending_tool = {"tool": tool_name, "args_keys": args_keys}
                            actions_taken.append(edata)
                            logger.info(f"🔧 Architect tool_call: {tool_name}({args_keys})")

                        elif etype == "tool_result":
                            success = edata.get("success", True)
                            preview = edata.get("preview", "")[:400]
                            result_msg = edata.get("message", "")
                            icon = "✅" if success else "❌"
                            action_log.append(f"  {icon} {result_msg or preview}")
                            if _pending_tool:
                                _pending_tool["success"] = success
                                _pending_tool["result_preview"] = preview
                                _pending_tool = None
                            logger.info(f"🔧 Architect tool_result: success={success} {result_msg[:80]}")

                        elif etype == "thinking":
                            msg = edata.get("message", "")
                            if msg:
                                action_log.append(f"💭 {msg}")

                        elif etype == "warning":
                            msg = edata.get("message", "")
                            if msg:
                                action_log.append(f"⚠️ {msg}")

                        elif etype == "summarizing":
                            action_log.append("📝 Generating summary...")

                        # ── Build Pipeline events ──
                        elif etype == "phase":
                            phase_name = edata.get("phase", "")
                            msg = edata.get("message", f"Phase: {phase_name}")
                            action_log.append(f"🔄 {msg}")

                        elif etype == "research_complete":
                            credits = edata.get("credits_remaining", "?")
                            integrations = edata.get("integrations_connected", [])
                            existing = edata.get("existing_agent_count", 0)
                            warnings = edata.get("warnings", [])
                            action_log.append(f"🔍 Research: credits={credits}, integrations={len(integrations)}, existing_agents={existing}")
                            for w in warnings:
                                action_log.append(f"  ⚠️ {w}")

                        elif etype == "plan_ready":
                            plan = edata.get("plan", {})
                            msg = edata.get("message", "Build plan ready")
                            action_log.append(f"📋 {msg}")
                            result["pipeline_plan"] = plan

                        elif etype == "verify_step":
                            msg = edata.get("message", "")
                            if msg:
                                action_log.append(f"  {msg}")

                        elif etype == "verify_complete":
                            all_ok = edata.get("all_ok", False)
                            warnings = edata.get("warnings", [])
                            icon = "✅" if all_ok else "⚠️"
                            action_log.append(f"{icon} Verification {'passed' if all_ok else 'has warnings'}")
                            for w in warnings:
                                action_log.append(f"  ⚠️ {w}")

                        elif etype == "prompt_step":
                            msg = edata.get("message", "")
                            if msg:
                                action_log.append(f"🧠 {msg}")

                        elif etype == "prompt_ready":
                            msg = edata.get("message", "")
                            length = edata.get("prompt_length", 0)
                            action_log.append(f"📝 Prompt generated ({length} chars)")

                        elif etype == "build_progress":
                            msg = edata.get("message", "")
                            if msg:
                                action_log.append(f"🔨 {msg}")

                        elif etype == "build_step":
                            msg = edata.get("message", "")
                            if msg:
                                action_log.append(f"  🔨 {msg}")

                        elif etype == "build_complete":
                            success = edata.get("success", False)
                            name = edata.get("name", "")
                            icon = "✅" if success else "❌"
                            action_log.append(f"{icon} Build {'succeeded' if success else 'failed'}: {name}")

                        elif etype == "test_step":
                            msg = edata.get("message", "")
                            if msg:
                                action_log.append(f"  🧪 {msg}")

                        elif etype == "test_result":
                            status = edata.get("status", "?")
                            msg = edata.get("message", f"Test: {status}")
                            action_log.append(f"🧪 {msg}")

                        elif etype == "offers_ready":
                            offers = edata.get("offers", [])
                            if offers:
                                action_log.append(f"💡 {len(offers)} post-build suggestions available")

                        elif etype == "options":
                            options_data = edata
                            result["present_options"] = self._map_architect_options(options_data)

                        elif etype == "complete":
                            resp_data = edata.get("response", edata)
                            accumulated_text = resp_data.get("text", accumulated_text)
                            options_data = resp_data.get("options")
                            if options_data:
                                result["present_options"] = self._map_architect_options(options_data)
                            resp_actions = resp_data.get("actions", [])
                            if resp_actions and not actions_taken:
                                actions_taken = resp_actions

                        elif etype == "error":
                            err = edata.get("error", edata.get("message", "Unknown error"))
                            action_log.append(f"❌ Error: {err}")
                            result["error"] = err

            # Build summary: prefer the formatted architect text (pipeline plan/response)
            # Only fall back to action log when there's no accumulated text
            if accumulated_text:
                result["summary"] = accumulated_text
            elif action_log:
                result["summary"] = "\n".join(action_log)
            else:
                result["summary"] = ""
            if actions_taken:
                result["actions"] = actions_taken
            return result

        except Exception as stream_err:
            logger.warning(f"Architect SSE stream failed, trying sync: {stream_err}")

        # Fallback: synchronous call
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    f"{AGENT_ARCHITECT_URL}/api/message",
                    json=svc_payload,
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("text", data.get("response", ""))
                    sync_actions = data.get("actions", [])
                    # Build grounded summary from sync response actions
                    if sync_actions:
                        log_lines = []
                        for a in sync_actions:
                            t = a.get("tool", "?")
                            r = a.get("result", {})
                            err = r.get("error", "") if isinstance(r, dict) else ""
                            icon = "❌" if err else "✅"
                            log_lines.append(f"  {icon} {t}: {str(r)[:200]}")
                        result["summary"] = f"ACTIONS PERFORMED (real API calls):\n" + "\n".join(log_lines) + f"\n\nARCHITECT RESPONSE:\n{text}"
                    else:
                        result["summary"] = text
                    options_data = data.get("options")
                    if options_data:
                        result["present_options"] = self._map_architect_options(options_data)
                    return result
                else:
                    result["success"] = False
                    result["error"] = f"Architect returned {resp.status_code}"
                    result["summary"] = "Agent Architect is temporarily unavailable."
                    return result
        except Exception as sync_err:
            raise Exception(f"Both SSE and sync calls failed: {stream_err}; {sync_err}")

    def _map_architect_options(self, options_data: Any) -> Dict[str, Any]:
        """Map architect present_options to the chat UI format."""
        if not isinstance(options_data, dict):
            return {}
        raw_options = options_data.get("options", [])
        mapped = []
        if isinstance(raw_options, list):
            for opt in raw_options[:4]:
                if isinstance(opt, str):
                    mapped.append({
                        "label": opt,
                        "value": f"Agent Architect: {opt}",
                        "description": opt,
                        "icon": "🔧",
                    })
                elif isinstance(opt, dict):
                    mapped.append({
                        "label": opt.get("label", opt.get("text", str(opt))),
                        "value": f"Agent Architect: {opt.get('value', opt.get('label', ''))}",
                        "description": opt.get("description", ""),
                        "icon": opt.get("icon", "🔧"),
                    })
        return {
            "_type": "present_options",
            "title": options_data.get("question", "What's next?"),
            "options": mapped,
            "allow_custom": True,
        }



# Global singleton
tool_executor = ToolExecutor()
