"""
Skill Executor Service
=======================

Executes skill actions on behalf of users within Resonant Chat.
Each skill has its own executor that handles the specific API calls
and returns structured results for the chat response.
"""

from __future__ import annotations

import logging
import re
import os
from typing import Any, Dict, List, Optional

import httpx

from .skills_registry import SkillDefinition, skills_registry
from .skills import INTEGRATION_SKILLS

try:
    from platform_tools.auth import AuthContext, build_service_headers
except ImportError:
    AuthContext = None
    build_service_headers = None

logger = logging.getLogger(__name__)

CODE_VISUALIZER_URL = os.getenv("AST_ANALYSIS_SERVICE_URL") or os.getenv("CODE_VISUALIZER_URL", "http://rg_ast_analysis:8000")
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://memory_service:8000")
AGENT_ENGINE_URL = os.getenv("AGENT_ENGINE_URL", "http://agent_engine_service:8000")
STATE_PHYSICS_URL = os.getenv("STATE_PHYSICS_URL", "http://rg_users_invarients_sim:8091")
IDE_SERVICE_URL = os.getenv("IDE_SERVICE_URL", "http://ide_platform_service:8080")
AGENT_ARCHITECT_URL = os.getenv("AGENT_ARCHITECT_URL", "http://agent_architect:8000")


class SkillExecutor:
    """Executes skill actions and returns structured results."""

    def __init__(self):
        self._executors = {
            "code_visualizer": self._execute_code_visualizer,
            "web_search": self._execute_web_search,
            "image_generation": self._execute_image_generation,
            "memory_search": self._execute_memory_search,
            "memory_library": self._execute_memory_library,
            "agents_os": self._execute_agents_os,
            "agent_architect": self._execute_agent_architect,
            "state_physics": self._execute_state_physics,
            "ide_workspace": self._execute_ide_workspace,
            "rabbit_post": self._execute_rabbit_post,
            "google_drive": self._execute_integration,
            "google_calendar": self._execute_integration,
            "figma": self._execute_integration,
            "sigma": self._execute_integration,
        }

    async def execute(
        self,
        skill: SkillDefinition,
        message: str,
        user_id: str,
        user_role: str = "user",
        is_superuser: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a skill and return results."""
        executor = self._executors.get(skill.id)
        if not executor:
            return {
                "skill_id": skill.id,
                "success": False,
                "error": f"No executor for skill: {skill.id}",
            }

        try:
            exec_context = dict(context or {})
            exec_context.setdefault("user_role", user_role)
            exec_context.setdefault("is_superuser", is_superuser)
            exec_context["_integration_skill_id"] = skill.id
            result = await executor(message, user_id, exec_context)
            result["skill_id"] = skill.id
            result["skill_name"] = skill.name
            return result
        except Exception as e:
            logger.error(f"Skill execution failed ({skill.id}): {e}")
            return {
                "skill_id": skill.id,
                "skill_name": skill.name,
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
    # AGENTS OS SKILL
    # ============================================

    def _detect_agents_os_action(self, message: str) -> str:
        """Detect which Agents OS action to perform.

        Uses tight patterns requiring the action verb to be immediately
        adjacent to 'agent' (with optional articles) to avoid false
        positives from casual mentions.
        """
        msg = (message or "").lower()

        # Multi-agent detection: message describes multiple named agents
        # e.g. "Context Agent: ..., Code Analysis Agent: ..., Content Creation Agent: ..."
        multi_agents = self._parse_multiple_agents(message)
        if len(multi_agents) >= 2:
            return "create_agents"

        # Team / workflow creation patterns (check BEFORE single-agent patterns)
        # But ONLY if message doesn't also contain agent descriptions
        if re.search(r"\b(create|build|make|set\s+up)\s+(?:me\s+)?(?:a\s+)?(?:an\s+)?(?:new\s+)?(?:agent\s+)?(?:team|workflow|pipeline)\b", msg):
            # If user describes specific agents alongside "workflow", treat as multi-create
            if re.search(r"agent\s*[:;]|:\s*(?:responsible|reads|builds|modifies|creates|handles)", msg):
                if len(multi_agents) >= 1:
                    return "create_agents"
            return "create_team"
        if re.search(r"\b(list|show|view)\s+(?:my\s+|all\s+|the\s+)?(?:agent\s+)?(?:teams|workflows|pipelines)\b", msg):
            return "list_teams"
        # ── Rename agent ──
        if re.search(r"\b(rename|change\s+(?:the\s+)?name\s+(?:of\s+)?)\b.*\bagents?\b", msg) or \
           re.search(r"\bagents?\b.*\b(rename|change\s+(?:the\s+)?name)\b", msg):
            return "rename_agent"

        # ── Delete agent ──
        if re.search(r"\b(delete|remove|destroy|kill)\s+(?:the\s+|my\s+|this\s+)?(?:agent|agents)\b", msg) or \
           re.search(r"\bagents?\b.*\b(delete|remove|destroy)\b", msg):
            return "delete_agent"

        # ── Update/edit agent ──
        if re.search(r"\b(update|edit|modify|change|configure)\s+(?:the\s+|my\s+|this\s+)?agents?\b", msg) or \
           re.search(r"\bagents?\b.*\b(update|edit|modify|configure)\b", msg):
            return "update_agent"

        # ── Start/run agent ──
        if re.search(r"\b(start|run|execute|launch|activate)\s+(?:the\s+|my\s+|this\s+)?agents?\b", msg):
            return "start_agent"

        # ── Stop/pause agent ──
        if re.search(r"\b(stop|pause|deactivate|disable|halt)\s+(?:the\s+|my\s+|this\s+)?agents?\b", msg):
            return "stop_agent"

        # Tight create pattern: verb + optional filler words + agent
        if re.search(r"\b(create|build|make|spin\s+up|configure|set\s+up)\s+(?:for\s+)?(?:me\s+)?(?:a\s+)?(?:an\s+)?(?:new\s+)?(?:the\s+)?(?:this\s+)?agents?\b", msg):
            return "create_agent"
        # Direct "create [Name] agent" — short command-like messages
        # Matches: "create Webhooks agent", "create this Webhooks agent for me"
        if len(msg) < 120 and re.search(r"\b(create|build|make)\s+(?:\w+\s+){0,4}agents?\b", msg):
            return "create_agent"
        # Broader create: verb and "agent" appear within 80 chars (natural language)
        if re.search(r"\b(create|build|make)\b.{0,80}\bagents?\b", msg) and "new" in msg:
            return "create_agent"
        # "new agent" anywhere is a strong create signal
        if re.search(r"\bnew\s+agents?\b", msg):
            return "create_agent"
        # "agent that will/to/for" — describing an agent to create
        if re.search(r"\bagents?\s+(?:that|which|who|to)\s+(?:will\s+)?", msg):
            return "create_agent"
        # Confirmation patterns: "yes create all", "yes do it", "go ahead", "create them"
        if re.search(r"\b(yes|yeah|yep|go ahead|do it|create them|create all|make them|build them)\b", msg):
            return "create_agent"
        if any(k in msg for k in [
            "list agents",
            "show agents",
            "my agents",
            "agents list",
        ]) or re.search(r"\b(list|show|open|view)\s+(my\s+|all\s+|the\s+)?agents?\b", msg):
            return "list_agents"
        return "open_panel"

    def _extract_agent_name(self, message: str) -> str:
        """Extract a meaningful agent name from natural language request.

        Priority order:
        1. Explicit naming: "named X" / "called X" / "name it X"
        2. Quoted name: "Create agent 'My Cool Bot'"
        3. Known domain keywords → clean canonical name
        4. "agent for/to TOPIC" → "Topic Agent"
        5. Purpose clause: strip boilerplate, take 2-3 meaningful words
        6. Fallback: "Resonant Agent"
        """
        text = (message or "").strip()
        lower = text.lower()

        # 1. Explicit naming: "named X" / "called X" / "name it X"
        explicit = re.search(
            r'(?:named?|called?|name\s+it|title[d]?)\s+["\']?([A-Za-z0-9][A-Za-z0-9 _\-]{2,50})["\']?',
            text, re.IGNORECASE,
        )
        if explicit:
            name = explicit.group(1).strip().strip(" .,!?:;")
            # Don't accept if it's just a common word like "agent" or "the"
            if len(name) >= 3 and name.lower() not in {"agent", "the", "this", "my", "new"}:
                return name.title()

        # 2. Quoted name: 'My Agent' or "My Agent"
        quoted = re.search(r'''["\u2018\u2019\u201c\u201d']([A-Za-z0-9][A-Za-z0-9 _\-]{2,50})["\u2018\u2019\u201c\u201d']''', text)
        if quoted:
            name = quoted.group(1).strip()
            if len(name) >= 3 and name.lower() not in {"agent", "the", "this"}:
                return name.title()

        # 3. Known domain keywords → clean canonical name (longest match first)
        _domain_names = [
            ("market research", "Market Research"), ("market reserch", "Market Research"),
            ("web penetration", "Web Penetration"), ("web penitration", "Web Penetration"),
            ("penetration test", "Penetration Testing"), ("penitration test", "Penetration Testing"),
            ("content writing", "Content Writing"), ("content creation", "Content Creation"),
            ("content marketing", "Content Marketing"),
            ("customer support", "Customer Support"), ("customer service", "Customer Service"),
            ("data analysis", "Data Analysis"), ("data analytics", "Data Analytics"),
            ("data processing", "Data Processing"), ("data collection", "Data Collection"),
            ("code review", "Code Review"), ("code analysis", "Code Analysis"),
            ("social media", "Social Media"), ("social post", "Social Media"),
            ("competitor analysis", "Competitor Analysis"), ("competitive analysis", "Competitive Analysis"),
            ("email marketing", "Email Marketing"), ("email automation", "Email Automation"),
            ("web scraping", "Web Scraping"), ("web crawl", "Web Crawling"),
            ("task automation", "Task Automation"), ("workflow automation", "Workflow Automation"),
            ("image generation", "Image Generation"),
            ("text generation", "Text Generation"),
            ("lead generation", "Lead Generation"),
            ("project management", "Project Management"),
            ("file management", "File Management"),
            ("security audit", "Security Audit"),
            ("penetration", "Penetration"), ("penitration", "Penetration"),
            ("security", "Security"), ("cybersecurity", "Cybersecurity"),
            ("monitoring", "Monitoring"), ("alerting", "Alerting"),
            ("research", "Research"), ("reserch", "Research"),
            ("testing", "Testing"), ("qa", "QA"),
            ("devops", "DevOps"), ("deployment", "Deployment"),
            ("api", "API"), ("database", "Database"),
            ("seo", "SEO"), ("marketing", "Marketing"),
            ("writing", "Writing"), ("copywriting", "Copywriting"),
            ("translation", "Translation"),
            ("summariz", "Summarization"),
            ("analyz", "Analysis"), ("analysis", "Analysis"), ("analyses", "Analysis"),
            ("automat", "Automation"),
            ("chatbot", "Chatbot"), ("assistant", "Assistant"),
            ("scraping", "Scraping"), ("crawling", "Crawling"),
            ("scheduling", "Scheduling"), ("planning", "Planning"),
            ("reporting", "Reporting"), ("dashboard", "Dashboard"),
            ("webhook", "Webhook"), ("webhooks", "Webhooks"),
            ("notification", "Notification"), ("alerting", "Alerting"),
            ("integration", "Integration"), ("pipeline", "Pipeline"),
        ]
        for kw, label in _domain_names:
            if kw in lower:
                return f"{label} Agent"

        # 4. "agent for/to TOPIC" → "Topic Agent"
        for_match = re.search(
            r'agent\s+(?:for|to|that\s+(?:will\s+|can\s+)?(?:do\s+)?)\s*(?:the\s+|a\s+|an\s+)?'
            r'([A-Za-z][A-Za-z0-9 ]{2,40})',
            text, re.IGNORECASE,
        )
        if for_match:
            topic = for_match.group(1).strip()
            # Take first 3 meaningful words from topic
            stop = {"the", "a", "an", "and", "or", "for", "in", "of", "to", "is", "do", "my", "me", "our"}
            words = [w for w in topic.split()[:4] if w.lower() not in stop and len(w) > 1]
            if words:
                topic_name = " ".join(words).title()
                # Strip trailing verbs/filler
                topic_name = re.sub(r'\s+(?:And|The|Of|In|To|For|A|An|Give|Provide|Report)\s*$', '', topic_name, flags=re.IGNORECASE).strip()
                if len(topic_name) >= 3:
                    return f"{topic_name} Agent"

        # 5. Purpose clause: strip preamble, take 2-3 meaningful words
        purpose = lower
        purpose = re.sub(r'^.*?\bagents?\b\s*(?:that|which|who|to)?\s*(?:will\s+|can\s+|should\s+)?(?:do\s+)?', '', purpose)
        purpose = re.sub(r'\s+(?:and\s+)?(?:give|provide|report|also|with|then)\b.*$', '', purpose).strip()

        if purpose and len(purpose) >= 5:
            stop = {"the", "a", "an", "and", "or", "for", "in", "of", "to", "is", "do", "deep",
                    "my", "me", "new", "create", "build", "make", "some", "please", "just"}
            words = [w for w in purpose.split()[:3] if w not in stop and len(w) > 2]
            if words:
                name = " ".join(words).title()
                if 3 <= len(name) <= 40:
                    return f"{name} Agent"

        return "Resonant Agent"

    def _parse_multiple_agents(self, message: str) -> List[Dict[str, str]]:
        """Parse multiple agent definitions from a natural language message.

        Handles many formats:
          - 'Context Agent: responsible for modifying the context...'  (any-case desc)
          - 'Code Analysis Agent - reads the codebase'
          - '1. Content Agent - builds help center page'
          - '* Context Agent: modifies context'
          - 'Context Agent that modifies the context'
          - Comma/newline separated role lists
        Returns list of {name, description} dicts.
        """
        agents: List[Dict[str, str]] = []
        text = (message or "").strip()
        if not text:
            return agents

        # Words that indicate a name-like token is actually a sentence start, not an agent name
        _skip_prefixes = (
            "to ", "the ", "we ", "i ", "you ", "this ", "that ", "my ", "our ",
            "all ", "it ", "he ", "she ", "they ", "please ", "just ", "now ",
            "create ", "build ", "make ", "set ", "add ", "and ", "or ",
        )

        def _is_valid_agent_entry(name: str, desc: str) -> bool:
            nl = name.lower()
            if any(nl.startswith(p) for p in _skip_prefixes):
                return False
            if len(name) < 3 or len(desc) < 8:
                return False
            return True

        # ── Pattern 1: "Name Agent: description" (any separator, ANY-CASE description)
        p1 = re.compile(
            r'(?:^|[\n;,])\s*'
            r'([A-Z][A-Za-z0-9 ]{2,45}(?:\s+Agent)?)'
            r'\s*[:;\-–—]+\s*'
            r'([^\n;,]{8,400})',
            re.MULTILINE,
        )
        for m in p1.finditer(text):
            name = m.group(1).strip()
            desc = m.group(2).strip().rstrip(".;,")
            if _is_valid_agent_entry(name, desc):
                agents.append({"name": name, "description": desc})

        # ── Pattern 2: numbered list  "1. Name - description"
        if not agents:
            p2 = re.compile(
                r'(?:^|\n)\s*\d+[.)\s]+'
                r'([A-Z][A-Za-z0-9 ]{2,45})'
                r'\s*[:;\-–—]+\s*'
                r'([^\n;,]{8,400})',
                re.MULTILINE,
            )
            for m in p2.finditer(text):
                name = m.group(1).strip()
                desc = m.group(2).strip().rstrip(".;,")
                if _is_valid_agent_entry(name, desc):
                    agents.append({"name": name, "description": desc})

        # ── Pattern 3: bullet list  "- Name: description" or "* Name: description"
        if not agents:
            p3 = re.compile(
                r'(?:^|\n)\s*[-*•·]\s+'
                r'([A-Z][A-Za-z0-9 ]{2,45}(?:\s+Agent)?)'
                r'\s*[:;\-–—]+\s*'
                r'([^\n;,]{8,400})',
                re.MULTILINE,
            )
            for m in p3.finditer(text):
                name = m.group(1).strip()
                desc = m.group(2).strip().rstrip(".;,")
                if _is_valid_agent_entry(name, desc):
                    agents.append({"name": name, "description": desc})

        # ── Pattern 4: "Name Agent that/to/which/for description"
        # e.g. "Context Agent that modifies the context" / "Code Agent to analyze code"
        if not agents:
            p4 = re.compile(
                r'([A-Z][A-Za-z0-9 ]{2,45}Agent)\s+(?:that|to|which|for|who)\s+'
                r'([^\n.;,]{8,400}?)(?=[,;.]|\n|$)',
                re.MULTILINE,
            )
            for m in p4.finditer(text):
                name = m.group(1).strip()
                desc = m.group(2).strip().rstrip(".;,")
                if _is_valid_agent_entry(name, desc):
                    agents.append({"name": name, "description": desc})

        # ── Pattern 5: inline comma list  "Context Agent, Code Agent, Content Agent"
        # When user lists agent names without descriptions, create minimal entries
        if not agents:
            inline = re.findall(r'([A-Z][A-Za-z0-9 ]{2,40}Agent)(?=[,;\n]|\s+and\s+|$)', text)
            if len(inline) >= 2:
                for raw_name in inline:
                    name = raw_name.strip()
                    if not any(name.lower().startswith(p) for p in _skip_prefixes) and len(name) >= 6:
                        agents.append({"name": name, "description": f"Agent: {name}"})

        # Deduplicate by name (case-insensitive)
        seen: set = set()
        unique: List[Dict[str, str]] = []
        for a in agents:
            key = a["name"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return unique

    # ── Shared helpers for intelligent agent configuration ──────────

    @staticmethod
    def _infer_tools(text: str) -> List[str]:
        """Infer the best set of tools from description keywords."""
        lower = text.lower()
        tools = ["web_search", "fetch_url"]  # always included

        # Platform action tools
        if any(k in lower for k in ["post", "rabbit", "community", "content", "publish", "blog", "write", "article"]):
            tools.extend(["create_rabbit_post", "list_rabbit_communities"])
        if any(k in lower for k in ["community", "subreddit", "forum", "create community"]):
            tools.append("create_rabbit_community")
        if any(k in lower for k in ["api", "http", "request", "endpoint", "webhook", "integration",
                                     "penetration", "penitration", "scan", "investigate", "scrape", "crawl"]):
            tools.append("http_request")
        if any(k in lower for k in ["memory", "remember", "context", "long-term", "persist",
                                     "research", "reserch", "analysis", "analyses", "analyze", "deep", "comprehensive"]):
            tools.extend(["memory.read", "memory.write"])
        if any(k in lower for k in ["github", "repo", "repository", "code", "codebase"]):
            tools.append("github")
        if any(k in lower for k in ["database", "sql", "data", "analytics", "query"]):
            tools.append("database")
        if any(k in lower for k in ["test", "qa", "quality", "sandbox", "execute", "run code"]):
            tools.append("code_execution")

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for t in tools:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    @staticmethod
    def _infer_provider_and_model(text: str) -> tuple:
        """Infer provider + model from user description keywords."""
        lower = text.lower()
        # Explicit provider mentions
        if any(k in lower for k in ["groq", "llama", "fast", "quick", "speed"]):
            return "groq", "llama-3.3-70b-versatile"
        if any(k in lower for k in ["claude", "anthropic"]):
            return "anthropic", "claude-3-5-sonnet-20241022"
        if any(k in lower for k in ["gemini", "google"]):
            return "google", "gemini-pro"
        if any(k in lower for k in ["gpt-4o", "openai", "chatgpt"]):
            return "openai", "gpt-4o"
        # Default: Groq for speed (it's free and fast)
        return "groq", "llama-3.3-70b-versatile"

    @staticmethod
    def _infer_mode(text: str) -> str:
        """Infer agent mode from description."""
        lower = text.lower()
        if any(k in lower for k in ["unbounded", "unrestricted", "no limits", "full access", "autonomous"]):
            return "unbounded"
        return "governed"

    @staticmethod
    def _build_system_prompt(name: str, description: str, tools: List[str]) -> str:
        """Build a rich, role-specific system prompt."""
        tool_instructions = []
        if "web_search" in tools:
            tool_instructions.append("- Use web_search to find real-time information, then fetch_url to read page content.")
        if "create_rabbit_post" in tools:
            tool_instructions.append("- Use create_rabbit_post to publish content on Rabbit communities.")
        if "http_request" in tools:
            tool_instructions.append("- Use http_request for internal platform API calls.")
        if "memory.read" in tools or "memory.write" in tools:
            tool_instructions.append("- Use memory.read/write to persist important information across sessions.")

        tools_section = "\n".join(tool_instructions) if tool_instructions else "- Use available tools as needed."

        return (
            f"You are '{name}', an advanced AI agent on the Resonant Genesis platform.\n\n"
            f"YOUR ROLE: {description}\n\n"
            f"TOOL USAGE:\n{tools_section}\n\n"
            "BEHAVIOR RULES:\n"
            "- For QUESTIONS: Research thoroughly, then respond with a comprehensive, well-structured answer.\n"
            "- For ACTIONS: Execute the requested action using the appropriate tool, then confirm results.\n"
            "- NEVER invent facts. If information is unavailable, state that clearly.\n"
            "- NEVER call action tools unless the user explicitly requests an action.\n"
            "- Keep responses focused, factual, and actionable.\n"
            "- Report outcomes clearly and summarize actions taken."
        )

    @staticmethod
    def _build_safety_config(tools: List[str], mode: str) -> Dict[str, Any]:
        """Build safety config based on tools and mode."""
        config: Dict[str, Any] = {
            "max_loops": 8,
            "max_tokens_per_run": 50000,
            "rate_limit_per_minute": 30,
        }
        # Require confirmation for destructive actions
        confirm_for = []
        if "http_request" in tools:
            confirm_for.append("http_request")
        if confirm_for:
            config["require_confirmation_for"] = confirm_for
        return config

    def _build_agent_create_payload(self, message: str) -> Dict[str, Any]:
        """Build a comprehensive Agent Engine create payload from user intent."""
        name = self._extract_agent_name(message)
        msg = (message or "").strip()

        # Extract a clean description: prefer the "that/to/which/for ..." clause
        desc_match = re.search(
            r'(?:that|to|which|who|for)\s+(.{10,300}?)(?=[.!?]|$)',
            msg, re.IGNORECASE | re.DOTALL,
        )
        if desc_match:
            description = desc_match.group(1).strip().rstrip(".;,!?")
        elif len(msg) > 10:
            description = msg[:300]
        else:
            description = f"Agent: {name}"

        tools = self._infer_tools(f"{name} {description} {msg}")
        provider, model = self._infer_provider_and_model(msg)
        mode = self._infer_mode(msg)
        system_prompt = self._build_system_prompt(name, description, tools)
        safety_config = self._build_safety_config(tools, mode)

        return {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "provider": provider,
            "model": model,
            "temperature": 0.6,
            "max_tokens": 4096,
            "tools": tools,
            "mode": mode,
            "safety_config": safety_config,
            "allowed_actions": tools,
            "blocked_actions": ["delete_community", "delete_user", "admin_override"],
        }

    def _build_payload_from_parsed(self, agent_def: Dict[str, str]) -> Dict[str, Any]:
        """Build a rich Agent Engine create payload from a parsed agent definition."""
        name = agent_def.get("name", "Resonant Agent")
        desc = agent_def.get("description", "Created by Resonant Chat")
        combined = f"{name} {desc}"

        tools = self._infer_tools(combined)
        provider, model = self._infer_provider_and_model(combined)
        mode = self._infer_mode(combined)
        system_prompt = self._build_system_prompt(name, desc, tools)
        safety_config = self._build_safety_config(tools, mode)

        return {
            "name": name,
            "description": desc,
            "system_prompt": system_prompt,
            "provider": provider,
            "model": model,
            "temperature": 0.6,
            "max_tokens": 4096,
            "tools": tools,
            "mode": mode,
            "safety_config": safety_config,
            "allowed_actions": tools,
            "blocked_actions": ["delete_community", "delete_user", "admin_override"],
        }

    def _extract_team_details(self, message: str) -> Dict[str, Any]:
        """Extract team name, agents, and workflow from natural language."""
        msg = (message or "").strip()
        msg_lower = msg.lower()

        # Extract team name
        name = "Custom Team"
        name_match = re.search(
            r'(?:named|called)\s+["\']?([A-Za-z0-9 _\-]{3,60})["\']?', msg, re.IGNORECASE
        )
        if name_match:
            name = name_match.group(1).strip()
        else:
            # Try "<type> team/workflow" pattern
            type_match = re.search(
                r'(?:team|workflow|pipeline)\s+(?:for\s+)?([A-Za-z0-9 _\-]{3,60})', msg, re.IGNORECASE
            )
            if type_match:
                name = type_match.group(1).strip().title() + " Team"

        # Extract agents from message
        from .team_engine import TeamEngine
        valid = TeamEngine.VALID_AGENT_TYPES
        agents_found = [a for a in valid if a in msg_lower]

        # If no explicit agents mentioned, infer from purpose
        if len(agents_found) < 2:
            purpose_map = {
                "code review": ["code", "review", "test"],
                "security": ["security", "review", "architecture"],
                "debug": ["debug", "test", "review"],
                "full stack": ["api", "database", "code", "test"],
                "refactor": ["review", "refactor", "test"],
                "performance": ["optimization", "review", "test"],
                "learning": ["explain", "research", "summary"],
                "documentation": ["documentation", "review", "code"],
                "testing": ["test", "review", "debug"],
                "api": ["api", "code", "test"],
                "database": ["database", "code", "review"],
                "devops": ["devops", "architecture", "review"],
                "migration": ["migration", "review", "test"],
            }
            for purpose, default_agents in purpose_map.items():
                if purpose in msg_lower:
                    agents_found = default_agents
                    if name == "Custom Team":
                        name = purpose.title() + " Team"
                    break
            if len(agents_found) < 2:
                agents_found = ["reasoning", "review", "code"]

        # Detect workflow type
        workflow = "sequential"
        if any(w in msg_lower for w in ["parallel", "concurrent", "simultaneously"]):
            workflow = "parallel_merge"

        # Build team_id from name
        team_id = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')

        return {
            "team_id": team_id,
            "name": name,
            "agents": agents_found,
            "workflow": workflow,
            "description": f"Custom team created from chat: {msg[:200]}",
        }

    async def _handle_team_action(
        self, action: str, message: str, user_id: str
    ) -> Dict[str, Any]:
        """Handle team/workflow creation and listing."""
        panel_url = "/agents?embed=1"
        try:
            from ..domain.agent.facade import team_engine, _init_engines
            _init_engines()

            if action == "list_teams":
                teams = team_engine.list_teams()
                summary = f"**Agent Teams** ({len(teams)} available):\n\n"
                for t in teams:
                    agents_str = ", ".join(t["agents"])
                    summary += f"- **{t['name']}** (`{t['id']}`) — {t['workflow']} — agents: {agents_str}\n"
                    if t.get("description"):
                        summary += f"  _{t['description']}_\n"
                return {
                    "success": True,
                    "action": "open_agents_panel",
                    "panel_url": panel_url,
                    "operation": "list_teams",
                    "teams": teams,
                    "summary": summary,
                }

            # create_team
            details = self._extract_team_details(message)
            result = team_engine.register_team(
                team_id=details["team_id"],
                name=details["name"],
                agents=details["agents"],
                workflow=details["workflow"],
                description=details["description"],
            )
            agents_str = ", ".join(result["agents"])
            summary = (
                "✅ **Agent team created successfully!**\n\n"
                f"- **Name:** {result['name']}\n"
                f"- **ID:** `{result['id']}`\n"
                f"- **Agents:** {agents_str}\n"
                f"- **Workflow:** {result['workflow']}\n"
                f"- **Description:** {result['description']}\n\n"
                "The team is now registered and can be triggered in future conversations. "
                "You can also trigger it by using its keywords or selecting it from the teams panel."
            )
            return {
                "success": True,
                "action": "open_agents_panel",
                "panel_url": panel_url,
                "operation": "create_team",
                "created_team": result,
                "summary": summary,
            }
        except Exception as e:
            logger.error(f"Team action failed: {e}", exc_info=True)
            return {
                "success": False,
                "action": "open_agents_panel",
                "panel_url": panel_url,
                "operation": action,
                "error": str(e),
                "summary": f"Failed to {action.replace('_', ' ')}: {e}",
            }

    async def _create_single_agent(
        self, client: httpx.AsyncClient, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Create a single agent via Agent Engine API. Returns the created agent dict or None."""
        try:
            resp = await client.post(f"{AGENT_ENGINE_URL}/agents/", headers=headers, json=payload)
            if resp.status_code == 404:
                resp = await client.post(f"{AGENT_ENGINE_URL}/agents", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                return None

            # Auto-create webhook trigger for agents with webhook/http_request tools
            agent_id = data.get("id")
            agent_name = data.get("name", "Agent")
            agent_tools = payload.get("tools", [])
            has_webhook_tools = any(
                t in agent_tools for t in ["http_request", "webhook"]
            ) or "webhook" in agent_name.lower()

            if has_webhook_tools and agent_id:
                try:
                    wh_resp = await client.post(
                        f"{AGENT_ENGINE_URL}/webhooks/agent/{agent_id}/create",
                        headers=headers,
                        json={"name": f"Webhook for {agent_name}"},
                    )
                    if wh_resp.status_code in (200, 201):
                        wh_data = wh_resp.json()
                        data["webhook_url"] = wh_data.get("webhook_url")
                        data["webhook_secret"] = wh_data.get("webhook_secret")
                        data["webhook_trigger_id"] = wh_data.get("id")
                        logger.info(f"✅ Auto-created webhook trigger for {agent_name}: {data['webhook_url']}")
                    else:
                        logger.warning(f"Webhook trigger creation returned {wh_resp.status_code}: {wh_resp.text[:200]}")
                except Exception as wh_err:
                    logger.warning(f"Failed to auto-create webhook trigger for {agent_name}: {wh_err}")

            return data
        except Exception as e:
            logger.warning(f"Failed to create agent '{payload.get('name')}': {e}")
            return None

    async def _execute_agents_os(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Open Agents OS and perform real agent operations when requested."""
        panel_url = "/agents?embed=1"
        action = self._detect_agents_os_action(message)
        headers = {
            "x-user-id": user_id,
            "x-user-role": str(context.get("user_role", "user")),
            "x-is-superuser": "true" if bool(context.get("is_superuser", False)) else "false",
            "x-unlimited-credits": "true" if bool(context.get("unlimited_credits", False)) else "false",
        }

        logger.info(f"🤖 Agents OS action detected: {action} for message: {message[:80]!r}")
        print(f"[AGENTS_OS] action={action} msg={message[:80]!r}", flush=True)

        # Handle team/workflow creation locally (no external API needed)
        if action in ("create_team", "list_teams"):
            return await self._handle_team_action(action, message, user_id)

        # Follow-up confirmation handling:
        # When user sends a short confirmation like "yes create all" and the previous
        # assistant message described agents, parse the agent descriptions from context
        # and upgrade to multi-agent creation.
        prev_content = context.get("prev_assistant_content", "")
        if action == "create_agent" and prev_content:
            parsed_from_context = self._parse_multiple_agents(prev_content)
            if len(parsed_from_context) >= 1:
                logger.info(f"🤖 Follow-up confirmation: parsed {len(parsed_from_context)} agents from previous assistant message")
                action = "create_agents"
                # Use the context-parsed agents instead of the short confirmation message
                context["_parsed_agents_from_context"] = parsed_from_context

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                created_agents: List[Dict[str, Any]] = []
                action_summary = ""

                if action == "create_agents":
                    parsed = context.get("_parsed_agents_from_context") or self._parse_multiple_agents(message)
                    logger.info(f"🤖 Multi-agent creation: {len(parsed)} agents to create")
                    for agent_def in parsed:
                        payload = self._build_payload_from_parsed(agent_def)
                        result = await self._create_single_agent(client, payload, headers)
                        if result:
                            created_agents.append(result)
                            logger.info(f"✅ Created agent: {result.get('name')} (ID: {result.get('id')})")

                elif action == "create_agent":
                    create_payload = self._build_agent_create_payload(message)
                    result = await self._create_single_agent(client, create_payload, headers)
                    if result:
                        created_agents.append(result)

                elif action == "rename_agent":
                    action_summary = await self._handle_rename_agent(client, message, headers, user_id)

                elif action == "delete_agent":
                    action_summary = await self._handle_delete_agent(client, message, headers, user_id)

                elif action == "update_agent":
                    action_summary = await self._handle_update_agent(client, message, headers, user_id)

                elif action == "start_agent":
                    action_summary = await self._handle_start_stop_agent(client, message, headers, user_id, start=True)

                elif action == "stop_agent":
                    action_summary = await self._handle_start_stop_agent(client, message, headers, user_id, start=False)

                # Always fetch current agent list
                resp = await client.get(
                    f"{AGENT_ENGINE_URL}/agents/",
                    headers=headers,
                    params={"limit": 20},
                )
                resp.raise_for_status()
                payload = resp.json()

            agents: List[Dict[str, Any]]
            if isinstance(payload, dict):
                raw = payload.get("agents") or payload.get("items") or payload.get("data") or []
                agents = raw if isinstance(raw, list) else []
            elif isinstance(payload, list):
                agents = payload
            else:
                agents = []

            top_agents = []
            for agent in agents[:5]:
                if isinstance(agent, dict):
                    top_agents.append(agent.get("name") or agent.get("id") or "agent")

            # Build summary
            summary = ""

            if action_summary:
                summary = action_summary + "\n\n"
            elif created_agents:
                summary = f"✅ **Created {len(created_agents)} agent(s) via Agent Engine API:**\n\n"
                for ca in created_agents:
                    ca_name = ca.get("name", "Agent")
                    ca_id = ca.get("id", "?")
                    ca_hash = ca.get("agent_public_hash", "")
                    ca_tools = ca.get("tools", [])
                    summary += f"- **{ca_name}** — ID: `{ca_id}`"
                    if ca_hash:
                        summary += f" — Hash: `{ca_hash}`"
                    summary += "\n"
                summary += "\n"

                # Show real webhook URLs if auto-created
                agents_with_webhooks = [
                    ca for ca in created_agents if ca.get("webhook_url")
                ]
                if agents_with_webhooks:
                    summary += "**Webhook Endpoint(s):**\n"
                    for ca in agents_with_webhooks:
                        summary += (
                            f"- **{ca.get('name', 'Agent')}**: `{ca['webhook_url']}`\n"
                            f"  - Method: `POST`\n"
                            f"  - Content-Type: `application/json`\n"
                        )
                    summary += "\n"

                # Add accurate post-creation guidance
                any_has_webhook = bool(agents_with_webhooks) or any(
                    "http_request" in (ca.get("tools") or []) or
                    "webhook" in (ca.get("name") or "").lower()
                    for ca in created_agents
                )
                any_has_discord = any(
                    "discord" in (ca.get("name") or "").lower() or
                    "discord" in (ca.get("description") or "").lower()
                    for ca in created_agents
                )

                summary += "**Next steps:**\n"
                summary += "1. Open the agent dashboard at **/agents** to view and configure your agent(s).\n"
                if agents_with_webhooks:
                    summary += (
                        "2. Use the webhook URL above as the Endpoint URL in your external service "
                        "(Discord, GitHub, Slack, etc.).\n"
                    )
                    summary += (
                        "3. To manage API keys and other integrations, go to **/connect-profiles**.\n"
                    )
                elif any_has_webhook or any_has_discord:
                    summary += (
                        "2. To connect external services (Discord webhooks, GitHub, Slack, etc.), "
                        "go to **/connect-profiles** — this is where you add webhook URLs, API keys, "
                        "and connect integrations.\n"
                    )
                    if any_has_discord:
                        summary += (
                            "3. For Discord: Go to your Discord server → Server Settings → Integrations → "
                            "Webhooks → New Webhook → Copy the webhook URL → Paste it at **/connect-profiles** "
                            "under the Discord card.\n"
                        )
                else:
                    summary += "2. To connect external services, go to **/connect-profiles**.\n"
                summary += "\n"

            if not summary:
                summary = "**Agents OS is ready.**\n\n"

            summary += (
                f"- Total agents: {len(agents)}\n"
                f"- Open panel: {panel_url}\n"
            )
            if top_agents:
                summary += "\n**Your agents:**\n" + "\n".join([f"- {name}" for name in top_agents])

            created_agent_id = created_agents[0].get("id") if created_agents else None
            created_agent_name = created_agents[0].get("name") if created_agents else None
            created_agent_hash = created_agents[0].get("agent_public_hash") if created_agents else None

            return {
                "success": True,
                "action": "open_agents_panel",
                "panel_url": panel_url,
                "operation": action,
                "agent_count": len(agents),
                "agents": agents[:10],
                "created_agent_id": created_agent_id,
                "created_agent_name": created_agent_name,
                "created_agent_public_hash": created_agent_hash,
                "created_agents": [
                    {"id": ca.get("id"), "name": ca.get("name"), "hash": ca.get("agent_public_hash")}
                    for ca in created_agents
                ],
                "summary": summary,
            }
        except Exception as e:
            return {
                "success": False,
                "action": "open_agents_panel",
                "panel_url": panel_url,
                "operation": action,
                "error": f"Agents OS check failed: {e}",
                "summary": (
                    "Agents OS panel can still be opened from split view, but the live agent list check failed. "
                    f"Open manually at {panel_url}."
                ),
            }

    async def _find_agent_by_name(
        self, client: httpx.AsyncClient, name_query: str, headers: Dict[str, str], user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Find an agent by name (fuzzy match) from the user's agent list."""
        try:
            resp = await client.get(
                f"{AGENT_ENGINE_URL}/agents/",
                headers=headers,
                params={"limit": 50},
            )
            resp.raise_for_status()
            payload = resp.json()
            agents_list = payload if isinstance(payload, list) else (
                payload.get("agents") or payload.get("items") or payload.get("data") or []
            )
            query_lower = name_query.lower().strip()
            # Exact match first
            for ag in agents_list:
                if isinstance(ag, dict) and (ag.get("name") or "").lower() == query_lower:
                    return ag
            # Partial match
            for ag in agents_list:
                if isinstance(ag, dict) and query_lower in (ag.get("name") or "").lower():
                    return ag
            return None
        except Exception as e:
            logger.warning(f"Failed to find agent by name '{name_query}': {e}")
            return None

    def _extract_agent_name_from_action(self, message: str, action_verb: str) -> str:
        """Extract the target agent name from an action message like 'rename agent X to Y'."""
        msg = (message or "").strip()
        # Try patterns like: rename agent "X" / rename "X" agent / delete agent named X
        patterns = [
            rf'{action_verb}\s+(?:the\s+|my\s+)?(?:agent\s+)?["\']([^"\']+)["\']',
            rf'{action_verb}\s+(?:the\s+|my\s+)?(?:agent\s+)?(?:named?\s+|called?\s+)?([A-Za-z0-9][A-Za-z0-9 _\-]{{2,50}}?)(?:\s+(?:to|agent|$))',
            rf'(?:agent\s+)["\']?([^"\']+?)["\']?\s*(?:$|to\s|from\s)',
        ]
        for p in patterns:
            m = re.search(p, msg, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return ""

    async def _handle_rename_agent(
        self, client: httpx.AsyncClient, message: str, headers: Dict[str, str], user_id: str
    ) -> str:
        """Handle renaming an agent via Agent Engine API."""
        msg = message.lower()
        # Extract old name and new name: "rename agent X to Y" or "change name of X to Y"
        rename_match = re.search(
            r'(?:rename|change\s+(?:the\s+)?name\s+(?:of\s+)?)\s*(?:the\s+|my\s+)?(?:agent\s+)?'
            r'["\']?(.+?)["\']?\s+to\s+["\']?(.+?)["\']?\s*$',
            message, re.IGNORECASE
        )
        if not rename_match:
            return "❌ Could not parse rename request. Use: **rename agent [old name] to [new name]**"

        old_name = rename_match.group(1).strip().strip("'\"")
        new_name = rename_match.group(2).strip().strip("'\"")

        if not old_name or not new_name:
            return "❌ Please specify both the current agent name and the new name."

        agent = await self._find_agent_by_name(client, old_name, headers, user_id)
        if not agent:
            return f"❌ Could not find an agent named **{old_name}**. Check the name and try again."

        agent_id = agent.get("id")
        try:
            resp = await client.patch(
                f"{AGENT_ENGINE_URL}/agents/{agent_id}",
                headers={**headers, "Content-Type": "application/json"},
                json={"name": new_name},
            )
            if resp.status_code == 405:
                resp = await client.put(
                    f"{AGENT_ENGINE_URL}/agents/{agent_id}",
                    headers={**headers, "Content-Type": "application/json"},
                    json={**agent, "name": new_name},
                )
            resp.raise_for_status()
            logger.info(f"✅ Renamed agent {agent_id}: '{old_name}' → '{new_name}'")
            return f"✅ **Agent renamed:** {old_name} → **{new_name}** (ID: `{agent_id}`)"
        except Exception as e:
            logger.error(f"Failed to rename agent {agent_id}: {e}")
            return f"❌ Failed to rename agent: {e}"

    async def _handle_delete_agent(
        self, client: httpx.AsyncClient, message: str, headers: Dict[str, str], user_id: str
    ) -> str:
        """Handle deleting an agent via Agent Engine API."""
        agent_name = self._extract_agent_name_from_action(message, "delete|remove|destroy")
        if not agent_name:
            return "❌ Could not determine which agent to delete. Use: **delete agent [name]**"

        agent = await self._find_agent_by_name(client, agent_name, headers, user_id)
        if not agent:
            return f"❌ Could not find an agent named **{agent_name}**."

        agent_id = agent.get("id")
        try:
            resp = await client.delete(
                f"{AGENT_ENGINE_URL}/agents/{agent_id}",
                headers=headers,
            )
            resp.raise_for_status()
            logger.info(f"🗑️ Deleted agent: {agent_name} ({agent_id})")
            return f"🗑️ **Agent deleted:** {agent_name} (ID: `{agent_id}`)"
        except Exception as e:
            logger.error(f"Failed to delete agent {agent_id}: {e}")
            return f"❌ Failed to delete agent: {e}"

    async def _handle_update_agent(
        self, client: httpx.AsyncClient, message: str, headers: Dict[str, str], user_id: str
    ) -> str:
        """Handle updating an agent's configuration via Agent Engine API."""
        agent_name = self._extract_agent_name_from_action(message, "update|edit|modify|configure")
        if not agent_name:
            return "❌ Could not determine which agent to update. Use: **update agent [name]**"

        agent = await self._find_agent_by_name(client, agent_name, headers, user_id)
        if not agent:
            return f"❌ Could not find an agent named **{agent_name}**."

        agent_id = agent.get("id")
        # Extract what to update from message
        updates: Dict[str, Any] = {}
        desc_match = re.search(r'description\s+(?:to\s+)?["\']?(.+?)["\']?(?:\s*$|\s+and\s)', message, re.IGNORECASE)
        if desc_match:
            updates["description"] = desc_match.group(1).strip()
        model_match = re.search(r'model\s+(?:to\s+)?(\S+)', message, re.IGNORECASE)
        if model_match:
            updates["model"] = model_match.group(1).strip()

        if not updates:
            return (
                f"Found agent **{agent_name}** (ID: `{agent_id}`), but couldn't determine what to update. "
                "Specify what to change, e.g.: **update agent X description to 'new description'**"
            )

        try:
            resp = await client.patch(
                f"{AGENT_ENGINE_URL}/agents/{agent_id}",
                headers={**headers, "Content-Type": "application/json"},
                json=updates,
            )
            if resp.status_code == 405:
                merged = {**agent, **updates}
                resp = await client.put(
                    f"{AGENT_ENGINE_URL}/agents/{agent_id}",
                    headers={**headers, "Content-Type": "application/json"},
                    json=merged,
                )
            resp.raise_for_status()
            changes_str = ", ".join(f"{k}={v!r}" for k, v in updates.items())
            logger.info(f"✏️ Updated agent {agent_id}: {changes_str}")
            return f"✏️ **Agent updated:** {agent_name} — {changes_str}"
        except Exception as e:
            logger.error(f"Failed to update agent {agent_id}: {e}")
            return f"❌ Failed to update agent: {e}"

    async def _handle_start_stop_agent(
        self, client: httpx.AsyncClient, message: str, headers: Dict[str, str], user_id: str, start: bool = True
    ) -> str:
        """Handle starting or stopping an agent."""
        verb = "start|run|execute|launch|activate" if start else "stop|pause|deactivate|disable|halt"
        agent_name = self._extract_agent_name_from_action(message, verb)
        if not agent_name:
            action_word = "start" if start else "stop"
            return f"❌ Could not determine which agent to {action_word}. Use: **{action_word} agent [name]**"

        agent = await self._find_agent_by_name(client, agent_name, headers, user_id)
        if not agent:
            return f"❌ Could not find an agent named **{agent_name}**."

        agent_id = agent.get("id")
        endpoint = "start" if start else "stop"
        try:
            resp = await client.post(
                f"{AGENT_ENGINE_URL}/agents/{agent_id}/{endpoint}",
                headers=headers,
            )
            if resp.status_code == 404:
                # Fallback: update enabled status
                status_val = True if start else False
                resp = await client.patch(
                    f"{AGENT_ENGINE_URL}/agents/{agent_id}",
                    headers={**headers, "Content-Type": "application/json"},
                    json={"enabled": status_val, "status": "active" if start else "paused"},
                )
                if resp.status_code == 405:
                    resp = await client.put(
                        f"{AGENT_ENGINE_URL}/agents/{agent_id}",
                        headers={**headers, "Content-Type": "application/json"},
                        json={**agent, "enabled": status_val, "status": "active" if start else "paused"},
                    )
            resp.raise_for_status()
            emoji = "▶️" if start else "⏸️"
            action_word = "started" if start else "stopped"
            logger.info(f"{emoji} Agent {action_word}: {agent_name} ({agent_id})")
            return f"{emoji} **Agent {action_word}:** {agent_name} (ID: `{agent_id}`)"
        except Exception as e:
            action_word = "start" if start else "stop"
            logger.error(f"Failed to {action_word} agent {agent_id}: {e}")
            return f"❌ Failed to {action_word} agent: {e}"

    # ============================================
    # RABBIT POST SKILL (uses shared/tools/rabbit.py)
    # ============================================

    async def _execute_rabbit_post(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a Rabbit post from a chat message using shared tools."""
        try:
            from platform_tools.rabbit import tool_create_rabbit_post, tool_list_rabbit_communities
            from platform_tools.auth import AuthContext

            auth = AuthContext(
                user_id=user_id,
                org_id=context.get("org_id"),
                user_role=str(context.get("user_role", "user")),
            )

            # Parse title and body from message
            title, body, community_slug = self._parse_rabbit_post_message(message)

            if not title or not body:
                # List available communities and return help
                communities_result = await tool_list_rabbit_communities(auth=auth)
                community_list = communities_result.get("communities", [])
                community_names = ", ".join([c["slug"] for c in community_list[:10]]) if community_list else "none found"

                return {
                    "success": True,
                    "action": "rabbit_post_help",
                    "summary": (
                        "**Create a Rabbit Post**\n\n"
                        "To create a post, include a title and body:\n"
                        "`create post titled 'My Title' body 'My content here' in r/community`\n\n"
                        f"**Available communities:** {community_names}\n"
                    ),
                }

            result = await tool_create_rabbit_post(
                title=title,
                body=body,
                community_slug=community_slug,
                auth=auth,
            )

            if result.get("success"):
                return {
                    "success": True,
                    "action": "rabbit_post_created",
                    "post_id": result.get("post_id"),
                    "summary": f"**Post Created!**\n\n{result.get('message', '')}",
                }
            else:
                return {
                    "success": False,
                    "action": "rabbit_post_failed",
                    "error": result.get("error", "Unknown error"),
                    "summary": f"Failed to create post: {result.get('error', 'Unknown error')}",
                }

        except ImportError:
            return {
                "success": False,
                "action": "rabbit_post",
                "error": "Shared tools not available",
            }
        except Exception as e:
            return {
                "success": False,
                "action": "rabbit_post",
                "error": str(e),
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
        skill_id = context.get("_integration_skill_id")
        if not skill_id:
            # Fallback: detect from message
            from .skills import is_integration_intent
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
        """Call the architect service via SSE streaming with sync fallback."""
        result: Dict[str, Any] = {
            "success": True,
            "action": "open_agents_panel",
            "panel_url": panel_url,
            "summary": "",
        }

        accumulated_text = ""
        actions_taken = []

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
                        if etype == "text":
                            accumulated_text += event.get("data", {}).get("content", "")
                        elif etype == "tool_call":
                            actions_taken.append(event.get("data", {}))
                        elif etype == "tool_result":
                            pass  # tracked via actions
                        elif etype == "complete":
                            resp_data = event.get("data", {}).get("response", {})
                            accumulated_text = resp_data.get("text", accumulated_text)
                            options_data = resp_data.get("options")
                            if options_data:
                                result["present_options"] = self._map_architect_options(options_data)
                        elif etype == "error":
                            err = event.get("data", {}).get("error", "Unknown error")
                            result["error"] = err

            if accumulated_text:
                result["summary"] = accumulated_text
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
                    result["summary"] = data.get("text", data.get("response", ""))
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

    def _parse_rabbit_post_message(self, message: str) -> tuple:
        """Parse title, body, and community_slug from a chat message."""
        title = ""
        body = ""
        community_slug = None

        # Try pattern: titled 'X' body 'Y' in r/Z
        title_match = re.search(r"titled?\s+['\"]([^'\"]+)['\"]", message, re.IGNORECASE)
        body_match = re.search(r"body\s+['\"]([^'\"]+)['\"]", message, re.IGNORECASE)
        community_match = re.search(r"in\s+(r/\S+)", message, re.IGNORECASE)

        if title_match:
            title = title_match.group(1)
        if body_match:
            body = body_match.group(1)
        if community_match:
            community_slug = community_match.group(1)

        # Fallback: title: X\nbody: Y
        if not title:
            t2 = re.search(r"title:\s*(.+?)(?:\n|body:|$)", message, re.IGNORECASE)
            if t2:
                title = t2.group(1).strip().strip("'\"")
        if not body:
            b2 = re.search(r"body:\s*(.+?)(?:\n|community:|$)", message, re.IGNORECASE | re.DOTALL)
            if b2:
                body = b2.group(1).strip().strip("'\"")

        # Fallback: use entire message as body, first line as title
        if not title and not body:
            lines = message.strip().split("\n")
            # Remove trigger keywords from first line
            first = re.sub(r"(?i)(create|make|post|rabbit)\s*(post)?\s*", "", lines[0]).strip()
            if first:
                title = first[:200]
                body = "\n".join(lines[1:]).strip() if len(lines) > 1 else first

        return title, body, community_slug


# Global singleton
skill_executor = SkillExecutor()
