"""
Owner Internal Catalog Router
===============================
Owner-only endpoint returning ALL internal platform agents, teams,
skills, RARA types, and autonomous infrastructure from live code.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/owner", tags=["owner-catalog"])


def _require_owner(request: Request) -> None:
    """Raise 403 if the caller is not a platform owner/superuser."""
    role = (request.headers.get("x-user-role") or "").lower()
    is_su = (request.headers.get("x-is-superuser") or "").lower() == "true"
    if not (is_su or role in ("owner", "platform_owner", "admin", "superuser")):
        raise HTTPException(status_code=403, detail="Owner access required")


@router.get("/internal-catalog")
async def get_internal_catalog(request: Request):
    """Return the full internal agent/team/skill catalog from live code.

    Only accessible to platform owners.
    """
    _require_owner(request)

    # ── Individual agent types from agent_engine ──
    from ..services.agent_engine import AgentEngine
    engine = AgentEngine.__new__(AgentEngine)
    raw_prompts: dict = {}
    try:
        # _get_agent_prompts returns a dict keyed by agent_type
        raw_prompts = engine._get_agent_prompts("__list__")  # returns default
        # Actually need ALL keys – the dict is defined inline, so instantiate properly
    except Exception:
        pass

    # Build the prompts dict directly from source
    try:
        import inspect, re, textwrap
        source = inspect.getsource(engine._get_agent_prompts)
        # Extract keys from `prompts = { "key": ...`
        agent_keys = re.findall(r'"(\w+)":\s*\[', source)
    except Exception:
        agent_keys = []

    if not agent_keys:
        # Fallback hardcoded from codebase analysis
        agent_keys = [
            "reasoning", "code", "debug", "review", "test", "research",
            "explain", "summary", "planning", "security", "architecture",
            "optimization", "documentation", "math", "api", "database",
            "devops", "migration", "refactor", "accessibility", "i18n",
            "regex", "git", "css",
        ]

    # ── Agent capability registry ──
    agent_capabilities: Dict[str, Any] = {}
    try:
        from ..services.agent_capability_registry import AGENT_CAPABILITIES
        for key, cap in AGENT_CAPABILITIES.items():
            agent_capabilities[key] = {
                "strengths": getattr(cap, "strengths", []),
                "weaknesses": getattr(cap, "weaknesses", []),
                "success_rate": getattr(cap, "success_rate", None),
                "avg_response_time": getattr(cap, "avg_response_time", None),
                "specializations": getattr(cap, "specializations", {}),
            }
    except Exception as e:
        logger.warning(f"Could not load AGENT_CAPABILITIES: {e}")

    agents_list = []
    for key in agent_keys:
        cap = agent_capabilities.get(key, {})
        agents_list.append({
            "id": key,
            "name": key.replace("_", " ").title() + " Agent",
            "category": _categorize_agent(key),
            "autonomous": True,
            "specializations": cap.get("specializations", {}),
            "success_rate": cap.get("success_rate"),
            "avg_response_time": cap.get("avg_response_time"),
            "strengths": cap.get("strengths", []),
            "weaknesses": cap.get("weaknesses", []),
        })

    # ── Internal teams from team_engine ──
    teams_list = []
    try:
        from ..services.team_engine import INTERNAL_TEAMS
        for tid, tdef in INTERNAL_TEAMS.items():
            teams_list.append({
                "id": tid,
                "name": tdef.name,
                "agents": tdef.agents,
                "workflow": tdef.workflow,
                "description": tdef.description,
                "trigger_keywords": tdef.trigger_keywords,
            })
    except Exception as e:
        logger.warning(f"Could not load INTERNAL_TEAMS: {e}")

    # ── Chat skills from skills_registry ──
    skills_list = []
    try:
        from ..services.skills_registry import skills_registry
        skills_list = skills_registry.list_skills()
    except Exception as e:
        logger.warning(f"Could not load skills: {e}")

    # ── RARA agent types ──
    rara_types = []
    try:
        from importlib import import_module
        mod = import_module("rara_service.app.invariants.agent_factory_invariants")
        AgentType = getattr(mod, "AgentType", None)
        if AgentType:
            rara_types = [{"id": t.value, "name": t.name} for t in AgentType]
    except Exception:
        # Fallback from codebase analysis
        rara_types = [
            {"id": "task_executor", "name": "Task Executor", "description": "Executes defined tasks with strict safety boundaries"},
            {"id": "business_operator", "name": "Business Operator", "description": "Manages business logic, workflows, and automated operations"},
            {"id": "tool_agent", "name": "Tool Agent", "description": "Interfaces with external tools, APIs, and integrations"},
            {"id": "swarm_member", "name": "Swarm Member", "description": "Participates in multi-agent swarms for distributed tasks"},
            {"id": "observer_auditor", "name": "Observer / Auditor", "description": "Monitors agent actions, enforces safety rules, audits compliance"},
        ]

    # ── Autonomous infrastructure ──
    infra = [
        {"name": "AutonomousAgentExecutor", "description": "Wraps any agent type for autonomous decision-making with KB lookup and LLM fallback.", "source": "chat_service/app/services/autonomous_agent_executor.py"},
        {"name": "AutonomousDaemon", "description": "Background daemon managing autonomous agent lifecycle, self-triggering, and health monitoring.", "source": "agent_engine_service/app/routers_autonomous.py"},
        {"name": "ParallelAgentRuntime", "description": "Enables parallel agent communication, capability registration, and multi-agent coordination.", "source": "agent_engine_service/app/parallel_runtime.py"},
        {"name": "AgentCapabilityRegistry", "description": "Tracks agent strengths, weaknesses, success rates, specialization scores for routing.", "source": "chat_service/app/services/agent_capability_registry.py"},
    ]

    return {
        "agents": agents_list,
        "teams": teams_list,
        "skills": skills_list,
        "rara_types": rara_types,
        "infrastructure": infra,
        "counts": {
            "agents": len(agents_list),
            "teams": len(teams_list),
            "skills": len(skills_list),
            "rara_types": len(rara_types),
            "infrastructure": len(infra),
        },
    }


def _categorize_agent(agent_type: str) -> str:
    """Map agent type to a category."""
    categories = {
        "reasoning": "Core", "explain": "Core", "summary": "Core",
        "research": "Core", "planning": "Core", "math": "Core",
        "code": "Development", "debug": "Development", "review": "Development",
        "test": "Development", "documentation": "Development", "migration": "Development",
        "refactor": "Development", "css": "Development",
        "security": "Security",
        "architecture": "Architecture", "api": "Architecture", "database": "Architecture",
        "optimization": "Performance",
        "devops": "Infrastructure",
        "accessibility": "Quality", "i18n": "Quality",
        "regex": "Utility", "git": "Utility",
    }
    return categories.get(agent_type, "Other")
