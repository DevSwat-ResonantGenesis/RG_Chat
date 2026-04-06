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

    # Fake agent_engine, agent_capability_registry, team_engine removed.
    # Chat no longer has "internal agents" — those were system prompt wrappers.
    # Real agents live in Agent Engine service (executor.py).
    agents_list = []
    teams_list = []

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

    # Fake autonomous infrastructure entries removed (all deleted modules).
    # Real infrastructure: Agent Engine executor.py, scheduler_daemon.py, autonomous_daemon.py
    infra = []

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
