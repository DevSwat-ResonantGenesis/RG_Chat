"""
Skills API Router
==================

Endpoints for managing Resonant Chat skills:
- List available skills
- Get user's skill preferences
- Enable/disable skills
- Execute skill actions directly
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/skills", tags=["skills"])


def _is_owner(request: Request) -> bool:
    """Check if the request comes from a platform owner/superuser."""
    role = (request.headers.get("x-user-role") or "").lower()
    is_su = (request.headers.get("x-is-superuser") or "").lower() == "true"
    return is_su or role in ("owner", "platform_owner", "admin", "superuser")


# ============================================
# REQUEST / RESPONSE MODELS
# ============================================

class SkillToggleRequest(BaseModel):
    skill_id: str
    enabled: bool


class SkillCreateRequest(BaseModel):
    name: str
    description: str = ""
    icon: str = "🧠"
    category: str = "utility"
    agent_type: Optional[str] = None
    trigger_keywords: List[str] = []
    capabilities: List[str] = []
    credit_cost: int = 0
    requires_api_key: Optional[str] = None
    is_default: bool = False


class SkillExecuteRequest(BaseModel):
    skill_id: str
    message: str
    context: Optional[Dict[str, Any]] = None


class SkillResponse(BaseModel):
    id: str
    name: str
    description: str
    icon: str
    category: str
    capabilities: List[str]
    credit_cost: int
    requires_api_key: Optional[str] = None
    is_default: bool = False
    enabled: bool = False


class SkillListResponse(BaseModel):
    skills: List[SkillResponse]


class SkillExecuteResponse(BaseModel):
    skill_id: str
    skill_name: str
    success: bool
    action: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


# ============================================
# ENDPOINTS
# ============================================

@router.get("/list", response_model=SkillListResponse)
async def list_skills(request: Request):
    """List all available skills with user's enabled/disabled status."""
    from ..services.skills_registry import skills_registry

    user_id = request.headers.get("x-user-id", "anonymous")
    all_skills = skills_registry.list_skills()
    user_prefs = skills_registry.get_user_skills(user_id)

    skills = []
    for s in all_skills:
        skills.append(SkillResponse(
            id=s["id"],
            name=s["name"],
            description=s["description"],
            icon=s["icon"],
            category=s["category"],
            capabilities=s["capabilities"],
            credit_cost=s["credit_cost"],
            requires_api_key=s.get("requires_api_key"),
            is_default=s.get("is_default", False),
            enabled=user_prefs.get(s["id"], s.get("is_default", False)),
        ))

    return SkillListResponse(skills=skills)


@router.post("/toggle")
async def toggle_skill(request: Request, body: SkillToggleRequest):
    """Enable or disable a skill for the current user."""
    from ..services.skills_registry import skills_registry

    user_id = request.headers.get("x-user-id", "anonymous")

    if body.enabled:
        success = skills_registry.enable_skill(user_id, body.skill_id)
    else:
        success = skills_registry.disable_skill(user_id, body.skill_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Skill '{body.skill_id}' not found")

    return {
        "skill_id": body.skill_id,
        "enabled": body.enabled,
        "message": f"Skill '{body.skill_id}' {'enabled' if body.enabled else 'disabled'}",
    }


@router.post("/execute", response_model=SkillExecuteResponse)
async def execute_skill(request: Request, body: SkillExecuteRequest):
    """Execute a skill action directly."""
    from ..services.skills_registry import skills_registry
    from ..services.skill_executor import skill_executor

    user_id = request.headers.get("x-user-id", "anonymous")

    skill = skills_registry.get_skill(body.skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{body.skill_id}' not found")

    # Check if skill is enabled for user
    user_prefs = skills_registry.get_user_skills(user_id)
    if not user_prefs.get(body.skill_id, False):
        raise HTTPException(
            status_code=403,
            detail=f"Skill '{body.skill_id}' is not enabled. Enable it first.",
        )

    result = await skill_executor.execute(
        skill=skill,
        message=body.message,
        user_id=user_id,
        context=body.context,
    )

    return SkillExecuteResponse(
        skill_id=result.get("skill_id", body.skill_id),
        skill_name=result.get("skill_name", skill.name),
        success=result.get("success", False),
        action=result.get("action"),
        summary=result.get("summary"),
        error=result.get("error"),
        data={k: v for k, v in result.items() if k not in {"skill_id", "skill_name", "success", "action", "summary", "error"}},
    )


@router.post("/create")
async def create_skill(request: Request, body: SkillCreateRequest):
    """Create a new skill. Owner/superuser only."""
    if not _is_owner(request):
        raise HTTPException(status_code=403, detail="Only platform owners can create skills")

    from ..services.skills_registry import skills_registry, SkillDefinition, SkillCategory

    # Validate category
    category_map = {
        "analysis": SkillCategory.ANALYSIS,
        "search": SkillCategory.SEARCH,
        "generation": SkillCategory.GENERATION,
        "memory": SkillCategory.MEMORY,
        "utility": SkillCategory.UTILITY,
    }
    cat = category_map.get(body.category.lower(), SkillCategory.UTILITY)

    skill_id = body.name.lower().replace(" ", "_").replace("-", "_")
    if skills_registry.get_skill(skill_id):
        raise HTTPException(status_code=409, detail=f"Skill '{skill_id}' already exists")

    skill = SkillDefinition(
        id=skill_id,
        name=body.name,
        description=body.description,
        icon=body.icon,
        category=cat,
        agent_type=body.agent_type,
        trigger_keywords=body.trigger_keywords,
        capabilities=body.capabilities,
        credit_cost=body.credit_cost,
        requires_api_key=body.requires_api_key,
        is_default=body.is_default,
    )
    skills_registry.register_skill(skill)
    logger.info(f"Owner created new skill: {skill_id}")

    return {
        "status": "created",
        "skill_id": skill_id,
        "name": body.name,
    }


@router.delete("/delete/{skill_id}")
async def delete_skill(request: Request, skill_id: str):
    """Delete a skill. Owner/superuser only."""
    if not _is_owner(request):
        raise HTTPException(status_code=403, detail="Only platform owners can delete skills")

    from ..services.skills_registry import skills_registry

    if not skills_registry.unregister_skill(skill_id):
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")

    return {"status": "deleted", "skill_id": skill_id}


@router.get("/enabled")
async def get_enabled_skills(request: Request):
    """Get list of enabled skills for the current user."""
    from ..services.skills_registry import skills_registry

    user_id = request.headers.get("x-user-id", "anonymous")
    enabled = skills_registry.get_enabled_skills(user_id)

    return {
        "enabled_skills": [
            {
                "id": s.id,
                "name": s.name,
                "icon": s.icon,
                "category": s.category.value,
            }
            for s in enabled
        ]
    }
