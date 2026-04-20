"""
Tools API Router
==================

Endpoints for managing Resonant Chat tools:
- List available tools
- Get user's tool preferences
- Enable/disable tools
- Execute tool actions directly
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["tools"])


def _is_owner(request: Request) -> bool:
    """Check if the request comes from a platform owner/superuser."""
    role = (request.headers.get("x-user-role") or "").lower()
    is_su = (request.headers.get("x-is-superuser") or "").lower() == "true"
    return is_su or role in ("owner", "platform_owner", "admin", "superuser")


# ============================================
# REQUEST / RESPONSE MODELS
# ============================================

class ToolToggleRequest(BaseModel):
    tool_id: str
    enabled: bool


class ToolCreateRequest(BaseModel):
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


class ToolExecuteRequest(BaseModel):
    tool_id: str
    message: str
    context: Optional[Dict[str, Any]] = None


class ToolResponse(BaseModel):
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


class ToolListResponse(BaseModel):
    tools: List[ToolResponse]


class ToolExecuteResponse(BaseModel):
    tool_id: str
    tool_name: str
    success: bool
    action: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


# ============================================
# ENDPOINTS
# ============================================

@router.get("/list", response_model=ToolListResponse)
async def list_tools(request: Request):
    """List all available tools with user's enabled/disabled status."""
    from ..services.tools_registry import tools_registry

    user_id = request.headers.get("x-user-id", "anonymous")
    all_tools = tools_registry.list_tools()
    user_prefs = tools_registry.get_user_tools(user_id)

    tools = []
    for s in all_tools:
        tools.append(ToolResponse(
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

    return ToolListResponse(tools=tools)


@router.post("/toggle")
async def toggle_tool(request: Request, body: ToolToggleRequest):
    """Enable or disable a tool for the current user."""
    from ..services.tools_registry import tools_registry

    user_id = request.headers.get("x-user-id", "anonymous")

    if body.enabled:
        success = tools_registry.enable_tool(user_id, body.tool_id)
    else:
        success = tools_registry.disable_tool(user_id, body.tool_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Tool '{body.tool_id}' not found")

    return {
        "tool_id": body.tool_id,
        "enabled": body.enabled,
        "message": f"Tool '{body.tool_id}' {'enabled' if body.enabled else 'disabled'}",
    }


@router.post("/execute", response_model=ToolExecuteResponse)
async def execute_tool(request: Request, body: ToolExecuteRequest):
    """Execute a tool action directly."""
    from ..services.tools_registry import tools_registry
    from ..services.tool_executor import tool_executor

    user_id = request.headers.get("x-user-id", "anonymous")

    tool = tools_registry.get_tool(body.tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{body.tool_id}' not found")

    # Check if tool is enabled for user
    user_prefs = tools_registry.get_user_tools(user_id)
    if not user_prefs.get(body.tool_id, False):
        raise HTTPException(
            status_code=403,
            detail=f"Tool '{body.tool_id}' is not enabled. Enable it first.",
        )

    result = await tool_executor.execute(
        tool=tool,
        message=body.message,
        user_id=user_id,
        context=body.context,
    )

    return ToolExecuteResponse(
        tool_id=result.get("tool_id", body.tool_id),
        tool_name=result.get("tool_name", tool.name),
        success=result.get("success", False),
        action=result.get("action"),
        summary=result.get("summary"),
        error=result.get("error"),
        data={k: v for k, v in result.items() if k not in {"tool_id", "tool_name", "success", "action", "summary", "error"}},
    )


@router.post("/create")
async def create_tool(request: Request, body: ToolCreateRequest):
    """Create a new tool. Owner/superuser only."""
    if not _is_owner(request):
        raise HTTPException(status_code=403, detail="Only platform owners can create tools")

    from ..services.tools_registry import tools_registry, ToolDefinition, ToolCategory

    # Validate category
    category_map = {
        "analysis": ToolCategory.ANALYSIS,
        "search": ToolCategory.SEARCH,
        "generation": ToolCategory.GENERATION,
        "memory": ToolCategory.MEMORY,
        "utility": ToolCategory.UTILITY,
    }
    cat = category_map.get(body.category.lower(), ToolCategory.UTILITY)

    tool_id = body.name.lower().replace(" ", "_").replace("-", "_")
    if tools_registry.get_tool(tool_id):
        raise HTTPException(status_code=409, detail=f"Tool '{tool_id}' already exists")

    tool = ToolDefinition(
        id=tool_id,
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
    tools_registry.register_tool(tool)
    logger.info(f"Owner created new tool: {tool_id}")

    return {
        "status": "created",
        "tool_id": tool_id,
        "name": body.name,
    }


@router.delete("/delete/{tool_id}")
async def delete_tool(request: Request, tool_id: str):
    """Delete a tool. Owner/superuser only."""
    if not _is_owner(request):
        raise HTTPException(status_code=403, detail="Only platform owners can delete tools")

    from ..services.tools_registry import tools_registry

    if not tools_registry.unregister_tool(tool_id):
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")

    return {"status": "deleted", "tool_id": tool_id}


@router.get("/enabled")
async def get_enabled_tools(request: Request):
    """Get list of enabled tools for the current user."""
    from ..services.tools_registry import tools_registry

    user_id = request.headers.get("x-user-id", "anonymous")
    enabled = tools_registry.get_enabled_tools(user_id)

    return {
        "enabled_tools": [
            {
                "id": s.id,
                "name": s.name,
                "icon": s.icon,
                "category": s.category.value,
            }
            for s in enabled
        ]
    }
