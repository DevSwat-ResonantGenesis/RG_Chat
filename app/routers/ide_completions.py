"""IDE Completions Router — lightweight LLM streaming for Resonant Local IDE.

Passes tool definitions to the LLM and streams back responses + tool_calls.
Does NOT execute any tools — the client handles all tool execution locally.
Uses rg_llm UnifiedLLMClient for all provider routing + streaming.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rg_llm import UnifiedLLMClient, LLMRequest, LLMStreamEvent
from rg_llm.models import StreamEventType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ide", tags=["ide"])

_llm_client = UnifiedLLMClient(
    fallback_order=["groq", "openai", "anthropic", "google", "deepseek", "mistral"],
)


class IDECompletionRequest(BaseModel):
    """Request for IDE completions — OpenAI-compatible messages + tools."""
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    preferred_provider: Optional[str] = None
    response_format: Optional[Dict[str, str]] = None


async def _get_user_keys(user_id: str) -> Dict[str, str]:
    """Fetch BYOK keys for a user (returns provider→key mapping)."""
    try:
        from ..services.user_api_keys import user_api_key_service
        return await user_api_key_service.get_user_api_keys(user_id) or {}
    except Exception as e:
        logger.warning(f"BYOK key lookup failed: {e}")
        return {}


@router.post("/completions")
async def ide_completions(
    request_body: IDECompletionRequest,
    request: Request,
):
    """Stream LLM completions with tool support for the local IDE.

    SSE Events:
    - chunk: {"content": "text..."}
    - tool_calls: {"tool_calls": [{id, type, function: {name, arguments}}]}
    - done: {"provider": "...", "model": "...", "usage": {...}}
    - error: {"error": "..."}
    """
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Fetch BYOK keys for provider selection
    user_keys = await _get_user_keys(user_id)

    logger.info(
        f"IDE completions: preferred={request_body.preferred_provider}, "
        f"model={request_body.model}, tools={len(request_body.tools or [])}"
    )

    async def generate():
        try:
            llm_request = LLMRequest(
                messages=request_body.messages,
                provider=request_body.preferred_provider,
                model=request_body.model,
                temperature=request_body.temperature,
                max_tokens=request_body.max_tokens,
                tools=request_body.tools,
                response_format=request_body.response_format,
                stream=True,
                user_id=user_id,
            )

            async for event in _llm_client.stream(llm_request, user_keys=user_keys):
                if event.event == StreamEventType.CHUNK:
                    yield f"event: chunk\ndata: {json.dumps({'content': event.content})}\n\n"

                elif event.event == StreamEventType.TOOL_CALLS:
                    # Convert ToolCall objects to OpenAI wire format for IDE
                    tc_list = [{
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    } for tc in event.tool_calls]
                    yield f"event: tool_calls\ndata: {json.dumps({'tool_calls': tc_list})}\n\n"

                elif event.event == StreamEventType.DONE:
                    yield f"event: done\ndata: {json.dumps({'provider': event.provider, 'model': event.model, 'usage': event.usage})}\n\n"

                elif event.event == StreamEventType.ERROR:
                    yield f"event: error\ndata: {json.dumps({'error': event.error})}\n\n"

        except Exception as e:
            logger.error(f"IDE completions error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
