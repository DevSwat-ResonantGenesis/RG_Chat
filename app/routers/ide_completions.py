"""IDE Completions Router — lightweight LLM streaming for Resonant Local IDE.

Passes tool definitions to the LLM and streams back responses + tool_calls.
Does NOT execute any tools — the client handles all tool execution locally.
Routes through the unified LLM service via HTTP (no rg_llm dependency).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ide", tags=["ide"])

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm_service:8000").rstrip("/")


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
            payload = {
                "messages": request_body.messages,
                "model": request_body.model,
                "temperature": request_body.temperature,
                "max_tokens": request_body.max_tokens,
                "stream": True,
                "user_id": user_id,
            }
            if request_body.tools:
                payload["tools"] = request_body.tools
            if request_body.preferred_provider:
                payload["provider"] = request_body.preferred_provider
            if request_body.response_format:
                payload["response_format"] = request_body.response_format
            if user_keys:
                payload["user_api_keys"] = user_keys

            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{LLM_SERVICE_URL}/v1/chat/completions",
                    json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"event: error\ndata: {json.dumps({'error': f'LLM service returned {resp.status_code}: {body.decode()[:200]}'})}\n\n"
                        return

                    buffer = ""
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                                content = delta.get("content")
                                tool_calls = delta.get("tool_calls")

                                if content:
                                    yield f"event: chunk\ndata: {json.dumps({'content': content})}\n\n"

                                if tool_calls:
                                    tc_list = []
                                    for tc in tool_calls:
                                        fn = tc.get("function", {})
                                        tc_list.append({
                                            "id": tc.get("id", ""),
                                            "type": "function",
                                            "function": {
                                                "name": fn.get("name", ""),
                                                "arguments": fn.get("arguments", ""),
                                            },
                                        })
                                    if tc_list:
                                        yield f"event: tool_calls\ndata: {json.dumps({'tool_calls': tc_list})}\n\n"
                            except json.JSONDecodeError:
                                continue

            # If streaming didn't work, try non-streaming fallback
            yield f"event: done\ndata: {json.dumps({'provider': request_body.preferred_provider or 'auto', 'model': request_body.model or ''})}\n\n"

        except Exception as e:
            logger.error(f"IDE completions error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
