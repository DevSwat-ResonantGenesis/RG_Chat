"""Provider facade for chat and agents.

Routes LLM queries through MultiAIRouter (direct provider calls) and
falls back to the unified LLM service HTTP endpoint when available.
"""

import logging
import os
from typing import Dict, List, Optional

import httpx

from .multi_ai_router import MultiAIRouter

logger = logging.getLogger(__name__)

# Primary router — handles all LLM calls via direct provider SDKs
_internal_router = MultiAIRouter()

# Unified LLM service (used as secondary path when available)
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm_service:8000").rstrip("/")


def set_user_api_keys(keys: Dict[str, str]) -> None:
    """Configure BYOK keys on the router."""
    _internal_router.set_user_api_keys(keys)


def clear_user_api_keys() -> None:
    """Clear user-specific API keys on the router."""
    _internal_router.set_user_api_keys({})


async def route_query(
    message: str,
    context: Optional[List[Dict]] = None,
    preferred_provider: Optional[str] = None,
    user_api_keys: Optional[Dict[str, str]] = None,
    images: Optional[List[Dict]] = None,
) -> Dict:
    """Route a chat/agent query to an LLM provider (non-streaming).

    Tries the unified LLM service first, falls back to MultiAIRouter.
    """
    # Build messages
    messages = []
    if context:
        for msg in context:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": message})

    user_keys = {k: v for k, v in (user_api_keys or {}).items() if not k.startswith("__")}

    # Try unified LLM service first
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{LLM_SERVICE_URL}/v1/chat/completions",
                json={
                    "messages": messages,
                    "provider": preferred_provider,
                    "user_api_keys": user_keys or None,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message", {})
                return {
                    "response": msg.get("content", ""),
                    "provider": data.get("provider", preferred_provider or "unknown"),
                    "model": data.get("model", ""),
                    "usage": data.get("usage", {}),
                }
    except Exception as e:
        logger.debug(f"Unified LLM service unavailable, falling back to MultiAIRouter: {e}")

    # Fallback: use MultiAIRouter directly
    if user_keys:
        _internal_router.set_user_api_keys(user_keys)
    try:
        result = await _internal_router.route_query(
            message=message,
            context=context,
            preferred_provider=preferred_provider,
            images=images,
        )
        return result
    finally:
        if user_keys:
            _internal_router.set_user_api_keys({})


def get_router_for_internal_use() -> MultiAIRouter:
    """Expose the internal router for legacy integrations."""
    return _internal_router


async def route_query_stream(
    message: str,
    context: Optional[List[Dict]] = None,
    preferred_provider: Optional[str] = None,
    user_api_keys: Optional[Dict[str, str]] = None,
):
    """Stream a query response from LLM provider.

    Yields:
        Dict with 'type' ('chunk', 'provider', 'error', 'done') and content
    """
    # Handle local provider separately — doesn't need an API key
    if preferred_provider and preferred_provider.lower() in ("local", "codellama"):
        provider = preferred_provider.lower()
        yield {"type": "provider", "provider": provider}
        messages = []
        system_content = ""
        if context:
            for msg in context:
                if msg.get("role") == "system":
                    system_content += msg.get("content", "") + "\n"
                else:
                    messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        messages.append({"role": "user", "content": message})
        if system_content:
            messages = [{"role": "system", "content": system_content.strip()}] + messages
        try:
            user_id = (user_api_keys or {}).get("__user_id__", "")
            model = "codellama:13b" if provider == "codellama" else "llama3.1:8b"
            async for chunk in _stream_local(messages, model, user_id):
                yield chunk
            yield {"type": "done"}
        except Exception as e:
            yield {"type": "error", "error": str(e)}
        return

    # Build messages from context
    messages = []
    if context:
        for msg in context:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": message})

    user_keys = {k: v for k, v in (user_api_keys or {}).items() if not k.startswith("__")}

    # Stream via MultiAIRouter
    try:
        if user_keys:
            _internal_router.set_user_api_keys(user_keys)
        provider_name = preferred_provider or "auto"
        yield {"type": "provider", "provider": provider_name}

        result = await _internal_router.route_query(
            message=message,
            context=context,
            preferred_provider=preferred_provider,
        )
        content = result.get("response", "")
        if content:
            yield {"type": "chunk", "content": content}
        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "error": str(e)}
    finally:
        if user_keys:
            _internal_router.set_user_api_keys({})


async def _stream_local(messages: list, model: str, user_id: str):
    """Route completion through gateway's local LLM tunnel proxy.
    
    The gateway holds a per-user WebSocket tunnel to the user's browser,
    which bridges to their local Ollama/LM Studio instance.
    """
    import httpx
    import os
    
    gateway_url = os.getenv("GATEWAY_URL", "http://gateway:8000")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{gateway_url}/api/v1/local-llm/tunnel/completions",
            headers={
                "Content-Type": "application/json",
                "x-user-id": user_id,
            },
            json={
                "messages": messages,
                "model": model,
                "temperature": 0.7,
                "max_tokens": 4096,
            },
        )
        
        if resp.status_code == 503:
            raise Exception("No local LLM tunnel active — open ResonantGenesis in your browser and connect your local LLM on the Integrations page.")
        
        resp.raise_for_status()
        data = resp.json()
        
        if "error" in data:
            raise Exception(data["error"])
        
        # Extract content from the completion response
        content = ""
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
        
        if content:
            yield {"type": "chunk", "content": content}
