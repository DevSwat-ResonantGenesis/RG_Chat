"""Provider facade for chat and agents.

All LLM calls go through UnifiedLLMClient (via MultiAIRouter).
Handles BYOK keys, provider preference, streaming, and local LLM tunnel.
"""

import logging
import os
from typing import Dict, List, Optional

import httpx

from rg_llm import UnifiedLLMClient, LLMRequest, LLMStreamEvent, StreamEventType

from .multi_ai_router import MultiAIRouter

logger = logging.getLogger(__name__)

_internal_router = MultiAIRouter()
_llm_client = UnifiedLLMClient()

# Provider name aliases
_ALIASES = {
    "chatgpt": "openai", "gpt": "openai",
    "gemini": "google", "claude": "anthropic",
}


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
    """Route a chat/agent query to an LLM provider (non-streaming)."""
    user_keys = {k: v for k, v in (user_api_keys or {}).items() if not k.startswith("__")}

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
        messages = _build_messages(context, message)
        try:
            user_id = (user_api_keys or {}).get("__user_id__", "")
            model = "codellama:13b" if provider == "codellama" else "llama3.1:8b"
            async for chunk in _stream_local(messages, model, user_id):
                yield chunk
            yield {"type": "done"}
        except Exception as e:
            yield {"type": "error", "error": str(e)}
        return

    # Stream via UnifiedLLMClient
    messages = _build_messages(context, message)
    user_keys = {k: v for k, v in (user_api_keys or {}).items() if not k.startswith("__")}

    norm = _ALIASES.get(
        (preferred_provider or "").lower(), (preferred_provider or "").lower()
    ) or None

    try:
        yield {"type": "provider", "provider": preferred_provider or "auto"}
        request = LLMRequest(
            messages=messages,
            provider=norm,
            temperature=0.7,
            max_tokens=16384,
            stream=True,
        )
        async for event in _llm_client.stream(request, user_keys=user_keys or None):
            if event.event == StreamEventType.CHUNK and event.content:
                yield {"type": "chunk", "content": event.content}
            elif event.event == StreamEventType.ERROR:
                yield {"type": "error", "error": event.error or "Unknown streaming error"}
                return
        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "error": str(e)}


def _build_messages(context: Optional[List[Dict]], message: str) -> List[Dict]:
    """Build message list from context + user message."""
    messages = []
    system_content = ""
    if context:
        for msg in context:
            if msg.get("role") == "system":
                system_content += msg.get("content", "") + "\n"
            else:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    if system_content:
        messages = [{"role": "system", "content": system_content.strip()}] + messages
    messages.append({"role": "user", "content": message})
    return messages


async def _stream_local(messages: list, model: str, user_id: str):
    """Route completion through gateway's local LLM tunnel proxy."""
    gateway_url = os.getenv("GATEWAY_URL", "http://gateway:8000")

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{gateway_url}/api/v1/local-llm/tunnel/completions",
            headers={"Content-Type": "application/json", "x-user-id": user_id},
            json={"messages": messages, "model": model, "temperature": 0.7, "max_tokens": 16384},
        )
        if resp.status_code == 503:
            raise Exception("No local LLM tunnel active — open DevSwat in your browser and connect your local LLM on the Integrations page.")
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise Exception(data["error"])
        content = ""
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
        if content:
            yield {"type": "chunk", "content": content}
