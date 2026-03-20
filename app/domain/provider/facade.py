"""Provider facade for chat and agents.

Routes LLM queries via rg_llm UnifiedLLMClient.
Keeps MultiAIRouter only for get_router_for_internal_use() backward compat.
"""

import logging
import os
from typing import Dict, List, Optional

from rg_llm import UnifiedLLMClient, LLMRequest
from rg_llm.models import StreamEventType

from .multi_ai_router import MultiAIRouter

logger = logging.getLogger(__name__)

# Unified client — single source of truth for all LLM calls
_llm_client = UnifiedLLMClient(
    fallback_order=["openai", "anthropic", "google", "groq"],
)

# Internal router kept ONLY for get_router_for_internal_use() backward compat
_internal_router = MultiAIRouter()


def set_user_api_keys(keys: Dict[str, str]) -> None:
    """Configure BYOK keys on the internal router for legacy paths."""
    _internal_router.set_user_api_keys(keys)


def clear_user_api_keys() -> None:
    """Clear user-specific API keys on the internal router."""
    _internal_router.set_user_api_keys({})


async def route_query(
    message: str,
    context: Optional[List[Dict]] = None,
    preferred_provider: Optional[str] = None,
    user_api_keys: Optional[Dict[str, str]] = None,
    images: Optional[List[Dict]] = None,
) -> Dict:
    """Route a chat/agent query to an LLM provider (non-streaming)."""
    messages = []
    if context:
        for msg in context:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": message})

    # Strip __user_id__ from BYOK keys dict if present
    user_keys = {k: v for k, v in (user_api_keys or {}).items() if not k.startswith("__")}

    response = await _llm_client.complete(
        LLMRequest(messages=messages, provider=preferred_provider),
        user_keys=user_keys or None,
    )
    return {
        "response": response.content,
        "provider": response.provider,
        "model": response.model,
        "usage": response.usage,
    }


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

    try:
        llm_request = LLMRequest(
            messages=messages,
            provider=preferred_provider,
            stream=True,
        )

        async for event in _llm_client.stream(llm_request, user_keys=user_keys or None):
            if event.event == StreamEventType.PROVIDER:
                yield {"type": "provider", "provider": event.provider}
            elif event.event == StreamEventType.CHUNK:
                yield {"type": "chunk", "content": event.content}
            elif event.event == StreamEventType.DONE:
                pass  # done emitted below
            elif event.event == StreamEventType.ERROR:
                yield {"type": "error", "error": event.error}
                return

        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "error": str(e)}


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
