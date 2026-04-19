"""
Multi-AI Routing Service — powered by UnifiedLLMClient.

All provider-specific code (key rotation, fallback chains, Anthropic/Gemini
format conversion, BYOK resolution) is handled by rg_llm.UnifiedLLMClient.
This module only provides the public API that the rest of chat_service expects.
"""
from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional

from rg_llm import UnifiedLLMClient, LLMRequest

logger = logging.getLogger(__name__)

# Provider name aliases used by the chat UI
_PROVIDER_ALIASES = {
    "chatgpt": "openai", "gpt": "openai",
    "gemini": "google", "claude": "anthropic",
}

# Singleton client instance (reused across requests)
_llm_client = UnifiedLLMClient()


class MultiAIRouter:
    """Route queries to different AI providers with automatic fallback.

    Thin wrapper around UnifiedLLMClient that keeps the same public API.
    """

    def __init__(self):
        self._user_api_keys: Dict[str, str] = {}

    def set_user_api_keys(self, keys: Dict[str, str]):
        """Set user-specific API keys for BYOK (Bring Your Own Key) users."""
        self._user_api_keys = keys or {}

    def get_available_providers(self) -> List[str]:
        """Get list of available providers based on configured keys."""
        providers = []
        for name, env_var in [
            ("openai", "OPENAI_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("google", "GOOGLE_API_KEY"),
            ("groq", "GROQ_API_KEY"),
        ]:
            if self._user_api_keys.get(name) or os.getenv(env_var):
                providers.append(name)
        return providers
    
    async def route_query(
        self,
        message: str,
        context: Optional[List[Dict]] = None,
        preferred_provider: Optional[str] = None,
        images: Optional[List[Dict]] = None,
    ) -> Dict:
        """Route query to an LLM provider with automatic fallback.

        Returns: {'provider': str, 'response': str, 'metadata': dict}
        """
        # Normalize provider alias
        norm = _PROVIDER_ALIASES.get(
            (preferred_provider or "").lower(), (preferred_provider or "").lower()
        ) or None

        # Build messages from context + user message
        messages: List[Dict] = []
        if context:
            for msg in context:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    })

        # Handle images — put them in OpenAI multimodal content format.
        # UnifiedLLMClient auto-converts for Anthropic/Gemini.
        if images:
            content_parts = [{"type": "text", "text": str(message)}]
            for img in images:
                if img.get("data"):
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": img["data"]},
                    })
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": str(message)})

        logger.info(
            f"[MultiAIRouter] preferred={preferred_provider} norm={norm} "
            f"msgs={len(messages)} images={len(images) if images else 0}"
        )

        try:
            request = LLMRequest(
                messages=messages,
                provider=norm,
                temperature=0.7,
                max_tokens=16384,
            )
            response = await _llm_client.complete(request, user_keys=self._user_api_keys or None)

            provider_name = response.provider or norm or "unknown"
            logger.info(f"[MultiAIRouter] Success via {provider_name}/{response.model}")

            return {
                "provider": provider_name,
                "response": response.content or "",
                "metadata": {
                    "model": response.model or "",
                    "usage": response.usage or {},
                    "was_fallback": response.was_fallback,
                    "fallback_chain": response.fallback_chain or [],
                    "preferred_provider": preferred_provider,
                },
            }
        except Exception as e:
            logger.error(f"[MultiAIRouter] All providers failed: {e}")
            return {
                "provider": "devswat",
                "response": (
                    "I apologize, but all AI providers are currently unavailable. "
                    "Please check your API keys or try again later."
                ),
                "metadata": {
                    "error": str(e),
                    "all_providers_failed": True,
                },
            }

    async def route_query_async(
        self,
        message: str,
        context: Optional[List[Dict]] = None,
        preferred_provider: Optional[str] = None,
    ) -> Dict:
        """Alias for route_query (now natively async)."""
        return await self.route_query(message, context, preferred_provider)
