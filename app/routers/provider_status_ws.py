"""
WebSocket Router for Live Provider Status Updates
==================================================

Implements WebSocket support for real-time provider availability,
latency, and status monitoring.
"""
from __future__ import annotations

import json
import logging
import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import time

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from rg_llm.providers import BUILTIN_PROVIDERS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["provider-status"])


class ProviderStatusManager:
    """Manages WebSocket connections for provider status updates."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.provider_cache: Dict[str, Any] = {}
        self.last_check: float = 0
        self.check_interval: float = 5.0  # Check every 5 seconds
        
    async def connect(self, websocket: WebSocket):
        """Register a WebSocket connection."""
        self.active_connections.add(websocket)
        logger.info(f"Provider status WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        logger.info(f"Provider status WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast provider status to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to connection: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def check_provider_status(self) -> Dict[str, Any]:
        """Check live provider status including latency."""
        providers = []
        
        # Platform keys (handle comma-separated keys by taking the first one)
        platform_tokenrouter = os.getenv("TOKENROUTER_API_KEY")
        raw_groq = os.getenv("GROQ_API_KEY") or os.getenv("CHAT_GROQ_API_KEY") or ""
        platform_groq = raw_groq.split(",")[0].strip() if raw_groq else None
        platform_openai = os.getenv("OPENAI_API_KEY") or os.getenv("CHAT_OPENAI_API_KEY")
        platform_gemini = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("CHAT_GOOGLE_API_KEY")
        platform_anthropic = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CHAT_ANTHROPIC_API_KEY")
        
        # Comprehensive model lists per provider
        TOKENROUTER_MODELS = [
            # Text
            "anthropic/claude-opus-4.7", "anthropic/claude-opus-4.6", "anthropic/claude-opus-4.5",
            "anthropic/claude-sonnet-4.6", "anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4",
            "anthropic/claude-haiku-4.5", "openai/gpt-5.5", "openai/gpt-5.4", "openai/gpt-5.2",
            "openai/gpt-5-mini", "openai/gpt-4o-mini", "x-ai/grok-4.3", "x-ai/grok-4.20-beta",
            "x-ai/grok-4.1-fast", "google/gemini-3.1-pro-preview", "google/gemini-3-flash-preview",
            "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "deepseek/deepseek-v3.2",
            "z-ai/glm-5.1", "z-ai/glm-5", "z-ai/glm-5-turbo", "z-ai/glm-4.7", "z-ai/glm-4.6",
            "z-ai/glm-4.6v", "z-ai/glm-4.5-air", "qwen/qwen3.6-plus", "qwen/qwen3.5-plus-02-15",
            "qwen/qwen3.5-flash", "qwen/qwen3.5-397b-a17b", "qwen/qwen3.5-122b-a10b",
            "qwen/qwen3.5-35b-a3b", "qwen/qwen3.5-9b", "moonshotai/kimi-k2.6", "moonshotai/kimi-k2.5",
            "minimax/minimax-m2.7", "minimax/minimax-m2.7-highspeed", "minimax/minimax-m2.5",
            "minimax/minimax-m2.1", "minimax/minimax-m2.1-highspeed", "minimax/minimax-m2-her",
            "xiaomi/mimo-v2.5-pro", "xiaomi/mimo-v2.5", "xiaomi/mimo-v2-pro", "xiaomi/mimo-v2-omni",
            "xiaomi/mimo-v2-flash", "stepfun/step-3.5-flash", "mistralai/devstral-2512",
            "nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            # Coding
            "openai/gpt-5.3-codex", "openai/gpt-5.2-codex", "openai/gpt-5.1-codex-max",
            "openai/gpt-5.1-codex-mini", "qwen/qwen3-coder-next",
            # Image
            "openai/gpt-5.4-image-2", "openai/gpt-5-image", "openai/gpt-5-image-mini",
            "google/gemini-3.1-flash-image-preview", "google/gemini-3-pro-image-preview",
            "google/gemini-2.5-flash-image", "bytedance-seed/seedream-4.5",
            # Video
            "happyhorse-1.0-i2v", "happyhorse-1.0-t2v", "dreamina-seedance-2-0-260128",
            "dreamina-seedance-2-0-fast-260128", "kling-v3", "kling-v2-6", "MiniMax-Hailuo-2.3",
            # Audio
            "openai/gpt-audio", "openai/gpt-audio-mini",
        ]
        # Sourced from rg_llm.providers.BUILTIN_PROVIDERS (single source of
        # truth shared with the actual BYOK call path) instead of a separate
        # hardcoded list that drifts out of sync and can list retired models.
        OPENAI_MODELS = BUILTIN_PROVIDERS["openai"].models
        ANTHROPIC_MODELS = BUILTIN_PROVIDERS["anthropic"].models
        GEMINI_MODELS = BUILTIN_PROVIDERS["google"].models
        GROQ_MODELS = BUILTIN_PROVIDERS["groq"].models

        # Check TokenRouter (Tier 0 — unified router, 72 models)
        tr_status = await self._check_provider_latency("tokenrouter", platform_tokenrouter)
        providers.append({
            "id": "tokenrouter",
            "name": "TokenRouter (72 Models)",
            "available": tr_status["available"],
            "latency": tr_status["latency"],
            "status": tr_status["status"],
            "model": "google/gemini-3-flash-preview",
            "models": TOKENROUTER_MODELS,
            "capabilities": ["chat", "coding", "vision", "tools", "image", "video", "audio"],
            "model_categories": {
                "text": [m for m in TOKENROUTER_MODELS if not any(t in m for t in ["image", "Image", "seedream", "happyhorse", "seedance", "kling", "Hailuo", "audio"])],
                "image": ["openai/gpt-5.4-image-2", "openai/gpt-5-image", "openai/gpt-5-image-mini",
                          "google/gemini-3.1-flash-image-preview", "google/gemini-3-pro-image-preview",
                          "google/gemini-2.5-flash-image", "bytedance-seed/seedream-4.5"],
                "video": ["happyhorse-1.0-i2v", "happyhorse-1.0-t2v", "dreamina-seedance-2-0-260128",
                          "dreamina-seedance-2-0-fast-260128", "kling-v3", "kling-v2-6", "MiniMax-Hailuo-2.3"],
                "audio": ["openai/gpt-audio", "openai/gpt-audio-mini"],
            },
            "supports_smart_routing": True,
        })
        
        # Check Groq
        groq_status = await self._check_provider_latency("groq", platform_groq)
        providers.append({
            "id": "groq",
            "name": "Groq",
            "available": groq_status["available"],
            "latency": groq_status["latency"],
            "status": groq_status["status"],
            "model": BUILTIN_PROVIDERS["groq"].default_model,
            "models": GROQ_MODELS,
            "capabilities": ["chat", "coding"],
        })
        
        # Check OpenAI
        openai_status = await self._check_provider_latency("openai", platform_openai)
        providers.append({
            "id": "chatgpt",
            "name": "ChatGPT",
            "available": openai_status["available"],
            "latency": openai_status["latency"],
            "status": openai_status["status"],
            "model": BUILTIN_PROVIDERS["openai"].default_model,
            "models": OPENAI_MODELS,
            "capabilities": ["chat", "coding", "vision", "image"],
        })
        
        # Check Gemini
        gemini_status = await self._check_provider_latency("gemini", platform_gemini)
        providers.append({
            "id": "gemini",
            "name": "Gemini",
            "available": gemini_status["available"],
            "latency": gemini_status["latency"],
            "status": gemini_status["status"],
            "model": BUILTIN_PROVIDERS["google"].default_model,
            "models": GEMINI_MODELS,
            "capabilities": ["chat", "coding", "vision"],
        })
        
        # Check Anthropic
        anthropic_status = await self._check_provider_latency("anthropic", platform_anthropic)
        providers.append({
            "id": "anthropic",
            "name": "Claude",
            "available": anthropic_status["available"],
            "latency": anthropic_status["latency"],
            "status": anthropic_status["status"],
            "model": BUILTIN_PROVIDERS["anthropic"].default_model,
            "models": ANTHROPIC_MODELS,
            "capabilities": ["chat", "coding", "vision"],
        })
        
        
        # ============================================
        # BYOK-only providers (OpenAI-compatible APIs)
        # These show up when user has added their own key
        # OR when a platform env key exists.
        # ============================================
        BYOK_PROVIDERS = [
            {
                "id": "openrouter", "name": "OpenRouter (100+ models)",
                "env_keys": ["OPENROUTER_API_KEY"],
                "base_url": "https://openrouter.ai/api/v1",
                "test_model": "openai/gpt-4o-mini",
                "default_model": "auto",
                "models": ["auto", "openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet",
                           "google/gemini-2.0-flash-exp", "meta-llama/llama-3.3-70b-instruct",
                           "qwen/qwen-2.5-72b-instruct", "deepseek/deepseek-chat",
                           "mistralai/mistral-large-latest"],
                "capabilities": ["chat", "coding", "vision"],
                "extra_headers": {"HTTP-Referer": "https://resonant.dev-swat.com"},
            },
            {
                "id": "mistral", "name": "Mistral AI",
                "env_keys": ["MISTRAL_API_KEY"],
                "base_url": "https://api.mistral.ai/v1",
                "test_model": "mistral-small-latest",
                "default_model": "mistral-large-latest",
                "models": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest",
                           "open-mixtral-8x22b", "codestral-latest"],
                "capabilities": ["chat", "coding"],
            },
            {
                "id": "deepseek", "name": "DeepSeek",
                "env_keys": ["DEEPSEEK_API_KEY"],
                "base_url": "https://api.deepseek.com/v1",
                "test_model": "deepseek-chat",
                "default_model": "deepseek-chat",
                "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
                "capabilities": ["chat", "coding", "reasoning"],
            },
            {
                "id": "together", "name": "Together AI",
                "env_keys": ["TOGETHER_API_KEY"],
                "base_url": "https://api.together.xyz/v1",
                "test_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "mistralai/Mixtral-8x22B-Instruct-v0.1",
                           "Qwen/Qwen2.5-72B-Instruct-Turbo"],
                "capabilities": ["chat", "coding"],
            },
            {
                "id": "perplexity", "name": "Perplexity AI",
                "env_keys": ["PERPLEXITY_API_KEY"],
                "base_url": "https://api.perplexity.ai",
                "test_model": "llama-3.1-sonar-small-128k-online",
                "default_model": "llama-3.1-sonar-large-128k-online",
                "models": ["llama-3.1-sonar-large-128k-online", "llama-3.1-sonar-small-128k-online",
                           "llama-3.1-sonar-huge-128k-online"],
                "capabilities": ["chat", "reasoning"],
            },
            {
                "id": "fireworks", "name": "Fireworks AI (Fast)",
                "env_keys": ["FIREWORKS_API_KEY"],
                "base_url": "https://api.fireworks.ai/inference/v1",
                "test_model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
                "default_model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
                "models": ["accounts/fireworks/models/llama-v3p1-70b-instruct",
                           "accounts/fireworks/models/mixtral-8x7b-instruct"],
                "capabilities": ["chat", "coding"],
            },
            {
                "id": "cohere", "name": "Cohere",
                "env_keys": ["COHERE_API_KEY"],
                "base_url": "https://api.cohere.ai/v1",
                "test_model": "command-r",
                "default_model": "command-r-plus",
                "models": ["command-r-plus", "command-r", "command"],
                "capabilities": ["chat"],
            },
            {
                "id": "grok", "name": "Grok (xAI)",
                "env_keys": ["XAI_API_KEY", "GROK_API_KEY"],
                "base_url": "https://api.x.ai/v1",
                "test_model": "grok-2",
                "default_model": "grok-2",
                "models": ["grok-2", "grok-2-mini", "grok-beta"],
                "capabilities": ["chat", "coding", "reasoning"],
            },
            {
                "id": "huggingface", "name": "Hugging Face",
                "env_keys": ["HUGGINGFACE_API_KEY", "HF_API_KEY"],
                "base_url": "https://api-inference.huggingface.co/models",
                "test_model": None,  # HF uses different API format, skip latency test
                "default_model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
                "models": ["meta-llama/Meta-Llama-3.1-70B-Instruct",
                           "mistralai/Mixtral-8x7B-Instruct-v0.1"],
                "capabilities": ["chat", "coding"],
            },
        ]
        
        for bp in BYOK_PROVIDERS:
            # Check platform env key
            platform_key = None
            for ek in bp["env_keys"]:
                platform_key = os.getenv(ek)
                if platform_key:
                    break
            
            if platform_key and bp.get("test_model"):
                status = await self._check_byok_provider_latency(
                    bp["base_url"], platform_key, bp["test_model"],
                    extra_headers=bp.get("extra_headers", {}),
                )
            elif platform_key:
                # Has key but can't test (e.g. HuggingFace) — assume available
                status = {"available": True, "latency": None, "status": "key_configured"}
            else:
                # No platform key — will be marked available per-user if they have BYOK
                status = {"available": False, "latency": None, "status": "byok_only"}
            
            providers.append({
                "id": bp["id"],
                "name": bp["name"],
                "available": status["available"],
                "latency": status.get("latency"),
                "status": status["status"],
                "model": bp["default_model"],
                "models": bp["models"],
                "capabilities": bp.get("capabilities", ["chat"]),
                "byok_only": not bool(platform_key),
            })
        
        # Check CodeLlama (Ollama) - tunneled from Mac - Coding
        codellama_status = await self._check_ollama_status("codellama:13b")
        providers.append({
            "id": "codellama",
            "name": "CodeLlama",
            "available": codellama_status["available"],
            "latency": codellama_status["latency"],
            "status": codellama_status["status"],
            "model": "codellama:13b",
            "capabilities": ["coding"],
        })
        
        return {
            "type": "provider_status",
            "providers": providers,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    async def _check_provider_latency(self, provider: str, api_key: Optional[str]) -> Dict[str, Any]:
        """Check if provider is available by testing actual chat completion (not just model listing)."""
        if not api_key:
            return {
                "available": False,
                "latency": None,
                "status": "offline",
                "error": "No API key configured",
            }
        
        try:
            start_time = time.time()
            
            # Test actual chat completion to detect quota/credit issues
            if provider == "groq":
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                        timeout=5.0
                    )
                    latency = int((time.time() - start_time) * 1000)
                    if response.status_code == 200:
                        return {"available": True, "latency": latency, "status": "online"}
                    elif response.status_code == 429:
                        return {"available": False, "latency": latency, "status": "quota_exceeded", "error": "Rate limit or quota exceeded"}
                    else:
                        return {"available": False, "latency": latency, "status": "error", "error": response.text[:100]}
            
            elif provider == "openai":
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                        timeout=5.0
                    )
                    latency = int((time.time() - start_time) * 1000)
                    if response.status_code == 200:
                        return {"available": True, "latency": latency, "status": "online"}
                    elif response.status_code == 429:
                        return {"available": False, "latency": latency, "status": "quota_exceeded", "error": "Quota exceeded"}
                    else:
                        return {"available": False, "latency": latency, "status": "error", "error": response.text[:100]}
            
            elif provider == "gemini":
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
                        headers={"Content-Type": "application/json"},
                        json={"contents": [{"parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 1}},
                        timeout=5.0
                    )
                    latency = int((time.time() - start_time) * 1000)
                    if response.status_code == 200:
                        return {"available": True, "latency": latency, "status": "online"}
                    elif response.status_code == 429:
                        return {"available": False, "latency": latency, "status": "quota_exceeded", "error": "Quota exceeded"}
                    else:
                        return {"available": False, "latency": latency, "status": "error", "error": response.text[:100]}
            
            elif provider == "anthropic":
                base_url = os.getenv("LLM_ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
                base_url = base_url.rstrip("/") + "/v1/messages"
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        base_url,
                        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                        json={"model": "claude-3-haiku-20240307", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                        timeout=5.0
                    )
                    latency = int((time.time() - start_time) * 1000)
                    if response.status_code == 200:
                        return {"available": True, "latency": latency, "status": "online"}
                    elif response.status_code == 429 or response.status_code == 400:
                        # Anthropic returns 400 for credit balance issues
                        return {"available": False, "latency": latency, "status": "quota_exceeded", "error": "Credit balance too low or quota exceeded"}
                    else:
                        return {"available": False, "latency": latency, "status": "error", "error": response.text[:100]}
            
            elif provider == "tokenrouter":
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.tokenrouter.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": "openai/gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
                        timeout=8.0
                    )
                    latency = int((time.time() - start_time) * 1000)
                    if response.status_code == 200:
                        return {"available": True, "latency": latency, "status": "online"}
                    elif response.status_code == 429:
                        return {"available": False, "latency": latency, "status": "quota_exceeded", "error": "Rate limit"}
                    else:
                        return {"available": False, "latency": latency, "status": "error", "error": response.text[:100]}
            
            else:
                return {"available": False, "latency": None, "status": "offline", "error": "Unknown provider"}
        
        except Exception as e:
            logger.warning(f"Provider {provider} check failed: {e}")
            return {"available": False, "latency": None, "status": "offline", "error": str(e)[:100]}
    
    async def _check_byok_provider_latency(
        self, base_url: str, api_key: str, test_model: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Generic health check for any OpenAI-compatible BYOK provider."""
        try:
            start_time = time.time()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                **(extra_headers or {}),
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={"model": test_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                    timeout=8.0,
                )
                latency = int((time.time() - start_time) * 1000)
                if response.status_code == 200:
                    return {"available": True, "latency": latency, "status": "online"}
                elif response.status_code == 429:
                    return {"available": False, "latency": latency, "status": "quota_exceeded"}
                else:
                    return {"available": False, "latency": latency, "status": "error", "error": response.text[:100]}
        except Exception as e:
            logger.warning(f"BYOK provider check failed ({base_url}): {e}")
            return {"available": False, "latency": None, "status": "offline", "error": str(e)[:100]}
    
    async def _check_ollama_status(self, model: str = "llama3.1:8b", user_id: str = "") -> Dict[str, Any]:
        """Check if local Ollama LLM is available via tunnel or direct connection."""
        import os
        
        # First: check if any user has an active tunnel via gateway
        gateway_url = os.getenv("GATEWAY_URL", "http://gateway:8000")
        if user_id:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{gateway_url}/api/v1/local-llm/tunnel/status",
                        headers={"x-user-id": user_id},
                        timeout=3.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("connected"):
                            return {
                                "available": True,
                                "latency": None,
                                "status": "tunnel_active",
                                "models": data.get("models", []),
                                "endpoint": data.get("endpoint_url", ""),
                            }
            except Exception:
                pass
        
        # Fallback: try direct localhost (legacy SSH tunnel or co-located Ollama)
        ollama_hosts = [
            "http://172.19.0.1:11435",
            "http://172.19.0.1:11434",
            "http://host.docker.internal:11434",
            "http://localhost:11434",
        ]
        
        for ollama_url in ollama_hosts:
            try:
                start_time = time.time()
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{ollama_url}/api/generate",
                        json={"model": model, "prompt": "hi", "stream": False},
                        timeout=5.0
                    )
                    latency = int((time.time() - start_time) * 1000)
                    if response.status_code == 200:
                        return {"available": True, "latency": latency, "status": "online"}
                    elif "model requires more system memory" in response.text:
                        return {"available": False, "latency": latency, "status": "insufficient_memory", "error": "Not enough RAM"}
                    else:
                        continue
            except Exception:
                continue
        
        return {"available": False, "latency": None, "status": "offline", "error": "No local LLM tunnel or direct connection available"}
    
    async def monitor_loop(self):
        """Background task to monitor provider status and broadcast updates."""
        while True:
            try:
                if self.active_connections:
                    current_time = time.time()
                    if current_time - self.last_check >= self.check_interval:
                        status = await self.check_provider_status()
                        
                        # Check if status changed
                        if status != self.provider_cache:
                            self.provider_cache = status
                            await self.broadcast(status)
                            logger.info(f"Broadcasted provider status to {len(self.active_connections)} clients")
                        
                        self.last_check = current_time
                
                await asyncio.sleep(1.0)
            
            except Exception as e:
                logger.error(f"Monitor loop error: {e}", exc_info=True)
                await asyncio.sleep(5.0)


# Global status manager
status_manager = ProviderStatusManager()


@router.websocket("/ws/provider-status")
async def websocket_provider_status(websocket: WebSocket):
    """
    WebSocket endpoint for live provider status updates.
    
    Protocol:
    1. Client connects
    2. Server immediately sends current provider status
    3. Server sends updates every 5 seconds if status changes
    4. Client can send {"type": "ping"} to check connection
    5. Server responds with {"type": "pong"}
    """
    try:
        # Accept connection
        await websocket.accept()
        
        # Register connection
        await status_manager.connect(websocket)
        
        # Send initial status immediately
        initial_status = await status_manager.check_provider_status()
        await websocket.send_json(initial_status)
        
        # Message loop
        while True:
            try:
                # Wait for messages with timeout
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0
                )
                
                msg_type = data.get("type")
                
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg_type == "refresh":
                    # Force refresh provider status
                    status = await status_manager.check_provider_status()
                    await websocket.send_json(status)
            
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})
    
    except WebSocketDisconnect:
        logger.info("Provider status WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"Provider status WebSocket error: {e}", exc_info=True)
    finally:
        status_manager.disconnect(websocket)


@router.get("/provider-status/health")
async def provider_status_health():
    """Health check for provider status service."""
    return {
        "status": "ok",
        "active_connections": len(status_manager.active_connections),
        "last_check": status_manager.last_check,
    }


# Monitor loop is started via lifespan in main.py
# Do NOT call asyncio.create_task at module load time
