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
        raw_groq = os.getenv("GROQ_API_KEY") or os.getenv("CHAT_GROQ_API_KEY") or ""
        platform_groq = raw_groq.split(",")[0].strip() if raw_groq else None
        platform_openai = os.getenv("OPENAI_API_KEY") or os.getenv("CHAT_OPENAI_API_KEY")
        platform_gemini = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("CHAT_GOOGLE_API_KEY")
        platform_anthropic = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CHAT_ANTHROPIC_API_KEY")
        
        # Comprehensive model lists per provider
        OPENAI_MODELS = [
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
            "o1", "o1-mini", "o1-pro",
            "gpt-3.5-turbo",
            "dall-e-3", "dall-e-2",
        ]
        ANTHROPIC_MODELS = [
            "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229", "claude-3-haiku-20240307",
        ]
        GEMINI_MODELS = [
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash",
            "gemini-pro-latest", "gemini-flash-latest",
        ]
        GROQ_MODELS = [
            "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768", "gemma2-9b-it",
        ]

        # Check Groq
        groq_status = await self._check_provider_latency("groq", platform_groq)
        providers.append({
            "id": "groq",
            "name": "Groq",
            "available": groq_status["available"],
            "latency": groq_status["latency"],
            "status": groq_status["status"],
            "model": "llama-3.3-70b-versatile",
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
            "model": "gpt-4o",
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
            "model": "gemini-2.0-flash",
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
            "model": "claude-sonnet-4-20250514",
            "models": ANTHROPIC_MODELS,
            "capabilities": ["chat", "coding", "vision"],
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
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",
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
            
            else:
                return {"available": False, "latency": None, "status": "offline", "error": "Unknown provider"}
        
        except Exception as e:
            logger.warning(f"Provider {provider} check failed: {e}")
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
