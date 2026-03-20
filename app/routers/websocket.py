"""
WebSocket Router for Resonant Chat
===================================

Implements WebSocket support for real-time bidirectional chat communication.
"""
from __future__ import annotations

import json
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session, async_session_factory
from ..models import ResonantChat, ResonantChatMessage
from ..domain.provider import route_query
from ..services.resonance_hashing import ResonanceHasher
from ..services.memory_merge import merge_and_rank_memories
from ..services.magnetic_pull import magnetic_pull_system

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manages WebSocket connections for chat rooms."""
    
    def __init__(self):
        # chat_id -> set of websocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # websocket -> user_id mapping
        self.connection_users: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket, chat_id: str, user_id: str):
        """Register a WebSocket connection (assumes already accepted)."""
        if chat_id not in self.active_connections:
            self.active_connections[chat_id] = set()
        
        self.active_connections[chat_id].add(websocket)
        self.connection_users[websocket] = user_id
        
        logger.info(f"WebSocket connected: user={user_id[:8]}..., chat={chat_id[:8]}...")
    
    def disconnect(self, websocket: WebSocket, chat_id: str):
        """Remove a WebSocket connection."""
        if chat_id in self.active_connections:
            self.active_connections[chat_id].discard(websocket)
            if not self.active_connections[chat_id]:
                del self.active_connections[chat_id]
        
        if websocket in self.connection_users:
            del self.connection_users[websocket]
        
        logger.info(f"WebSocket disconnected: chat={chat_id[:8]}...")
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send a message to a specific connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
    
    async def broadcast(self, chat_id: str, message: dict, exclude: Optional[WebSocket] = None):
        """Broadcast a message to all connections in a chat room."""
        if chat_id not in self.active_connections:
            return
        
        for connection in self.active_connections[chat_id]:
            if connection != exclude:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to broadcast: {e}")
    
    def get_user_id(self, websocket: WebSocket) -> Optional[str]:
        """Get user ID for a connection."""
        return self.connection_users.get(websocket)
    
    def get_connection_count(self, chat_id: str) -> int:
        """Get number of active connections for a chat."""
        return len(self.active_connections.get(chat_id, set()))


# Global connection manager
manager = ConnectionManager()


def _simple_hash(text: str) -> str:
    """Simple hash for fallback."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def _hash_to_xyz_simple(hash_str: str) -> tuple:
    """Convert hash to XYZ coordinates."""
    try:
        x = int(hash_str[:8], 16) / 0xFFFFFFFF
        y = int(hash_str[8:16], 16) / 0xFFFFFFFF
        z = int(hash_str[16:24], 16) / 0xFFFFFFFF
        return (x, y, z)
    except:
        return (0.5, 0.5, 0.5)


async def _extract_memories_ws(
    user_id: str,
    org_id: str,
    message: str,
) -> List[Dict[str, Any]]:
    """Extract memories for WebSocket context using FULL Hash Sphere extraction."""
    memories = []
    
    try:
        async with httpx.AsyncClient() as client:
            # Use FULL Hash Sphere extraction endpoint
            response = await client.post(
                "http://memory_service:8000/memory/hash-sphere/extract",
                json={
                    "query": message,
                    "user_id": user_id,
                    "org_id": org_id,
                    "limit": 5,
                    # Full 9-Layer extraction
                    "use_anchors": True,
                    "use_proximity": True,
                    "use_resonance": True,
                    "use_clusters": True,
                    "use_rag_fallback": True,  # RAG as LAST RESORT
                    "apply_magnetic_pull": True,
                },
                timeout=5.0,
            )
            if response.status_code == 200:
                result = response.json()
                for mem in result.get("memories", []):
                    memories.append({
                        "content": mem.get("content", ""),
                        "hash": mem.get("hash"),
                        "xyz": mem.get("xyz"),
                        "hybrid_score": mem.get("hybrid_score", 0.0),
                        "resonance_score": mem.get("resonance_score", 0.0),
                        "magnetic_score": mem.get("magnetic_score", 0.0),
                    })
    except:
        pass
    
    return memories


@router.websocket("/ws/chat/{chat_id}")
async def websocket_chat(
    websocket: WebSocket,
    chat_id: str,
):
    """
    WebSocket endpoint for real-time chat.
    
    Protocol:
    1. Client connects with chat_id in URL
    2. Client sends authentication message: {"type": "auth", "token": "jwt_token"}
    3. Server responds: {"type": "auth_success", "user_id": "..."}
    4. Client sends messages: {"type": "message", "content": "...", "provider": "..."}
    5. Server responds with:
       - {"type": "typing"} - AI is generating
       - {"type": "chunk", "content": "..."} - Streaming chunk
       - {"type": "message", "message": {...}} - Complete message
       - {"type": "error", "error": "..."} - Error occurred
    6. Client can send: {"type": "ping"} and receive {"type": "pong"}
    """
    user_id = None
    org_id = None
    
    try:
        # Accept WebSocket connection
        # Note: When proxied through gateway, connection may already be accepted
        try:
            await websocket.accept()
        except RuntimeError:
            # Already accepted by proxy - this is fine
            pass
        
        # First message must be auth
        auth_data = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=30.0
        )
        
        if auth_data.get("type") != "auth":
            await websocket.send_json({
                "type": "error",
                "error": "First message must be authentication"
            })
            await websocket.close(code=4001)
            return
        
        # Validate token (simplified - in production, verify JWT)
        token = auth_data.get("token")
        if not token:
            # Allow user_id directly for development
            user_id = auth_data.get("user_id")
            org_id = auth_data.get("org_id") or user_id
        else:
            # In production, decode JWT and extract user_id
            # For now, accept token as user_id
            user_id = token[:36] if len(token) >= 36 else token
            org_id = user_id
        
        if not user_id:
            await websocket.send_json({
                "type": "error",
                "error": "Invalid authentication"
            })
            await websocket.close(code=4002)
            return
        
        # Register connection
        await manager.connect(websocket, chat_id, user_id)
        
        # Send auth success
        await websocket.send_json({
            "type": "auth_success",
            "user_id": user_id,
            "chat_id": chat_id,
            "connections": manager.get_connection_count(chat_id),
        })
        
        # Create database session
        async with async_session_factory() as session:
            # Verify chat exists and user has access
            try:
                result = await session.execute(
                    select(ResonantChat).where(ResonantChat.id == UUID(chat_id))
                )
                chat = result.scalar_one_or_none()
                
                if not chat:
                    # Create new chat
                    chat = ResonantChat(
                        user_id=UUID(user_id),
                        org_id=UUID(org_id),
                        title="WebSocket Chat",
                        status="active",
                    )
                    session.add(chat)
                    await session.commit()
                    await session.refresh(chat)
                
            except ValueError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid chat ID format"
                })
                await websocket.close(code=4003)
                return
            
            # Main message loop
            while True:
                try:
                    data = await websocket.receive_json()
                    msg_type = data.get("type")
                    
                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue
                    
                    if msg_type == "message":
                        content = data.get("content", "").strip()
                        if not content:
                            await websocket.send_json({
                                "type": "error",
                                "error": "Message content required"
                            })
                            continue
                        
                        preferred_provider = data.get("provider")
                        
                        # Hash user message
                        try:
                            hasher = ResonanceHasher()
                            user_hash = hasher.hash_text(content)
                            user_xyz = hasher.hash_to_coords(user_hash)
                        except:
                            user_hash = _simple_hash(content)
                            user_xyz = _hash_to_xyz_simple(user_hash)
                        
                        # Store user message
                        user_message = ResonantChatMessage(
                            chat_id=UUID(chat_id),
                            role="user",
                            content=content,
                            hash=user_hash,
                            resonance_score=0.5,
                            xyz_x=user_xyz[0],
                            xyz_y=user_xyz[1],
                            xyz_z=user_xyz[2],
                        )
                        session.add(user_message)
                        await session.commit()
                        await session.refresh(user_message)
                        
                        # Broadcast user message to other connections
                        await manager.broadcast(chat_id, {
                            "type": "user_message",
                            "message": {
                                "id": str(user_message.id),
                                "role": "user",
                                "content": content,
                                "timestamp": datetime.utcnow().isoformat(),
                                "hash": user_hash,
                            }
                        }, exclude=websocket)
                        
                        # Send typing indicator
                        await websocket.send_json({"type": "typing"})
                        
                        # Extract memories
                        memories = await _extract_memories_ws(
                            user_id=user_id,
                            org_id=org_id,
                            message=content,
                        )
                        
                        # Get recent messages for context
                        result = await session.execute(
                            select(ResonantChatMessage)
                            .where(ResonantChatMessage.chat_id == UUID(chat_id))
                            .order_by(ResonantChatMessage.created_at.desc())
                            .limit(10)
                        )
                        recent_messages = list(reversed(result.scalars().all()))
                        
                        # Build context
                        context_messages = [{
                            "role": "system",
                            "content": "You are Resonant AI, an intelligent assistant. Be helpful and conversational."
                        }]
                        
                        for msg in recent_messages[:-1]:
                            context_messages.append({
                                "role": msg.role,
                                "content": msg.content
                            })
                        
                        if memories:
                            memory_context = "RELEVANT MEMORIES:\n"
                            for i, mem in enumerate(memories[:3], 1):
                                mem_content = mem.get("content", "") or mem.get("anchor_text", "")
                                if mem_content:
                                    memory_context += f"{i}. {mem_content[:150]}\n"
                            context_messages.append({
                                "role": "system",
                                "content": memory_context
                            })
                        
                        # Get AI response
                        try:
                            ai_response = await route_query(
                                message=content,
                                context=context_messages,
                                preferred_provider=preferred_provider,
                            )
                            response_text = ai_response.get("response", "")
                            provider = ai_response.get("provider", "unknown")
                        except Exception as e:
                            logger.error(f"AI response error: {e}")
                            response_text = "I apologize, but I couldn't generate a response. Please try again."
                            provider = "error"
                        
                        # Hash assistant response
                        try:
                            assistant_hash = hasher.hash_text(response_text)
                            assistant_xyz = hasher.hash_to_coords(assistant_hash)
                        except:
                            assistant_hash = _simple_hash(response_text)
                            assistant_xyz = _hash_to_xyz_simple(assistant_hash)
                        
                        # Calculate resonance
                        resonance_score = 0.5
                        if memories:
                            resonance_score = min(0.9, 0.5 + len(memories) * 0.05)
                        
                        # Store assistant message
                        assistant_message = ResonantChatMessage(
                            chat_id=UUID(chat_id),
                            role="assistant",
                            content=response_text,
                            ai_provider=provider,
                            hash=assistant_hash,
                            resonance_score=resonance_score,
                            xyz_x=assistant_xyz[0],
                            xyz_y=assistant_xyz[1],
                            xyz_z=assistant_xyz[2],
                        )
                        session.add(assistant_message)
                        await session.commit()
                        await session.refresh(assistant_message)
                        
                        # Send complete message
                        message_data = {
                            "type": "message",
                            "message": {
                                "id": str(assistant_message.id),
                                "role": "assistant",
                                "content": response_text,
                                "timestamp": datetime.utcnow().isoformat(),
                                "provider": provider,
                                "hash": assistant_hash,
                                "resonance_score": resonance_score,
                                "xyz": list(assistant_xyz),
                            },
                            "anchors": [
                                (mem.get("anchor_text", "") or mem.get("content", ""))[:50]
                                for mem in memories[:5]
                                if mem.get("anchor_text") or mem.get("content")
                            ],
                        }
                        
                        # Send to requesting client
                        await websocket.send_json(message_data)
                        
                        # Broadcast to other connections
                        await manager.broadcast(chat_id, message_data, exclude=websocket)
                        
                        # Background: Ingest to memory
                        try:
                            async with httpx.AsyncClient() as client:
                                await client.post(
                                    "http://memory_service:8000/memory/ingest",
                                    json={
                                        "chat_id": chat_id,
                                        "user_id": user_id,
                                        "source": "chat",
                                        "content": content,
                                        "role": "user",
                                    },
                                    timeout=2.0,
                                )
                                await client.post(
                                    "http://memory_service:8000/memory/ingest",
                                    json={
                                        "chat_id": chat_id,
                                        "user_id": user_id,
                                        "source": "chat",
                                        "content": response_text,
                                        "role": "assistant",
                                    },
                                    timeout=2.0,
                                )
                        except:
                            pass
                    
                    elif msg_type == "typing":
                        # Broadcast typing indicator to other users
                        await manager.broadcast(chat_id, {
                            "type": "user_typing",
                            "user_id": user_id,
                        }, exclude=websocket)
                    
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "error": f"Unknown message type: {msg_type}"
                        })
                
                except asyncio.TimeoutError:
                    # Send keepalive
                    await websocket.send_json({"type": "keepalive"})
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if user_id:
            manager.disconnect(websocket, chat_id)
            # Notify others of disconnect
            await manager.broadcast(chat_id, {
                "type": "user_disconnected",
                "user_id": user_id,
                "connections": manager.get_connection_count(chat_id),
            })


@router.get("/ws/status/{chat_id}")
async def websocket_status(chat_id: str):
    """Get WebSocket connection status for a chat."""
    return {
        "chat_id": chat_id,
        "connections": manager.get_connection_count(chat_id),
        "active": chat_id in manager.active_connections,
    }
