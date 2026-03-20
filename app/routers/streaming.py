"""
Real-time Streaming Router for Resonant Chat
=============================================

Implements Server-Sent Events (SSE) streaming for chat responses.
This allows users to see responses as they are generated.
"""
from __future__ import annotations

import json
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..models import ResonantChat, ResonantChatMessage
from ..domain.provider import route_query_stream
from ..services.resonance_hashing import ResonanceHasher
from ..services.memory_merge import merge_and_rank_memories
from ..services.magnetic_pull import magnetic_pull_system
from ..services.intent_engine import intent_engine
from ..services.emotional_normalizer import emotional_normalizer
from ..services.evidence_graph import evidence_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resonant-chat", tags=["resonant-chat-streaming"])


class StreamMessageRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    preferred_provider: Optional[str] = None
    agent_hash: Optional[str] = None
    teamId: Optional[str] = None


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


async def _extract_memories_for_stream(
    user_id: str,
    org_id: str,
    message: str,
    agent_hash: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract memories for streaming context using FULL Hash Sphere extraction."""
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
                    "agent_hash": agent_hash,
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


@router.post("/message/stream")
async def stream_message(
    request_body: StreamMessageRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Stream a chat response using Server-Sent Events (SSE).
    
    This endpoint allows real-time streaming of AI responses,
    so users can see the response as it's being generated.
    
    Events:
    - start: Initial metadata (chat_id, message_id)
    - chunk: Response text chunk
    - metadata: Hash, resonance score, xyz coordinates
    - done: Completion signal with final data
    - error: Error information
    """
    user_id = request.headers.get("x-user-id")
    org_id = request.headers.get("x-org-id") or user_id
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    async def generate_stream():
        try:
            # Get or create chat
            chat_id = request_body.chat_id
            chat = None
            
            if chat_id:
                try:
                    result = await session.execute(
                        select(ResonantChat).where(ResonantChat.id == UUID(chat_id))
                    )
                    chat = result.scalar_one_or_none()
                except ValueError:
                    pass
            
            if not chat:
                chat = ResonantChat(
                    user_id=UUID(user_id),
                    org_id=UUID(org_id),
                    title=request_body.message[:50] + "..." if len(request_body.message) > 50 else request_body.message,
                    status="active",
                    agent_hash=request_body.agent_hash,
                )
                session.add(chat)
                await session.commit()
                await session.refresh(chat)
                chat_id = str(chat.id)
            else:
                chat_id = str(chat.id)
            
            # Hash user message
            try:
                hasher = ResonanceHasher()
                user_hash = hasher.hash_text(request_body.message)
                user_xyz = hasher.hash_to_coords(user_hash)
            except:
                user_hash = _simple_hash(request_body.message)
                user_xyz = _hash_to_xyz_simple(user_hash)
            
            # Store user message
            user_message = ResonantChatMessage(
                chat_id=UUID(chat_id),
                role="user",
                content=request_body.message,
                hash=user_hash,
                resonance_score=0.5,
                xyz_x=user_xyz[0],
                xyz_y=user_xyz[1],
                xyz_z=user_xyz[2],
                agent_hash=request_body.agent_hash,
            )
            session.add(user_message)
            await session.commit()
            await session.refresh(user_message)
            
            # Send start event
            start_data = {
                "event": "start",
                "chat_id": chat_id,
                "user_message_id": str(user_message.id),
                "user_hash": user_hash,
            }
            yield f"data: {json.dumps(start_data)}\n\n"
            
            # --- Step: Hashing ---
            yield f'data: {json.dumps({"event": "step", "step": "hashing", "message": "Resonance hashing user message..."})}\n\n'
            
            # --- Step: Memory extraction ---
            yield f'data: {json.dumps({"event": "step", "step": "memory_search", "message": "Searching Hash Sphere memories..."})}\n\n'
            
            # Extract memories
            memories = await _extract_memories_for_stream(
                user_id=user_id,
                org_id=org_id,
                message=request_body.message,
                agent_hash=request_body.agent_hash,
            )
            
            mem_count = len(memories)
            if mem_count > 0:
                yield f'data: {json.dumps({"event": "step", "step": "memory_found", "message": f"Found {mem_count} relevant memories", "count": mem_count})}\n\n'
            else:
                yield f'data: {json.dumps({"event": "step", "step": "memory_found", "message": "No relevant memories found", "count": 0})}\n\n'
            
            # Get recent messages for context
            result = await session.execute(
                select(ResonantChatMessage)
                .where(ResonantChatMessage.chat_id == UUID(chat_id))
                .order_by(ResonantChatMessage.created_at.desc())
                .limit(10)
            )
            recent_messages = list(reversed(result.scalars().all()))
            
            # Build context
            context_messages = []
            context_messages.append({
                "role": "system",
                "content": "You are Resonant AI, an intelligent assistant with persistent memory. Be helpful, accurate, and conversational."
            })
            
            for msg in recent_messages[:-1]:
                context_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            if memories:
                memory_context = "RELEVANT MEMORIES:\n"
                for i, mem in enumerate(memories[:3], 1):
                    content = mem.get("content", "") or mem.get("anchor_text", "")
                    if content:
                        memory_context += f"{i}. {content[:150]}\n"
                context_messages.append({
                    "role": "system",
                    "content": memory_context
                })
            
            # --- Step: Building context ---
            yield f'data: {json.dumps({"event": "step", "step": "context", "message": "Building conversation context...", "history_count": len(recent_messages)})}\n\n'
            
            # --- Step: Routing to provider ---
            _prov_label = request_body.preferred_provider or "auto"
            yield f'data: {json.dumps({"event": "step", "step": "routing", "message": f"Routing to AI provider: {_prov_label}..."})}\n\n'
            
            # Stream response from LLM
            full_response = ""
            provider = "unknown"
            
            try:
                async for chunk_data in route_query_stream(
                    message=request_body.message,
                    context=context_messages,
                    preferred_provider=request_body.preferred_provider,
                ):
                    if chunk_data.get("type") == "chunk":
                        chunk_text = chunk_data.get("content", "")
                        full_response += chunk_text
                        
                        chunk_event = {
                            "event": "chunk",
                            "content": chunk_text,
                        }
                        yield f"data: {json.dumps(chunk_event)}\n\n"
                    
                    elif chunk_data.get("type") == "provider":
                        provider = chunk_data.get("provider", "unknown")
                    
                    elif chunk_data.get("type") == "error":
                        error_event = {
                            "event": "error",
                            "error": chunk_data.get("error", "Unknown error"),
                        }
                        yield f"data: {json.dumps(error_event)}\n\n"
                        return
                    
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                # Fallback to non-streaming
                from ..domain.provider import route_query
                ai_response = await route_query(
                    message=request_body.message,
                    context=context_messages,
                    preferred_provider=request_body.preferred_provider,
                )
                full_response = ai_response.get("response", "")
                provider = ai_response.get("provider", "unknown")
                
                # Send full response as single chunk
                chunk_event = {
                    "event": "chunk",
                    "content": full_response,
                }
                yield f"data: {json.dumps(chunk_event)}\n\n"
            
            # --- Step: Generation complete ---
            yield f'data: {json.dumps({"event": "step", "step": "generating_done", "message": "Response generated", "length": len(full_response)})}\n\n'
            
            if not full_response:
                full_response = "I apologize, but I couldn't generate a response. Please try again."
            
            # Hash and store assistant message
            try:
                assistant_hash = hasher.hash_text(full_response)
                assistant_xyz = hasher.hash_to_coords(assistant_hash)
            except:
                assistant_hash = _simple_hash(full_response)
                assistant_xyz = _hash_to_xyz_simple(assistant_hash)
            
            # Calculate resonance score
            resonance_score = 0.5
            if memories:
                resonance_score = min(0.9, 0.5 + len(memories) * 0.05)
            
            assistant_message = ResonantChatMessage(
                chat_id=UUID(chat_id),
                role="assistant",
                content=full_response,
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
            
            # Send metadata event
            metadata_event = {
                "event": "metadata",
                "hash": assistant_hash,
                "resonance_score": resonance_score,
                "xyz": list(assistant_xyz),
                "provider": provider,
            }
            yield f"data: {json.dumps(metadata_event)}\n\n"
            
            # --- Step: Post-processing ---
            yield f'data: {json.dumps({"event": "step", "step": "post_processing", "message": "Hashing response & building evidence graph..."})}\n\n'
            
            # Build evidence graph
            intents = intent_engine.extract(request_body.message)
            emotion = emotional_normalizer.detect(request_body.message)
            
            evidence_data = evidence_graph.build_graph(
                user_hash=user_hash,
                assistant_hash=assistant_hash,
                memories=memories,
                provider=provider,
                intents=intents,
                emotion=emotion,
            )
            
            # Send done event
            done_data = {
                "event": "done",
                "message_id": str(assistant_message.id),
                "chat_id": chat_id,
                "total_length": len(full_response),
                "anchors": [
                    (mem.get("anchor_text", "") or mem.get("content", ""))[:50]
                    for mem in memories[:5]
                    if mem.get("anchor_text") or mem.get("content")
                ],
                "evidence_graph": evidence_data,
            }
            yield f"data: {json.dumps(done_data)}\n\n"
            
            # --- Step: Memory ingestion ---
            yield f'data: {json.dumps({"event": "step", "step": "memory_ingest", "message": "Ingesting to Hash Sphere memory..."})}\n\n'
            
            # Background: Ingest to memory service
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        "http://memory_service:8000/memory/ingest",
                        json={
                            "chat_id": chat_id,
                            "user_id": user_id,
                            "source": "chat",
                            "content": request_body.message,
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
                            "content": full_response,
                            "role": "assistant",
                        },
                        timeout=2.0,
                    )
            except:
                pass
            
        except Exception as e:
            logger.error(f"Stream generation error: {e}", exc_info=True)
            error_event = {
                "event": "error",
                "error": str(e),
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
