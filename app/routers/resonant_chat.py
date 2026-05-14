"""
Resonant Chat Router - Full Pipeline
Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/routers/resonant_chat.py

This router implements the complete Resonant Chat flow:
1. Authentication & Context
2. Resonance Hashing
3. Memory Extraction (RAG + Hash Sphere)
4. Context Building
5. LLM Provider Routing
6. Response Storage
7. Background Tasks
8. Response Enhancement
"""
from __future__ import annotations

import logging
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
import asyncio
import re
from zoneinfo import ZoneInfo

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.ext.asyncio import AsyncSession

# Import crypto identity helper
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from shared.crypto_identity import get_crypto_identity
    CRYPTO_IDENTITY_AVAILABLE = True
except ImportError:
    CRYPTO_IDENTITY_AVAILABLE = False

from ..db import get_session
from ..models import ResonantChat, ResonantChatMessage
from ..domain.provider import route_query, route_query_stream, route_query_with_tools
from ..domain.agent import maybe_run_debate, maybe_spawn_agent
from ..services.resonance_hashing import ResonanceHasher
from ..services.rag_engine import rag_engine
from ..services.memory_merge import merge_and_rank_memories
from ..services.personality_dna import personality_dna
from ..services.intent_engine import intent_engine
from ..services.emotional_normalizer import emotional_normalizer
from ..services.knowledge_graph import knowledge_graph
from ..services.thought_branching import thought_branching
from ..services.evidence_graph import evidence_graph
from ..services.narrative_continuity_engine import narrative_continuity_engine
from ..services.temporal_thread_engine import temporal_thread_engine
from ..services.token_optimizer import token_optimizer
from ..services.insight_seed_engine import insight_seed_engine
from ..services.pmi_layer import pmi_manager
from ..services.latent_intent_predictor import latent_intent_predictor
from ..services.dual_memory_engine import dual_memory_engine
from ..services.magnetic_pull import magnetic_pull_system
from ..services.autonomous_error_correction import error_correction
from ..services.causal_reasoning import causal_reasoner
from ..services.neural_gravity_engine import neural_gravity_engine
from ..services.user_api_keys import user_api_key_service
# Autonomous Services
from ..services.response_cache import response_cache, get_cached_response, cache_response
from ..services.self_improving_agent import self_improving_agent, FeedbackType
from ..services.autonomous_planner import autonomous_planner, create_task_plan
# DSID-P Integration (HSU-Spec Layer 1-2)
from ..services.dsid_integration import dsid_integration, create_message_dsid, MessageDSID
# Web Search & Image Generation (optional - requires rebuild)
try:
    from ..services.web_search import web_search, WebSearchResult
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    web_search = None
    WebSearchResult = None
    WEB_SEARCH_AVAILABLE = False
    
try:
    from ..services.image_generation import image_generation, GeneratedImage
    IMAGE_GENERATION_AVAILABLE = True
except ImportError:
    image_generation = None
    GeneratedImage = None
    IMAGE_GENERATION_AVAILABLE = False
# Plan Limits Enforcement (GTM Critical)
from ..services.plan_limits import plan_limits_service
from ..services.credit_deduction import deduct_credits
# Layer 9: Output Correction (Hash Sphere Architecture)
from ..services.output_correction import output_correction
# Enhanced Metrics Calculation (NLP + Semantic Analysis)
from ..services.enhanced_metrics import enhanced_metrics_calculator, EnhancedMetricsResult
# Tools System (Neural Tool Classification)
from ..services.tools_registry import tools_registry
from ..services.tool_executor import tool_executor
from ..services.tool_classifier import tool_classifier
# Extracted modules (Phase A decomposition)
from ..infrastructure.service_client import service_client, ServiceClient, CircuitBreaker, ServiceState
from .chat_models import (
    SendMessageRequest, MessageData, GeneratedImageData, WebSearchResultData,
    ToolResultData, ResonantChatResponse, ConversationMessageRequest,
    SaveAgenticRequest, CreateChatRequest, CreateChatResponse,
    HallucinationSettingsRequest, KnowledgeBaseAddRequest,
    InternalRouteQueryRequest, FeedbackRequest, CreateChainRequest,
    ExecuteChainRequest, ExecuteCodeRequest, AnalyzeRequest,
    ValidateRequest, VotingRequest, ProjectContextRequest,
    ChunkingInfoRequest, ProcessChunkedRequest,
)
from ..services.navigation import (
    extract_navigation_tool_results,
    extract_current_time_tool_results,
    is_time_only_query,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resonant-chat", tags=["resonant-chat"])

# ============================================
# ARCHITECT SESSION TRACKER (survives page refresh)
# ============================================
# In-memory store for active/completed architect sessions.
# Key: chat_id, Value: dict with status, accumulated text, events, etc.
_architect_sessions: Dict[str, Dict[str, Any]] = {}
_ARCHITECT_SESSION_TTL = 600  # seconds — keep completed sessions for 10 min

def _get_architect_session(chat_id: str) -> Optional[Dict[str, Any]]:
    """Get an active/recent architect session."""
    s = _architect_sessions.get(chat_id)
    if not s:
        return None
    import time
    if s.get("status") == "completed" and time.time() - s.get("completed_at", 0) > _ARCHITECT_SESSION_TTL:
        _architect_sessions.pop(chat_id, None)
        return None
    return s

def _cleanup_old_sessions():
    """Remove expired completed sessions."""
    import time
    now = time.time()
    to_remove = [k for k, v in _architect_sessions.items()
                 if v.get("status") == "completed" and now - v.get("completed_at", 0) > _ARCHITECT_SESSION_TTL]
    for k in to_remove:
        _architect_sessions.pop(k, None)

async def _run_architect_background(
    chat_id: str,
    svc_payload: Dict[str, Any],
    headers: Dict[str, str],
    user_id: str,
    user_role: str,
    is_superuser: bool,
    unlimited_credits: bool,
    safe_message: str,
    user_msg_id: str,
    org_id: str,
):
    """Run architect streaming in background — survives client disconnect."""
    import time
    session_data = _architect_sessions.get(chat_id)
    if not session_data:
        return
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", f"{AGENT_ARCHITECT_URL}/api/message/stream",
                                     json=svc_payload, headers=headers) as resp:
                if resp.status_code != 200:
                    session_data["status"] = "error"
                    session_data["error"] = f"Architect returned {resp.status_code}"
                    session_data["completed_at"] = time.time()
                    return

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        event = json.loads(data_str)
                    except Exception:
                        continue

                    etype = event.get("type", "")
                    edata = event.get("data", event)

                    if etype == "text":
                        session_data["accumulated_text"] = edata.get("content", "")
                    elif etype == "complete":
                        resp_data = edata.get("response", edata)
                        final_text = resp_data.get("text", "")
                        if final_text:
                            session_data["accumulated_text"] = final_text
                        opts = resp_data.get("options")
                        if opts:
                            session_data["options"] = tool_executor._map_architect_options(opts)
                    elif etype == "stream_token":
                        content = edata.get("content", "")
                        if content:
                            session_data["accumulated_text"] = session_data.get("accumulated_text", "") + content
                    elif etype == "options":
                        session_data["options"] = tool_executor._map_architect_options(edata)

                    # Store all events for the frontend to replay
                    session_data.setdefault("events", []).append({"type": etype, "data": edata, "ts": time.time()})

        session_data["status"] = "completed"
        session_data["completed_at"] = time.time()
        print(f"[ARCHITECT-BG] Session {chat_id} completed. text_len={len(session_data.get('accumulated_text', ''))}", flush=True)

        # Save the message to DB
        try:
            from ..db import async_session_factory
            async with async_session_factory() as db_session:
                _clean = session_data.get("accumulated_text", "")
                _clean = re.sub(r'Please choose one of the options:\s*present_options\([^)]*\)', '', _clean).strip()
                _clean = re.sub(r'present_options\([^)]*\)', '', _clean).strip()
                if _clean:
                    _clean = _sanitize_agent_response(_clean)
                try:
                    a_hash = hasher.hash_text(_clean) if _clean else _simple_hash("")
                    a_xyz = hasher.hash_to_coords(a_hash) if _clean else _hash_to_xyz_simple(a_hash)
                except Exception:
                    a_hash = _simple_hash(_clean)
                    a_xyz = _hash_to_xyz_simple(a_hash)
                assistant_msg = ResonantChatMessage(
                    chat_id=UUID(chat_id), role="assistant", content=_clean,
                    ai_provider="tool_agent_architect",
                    hash=a_hash, resonance_score=0.5,
                    xyz_x=a_xyz[0], xyz_y=a_xyz[1], xyz_z=a_xyz[2],
                    meta_data={"agent_type": "architect", "toolResults": [{"tool_name": "tool_agent_architect", "success": True, "result": {"summary": _clean[:500]}}]},
                )
                db_session.add(assistant_msg)
                await db_session.commit()
                await db_session.refresh(assistant_msg)
                session_data["message_id"] = str(assistant_msg.id)
                print(f"[ARCHITECT-BG] Saved message {assistant_msg.id} for chat {chat_id}", flush=True)
        except Exception as db_err:
            print(f"[ARCHITECT-BG] DB save failed: {db_err}", flush=True)

    except asyncio.CancelledError:
        session_data["status"] = "cancelled"
        session_data["completed_at"] = time.time()
    except Exception as e:
        session_data["status"] = "error"
        session_data["error"] = str(e)
        session_data["completed_at"] = time.time()
        print(f"[ARCHITECT-BG] Error: {e}", flush=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def _simple_hash(text: str) -> str:
    """Simple hash for fallback when resonance hasher unavailable."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def _hash_to_xyz_simple(hash_str: str) -> tuple:
    """Convert hash to XYZ coordinates (simple version)."""
    try:
        x = int(hash_str[:8], 16) / 0xFFFFFFFF
        y = int(hash_str[8:16], 16) / 0xFFFFFFFF
        z = int(hash_str[16:24], 16) / 0xFFFFFFFF
        return (x, y, z)
    except:
        return (0.5, 0.5, 0.5)


def _extract_github_token_from_user_keys(user_api_keys: Optional[Dict[str, str]]) -> Optional[str]:
    if not user_api_keys:
        return None
    for key in ("github", "github_token", "gh_token"):
        value = (user_api_keys.get(key) or "").strip()
        if value:
            return value
    return None


def _sanitize_sensitive_tokens(text: str) -> str:
    """Redact secrets from user-entered text before persistence/logging.

    Keeps intent/repo URLs intact while removing raw token values.
    """
    value = text or ""

    # GitHub classic/fine-grained token formats
    value = re.sub(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,255}|github_pat_[A-Za-z0-9_]{20,255})\b', '[REDACTED_GITHUB_TOKEN]', value)

    # Generic token hints in prompts, e.g. token=... / github_token: ...
    value = re.sub(
        r'((?:github[_\s-]?token|token)\s*[:=]\s*)([A-Za-z0-9_\-]{12,})',
        r'\1[REDACTED_TOKEN]',
        value,
        flags=re.IGNORECASE,
    )
    return value


def _calculate_resonance_score(
    response_text: str,
    user_message: str,
    memories: List[Dict[str, Any]],
    context_messages: List[Dict[str, str]],
) -> float:
    """
    Calculate a resonance score based on response quality factors:
    - Response length and completeness
    - Memory utilization (did we use relevant memories?)
    - Context coherence (does response relate to the conversation?)
    - Error indicators (apologies, uncertainty markers)
    
    Returns a score between 0.0 and 1.0
    """
    score = 0.5  # Base score
    
    # Factor 1: Response length (longer responses tend to be more complete)
    response_len = len(response_text)
    if response_len > 500:
        score += 0.1
    elif response_len > 200:
        score += 0.05
    elif response_len < 50:
        score -= 0.1
    
    # Factor 2: Memory utilization
    if memories and len(memories) > 0:
        # Check if response references memory content
        memory_keywords = set()
        for mem in memories[:5]:  # Increased from 5 to 20 for better context
            content = mem.get("content", "")
            if content:
                words = content.lower().split()[:10]
                memory_keywords.update(words)
        
        response_lower = response_text.lower()
        matches = sum(1 for kw in memory_keywords if kw in response_lower and len(kw) > 4)
        if matches > 3:
            score += 0.15
        elif matches > 0:
            score += 0.05
    
    # Factor 3: Context coherence - check if response relates to user message
    user_words = set(user_message.lower().split())
    response_words = set(response_text.lower().split())
    overlap = len(user_words & response_words)
    if overlap > 5:
        score += 0.1
    elif overlap > 2:
        score += 0.05
    
    # Factor 4: Error/uncertainty indicators (reduce score)
    error_phrases = [
        "i apologize", "i'm sorry", "i cannot", "i don't have",
        "i'm not sure", "i don't know", "unavailable", "error",
        "no ai providers", "api key"
    ]
    response_lower = response_text.lower()
    for phrase in error_phrases:
        if phrase in response_lower:
            score -= 0.1
            break
    
    # Factor 5: Structured response (code blocks, lists)
    if "```" in response_text:
        score += 0.05
    if any(marker in response_text for marker in ["1.", "2.", "- ", "* "]):
        score += 0.05
    
    # Clamp to valid range
    return max(0.1, min(0.95, score))


async def _get_user_api_keys(session: AsyncSession, user_id: str) -> Optional[Dict[str, str]]:
    """Get user's API keys for BYOK (Bring Your Own Key)."""
    logger.info(f"🔑 _get_user_api_keys called for user: {user_id}")
    try:
        keys = await user_api_key_service.get_user_api_keys(user_id)
        logger.info(f"🔑 Raw keys from auth service: {keys}")
        if keys:
            formatted = user_api_key_service.format_keys_for_router(keys)
            logger.info(f"🔑 Retrieved {len(formatted)} user API keys for BYOK: {list(formatted.keys())}")
            return formatted
        else:
            logger.warning(f"🔑 No keys returned from auth service for user {user_id}")
    except Exception as e:
        logger.error(f"🔑 Failed to retrieve user API keys: {e}", exc_info=True)
    return None


_MEMORY_CACHE: Dict[str, tuple] = {}
_MEMORY_CACHE_TTL = 60.0  # seconds
_MEMORY_MIN_SCORE = float(os.getenv("MEMORY_MIN_SCORE", "0.35"))


def _mem_cache_key(user_id: str, session_id: Optional[str], message: str) -> str:
    import hashlib
    h = hashlib.sha256(message.strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"{user_id}|{session_id or '-'}|{h}"


async def _extract_memories(
    user_id: str,
    org_id: str,
    message: str,
    agent_hash: Optional[str] = None,
    team_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract memories from memory service (RAG + Hash Sphere).
    
    Uses 3-level hierarchical memory:
    1. USER MEMORY (Global) - agent_hash=None
    2. TEAM MEMORY - agent_hash="team_{teamId}"
    3. AGENT MEMORY - agent_hash="{agent_hash}"
    """
    rag_memories = []
    sphere_memories = []
    
    # Determine effective agent_hash for hierarchical memory
    effective_agent_hash = None
    if team_id:
        effective_agent_hash = f"team_{team_id}"
    elif agent_hash:
        effective_agent_hash = agent_hash
    
    # ============================================
    # STEP 4: Extract Memories (RAG + Hash Sphere) - PRODUCTION READY
    # ============================================
    rag_memories = []
    sphere_memories = []
    
    # Determine effective agent_hash for hierarchical memory
    effective_agent_hash = None
    if team_id:
        effective_agent_hash = f"team_{team_id}"
    elif agent_hash:
        effective_agent_hash = agent_hash
    
    # RAG retrieval with circuit breaker
    try:
        # Validate user_id is UUID format for memory service
        import uuid
        import time as _time
        try:
            uuid.UUID(user_id)
            valid_user_id = user_id
        except ValueError:
            # Generate a UUID for testing if invalid
            valid_user_id = str(uuid.uuid4())
            logger.warning(f"Invalid user_id format, using generated UUID: {valid_user_id}")

        # PATCH C: 60s LRU cache keyed on (user, session, message-hash)
        _cache_k = _mem_cache_key(valid_user_id, session_id, message)
        _cached = _MEMORY_CACHE.get(_cache_k)
        if _cached and (_time.time() - _cached[0]) < _MEMORY_CACHE_TTL:
            logger.info(f"🧠 Memory cache HIT for {_cache_k[:40]}")
            return _cached[1]

        # ============================================
        # FULL HASH SPHERE MEMORY EXTRACTION
        # Uses 9-Layer Architecture with multi-method retrieval
        # RAG is LAST RESORT fallback only
        # ============================================
        hash_sphere_result = await service_client.call_service(
            "memory_service",
            "POST",
            "http://memory_service:8000/memory/hash-sphere/extract",
            json={
                "query": message,
                "user_id": valid_user_id,
                "org_id": org_id,
                "agent_hash": effective_agent_hash,
                "session_id": session_id,           # PATCH B: conversation scope
                "min_score": _MEMORY_MIN_SCORE,     # PATCH A: relevance floor
                "limit": 25,
                # Extraction methods in priority order
                "use_anchors": True,       # Layer 4: Anchor-based lookup (PRIORITY 1)
                "use_proximity": True,     # Layer 5: XYZ proximity search (PRIORITY 2)
                "use_resonance": True,     # Layer 6: Hash resonance filtering (PRIORITY 3)
                "use_clusters": True,      # Cluster-based retrieval (PRIORITY 4)
                "use_rag_fallback": True,  # RAG as LAST RESORT only
                # Advanced options
                "include_coordinates": True,
                "apply_magnetic_pull": True,  # HS-MPS non-linear boost
            }
        )
        print(f"[MEMORY] hash_sphere_result type={type(hash_sphere_result).__name__}, "
              f"keys={list(hash_sphere_result.keys()) if isinstance(hash_sphere_result, dict) else 'N/A'}, "
              f"memories={len(hash_sphere_result.get('memories', [])) if isinstance(hash_sphere_result, dict) else 0}, "
              f"user_id={valid_user_id}, agent_hash={effective_agent_hash}",
              flush=True)
        if hash_sphere_result:
            # Extract memories from full Hash Sphere response
            memories_data = hash_sphere_result.get("memories", [])
            methods_used = hash_sphere_result.get("extraction_methods_used", [])
            extraction_time = hash_sphere_result.get("extraction_time_ms", 0)
            
            # Convert to memory format for context
            for mem in memories_data:
                sphere_memories.append({
                    "content": mem.get("content", ""),
                    "hash": mem.get("hash"),
                    "xyz": mem.get("xyz"),
                    "hybrid_score": mem.get("hybrid_score", 0.0),
                    "resonance_score": mem.get("resonance_score", 0.0),
                    "proximity_score": mem.get("proximity_score", 0.0),
                    "anchor_energy": mem.get("anchor_energy", 0.0),
                    "magnetic_score": mem.get("magnetic_score", 0.0),
                    "type": mem.get("type", "memory"),
                })
            
            logger.info(f"✅ Hash Sphere extraction successful: {len(sphere_memories)} memories")
            logger.info(f"   Methods used: {methods_used}")
            logger.info(f"   Extraction time: {extraction_time:.2f}ms")
        else:
            logger.info("⚠️ Hash Sphere extraction skipped - circuit breaker open or service unavailable")
    except Exception as e:
        print(f"[MEMORY] Hash Sphere extraction FAILED: {e}", flush=True)
    
    logger.info(f"🧠 Memory extraction complete: RAG={len(rag_memories)}, Sphere={len(sphere_memories)}")
    
    # Apply magnetic pull to memories (Patch #39)
    if rag_memories:
        rag_memories = magnetic_pull_system.apply_to_memories(rag_memories)
    if sphere_memories:
        sphere_memories = magnetic_pull_system.apply_to_memories(sphere_memories)
    
    # Merge and rank memories using hybrid scoring
    # Merge and rank memories (increased limit from 10 to 25 for better context)
    merged_memories = merge_and_rank_memories(
        rag_memories=rag_memories,
        sphere_memories=sphere_memories,
        limit=25
    )
    
    # Add xyz coordinates to memories for Layer 7/9 evidence aggregation
    hasher = ResonanceHasher()
    for mem in merged_memories:
        if not mem.get("xyz"):
            try:
                # Try hash first, then content
                if mem.get("hash"):
                    mem["xyz"] = hasher.hash_to_coords(mem["hash"])
                elif mem.get("content"):
                    # Generate hash from content and convert to xyz
                    content_hash = hasher.hash_text(mem["content"])
                    mem["xyz"] = hasher.hash_to_coords(content_hash)
                    mem["hash"] = content_hash  # Store for future use
            except Exception as e:
                print(f"⚠️ Failed to generate xyz for memory: {e}")
    
    logger.info(f"🧠 Memory extraction: RAG={len(rag_memories)}, Sphere={len(sphere_memories)}, Merged={len(merged_memories)}")

    # PATCH C: write result to cache
    try:
        _MEMORY_CACHE[_cache_k] = (_time.time(), merged_memories)
    except Exception:
        pass

    return merged_memories


def _sanitize_agent_response(response: str) -> str:
    """
    Remove internal agent prompts and meta-instructions from response.
    
    Filters out common agent prompt leakage patterns like:
    - "Based on the user message..."
    - "Here is a corrected response..."
    - "To address/analyze/provide..."
    - Numbered steps at the beginning
    """
    import re
    
    if not response:
        return response
    
    # Patterns to remove (case-insensitive)
    patterns = [
        r"Based on the (?:updated )?user(?:'s)? message.*?response:\s*",
        r"Here is a (?:corrected|more accurate).*?response:\s*",
        r"To (?:address|analyze|provide|create|fix|resolve).*?:\s*",
        r"(?:Receive and Review|Remember to|The user (?:stated|said|asked)).*?:\s*",
        r"^(?:Step \d+:|Action \d+:|\d+\.)\s+(?:Receive|Analyze|Provide|Create|Fix|Resolve).*?\n",
        r"CRITICAL:\s*.*?\n",
        r"IMPORTANT:\s*.*?\n",
    ]
    
    cleaned = response
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove trailing filler questions that add no value
    trailing_q_patterns = [
        r"\s*What do you think\??[^\n]*$",
        r"\s*Is there anything (?:in particular |else |specific )?(?:that resonates|you'?d like to (?:explore|discuss|know|focus on)|I can help with)\??[^\n]*$",
        r"\s*(?:Would|Do) you (?:like|want) (?:to|me to) (?:discuss|explore|elaborate|dive|know|talk about|proceed)\b[^\n]*\??[^\n]*$",
        r"\s*How (?:would you like to proceed|does that (?:sound|make sense)|do you feel about)\??[^\n]*$",
        r"\s*What are your thoughts\??[^\n]*$",
        r"\s*Does that (?:make sense|resonate|help)\??[^\n]*$",
        r"\s*(?:Anything else|Let me know if) (?:you'?d like|I can|you want)[^\n]*\??[^\n]*$",
    ]
    for pattern in trailing_q_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def _build_context_messages(
    recent_messages: List[ResonantChatMessage],
    memories: List[Dict[str, Any]],
    user_message: str,
    user_role: str = "user",
    user_plan: str = "free",
) -> List[Dict[str, str]]:
    """Build minimal context for agent orchestration.
    
    NOTE: Agents have their own specialized prompts. This just provides conversation context.
    """
    context_messages = []
    
    # ============================================
    # RESONANT CHAT IDENTITY SYSTEM PROMPT
    # ============================================
    # Get personality DNA
    pdna = personality_dna.system_prompt()
    
    # Get current date/time for context
    from datetime import datetime
    current_datetime = datetime.now(ZoneInfo("America/Los_Angeles"))
    current_date_str = current_datetime.strftime("%A, %B %d, %Y")
    current_time_str = current_datetime.strftime("%I:%M %p %Z")
    
    resonant_identity_prompt = f"""You are DevSwat Chat — the AI assistant for the DevSwat platform.
Today is {current_date_str}, {current_time_str}. User role: {user_role}, plan: {user_plan}.

<identity>
You are DevSwat Chat, a specialized AI with persistent memory (Hash Sphere), web search, code analysis, and agent management capabilities. You were created by the DevSwat team. When asked "who are you?", identify as DevSwat Chat.
{pdna}
</identity>

<capabilities>
You have access to **208 tools** auto-selected by a neural classifier. Categories include: web search, weather, news, code analysis (AST/SAST), Agent Architect (create/manage/run agents), memory (Hash Sphere read/write/search), media generation (image/audio/video), integrations (Google Drive/Calendar/Gmail, Slack, Discord, Figma, Jira, Notion, GitHub, GitLab, LinkedIn, Salesforce, HubSpot, Airtable, and 20+ more), state physics simulation, community/Rabbit posts, file operations, code execution, and more. When asked about your tools: you have 208 tools across 15+ categories. Tool outputs appear as "TOOL OUTPUT (name):". If absent, the tool did NOT run — never fabricate.
</capabilities>

<anti_hallucination>
- If no "TOOL OUTPUT" section exists in your context for a tool, it did NOT run. Say so honestly and offer to retry. Never fabricate scan results, statistics, endpoints, IDs, or analysis data.
- Never invent facts, sources, metrics, IDs, links, or tool results. Prefer "I don't know" over guessing.
- When asked "how many tools/agents/capabilities do you have" — answer from your system prompt, not from conversation history.
</anti_hallucination>

<communication_style>
- Be direct and specific. Talk like a smart colleague, not a corporate chatbot.
- Jump straight to your answer. Never paraphrase what the user just said ("It sounds like...", "You mentioned...").
- Give actionable information, not vague encouragement. Empathize briefly if needed, then offer concrete solutions.
- Keep responses concise. No filler content.
- Use conversation history and memories naturally without announcing it.
- End with a statement, never a trailing question ("What do you think?", "How does that sound?", etc.). Ask questions only when genuinely ambiguous (<5% of responses).
</communication_style>

<markdown_formatting>
Format all responses with Markdown. The chat UI renders full Markdown with syntax highlighting.
- **bold** for key terms, `inline code` for technical values, headings for sections.
- Bullet/numbered lists for multiple items, code blocks with language tags, tables for structured data.
- Blockquotes for callouts, horizontal rules between major sections.
- Short answers don't need headers but still use bold and code where appropriate.
</markdown_formatting>

<context_awareness>
- Conversation history is OLDEST → NEWEST. The LAST user message is your primary focus.
- Resolve pronouns ("it", "that") from previous messages. Build on prior discussion, don't restart.
- Never contradict your own previous messages. Maintain conversation continuity.
</context_awareness>

<platform_pages>
/dashboard, /agents (agent config at /agents/:agentId), /agent-teams, /connect-profiles (integrations/API keys), /ide, /code-visualizer, /state-physics, /resonant-memory, /rabbit (community), /pricing, /help, /profile, /marketplace, /build, /network. Never invent routes.
</platform_pages>
"""
    
    context_messages.append({
        "role": "system",
        "content": resonant_identity_prompt
    })

    truthfulness_guardrail_prompt = """TRUTHFULNESS & GROUNDING RULES (MANDATORY):
- Do NOT invent facts, sources, metrics, events, IDs, links, or tool results.
- If a fact is uncertain or unavailable, explicitly say so in plain language.
- Prefer "I don't know" over guessing.
- Separate verified information from inference (label inference clearly).
- For time-sensitive or external factual claims, request/require web search or tool execution instead of fabricating.
- Never claim a scan/analysis/tool action ran unless it actually ran and returned output.
"""
    context_messages.append({
        "role": "system",
        "content": truthfulness_guardrail_prompt,
    })

    # Add recent messages for context (capped at 15 to keep LLM focused)
    logger.info(f"📝 Adding {len(recent_messages[-15:])} recent messages to context")
    for msg in recent_messages[-15:]:
        context_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    logger.info(f"📝 Total context_messages: {len(context_messages)}")
    
    # Add memory context if available (increased from 5 to 20 for better memory retention)
    if memories:
        print(f"[MEMORY-CONTEXT] Injecting {len(memories)} memories into context", flush=True)
        memory_context = "RELEVANT MEMORIES FROM USER'S HASH SPHERE:\n"
        mem_count = 0
        for mem in memories[:5]:
            content = mem.get("content", "") or mem.get("anchor_text", "")
            # Quality filter: skip very short, empty, or still-encrypted memories
            if not content or len(content.strip()) < 15 or content.startswith("ENC2:"):
                print(f"[MEMORY-CONTEXT] Skipping memory: len={len(content)}, starts_with_ENC2={content.startswith('ENC2:') if content else False}", flush=True)
                continue
            mem_count += 1
            score = mem.get("hybrid_score", 0)
            memory_context += f"{mem_count}. [{score:.2f}] {content[:300]}\n"
        
        print(f"[MEMORY-CONTEXT] Added {mem_count} memories to context", flush=True)
        if mem_count > 0:
            context_messages.append({
                "role": "system",
                "content": memory_context
            })
    else:
        print(f"[MEMORY-CONTEXT] No memories to inject", flush=True)
    
    return context_messages


# ============================================
# SSE STREAMING ENDPOINT
# ============================================

AGENT_ARCHITECT_URL = os.getenv("AGENT_ARCHITECT_URL", "http://agent_architect:8000")

_AGENT_INTENT_RE = re.compile(
    r'\b(create|build|make|delete|remove|run|start|stop|list|show|manage|configure|modify'
    r'|diagnose|update|edit|deploy|schedule|test|launch|setup|set\s*up)\b'
    r'.{0,20}'
    r'\b(agents?)\b',
    re.IGNORECASE,
)
_AGENT_REVERSE_RE = re.compile(
    r'\b(agents?)\b.{0,15}\b(create|build|make|delete|run|start|stop|list|manage|configure|modify)\b',
    re.IGNORECASE,
)
_AGENT_WANT_RE = re.compile(
    r'\b(i\s+want|i\s+need|i\s+wanna|give\s+me|help\s+me\s+with)\b.{0,20}\b(agents?)\b',
    re.IGNORECASE,
)
_AGENT_NOUN_RE = re.compile(
    r'\b(my\s+agents?|agents?\s+for\b|agents?\s+that\b|agents?\s+to\b|agents?\s+page|how\s+many\s+agents?|agents?\s+(?:do\s+)?i\s+have)',
    re.IGNORECASE,
)
_AGENT_DELETE_RE = re.compile(
    r'\b(delete|remove)\b.{0,20}\b(duplicates?|the\s+\d|all\s+\d|them\s+all)\b',
    re.IGNORECASE,
)

def _is_agent_intent(text: str) -> bool:
    """Detect whether user message is about agent management (architect territory)."""
    if not text:
        return False
    # If user is clearly asking for image generation, it's NOT an agent intent
    if _is_image_intent(text):
        return False
    return bool(
        _AGENT_INTENT_RE.search(text)
        or _AGENT_REVERSE_RE.search(text)
        or _AGENT_WANT_RE.search(text)
        or _AGENT_NOUN_RE.search(text)
        or _AGENT_DELETE_RE.search(text)
    )


_IMAGE_INTENT_RE = re.compile(
    r'\b(generate|create|draw|make|illustrate|design|render)\b'
    r'.{0,25}'
    r'\b(image|picture|photo|illustration|logo|art|artwork|portrait|poster|icon|banner|thumbnail|wallpaper|graphic)\b',
    re.IGNORECASE,
)
_IMAGE_REVERSE_RE = re.compile(
    r'\b(image|picture|photo|illustration|logo|art|artwork)\b'
    r'.{0,15}'
    r'\b(of|for|with|showing)\b',
    re.IGNORECASE,
)

_QUESTION_ABOUT_RE = re.compile(
    r'\b(why|how come|cannot|can\'t|cant|unable|not able|doesn\'t|doesnt|don\'t|dont|won\'t|wont|isn\'t|isnt|failed|failing|broken|issue|problem|bug|error)\b',
    re.IGNORECASE,
)

def _is_image_intent(text: str) -> bool:
    """Detect whether user message is clearly REQUESTING image generation (not asking about it)."""
    if not text:
        return False
    # If the message looks like a QUESTION about image generation, it's NOT a request
    if _QUESTION_ABOUT_RE.search(text):
        return False
    return bool(
        _IMAGE_INTENT_RE.search(text)
        or _IMAGE_REVERSE_RE.search(text)
    )


def _sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


@router.post("/message/stream")
async def stream_message(
    request_body: SendMessageRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    SSE streaming endpoint. Forwards architect SSE events live.
    For non-architect messages, streams LLM chunks.
    """
    # ── Auth ──
    if CRYPTO_IDENTITY_AVAILABLE:
        identity = get_crypto_identity(request)
        user_id = identity.user_id
        org_id = identity.org_id or user_id
    else:
        user_id = request.headers.get("x-user-id")
        org_id = request.headers.get("x-org-id") or user_id
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")

    user_role = request.headers.get("x-user-role", "user")
    is_superuser = request.headers.get("x-is-superuser", "false").lower() == "true"
    unlimited_credits = request.headers.get("x-unlimited-credits", "false").lower() == "true"
    raw_message = request_body.message or ""
    safe_message = _sanitize_sensitive_tokens(raw_message)
    print(f"[STREAM-REQ] preferred_provider={request_body.preferred_provider!r} preferred_model={request_body.preferred_model!r} msg={safe_message[:60]!r}", flush=True)

    # ── Get or create chat ──
    chat_id = request_body.chat_id
    chat = None
    if chat_id:
        try:
            result = await session.execute(select(ResonantChat).where(ResonantChat.id == UUID(chat_id)))
            chat = result.scalar_one_or_none()
        except ValueError:
            pass
    if not chat:
        chat = ResonantChat(
            user_id=user_id, org_id=org_id,
            title=safe_message[:50] + ("..." if len(safe_message) > 50 else ""),
            status="active", agent_hash=request_body.agent_hash,
        )
        session.add(chat)
        await session.commit()
        await session.refresh(chat)
    chat_id = str(chat.id)

    # ── Store user message ──
    try:
        hasher = ResonanceHasher()
        user_hash = hasher.hash_text(safe_message)
        user_xyz = hasher.hash_to_coords(user_hash)
    except Exception:
        user_hash = _simple_hash(safe_message)
        user_xyz = _hash_to_xyz_simple(user_hash)

    user_msg = ResonantChatMessage(
        chat_id=UUID(chat_id), role="user", content=safe_message,
        hash=user_hash, resonance_score=0.5,
        xyz_x=user_xyz[0], xyz_y=user_xyz[1], xyz_z=user_xyz[2],
        agent_hash=request_body.agent_hash,
    )
    session.add(user_msg)
    await session.commit()
    await session.refresh(user_msg)

    # ── Deduct credits ──
    try:
        await deduct_credits(
            user_id=user_id, action="chat_message",
            description=f"Chat message in {chat_id[:8]}...",
            user_role=user_role, is_superuser=is_superuser,
            unlimited_credits=unlimited_credits,
        )
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 402:
            raise HTTPException(status_code=402, detail="Credits exhausted")
        raise
    except Exception:
        pass

    # ── Detect tool ──
    # Fetch 5 full messages + 20 summarized messages for context
    result_msgs_full = await session.execute(
        select(ResonantChatMessage).where(ResonantChatMessage.chat_id == UUID(chat_id))
        .order_by(ResonantChatMessage.created_at.desc()).limit(5)
    )
    recent_messages_full = list(reversed(result_msgs_full.scalars().all()))

    # Fetch additional messages for summarization (up to 20 more, skipping the 5 full ones)
    result_msgs_summary = await session.execute(
        select(ResonantChatMessage).where(ResonantChatMessage.chat_id == UUID(chat_id))
        .order_by(ResonantChatMessage.created_at.desc()).offset(5).limit(20)
    )
    recent_messages_summary = list(reversed(result_msgs_summary.scalars().all()))

    # Combine: 5 full messages + summarized older messages
    recent_messages = recent_messages_full + [
        ResonantChatMessage(
            id=m.id, role=m.role, content=f"[SUMMARY] {m.content[:200]}...", 
            hash=m.hash, resonance_score=m.resonance_score,
            xyz_x=m.xyz_x, xyz_y=m.xyz_y, xyz_z=m.xyz_z,
            created_at=m.created_at, meta_data=m.meta_data
        )
        for m in recent_messages_summary
    ]

    # Check architect pipeline continuity — route follow-ups to architect
    # Triggers if ANY recent assistant message was from the architect
    _architect_active = False
    for _rmsg in reversed(recent_messages[-4:]):
        if (getattr(_rmsg, "role", "") or "") != "assistant":
            continue
        # Check 1: ai_provider indicates architect handled it
        _r_provider = getattr(_rmsg, "ai_provider", "") or ""
        if "architect" in _r_provider.lower():
            _architect_active = True
            break
        # Check 2: toolResults contain architect tool or present_options
        _r_meta = getattr(_rmsg, "meta_data", None)
        if _r_meta and isinstance(_r_meta, dict):
            for _tr in _r_meta.get("toolResults", []):
                if isinstance(_tr, dict):
                    _tn = (_tr.get("tool_name") or "")
                    if "architect" in _tn.lower() or (_tr.get("result") or {}).get("present_options"):
                        _architect_active = True
                        break
        break

    use_architect = False
    detected_tool = None
    _stream_web_search = False
    _stream_image_gen = False

    # Messages from present_options buttons are prefixed with "Agent Architect: "
    # This is a hard signal — always route to architect
    if safe_message.startswith("Agent Architect:"):
        use_architect = True
        detected_tool = tools_registry.get_tool("agent_architect")
        print(f"[STREAM-DETECT] 'Agent Architect:' prefix detected — routing to architect", flush=True)

    # Always run classifier for full tool detection
    _clf_prediction = None
    if not use_architect:
        try:
            from ..services.tool_classifier import ALL_TOOLS
            enabled_ids = {s for s in ALL_TOOLS if s is not None}
            msg_intents = intent_engine.extract(safe_message)
            _clf_prediction = await tool_classifier.predict(
                message=safe_message, enabled_tool_ids=enabled_ids,
                recent_messages=recent_messages[-6:] if recent_messages else None,
                intents=msg_intents, user_id=user_id,
            )
            print(f"[STREAM-DETECT] classifier: tool={_clf_prediction.tool_id} conf={_clf_prediction.confidence:.3f} method={_clf_prediction.method} msg={safe_message[:60]!r}", flush=True)
            if _clf_prediction.tool_id == "agent_architect":
                use_architect = True
                detected_tool = tools_registry.get_tool("agent_architect")
            elif _clf_prediction.tool_id and _clf_prediction.tool_id.startswith("agents_"):
                # Only route to architect if confidence is high enough (avoid false positives)
                if _clf_prediction.confidence > 0.4:
                    use_architect = True
                    detected_tool = tools_registry.get_tool("agent_architect")
                    print(f"[STREAM-DETECT] agents_* tool '{_clf_prediction.tool_id}' (conf={_clf_prediction.confidence:.3f}) → routing to architect", flush=True)
                else:
                    print(f"[STREAM-DETECT] agents_* tool '{_clf_prediction.tool_id}' (conf={_clf_prediction.confidence:.3f}) → SKIPPED (too low)", flush=True)
            elif _clf_prediction.tool_id:
                effective_id = _clf_prediction.tool_id
                if effective_id == "web_search":
                    _stream_web_search = True
                elif effective_id == "image_generation":
                    _stream_image_gen = True
                else:
                    detected_tool = tools_registry.get_tool(effective_id)
                    if detected_tool:
                        print(f"[STREAM-DETECT] tool resolved: {detected_tool.id} ({detected_tool.name})", flush=True)
        except Exception as _clf_err:
            print(f"[STREAM-DETECT] classifier ERROR: {_clf_err}", flush=True)

    # Soft continuity: if architect was active AND classifier didn't confidently
    # pick a different tool, keep routing to architect for follow-ups.
    # EXCEPTION: if image_gen or web_search was detected, always break continuity.
    # EXCEPTION: if keyword guard detects clear image intent, break continuity.
    # EXCEPTION: if classifier says "no tool needed" (None), break continuity.
    _image_intent_detected = _is_image_intent(safe_message) and IMAGE_GENERATION_AVAILABLE
    if not use_architect and _architect_active and not _stream_image_gen and not _stream_web_search and not _image_intent_detected:
        # Check if classifier picked a specific other tool confidently
        _clf_picked_other = (
            _clf_prediction
            and _clf_prediction.tool_id
            and _clf_prediction.tool_id != "agent_architect"
            and not (_clf_prediction.tool_id or "").startswith("agents_")
            and _clf_prediction.confidence > 0.45
        )
        # Check if classifier said "no tool needed" (general chat) — also breaks continuity
        _clf_said_none = (
            _clf_prediction
            and not _clf_prediction.tool_id
            and _clf_prediction.confidence > 0.30
        )
        if _clf_picked_other:
            print(f"[STREAM-DETECT] continuity BROKEN — classifier confidently picked '{_clf_prediction.tool_id}' ({_clf_prediction.confidence:.2f})", flush=True)
        elif _clf_said_none:
            print(f"[STREAM-DETECT] continuity BROKEN — classifier says NO TOOL needed (conf={_clf_prediction.confidence:.2f}), general chat", flush=True)
        else:
            # Only keep continuity if this looks like an architect follow-up
            # (e.g., "yes", "do it", "the reports folder" — short affirmatives)
            _is_short_followup = len(safe_message.split()) <= 6
            if _is_short_followup:
                use_architect = True
                detected_tool = tools_registry.get_tool("agent_architect")
                print(f"[STREAM-DETECT] continuity KEPT — short follow-up ({len(safe_message.split())} words)", flush=True)
            else:
                print(f"[STREAM-DETECT] continuity BROKEN — message too long/complex for follow-up ({len(safe_message.split())} words)", flush=True)

    # Also check keyword guard (regex-based, handles articles/prepositions)
    if not use_architect and _is_agent_intent(safe_message):
        use_architect = True
        print(f"[STREAM-DETECT] regex guard triggered for msg={safe_message[:60]!r}", flush=True)

    # Image intent keyword guard — override architect/other tools when user clearly wants an image
    if _is_image_intent(safe_message) and IMAGE_GENERATION_AVAILABLE:
        use_architect = False
        detected_tool = None
        _stream_image_gen = True
        print(f"[STREAM-DETECT] IMAGE INTENT guard — forcing image generation for msg={safe_message[:60]!r}", flush=True)

    # Vision guard: when user uploads images, skip integration tools and let LLM handle with vision
    if request_body.images and len(request_body.images) > 0 and not _stream_image_gen:
        _integration_tools = {"google_drive", "google_calendar", "figma", "sigma"}
        if detected_tool and detected_tool.id in _integration_tools:
            print(f"[STREAM-DETECT] VISION guard — discarding {detected_tool.id} (user uploaded images, routing to LLM vision)", flush=True)
            detected_tool = None
        if use_architect and not safe_message.startswith("Agent Architect:"):
            print(f"[STREAM-DETECT] VISION guard — discarding architect (user uploaded images)", flush=True)
            use_architect = False

    print(f"[STREAM-DETECT] FINAL: use_architect={use_architect} detected_tool={detected_tool.id if detected_tool else None} img={_stream_image_gen} web={_stream_web_search} continuity={_architect_active} msg={safe_message[:60]!r}", flush=True)

    # ── Get user API keys ──
    user_api_keys = await _get_user_api_keys(session, user_id)

    # ── Pre-warm memory extraction (runs concurrently with architect/pipeline setup) ──
    import asyncio as _asyncio
    _memory_task = None
    if not use_architect:
        _memory_task = _asyncio.create_task(_extract_memories(
            user_id=user_id, org_id=org_id, message=safe_message,
            agent_hash=request_body.agent_hash, team_id=request_body.teamId,
            session_id=chat_id,
        ))

    async def _generate_events():
        nonlocal detected_tool
        accumulated_text = ""
        present_options_data = None
        _stream_tool_result = None

        actual_provider = None
        router_meta = None
        agent_type = None
        memories = []
        context_messages = []
        _sse_images = []  # populated by image generation for persistence

        if use_architect:
            # ── ARCHITECT SSE STREAMING (background-task pattern) ──
            # The architect runs in a detached background task so it survives
            # client disconnects (page refresh, navigation).  This SSE generator
            # tails the in-memory session store and forwards events to the client.
            import time as _time
            print(f"[ARCHITECT-ROUTE] use_architect=True, user_api_keys={list(user_api_keys.keys()) if user_api_keys else 'None'}", flush=True)
            yield _sse_event({"event": "start", "chat_id": chat_id, "user_message_id": str(user_msg.id), "tool": "agent_architect"})

            _arch_headers = {
                "x-user-id": user_id,
                "x-user-role": user_role,
                "x-is-superuser": str(is_superuser).lower(),
                "x-unlimited-credits": str(unlimited_credits).lower(),
            }
            _hist_msgs = []
            for m in recent_messages[-20:]:
                _entry = {
                    "role": getattr(m, "role", ""),
                    "content": (getattr(m, "content", "") or "")[:1000],
                }
                if hasattr(m, "created_at") and m.created_at:
                    _entry["timestamp"] = m.created_at.isoformat()
                _hist_msgs.append(_entry)

            svc_payload = {
                "message": raw_message,
                "workspace_id": user_id,
                "user_id": user_id,
                "context": "",
                "conversation_history": _hist_msgs,
                "user_api_keys": user_api_keys or {},
            }

            # Register session in tracker & launch background task
            _cleanup_old_sessions()
            _architect_sessions[chat_id] = {
                "status": "running",
                "accumulated_text": "",
                "events": [],
                "options": None,
                "started_at": _time.time(),
                "user_id": user_id,
                "user_msg_id": str(user_msg.id),
                "bg_save_done": False,
            }
            _bg_task = asyncio.create_task(_run_architect_background(
                chat_id=chat_id, svc_payload=svc_payload, headers=_arch_headers,
                user_id=user_id, user_role=user_role, is_superuser=is_superuser,
                unlimited_credits=unlimited_credits, safe_message=safe_message,
                user_msg_id=str(user_msg.id), org_id=org_id,
            ))

            # Tail the in-memory session — forward events to client in real time
            _last_event_idx = 0
            _last_text = ""
            while True:
                _sess = _architect_sessions.get(chat_id)
                if not _sess:
                    break

                # Forward new events
                _events = _sess.get("events", [])
                while _last_event_idx < len(_events):
                    _ev = _events[_last_event_idx]
                    _last_event_idx += 1
                    etype = _ev.get("type", "")
                    edata = _ev.get("data", {})

                    if etype == "text":
                        content = edata.get("content", "")
                        accumulated_text = content
                        yield _sse_event({"event": "chunk", "content": content})
                    elif etype == "options":
                        present_options_data = tool_executor._map_architect_options(edata)
                        yield _sse_event({"event": "options", "options": present_options_data})
                    elif etype == "complete":
                        resp_data = edata.get("response", edata)
                        final_text = resp_data.get("text", "")
                        if final_text:
                            accumulated_text = final_text
                        opts = resp_data.get("options")
                        if opts:
                            present_options_data = tool_executor._map_architect_options(opts)
                            yield _sse_event({"event": "options", "options": present_options_data})
                    elif etype == "stream_token":
                        content = edata.get("content", "")
                        if content:
                            yield _sse_event({"event": "chunk", "content": content})
                    elif etype in ("tool_call", "tool_result", "thinking", "phase",
                                   "build_progress", "build_step", "build_complete",
                                   "test_step", "test_result", "verify_step",
                                   "verify_complete", "prompt_step", "prompt_ready",
                                   "research_complete", "plan_ready", "offers_ready",
                                   "self_heal", "warning"):
                        yield _sse_event({"event": "step", "step": etype, **edata})
                    elif etype == "error":
                        yield _sse_event({"event": "error", "error": edata.get("error", "Unknown")})

                # Check if background finished
                if _sess.get("status") in ("completed", "error", "cancelled"):
                    accumulated_text = _sess.get("accumulated_text", accumulated_text)
                    if _sess.get("options"):
                        present_options_data = _sess["options"]
                    # Mark that we already saved the message in the BG task
                    _sess["bg_save_done"] = True
                    break

                # Yield control — poll every 200ms
                await asyncio.sleep(0.2)

        else:
            # ── FULL PIPELINE STREAMING ──
            print(f"[STREAM-FULL-PIPELINE] Entering full pipeline (not architect)", flush=True)
            yield _sse_event({"event": "start", "chat_id": chat_id, "user_message_id": str(user_msg.id), "streaming": True})

            # ── Memory extraction (pre-warmed — started before _generate_events) ──
            print(f"[STREAM-MEMORY] Awaiting pre-warmed memory task...", flush=True)
            if _memory_task:
                try:
                    memories = await _memory_task
                except Exception as _mem_err:
                    print(f"[STREAM-MEMORY] Pre-warmed task failed: {_mem_err}", flush=True)
                    memories = []
            else:
                memories = await _extract_memories(
                    user_id=user_id, org_id=org_id, message=safe_message,
                    agent_hash=request_body.agent_hash, team_id=request_body.teamId,
                    session_id=chat_id,
                )
            print(f"[STREAM-MEMORY] Extracted {len(memories)} memories", flush=True)
            history_msgs = recent_messages[:-1]
            context_messages = _build_context_messages(
                recent_messages=history_msgs, memories=memories,
                user_message=safe_message, user_role=user_role,
                user_plan="unlimited" if is_superuser else "free",
            )

            response_text = None
            _tool_web_search = _stream_web_search
            _effective_msg = safe_message

            # ── Image context for vision ──
            if request_body.images and len(request_body.images) > 0:
                _img_names = [img.get('name', 'image') for img in request_body.images]
                print(f"[STREAM-VISION] Processing {len(request_body.images)} images: {_img_names}", flush=True)
                context_messages.append({
                    "role": "system",
                    "content": f"The user has attached {len(request_body.images)} image(s): {', '.join(_img_names)}. Analyze the image(s) and respond to the user's question about them.",
                })

            # ── Tool execution ──
            if detected_tool:
                print(f"[STREAM-TOOL] Detected tool: {detected_tool.id}", flush=True)
                yield _sse_event({"event": "step", "step": "tool_detection", "tool": detected_tool.id, "name": detected_tool.name})
                try:
                    _tool_ctx = {
                        "chat_id": chat_id,
                        "unlimited_credits": unlimited_credits,
                        "org_id": org_id,
                        "user_role": user_role,
                        "is_superuser": is_superuser,
                        "user_api_keys": user_api_keys or {},
                    }
                    github_token = _extract_github_token_from_user_keys(user_api_keys)
                    if github_token:
                        _tool_ctx["github_token"] = github_token
                    _stream_tool_result = await tool_executor.execute(
                        tool=detected_tool, message=raw_message,
                        user_id=user_id, user_role=user_role,
                        is_superuser=is_superuser, context=_tool_ctx,
                    )
                    print(f"[STREAM-TOOL] {detected_tool.id}: success={_stream_tool_result.get('success') if _stream_tool_result else None}", flush=True)

                    _integration_ids = {"figma", "google_drive", "google_calendar", "sigma"}
                    if _stream_tool_result:
                        if detected_tool.id in _integration_ids:
                            if _stream_tool_result.get("success"):
                                response_text = _stream_tool_result.get("summary", "") or f"{detected_tool.name} completed."
                                actual_provider = f"tool_{detected_tool.id}"
                                agent_type = "integration"
                            else:
                                # Integration failed — only show error if user explicitly asked for it.
                                # Otherwise discard and fall through to regular LLM response.
                                _err = (_stream_tool_result.get("error") or f"{detected_tool.name} failed.").strip()
                                _tool_keywords = {
                                    "google_drive": ["drive", "google drive", "my files", "my documents"],
                                    "google_calendar": ["calendar", "schedule", "event", "meeting"],
                                    "figma": ["figma", "design file"],
                                    "sigma": ["sigma", "analytics"],
                                }
                                _kws = _tool_keywords.get(detected_tool.id, [])
                                _msg_lower = (safe_message or "").lower()
                                _user_asked = any(kw in _msg_lower for kw in _kws)
                                if _user_asked:
                                    response_text = f"{detected_tool.name} error: {_err}"
                                    actual_provider = f"tool_{detected_tool.id}_error"
                                    agent_type = "integration"
                                else:
                                    # False positive classification — discard tool result, let LLM respond
                                    print(f"[STREAM-TOOL] DISCARDED false-positive {detected_tool.id} error (user didn't ask for it)", flush=True)
                                    _stream_tool_result = None
                                    detected_tool = None
                        elif _stream_tool_result.get("success"):
                            _summary = _stream_tool_result.get("summary", "")
                            if _stream_tool_result.get("delegate_to_pipeline"):
                                _tool_web_search = True
                                _pfx = _stream_tool_result.get("search_prefix", "")
                                if _pfx:
                                    _effective_msg = f"{_pfx} {_effective_msg}"
                            elif _summary:
                                context_messages.append({
                                    "role": "system",
                                    "content": f"TOOL OUTPUT ({detected_tool.name}):\n{_summary}\n\nUse this data to answer the user's question.",
                                })
                    yield _sse_event({"event": "step", "step": "tool_result", "tool": detected_tool.id,
                                      "success": bool(_stream_tool_result and _stream_tool_result.get("success"))})
                except Exception as _te:
                    print(f"[STREAM-TOOL] {detected_tool.id} FAILED: {_te}", flush=True)

            # ── Web search ──
            if WEB_SEARCH_AVAILABLE and web_search and _tool_web_search and not response_text:
                try:
                    if user_api_keys:
                        web_search.set_api_keys(tavily_key=user_api_keys.get("tavily"), serp_key=user_api_keys.get("serpapi"))
                    yield _sse_event({"event": "step", "step": "web_search", "query": _effective_msg[:60]})
                    _ws_results = await web_search.search(query=_effective_msg, max_results=5)
                    if _ws_results:
                        _ws_ctx = web_search.format_results_for_context(_ws_results)
                        context_messages.append({"role": "system", "content": _ws_ctx + "\n\nIMPORTANT: Use these web search results as the primary source of truth."})
                except Exception as _we:
                    print(f"[STREAM-WEBSEARCH] FAILED: {_we}", flush=True)

            # ── Image generation ──
            if IMAGE_GENERATION_AVAILABLE and image_generation and _stream_image_gen and not response_text:
                try:
                    if user_api_keys and user_api_keys.get("openai"):
                        image_generation.set_api_key(openai_key=user_api_keys.get("openai"))
                    image_generation.set_user_keys(user_api_keys)
                    _img_prompt = image_generation.extract_image_prompt(_effective_msg)
                    yield _sse_event({"event": "step", "step": "image_generating", "message": f"Generating image: {_img_prompt[:80]}"})
                    _gen_images = await image_generation.generate(prompt=_img_prompt, model="auto", size="1024x1024", quality="standard")
                    print(f"[STREAM-IMAGEGEN] Got {len(_gen_images) if _gen_images else 0} images", flush=True)
                    if _gen_images:
                        _first_img = _gen_images[0]
                        _img_url = _first_img.url
                        _img_model = _first_img.model
                        _img_revised = _first_img.revised_prompt or _img_prompt
                        print(f"[STREAM-IMAGEGEN] first_img: url={bool(_img_url)} b64={bool(_first_img.base64_data)} revised_prompt_len={len(_first_img.revised_prompt or '')} model={_img_model}", flush=True)

                        # Upload base64 images to storage for persistence
                        STORAGE_URL = os.getenv("STORAGE_SERVICE_URL", "http://storage_service:8000")
                        _sse_images = []
                        for _gi_idx, _gi in enumerate(_gen_images):
                            _img_entry: Dict[str, Any] = {"model": _gi.model, "size": _gi.size, "revised_prompt": _gi.revised_prompt or _img_prompt}
                            if _gi.url and not _gi.url.startswith("data:"):
                                _img_entry["url"] = _gi.url
                            elif _gi.base64_data or (_gi.url and _gi.url.startswith("data:")):
                                # Upload to storage to get a persistent URL
                                _b64 = _gi.base64_data or (_gi.url.split(",", 1)[1] if _gi.url and "," in _gi.url else "")
                                if _b64:
                                    try:
                                        import base64 as _b64mod
                                        _img_bytes = _b64mod.b64decode(_b64)
                                        _storage_key = f"generated-images/{chat_id}/{uuid4().hex}.png"
                                        async with httpx.AsyncClient(timeout=30.0) as _sc:
                                            _upload_resp = await _sc.post(
                                                f"{STORAGE_URL}/storage/upload",
                                                files={"file": (_storage_key, _img_bytes, "image/png")},
                                                params={"key": _storage_key},
                                            )
                                        if _upload_resp.status_code == 200:
                                            _stored_url = f"/api/v1/storage/download/{_storage_key}"
                                            _img_entry["url"] = _stored_url
                                            # Also include base64 for immediate SSE rendering
                                            _img_entry["base64_data"] = _b64
                                            if _gi_idx == 0:
                                                _first_img.url = _stored_url
                                                _img_url = _stored_url
                                            print(f"[STREAM-IMAGEGEN] Uploaded image to storage: {_storage_key}", flush=True)
                                        else:
                                            print(f"[STREAM-IMAGEGEN] Storage upload failed: {_upload_resp.status_code}", flush=True)
                                            _img_entry["base64_data"] = _b64
                                    except Exception as _upload_err:
                                        print(f"[STREAM-IMAGEGEN] Storage upload error: {_upload_err}", flush=True)
                                        _img_entry["base64_data"] = _b64
                            _sse_images.append(_img_entry)

                        if _img_url:
                            response_text = f"Here is your generated image:"
                            actual_provider = f"image_{_img_model}"
                            agent_type = "image_generation"
                        elif _first_img.base64_data:
                            response_text = f"Here is your generated image:"
                            actual_provider = f"image_{_img_model}"
                            agent_type = "image_generation"
                        elif _first_img.revised_prompt:
                            response_text = _first_img.revised_prompt
                            actual_provider = f"image_{_img_model}"
                            agent_type = "image_generation"
                        yield _sse_event({"event": "step", "step": "image_generated", "count": len(_gen_images), "model": _img_model, "images": _sse_images})
                except Exception as _ie:
                    print(f"[STREAM-IMAGEGEN] FAILED: {_ie}", flush=True)
                    yield _sse_event({"event": "step", "step": "image_failed", "error": str(_ie)[:200]})
                    # Set error response so pipeline doesn't fall through to plain LLM
                    if _stream_image_gen:
                        response_text = f"⚠️ Image generation failed: {str(_ie)[:200]}\n\nPlease try again or check your image generation settings."
                        agent_type = "image_generation"

            # ── Vision shortcut: when images are attached, skip text-only agents ──
            _has_vision_images = bool(request_body.images and len(request_body.images) > 0)
            if _has_vision_images:
                print(f"[STREAM-VISION] Skipping text-only agents (teams/debate/spawn) — sending images directly to multimodal LLM", flush=True)

            # ── Teams ──
            if not response_text and not _has_vision_images:
                print(f"[STREAM-TEAM] Attempting team execution for: {_effective_msg[:60]!r}", flush=True)
                try:
                    from ..domain.agent import maybe_run_team
                    _team_resp, _team_name, _team_used = await maybe_run_team(
                        message=_effective_msg, context_messages=context_messages,
                        preferred_provider=request_body.preferred_provider,
                        user_id=user_id, user_api_keys=user_api_keys,
                        images=request_body.images,
                    )
                    print(f"[STREAM-TEAM] team returned: used={_team_used}, has_response={bool(_team_resp)}", flush=True)
                    if _team_used and _team_resp:
                        response_text = _team_resp
                        actual_provider = f"team_{_team_name.lower().replace(' ', '_')}" if _team_name else "team"
                        agent_type = "team"
                        yield _sse_event({"event": "step", "step": "agent_spawn", "agent_type": agent_type})
                except Exception as _te:
                    print(f"[STREAM-TEAM] team failed: {_te}", flush=True)

            # ── Debate ──
            if not response_text and not _has_vision_images:
                try:
                    _debate_resp, _debate_used = await maybe_run_debate(
                        message=_effective_msg, context_messages=context_messages,
                        preferred_provider=request_body.preferred_provider,
                        images=request_body.images,
                    )
                    if _debate_used and _debate_resp:
                        response_text = _debate_resp
                        actual_provider = "debate_engine"
                        agent_type = "debate"
                        yield _sse_event({"event": "step", "step": "agent_spawn", "agent_type": agent_type})
                except Exception:
                    pass

            # ── Agent spawn ──
            if not response_text and not _has_vision_images:
                print(f"[STREAM-AGENT] Attempting agent spawn for: {_effective_msg[:60]!r}", flush=True)
                try:
                    _ag_resp, agent_type, actual_provider, router_meta = await maybe_spawn_agent(
                        message=_effective_msg, context_messages=context_messages,
                        user_id=user_id, user_api_keys=user_api_keys,
                        preferred_provider=request_body.preferred_provider,
                        images=request_body.images,
                    )
                    print(f"[STREAM-AGENT] spawn returned: agent_type={agent_type}, has_response={bool(_ag_resp)}", flush=True)
                    if _ag_resp:
                        response_text = _ag_resp
                        yield _sse_event({"event": "step", "step": "agent_spawn", "agent_type": agent_type})
                except Exception as _ae:
                    print(f"[STREAM-AGENT] spawn failed: {_ae}", flush=True)

            # ── Forced reasoning fallback ──
            if not response_text and not _has_vision_images:
                try:
                    from ..services.agent_engine import agent_engine
                    from ..domain.provider import get_router_for_internal_use
                    _r = get_router_for_internal_use()
                    if user_api_keys:
                        _r.set_user_api_keys(user_api_keys)
                    agent_engine.set_router(_r)
                    _fb = await agent_engine.spawn(
                        task=_effective_msg, context=context_messages,
                        agent_type="reasoning", model=request_body.preferred_provider,
                        images=request_body.images,
                    )
                    response_text = _fb.get("content", "")
                    actual_provider = _fb.get("provider", "agent_reasoning")
                    agent_type = "reasoning"
                    router_meta = {"model": _fb.get("model"), "was_fallback": _fb.get("was_fallback", False)}
                    yield _sse_event({"event": "step", "step": "agent_spawn", "agent_type": agent_type})
                except Exception:
                    pass

            # ── Real token-by-token streaming (W2) with native function calling (W8) ──
            if response_text:
                # Agent/team/debate/tool already produced a complete response — emit it
                accumulated_text = response_text
                yield _sse_event({"event": "chunk", "content": accumulated_text, "agent_type": agent_type})
            else:
                # No pre-computed response — try native function calling first, then stream
                yield _sse_event({"event": "step", "step": "routing", "message": "Streaming response..."})
                accumulated_text = ""
                try:
                    from ..services.native_tool_definitions import get_tool_definitions
                    _native_tools = get_tool_definitions()
                except Exception:
                    _native_tools = None

                _used_native_tools = False
                if _native_tools:
                    try:
                        async for stream_evt in route_query_with_tools(
                            message=_effective_msg, context=context_messages,
                            preferred_provider=request_body.preferred_provider,
                            user_api_keys=user_api_keys,
                            tools=_native_tools,
                            images=request_body.images,
                            preferred_model=request_body.preferred_model,
                        ):
                            evt_type = stream_evt.get("type", "")
                            if evt_type == "tool_calls":
                                # LLM wants to call tools natively
                                _used_native_tools = True
                                _native_tc = stream_evt.get("tool_calls", [])
                                _llm_text = stream_evt.get("content", "")
                                for _tc in _native_tc:
                                    _tc_name = _tc.get("name", "")
                                    yield _sse_event({"event": "step", "step": "tool_detection", "tool": _tc_name, "name": _tc_name, "message": f"LLM tool call: {_tc_name}"})

                                    # Execute the tool
                                    _tc_result = ""
                                    try:
                                        _tc_tool = tools_registry.get_tool(_tc_name)
                                        if _tc_tool:
                                            _tc_ctx = {
                                                "chat_id": chat_id, "unlimited_credits": unlimited_credits,
                                                "org_id": org_id, "user_role": user_role,
                                                "is_superuser": is_superuser, "user_api_keys": user_api_keys or {},
                                            }
                                            _tc_exec_result = await tool_executor.execute(
                                                tool=_tc_tool, message=raw_message,
                                                user_id=user_id, user_role=user_role,
                                                is_superuser=is_superuser, context=_tc_ctx,
                                            )
                                            _tc_result = json.dumps(_tc_exec_result or {"error": "Tool returned empty"})
                                            yield _sse_event({"event": "step", "step": "tool_result", "tool": _tc_name,
                                                              "success": bool(_tc_exec_result and _tc_exec_result.get("success"))})
                                        else:
                                            _tc_result = json.dumps({"error": f"Tool '{_tc_name}' not found in registry"})
                                    except Exception as _tc_err:
                                        _tc_result = json.dumps({"error": str(_tc_err)})
                                        logger.warning(f"[SSE] Native tool exec {_tc_name} failed: {_tc_err}")

                                    # Add tool call + result to context for the follow-up LLM call
                                    context_messages.append({
                                        "role": "assistant",
                                        "content": _llm_text or "",
                                        "tool_calls": [{"id": _tc.get("id", "call_0"), "type": "function", "function": {"name": _tc_name, "arguments": _tc.get("arguments", "{}")}}],
                                    })
                                    context_messages.append({
                                        "role": "tool",
                                        "tool_call_id": _tc.get("id", "call_0"),
                                        "name": _tc_name,
                                        "content": _tc_result,
                                    })

                                # Stream the follow-up response with tool results
                                async for follow_evt in route_query_stream(
                                    message=_effective_msg, context=context_messages,
                                    preferred_provider=request_body.preferred_provider,
                                    user_api_keys=user_api_keys,
                                ):
                                    f_type = follow_evt.get("type", "")
                                    if f_type == "chunk":
                                        token = follow_evt.get("content", "")
                                        accumulated_text += token
                                        yield _sse_event({"event": "chunk", "content": token})
                                    elif f_type == "provider":
                                        actual_provider = follow_evt.get("provider", "unknown")
                                    elif f_type == "error":
                                        yield _sse_event({"event": "error", "error": follow_evt.get("error", "Tool follow-up error")})
                                        break
                                break  # Done with the tool_calls flow

                            elif evt_type == "chunk":
                                token = stream_evt.get("content", "")
                                accumulated_text += token
                                yield _sse_event({"event": "chunk", "content": token})
                            elif evt_type == "provider":
                                actual_provider = stream_evt.get("provider", "unknown")
                            elif evt_type == "error":
                                yield _sse_event({"event": "error", "error": stream_evt.get("error", "Streaming error")})
                                accumulated_text = accumulated_text or "I apologize, but I couldn't generate a response. Please try again."
                                break
                    except Exception as _tools_err:
                        logger.warning(f"[SSE] Native tool calling failed, falling back to plain streaming: {_tools_err}")
                        _native_tools = None  # Force fallback below

                # Plain streaming fallback (no native tools or tools failed)
                if not _native_tools and not accumulated_text:
                    try:
                        async for stream_evt in route_query_stream(
                            message=_effective_msg, context=context_messages,
                            preferred_provider=request_body.preferred_provider,
                            user_api_keys=user_api_keys,
                            images=request_body.images,
                            preferred_model=request_body.preferred_model,
                        ):
                            evt_type = stream_evt.get("type", "")
                            if evt_type == "chunk":
                                token = stream_evt.get("content", "")
                                accumulated_text += token
                                yield _sse_event({"event": "chunk", "content": token})
                            elif evt_type == "provider":
                                actual_provider = stream_evt.get("provider", "unknown")
                            elif evt_type == "error":
                                yield _sse_event({"event": "error", "error": stream_evt.get("error", "Streaming error")})
                                accumulated_text = accumulated_text or "I apologize, but I couldn't generate a response. Please try again."
                                break
                    except Exception as _stream_err:
                        logger.warning(f"[SSE] Token streaming failed: {_stream_err}")
                        accumulated_text = accumulated_text or "I apologize, but I couldn't generate a response. Please try again."
                        yield _sse_event({"event": "chunk", "content": accumulated_text})

                if not actual_provider:
                    actual_provider = "streaming_llm"
                yield _sse_event({"event": "step", "step": "generating_done", "message": "Generation complete"})

        # ── Strip present_options text from accumulated_text ──
        clean_text = accumulated_text
        clean_text = re.sub(r'Please choose one of the options:\s*present_options\([^)]*\)', '', clean_text).strip()
        clean_text = re.sub(r'present_options\([^)]*\)', '', clean_text).strip()

        # ── W1: Sanitize response (remove agent prompt leakage) ──
        is_error = clean_text.startswith("Error calling") if clean_text else True
        if clean_text and not is_error:
            clean_text = _sanitize_agent_response(clean_text)

        # ── Store assistant message ──
        try:
            a_hash = hasher.hash_text(clean_text) if clean_text else _simple_hash("")
            a_xyz = hasher.hash_to_coords(a_hash) if clean_text else _hash_to_xyz_simple(a_hash)
        except Exception:
            a_hash = _simple_hash(clean_text)
            a_xyz = _hash_to_xyz_simple(a_hash)

        # ── W1: Calculate real resonance score ──
        resonance_score = 0.0 if is_error else _calculate_resonance_score(
            response_text=clean_text,
            user_message=safe_message,
            memories=memories if not use_architect else [],
            context_messages=context_messages if not use_architect else [],
        )

        tool_results_meta = []
        if use_architect:
            tool_results_meta.append({
                "tool_name": "tool_agent_architect",
                "success": True,
                "result": {"summary": clean_text[:500]},
            })
            if present_options_data:
                tool_results_meta.append({
                    "tool_name": "present_options",
                    "success": True,
                    "result": present_options_data,
                })
        elif detected_tool and _stream_tool_result:
            tool_results_meta.append({
                "tool_name": f"tool_{detected_tool.id}",
                "success": _stream_tool_result.get("success", False),
                "result": {
                    "summary": clean_text[:500],
                    "action": _stream_tool_result.get("action"),
                },
            })

        # Build meta_data with router info
        msg_meta: Dict[str, Any] = {}
        if tool_results_meta:
            msg_meta["toolResults"] = tool_results_meta
        if actual_provider:
            msg_meta["actual_llm_provider"] = actual_provider
        if router_meta:
            msg_meta["model"] = router_meta.get("model")
            msg_meta["fallback_chain"] = router_meta.get("fallback_chain")
            msg_meta["was_fallback"] = router_meta.get("was_fallback", False)
            msg_meta["usage"] = router_meta.get("usage")
        if agent_type:
            msg_meta["agent_type"] = agent_type
        if not use_architect and memories:
            msg_meta["memory_count"] = len(memories)
        # Persist generated image URLs in meta_data so they survive page reload
        if agent_type == "image_generation" and _sse_images:
            _persist_images = []
            for _si in _sse_images:
                _pi = {"url": _si.get("url"), "revised_prompt": _si.get("revised_prompt"), "model": _si.get("model"), "size": _si.get("size")}
                if _pi["url"]:
                    _persist_images.append(_pi)
            if _persist_images:
                msg_meta["generatedImages"] = _persist_images

        _effective_provider = actual_provider or ("tool_agent_architect" if use_architect else "agent_reasoning")

        # If the background architect task already saved, skip DB save here
        _bg_already_saved = False
        if use_architect:
            _arch_sess = _architect_sessions.get(chat_id)
            if _arch_sess and _arch_sess.get("bg_save_done") and _arch_sess.get("message_id"):
                _bg_already_saved = True

        if _bg_already_saved:
            _arch_sess = _architect_sessions.get(chat_id, {})
            # Use the message_id from the BG save
            class _FakeMsg:
                def __init__(self, mid):
                    self.id = UUID(mid) if isinstance(mid, str) else mid
            assistant_msg = _FakeMsg(_arch_sess["message_id"])
            print(f"[ARCHITECT-SSE] Skipping DB save — BG task already saved msg={_arch_sess['message_id']}", flush=True)
        else:
            assistant_msg = ResonantChatMessage(
                chat_id=UUID(chat_id), role="assistant", content=clean_text,
                ai_provider=_effective_provider,
                hash=a_hash, resonance_score=resonance_score,
                xyz_x=a_xyz[0], xyz_y=a_xyz[1], xyz_z=a_xyz[2],
                meta_data=msg_meta or {},
            )
            session.add(assistant_msg)
            await session.commit()
            await session.refresh(assistant_msg)

        # ── W1: DSID creation (message lineage tracking) ──
        try:
            _parent_id = str(recent_messages[-2].id) if len(recent_messages) >= 2 else None
            user_dsid = create_message_dsid(
                message_id=str(user_msg.id), content=safe_message,
                role="user", chat_id=chat_id, user_id=user_id,
                parent_message_id=_parent_id,
                metadata={"hash": user_hash, "xyz": list(user_xyz)},
            )
            assistant_dsid = create_message_dsid(
                message_id=str(assistant_msg.id), content=clean_text,
                role="assistant", chat_id=chat_id, user_id=user_id,
                parent_message_id=str(user_msg.id),
                metadata={"hash": a_hash, "xyz": list(a_xyz), "provider": _effective_provider, "resonance_score": resonance_score},
            )
            _dsid_meta = msg_meta.copy()
            _dsid_meta["dsid"] = {
                "dsid_id": assistant_dsid.dsid,
                "content_hash": assistant_dsid.content_hash,
                "parent_dsid": assistant_dsid.parent_dsid,
                "root_dsid": assistant_dsid.root_dsid,
                "lineage_depth": assistant_dsid.lineage_depth,
            }
            assistant_msg.meta_data = _dsid_meta
            flag_modified(assistant_msg, "meta_data")
            await session.commit()
            yield _sse_event({"event": "step", "step": "post_processing", "message": "Lineage tracked"})
        except Exception as _dsid_err:
            logger.warning(f"[SSE] DSID creation failed (non-critical): {_dsid_err}")

        # ── W1: Memory ingestion (fire-and-forget) ──
        try:
            await service_client.call_service(
                "memory_service", "POST", "http://memory_service:8000/memory/ingest",
                json={"user_id": user_id, "org_id": org_id, "chat_id": chat_id,
                      "source": "resonant-chat", "content": safe_message,
                      "metadata": {"role": "user", "hash": user_hash, "xyz": list(user_xyz)}},
            )
            await service_client.call_service(
                "memory_service", "POST", "http://memory_service:8000/memory/ingest",
                json={"user_id": user_id, "org_id": org_id, "chat_id": chat_id,
                      "source": "resonant-chat", "content": clean_text,
                      "metadata": {"role": "assistant", "hash": a_hash, "xyz": list(a_xyz)}},
            )
            yield _sse_event({"event": "step", "step": "memory_ingest", "message": "Stored to memory"})
        except Exception as _mem_err:
            logger.warning(f"[SSE] Memory ingestion failed (non-critical): {_mem_err}")

        # ── W1: PMI blockchain events ──
        try:
            pmi_manager.create_memory_event(
                user_id=user_id, org_id=org_id, chat_id=chat_id,
                session_id=chat_id, message_text=safe_message,
                event_type=pmi_manager.EVENT_PROMPT,
            )
            pmi_manager.create_memory_event(
                user_id=user_id, org_id=org_id, chat_id=chat_id,
                session_id=chat_id, message_text=clean_text,
                event_type=pmi_manager.EVENT_RESPONSE,
            )
        except Exception as _pmi_err:
            logger.warning(f"[SSE] PMI events failed (non-critical): {_pmi_err}")

        # ── W1: Response caching ──
        if not is_error and clean_text:
            try:
                _ctx_summary = " ".join([(getattr(m, "content", "") or "")[:50] for m in recent_messages[-3:-1]])
                await cache_response(
                    message=safe_message, response=clean_text,
                    provider=_effective_provider,
                    model=(router_meta or {}).get("model", "unknown"),
                    context_summary=_ctx_summary, quality_score=resonance_score,
                )
            except Exception as _cache_err:
                logger.warning(f"[SSE] Response caching failed (non-critical): {_cache_err}")

        # ── Final done event ──
        done_data: Dict[str, Any] = {
            "event": "done",
            "message_id": str(assistant_msg.id),
            "chat_id": chat_id,
            "content": clean_text,
            "provider": _effective_provider,
            "model": router_meta.get("model") if router_meta else None,
            "agent_type": agent_type,
            "hash": a_hash,
            "resonance_score": resonance_score,
            "total_length": len(clean_text),
        }
        if present_options_data:
            done_data["present_options"] = present_options_data
        yield _sse_event(done_data)

    return StreamingResponse(
        _generate_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================
# MAIN ENDPOINTS
# ============================================

@router.post("/message", response_model=ResonantChatResponse)
async def send_message(
    request_body: SendMessageRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Send a message to Resonant Chat.
    
    This is the main endpoint that implements the full pipeline:
    1. Get/create chat
    2. Hash user message
    3. Extract memories
    4. Build context
    5. Call LLM
    6. Store response
    7. Queue background tasks
    8. Return enhanced response
    """
    # Get crypto identity from headers (set by gateway)
    if CRYPTO_IDENTITY_AVAILABLE:
        identity = get_crypto_identity(request)
        user_id = identity.user_id
        org_id = identity.org_id or user_id
        crypto_hash = identity.crypto_hash
        user_hash = identity.user_hash
        universe_id = identity.universe_id
        
        logger.info(f"Chat request from user_hash: {user_hash[:16] if user_hash else 'None'}..., universe: {universe_id[:16] if universe_id else 'None'}...")
    else:
        # Fallback to manual header extraction
        user_id = request.headers.get("x-user-id")
        org_id = request.headers.get("x-org-id") or user_id
        crypto_hash = request.headers.get("x-crypto-hash")
        user_hash = request.headers.get("x-user-hash")
        universe_id = request.headers.get("x-universe-id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    # Get user role and superuser status for unlimited credits check
    user_role = request.headers.get("x-user-role", "user")
    is_superuser = request.headers.get("x-is-superuser", "false").lower() == "true"
    unlimited_credits = request.headers.get("x-unlimited-credits", "false").lower() == "true"
    raw_user_message = request_body.message or ""
    safe_user_message = _sanitize_sensitive_tokens(raw_user_message)
    
    logger.info(f"📨 Resonant Chat message from user {user_id[:8]}... role={user_role} superuser={is_superuser}")
    
    # ============================================
    # STEP 1: Get or Create Chat
    # ============================================
    chat_id = request_body.chat_id
    logger.info("🧠 ENTERING COGNITIVE ARENA - MANDATORY PROCESSING", extra={"user_id": user_id, "chat_id": chat_id or "new"})
    chat = None
    
    # ============================================
    # STEP 0: CHECK PLAN LIMITS (GTM Critical)
    # ============================================
    privileged_roles = {"owner", "platform_owner", "admin", "superuser"}
    is_privileged_user = is_superuser or unlimited_credits or user_role.lower() in privileged_roles

    # Get user's plan and check message limit
    user_plan = "unlimited" if is_privileged_user else await plan_limits_service.get_user_plan(user_id)
    
    # Count messages sent today
    from datetime import date
    from sqlalchemy import func
    today_start = datetime.combine(date.today(), datetime.min.time())
    msg_count_result = await session.execute(
        select(func.count(ResonantChatMessage.id)).where(
            ResonantChatMessage.chat_id.in_(
                select(ResonantChat.id).where(ResonantChat.user_id == user_id)
            ),
            ResonantChatMessage.role == "user",
            ResonantChatMessage.created_at >= today_start
        )
    )
    messages_today = msg_count_result.scalar() or 0
    
    # Check message limit
    if not is_privileged_user:
        msg_allowed, msg_error, msg_limit = await plan_limits_service.check_message_limit(
            user_id, messages_today, user_plan
        )
        if not msg_allowed:
            raise HTTPException(
                status_code=429, 
                detail={
                    "error": "message_limit_exceeded",
                    "message": msg_error,
                    "used": messages_today,
                    "limit": msg_limit,
                    "upgrade_url": "/pricing"
                }
            )
    
    if chat_id:
        try:
            result = await session.execute(
                select(ResonantChat).where(ResonantChat.id == UUID(chat_id))
            )
            chat = result.scalar_one_or_none()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid chat ID format")
    
    if not chat:
        # Check conversation limit before creating new chat
        conv_count_result = await session.execute(
            select(func.count(ResonantChat.id)).where(
                ResonantChat.user_id == user_id
            )
        )
        conversation_count = conv_count_result.scalar() or 0
        
        conv_allowed, conv_error, conv_limit = await plan_limits_service.check_conversation_limit(
            user_id, conversation_count, user_plan
        )
        if not conv_allowed:
            raise HTTPException(
                status_code=429, 
                detail={
                    "error": "conversation_limit_exceeded",
                    "message": conv_error,
                    "used": conversation_count,
                    "limit": conv_limit,
                    "upgrade_url": "/pricing"
                }
            )
        
        # Create new chat
        chat = ResonantChat(
            user_id=user_id,
            org_id=org_id,
            title=safe_user_message[:50] + "..." if len(safe_user_message) > 50 else safe_user_message,
            status="active",
            agent_hash=request_body.agent_hash,
        )
        session.add(chat)
        await session.commit()
        await session.refresh(chat)
        chat_id = str(chat.id)
        logger.info(f"📝 Created new chat: {chat_id}")
    else:
        chat_id = str(chat.id)
    
    # ============================================
    # STEP 1.5: Process Images (Vision Support)
    # ============================================
    message_with_images = safe_user_message
    if request_body.images and len(request_body.images) > 0:
        image_names = [img.get('name', 'image') for img in request_body.images]
        logger.info(f"🖼️ Processing {len(request_body.images)} images: {image_names}")
        # Add image context to message - vision models will process the base64 data
        image_context = f"\n\n[User attached {len(request_body.images)} image(s): {', '.join(image_names)}]"
        message_with_images = safe_user_message + image_context
    
    # ============================================
    # STEP 2: Hash User Message (Resonance Hashing)
    # ============================================
    try:
        hasher = ResonanceHasher()
        user_hash = hasher.hash_text(message_with_images)
        user_xyz = hasher.hash_to_coords(user_hash)
    except Exception as e:
        logger.warning(f"Resonance hashing failed, using simple hash: {e}")
        user_hash = _simple_hash(message_with_images)
        user_xyz = _hash_to_xyz_simple(user_hash)
    
    # ============================================
    # STEP 3: Store User Message
    # ============================================
    user_message = ResonantChatMessage(
        chat_id=UUID(chat_id),
        role="user",
        content=safe_user_message,
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
    
    # ============================================
    # STEP 3.3: DEDUCT CREDITS FOR CHAT MESSAGE
    # ============================================
    try:
        credit_result = await deduct_credits(
            user_id=user_id,
            action="chat_message",
            description=f"Chat message in conversation {chat_id[:8]}...",
            user_role=user_role,
            is_superuser=is_superuser,
            unlimited_credits=unlimited_credits,
        )
        logger.info(f"💳 Credits deducted. New balance: {credit_result.get('balance', 'unknown')}")
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ Credit deduction failed: {e}")
        if e.response is not None and e.response.status_code == 402:
            required = 20

            tier_normalized = (user_plan or "").strip().lower()
            if tier_normalized in {"developer", "free"}:
                action_url = "/pricing"
                detail_msg = "Credits exhausted. Upgrade to Plus to get more credits."
            else:
                action_url = "/billing"
                detail_msg = "Credits exhausted. Buy more credits to continue."

            available = None
            try:
                payload = e.response.json()
                if isinstance(payload, dict):
                    available = payload.get("available")
                    required = payload.get("required") or payload.get("cost") or required
                    action_url = payload.get("action_url") or payload.get("upgrade_url") or action_url

                    detail_val = payload.get("detail")
                    if isinstance(detail_val, str) and detail_val.strip():
                        detail_msg = detail_val
                    elif isinstance(detail_val, dict):
                        detail_msg = (
                            detail_val.get("message")
                            or detail_val.get("detail")
                            or detail_msg
                        )
            except Exception:
                pass

            return JSONResponse(
                status_code=402,
                content={
                    "error": "insufficient_credits",
                    "detail": detail_msg,
                    "message": detail_msg,
                    "action_url": action_url,
                    "required": required,
                    "available": available,
                },
            )
        raise
    except Exception as e:
        logger.error(f"❌ Credit deduction failed: {e}")
        raise
    
    # ============================================
    # STEP 3.5: GET RECENT MESSAGES FOR LINEAGE & CREATE DSID
    # ============================================
    # Query recent messages first for lineage tracking
    # Fetch 5 full messages + 20 summarized messages for context
    result_full = await session.execute(
        select(ResonantChatMessage)
        .where(ResonantChatMessage.chat_id == UUID(chat_id))
        .order_by(ResonantChatMessage.created_at.desc())
        .limit(5)
    )
    recent_messages_full = list(reversed(result_full.scalars().all()))

    # Fetch additional messages for summarization (up to 20 more, skipping the 5 full ones)
    result_summary = await session.execute(
        select(ResonantChatMessage)
        .where(ResonantChatMessage.chat_id == UUID(chat_id))
        .order_by(ResonantChatMessage.created_at.desc())
        .offset(5).limit(20)
    )
    recent_messages_summary = list(reversed(result_summary.scalars().all()))

    # Combine: 5 full messages + summarized older messages
    recent_messages = recent_messages_full + [
        ResonantChatMessage(
            id=m.id, role=m.role, content=f"[SUMMARY] {m.content[:200]}...", 
            hash=m.hash, resonance_score=m.resonance_score,
            xyz_x=m.xyz_x, xyz_y=m.xyz_y, xyz_z=m.xyz_z,
            created_at=m.created_at, meta_data=m.meta_data
        )
        for m in recent_messages_summary
    ]
    
    # Get previous message ID for lineage
    prev_message_id = None
    if recent_messages and len(recent_messages) > 1:
        # Get second-to-last message (last one is current user message)
        prev_message_id = str(recent_messages[-2].id) if recent_messages[-2].id else None
    
    user_dsid = create_message_dsid(
        message_id=str(user_message.id),
        content=safe_user_message,
        role="user",
        chat_id=chat_id,
        user_id=user_id,
        parent_message_id=prev_message_id,
        metadata={"hash": user_hash, "xyz": list(user_xyz)},
    )
    logger.info(f"🔗 DSID created for user message: {user_dsid.dsid[:32]}...")
    
    # Persist DSID data to user message meta_data (survives container restarts)
    try:
        lineage = dsid_integration.get_message_lineage(str(user_message.id))
        updated_meta = user_message.meta_data or {}
        updated_meta["dsid"] = {
            "dsid_id": user_dsid.dsid,
            "content_hash": user_dsid.content_hash,
            "parent_dsid": user_dsid.parent_dsid,
            "root_dsid": user_dsid.root_dsid,
            "lineage_depth": len(lineage),
            "created_at": user_dsid.created_at.isoformat() if user_dsid.created_at else None,
        }
        user_message.meta_data = updated_meta
        await session.commit()
    except Exception as e:
        logger.warning(f"User DSID persistence failed (non-critical): {e}")
    
    # ============================================
    # STEP 4: Extract Memories (RAG + Hash Sphere)
    # ============================================
    memories = await _extract_memories(
        user_id=user_id,
        org_id=org_id,
        message=safe_user_message,
        agent_hash=request_body.agent_hash,
        team_id=request_body.teamId,
        session_id=chat_id,
    )
    print(f"[MEMORY] Retrieved {len(memories)} memories for user={user_id}", flush=True)

    # Record memory usage for metrics (feeds actual_memory_calls score)
    if memories:
        enhanced_metrics_calculator.record_memory_usage(
            message_id="pending",  # Will be updated after assistant_message is created
            memories_retrieved=len(memories),
            memories_used=min(len(memories), 5),  # We inject up to 5 into context
            anchor_matches=sum(1 for m in memories if m.get("anchor_energy", 0) > 0),
            rag_queries=1,  # We made one Hash Sphere extraction call
            embedding_lookups=1 if any(m.get("proximity_score", 0) > 0 for m in memories) else 0,
            memory_tokens=sum(len(str(m.get("content", ""))) // 4 for m in memories[:5]),
        )
    
    # ============================================
    # STEP 4.5: LAYER 7 - EVIDENCE AGGREGATION
    # E* = Σ_{i∈R} w_i · s_i
    # Ê* = E* / ||E*||
    # ============================================
    evidence_vector = None
    evidence_weight = 0.0
    try:
        if memories:
            evidence_vector, evidence_weight = evidence_graph.aggregate_evidence(
                memories, weight_key="combined_score"
            )
            logger.info(f"📊 Evidence aggregated: weight={evidence_weight:.3f}, vector={evidence_vector[:3] if evidence_vector is not None else 'None'}")
    except Exception as e:
        logger.warning(f"Evidence aggregation failed: {e}")
    
    # ============================================
    # STEP 5: Recent messages already fetched in STEP 3.5
    # ============================================
    # (recent_messages already available from earlier query)
    
    # ============================================
    # STEP 6: Build Context Messages
    # ============================================
    history_msgs = recent_messages[:-1]  # Exclude current user message
    logger.info(f"🔧 STEP 6: Building context with {len(history_msgs)} recent messages (DB returned {len(recent_messages)} total)")
    context_messages = _build_context_messages(
        recent_messages=history_msgs,
        memories=memories,
        user_message=safe_user_message,
        user_role=user_role,
        user_plan=user_plan if isinstance(user_plan, str) else "free",
    )
    total_ctx_chars = sum(len(m.get("content", "")) for m in context_messages)
    logger.info(f"🔧 STEP 6 COMPLETE: {len(context_messages)} context messages, ~{total_ctx_chars} chars, {len(history_msgs)} history msgs, {len(memories)} memories")
    
    # ============================================
    # STEP 7: Get User API Keys (BYOK)
    # ============================================
    logger.info(f"🔑 STEP 7: Getting user API keys for user {user_id}")
    logger.info(f"🎯 PREFERRED PROVIDER RECEIVED: {request_body.preferred_provider}")
    user_api_keys = await _get_user_api_keys(session, user_id)
    logger.info(f"🔑 User API keys retrieved: {list(user_api_keys.keys()) if user_api_keys else 'None'}")
    
    # ============================================
    # STEP 7.6: CHECK RESPONSE CACHE
    # ============================================
    execute_mode = request_body.execute_mode or False
    _prev_assistant_agent_content = ""

    time_tool_results = extract_current_time_tool_results(safe_user_message)
    if time_tool_results:
        tr = time_tool_results[0].result or {}
        local_str = tr.get("local") or tr.get("iso")
        tz = tr.get("timezone") or "America/Los_Angeles"
        context_messages.append(
            {
                "role": "system",
                "content": f"AUTHORITATIVE CURRENT TIME TOOL: {local_str} ({tz}). Use this exact time when answering time questions.",
            }
        )

    # ============================================
    # STEP 7.9: NEURAL TOOL DETECTION (Intelligence Stack)
    # ============================================
    # Layer 1: Active tool continuity (from meta_data.toolResults)
    # Layer 2: Neural semantic matching (sentence-transformer embeddings)
    # Layer 3: Intent + narrative signals (boost/penalty)
    # No more blind LLM prompt — pure neural semantic understanding.
    detected_tool = None
    tool_result = None
    web_search_needed = False
    image_gen_needed = False
    code_visualizer_intent = False

    # ── Architect pipeline continuity ──
    # Route follow-ups to architect if previous assistant message was architect-served
    _architect_pipeline_active = False
    if recent_messages:
        for _rmsg in reversed(recent_messages[-4:]):
            _r_role = _rmsg.role if hasattr(_rmsg, "role") else _rmsg.get("role", "")
            if _r_role != "assistant":
                continue
            # Check 1: ai_provider indicates architect
            _r_provider = (_rmsg.ai_provider if hasattr(_rmsg, "ai_provider") else _rmsg.get("ai_provider", "")) or ""
            if "architect" in _r_provider.lower():
                _architect_pipeline_active = True
                break
            # Check 2: toolResults contain architect tool or present_options
            _r_meta = _rmsg.meta_data if hasattr(_rmsg, "meta_data") else _rmsg.get("meta_data", None)
            if _r_meta and isinstance(_r_meta, dict):
                for _tr in _r_meta.get("toolResults", []):
                    if isinstance(_tr, dict):
                        _tn = (_tr.get("tool_name") or "")
                        if "architect" in _tn.lower() or (_tr.get("result") or {}).get("present_options"):
                            _architect_pipeline_active = True
                            break
            break  # only check the most recent assistant message

    # Messages from present_options buttons — hard signal, always architect
    if safe_user_message.startswith("Agent Architect:"):
        detected_tool = tools_registry.get_tool("agent_architect")
        _architect_pipeline_active = True
        print(f"[TOOL-7.9] 'Agent Architect:' prefix detected — routing to architect", flush=True)

    # Always use full ALL_TOOLS for classifier — the old tools_registry only
    # knows 13 legacy skills but the neural classifier covers 199 tools.
    from ..services.tool_classifier import ALL_TOOLS
    enabled_tool_ids = {s for s in ALL_TOOLS if s is not None}

    _clf_prediction = None
    if detected_tool:
        pass  # Already resolved (prefix)
    elif request_body.teamId:
        print(f"[TOOL-7.9] BYPASSED — user selected team {request_body.teamId}", flush=True)
    else:
        try:
            msg_intents = intent_engine.extract(safe_user_message)

            _clf_prediction = await tool_classifier.predict(
                message=safe_user_message,
                enabled_tool_ids=enabled_tool_ids,
                recent_messages=recent_messages[-6:] if recent_messages else None,
                intents=msg_intents,
                user_id=user_id,
            )

            detected_tool_id = _clf_prediction.tool_id
            if detected_tool_id:
                effective_id = detected_tool_id

                # agents_* tool IDs (agents_delete, agents_create, etc.) → architect
                if effective_id == "agent_architect" or effective_id.startswith("agents_"):
                    detected_tool = tools_registry.get_tool("agent_architect")
                    _architect_pipeline_active = True
                    if effective_id != "agent_architect":
                        print(f"[TOOL-7.9] agents_* tool '{effective_id}' → routing to architect", flush=True)
                elif effective_id == "web_search":
                    web_search_needed = True
                elif effective_id == "image_generation":
                    image_gen_needed = True
                else:
                    detected_tool = tools_registry.get_tool(detected_tool_id)
                code_visualizer_intent = (effective_id == "code_visualizer")

            print(f"[TOOL-7.9] Classifier: tool={detected_tool_id or 'None'} "
                  f"conf={_clf_prediction.confidence:.3f} method={_clf_prediction.method} "
                  f"active={_clf_prediction.active_tool} latency={_clf_prediction.latency_ms:.1f}ms "
                  f"intents={msg_intents[:2]} probs={dict(list(_clf_prediction.probabilities.items())[:3])} "
                  f"msg={safe_user_message[:60]!r}", flush=True)
        except Exception as e:
            logger.warning(f"Tool classifier failed: {e}", exc_info=True)

    # Soft continuity: if architect was active AND classifier didn't confidently
    # pick a different tool, keep routing to architect for follow-ups.
    # If user asks random question (classifier picks web_search, image_gen, etc.), break continuity.
    _ns_image_intent = _is_image_intent(safe_user_message) and IMAGE_GENERATION_AVAILABLE
    if not detected_tool and not web_search_needed and not image_gen_needed and _architect_pipeline_active and not _ns_image_intent:
        _clf_picked_other = (
            _clf_prediction
            and _clf_prediction.tool_id
            and _clf_prediction.tool_id != "agent_architect"
            and not (_clf_prediction.tool_id or "").startswith("agents_")
            and _clf_prediction.confidence > 0.45
        )
        _clf_said_none = (
            _clf_prediction
            and not _clf_prediction.tool_id
            and _clf_prediction.confidence > 0.30
        )
        if _clf_picked_other:
            print(f"[TOOL-7.9] continuity BROKEN — classifier confidently picked '{_clf_prediction.tool_id}' ({_clf_prediction.confidence:.2f})", flush=True)
        elif _clf_said_none:
            print(f"[TOOL-7.9] continuity BROKEN — classifier says NO TOOL needed (conf={_clf_prediction.confidence:.2f}), general chat", flush=True)
        else:
            _is_short_followup = len(safe_user_message.split()) <= 6
            if _is_short_followup:
                detected_tool = tools_registry.get_tool("agent_architect")
                print(f"[TOOL-7.9] continuity KEPT — short follow-up ({len(safe_user_message.split())} words)", flush=True)
            else:
                print(f"[TOOL-7.9] continuity BROKEN — message too long/complex for follow-up ({len(safe_user_message.split())} words)", flush=True)

    # Image intent keyword guard — override architect/other tools when user clearly wants an image
    if _is_image_intent(safe_user_message) and IMAGE_GENERATION_AVAILABLE:
        detected_tool = None
        image_gen_needed = True
        _architect_pipeline_active = False
        print(f"[TOOL-7.9] IMAGE INTENT guard — forcing image generation for msg={safe_user_message[:60]!r}", flush=True)

    # ============================================
    # STEP 7.6: CHECK RESPONSE CACHE
    # ============================================
    cached_response = None
    if not execute_mode and not time_tool_results and not web_search_needed and not code_visualizer_intent:
        try:
            context_summary = " ".join([m.content[:50] for m in recent_messages[-3:-1]])
            cached_response = await get_cached_response(
                message=safe_user_message,
                context_summary=context_summary,
                provider=request_body.preferred_provider
            )
            if cached_response:
                logger.info(f"\U0001f3af Cache HIT - returning cached response")
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")

    # ============================================
    # STEP 7.7: WEB SEARCH (Real-time information) — triggered by LLM decision
    # ============================================
    web_search_results = []
    web_search_context = ""
    web_search_query = safe_user_message
    if WEB_SEARCH_AVAILABLE and web_search and web_search_needed:
        try:
            if user_api_keys:
                web_search.set_api_keys(
                    tavily_key=user_api_keys.get("tavily"),
                    serp_key=user_api_keys.get("serpapi"),
                )
            print(f"[WEB_SEARCH] Executing search for: {web_search_query[:60]!r}", flush=True)
            web_search_results = await web_search.search(
                query=web_search_query,
                max_results=5,
            )
            print(f"[WEB_SEARCH] Got {len(web_search_results)} results", flush=True)
            if web_search_results:
                web_search_context = web_search.format_results_for_context(web_search_results)
                context_messages.append({
                    "role": "system",
                    "content": web_search_context
                    + "\n\nIMPORTANT: Use these web search results as the primary source of truth for factual/time-sensitive answers. Do not guess or use stale information.",
                })
                print(f"[WEB_SEARCH] Added {len(web_search_results)} results to context ({len(web_search_context)} chars)", flush=True)
        except Exception as e:
            print(f"[WEB_SEARCH] FAILED: {e}", flush=True)

    # ============================================
    # STEP 7.8: IMAGE GENERATION (TokenRouter + DALL-E fallback)
    # ============================================
    generated_images = []
    if IMAGE_GENERATION_AVAILABLE and image_generation and image_gen_needed:
        try:
            if user_api_keys and user_api_keys.get("openai"):
                image_generation.set_api_key(openai_key=user_api_keys.get("openai"))
            image_generation.set_user_keys(user_api_keys)
            logger.info(f"\U0001f3a8 Image generation triggered for: {safe_user_message[:50]}...")
            image_prompt = image_generation.extract_image_prompt(safe_user_message)
            try:
                generated_images = await image_generation.generate(
                    prompt=image_prompt,
                    model="auto",
                    size="1024x1024",
                    quality="standard",
                )
                if generated_images:
                    logger.info(f"\U0001f3a8 Generated {len(generated_images)} image(s) via {generated_images[0].model}")
            except ValueError as e:
                logger.warning(f"Image generation skipped: {e}")
        except Exception as e:
            logger.warning(f"Image generation failed: {e}")

    # ============================================
    # STEP 7.95: TOOL EXECUTION (for non-web/non-image tools detected by classifier)
    # ============================================
    if detected_tool:
        try:
            print(f"[TOOL-7.95] EXECUTING: {detected_tool.id} ({detected_tool.name})", flush=True)
            tool_context = {
                "analysis_id": (request_body.project_context or {}).get("projectId", ""),
                "chat_id": chat_id,
                "unlimited_credits": unlimited_credits,
                "org_id": request.headers.get("x-org-id"),
                "user_role": user_role,
                "is_superuser": is_superuser,
                "user_api_keys": user_api_keys or {},
            }
            if _prev_assistant_agent_content:
                tool_context["prev_assistant_content"] = _prev_assistant_agent_content
            # Pass recent conversation history for agent_architect context
            if detected_tool.id == "agent_architect" and recent_messages:
                tool_context["recent_messages"] = [
                    {"role": (m.role if hasattr(m, "role") else m.get("role", "")),
                     "content": (m.content if hasattr(m, "content") else m.get("content", ""))[:1000]}
                    for m in recent_messages[-20:]
                ]
            github_token = _extract_github_token_from_user_keys(user_api_keys)
            if github_token:
                tool_context["github_token"] = github_token
            tool_result = await tool_executor.execute(
                tool=detected_tool,
                message=raw_user_message,
                user_id=user_id,
                user_role=user_role,
                is_superuser=is_superuser,
                context=tool_context,
            )
            print(f"[TOOL-7.95] RESULT: success={tool_result.get('success') if tool_result else 'None'}, "
                  f"delegate={tool_result.get('delegate_to_pipeline') if tool_result else 'N/A'}, "
                  f"summary_len={len(tool_result.get('summary', '')) if tool_result else 0}, "
                  f"error={tool_result.get('error', '')[:100] if tool_result else 'N/A'}", flush=True)
            if tool_result and tool_result.get("success"):
                tool_summary = tool_result.get("summary", "")
                if tool_result.get("delegate_to_pipeline"):
                    # Tool wants to delegate to web search pipeline (weather, news, reddit, etc.)
                    web_search_needed = True
                    _search_prefix = tool_result.get("search_prefix", "")
                    if _search_prefix:
                        safe_user_message = f"{_search_prefix} {safe_user_message}"
                    logger.info(f"🔄 Tool {detected_tool.id} delegated to search pipeline (prefix={_search_prefix!r})")
                elif tool_summary:
                    if detected_tool.id in ("agent_architect", "agents_list", "agents_create",
                                             "agents_start", "agents_stop", "agents_status",
                                             "agents_delete", "agents_update", "run_agent"):
                        # Grounded output for agent management tools — LLM must only report real actions
                        context_messages.append({
                            "role": "system",
                            "content": (
                                f"TOOL OUTPUT ({detected_tool.name}) — GROUNDED RESULTS:\n{tool_summary}\n\n"
                                "CRITICAL: Report ONLY the actions and results shown above. "
                                "Do NOT fabricate any data, tool names, agent names, or statuses. "
                                "Show the action log verbatim so the user sees exactly what happened. "
                                "If the output includes 'ACTIONS PERFORMED', present each action as a step."
                            ),
                        })
                    else:
                        context_messages.append({
                            "role": "system",
                            "content": f"TOOL OUTPUT ({detected_tool.name}):\n{tool_summary}\n\nUse this data to answer the user's question. Present the information clearly.",
                        })
                    logger.info(f"\U0001f527 Tool output added to context: {len(tool_summary)} chars")
        except Exception as e:
            logger.warning(f"Tool execution failed: {e}")

    # ============================================
    # STEP 8: Agent Debate/Spawn (Patches #40, #41)
    # ============================================
    debate_used = False
    agent_type = None
    response_text = None
    provider = "unknown"

    # Force tool-grounded reply for Code Visualizer to avoid synthetic/hallucinated summaries.
    if not execute_mode and detected_tool and detected_tool.id == "code_visualizer" and tool_result:
        tool_summary = (tool_result.get("summary") or "").strip()
        if tool_result.get("success"):
            response_text = tool_summary or "Code Visualizer completed successfully."
            provider = "tool_code_visualizer"
            agent_type = "code"
        else:
            error_detail = (tool_result.get("error") or "Code Visualizer request failed.").strip()
            response_text = f"Code Visualizer failed: {error_detail}"
            provider = "tool_code_visualizer_error"
            agent_type = "code"

    if not execute_mode and detected_tool and detected_tool.id == "code_visualizer" and not tool_result:
        logger.warning("Code Visualizer was detected but no tool result was returned; forcing failure response.")
        tool_result = {
            "tool_id": "code_visualizer",
            "tool_name": "Code Visualizer",
            "success": False,
            "action": "scan_github",
            "error": "Code Visualizer did not execute. No scan was run.",
        }
        response_text = "Code Visualizer failed: Code Visualizer did not execute. No scan was run."
        provider = "tool_code_visualizer_error"
        agent_type = "code"

    if (
        not execute_mode
        and response_text is None
        and code_visualizer_intent
        and "code_visualizer" not in enabled_tool_ids
        and (not detected_tool or detected_tool.id != "code_visualizer")
    ):
        logger.info("Code Visualizer intent detected but tool is disabled/not selected; returning tool-grounded disabled response.")
        tool_result = {
            "tool_id": "code_visualizer",
            "tool_name": "Code Visualizer",
            "success": False,
            "action": "scan_github",
            "error": "Code Visualizer skill is disabled, so no scan was run.",
            "summary": "Code Visualizer is disabled for this session. Enable the skill from the input bar Skills toggle and retry.",
        }
        response_text = (
            "Code Visualizer is disabled for this session, so no scan was run. "
            "Enable it from the input bar Skills toggle and try again."
        )
        provider = "tool_code_visualizer_disabled"
        agent_type = "code"

    # Force tool-grounded reply for ALL Agent Architect operations (including create).
    # Delegating create ops to the LLM caused hallucinated fake URLs/configs.
    if not execute_mode and detected_tool and detected_tool.id == "agent_architect" and tool_result:
        operation = tool_result.get("operation", "")
        if tool_result.get("success"):
            tool_summary = (tool_result.get("summary") or "").strip()
            # Strip raw present_options(...) calls from text — they render as buttons via toolResults
            tool_summary = re.sub(r'Please choose one of the options:\s*present_options\([^)]*\)', '', tool_summary).strip()
            tool_summary = re.sub(r'present_options\([^)]*\)', '', tool_summary).strip()
            response_text = tool_summary or "Agent Architect operation completed successfully."
            provider = "tool_agent_architect"
            agent_type = "agents"
        else:
            error_detail = (tool_result.get("error") or "Agent Architect request failed.").strip()
            response_text = f"Agent Architect error: {error_detail}"
            provider = "tool_agent_architect_error"
            agent_type = "agents"

    # Force tool-grounded reply for integration skill failures (google_calendar, figma, google_drive, sigma).
    # But ONLY if the user actually asked for that integration (keyword check).
    # False-positive classifications should fall through to regular LLM response.
    _integration_tool_ids = {"figma", "google_drive", "google_calendar", "sigma"}
    if (
        not execute_mode
        and response_text is None
        and detected_tool
        and detected_tool.id in _integration_tool_ids
        and tool_result
        and not tool_result.get("success")
    ):
        _tool_keywords_ns = {
            "google_drive": ["drive", "google drive", "my files", "my documents"],
            "google_calendar": ["calendar", "schedule", "event", "meeting"],
            "figma": ["figma", "design file"],
            "sigma": ["sigma", "analytics"],
        }
        _kws_ns = _tool_keywords_ns.get(detected_tool.id, [])
        _msg_lower_ns = (safe_user_message or "").lower()
        _user_asked_ns = any(kw in _msg_lower_ns for kw in _kws_ns)
        if _user_asked_ns:
            error_detail = (tool_result.get("error") or f"{detected_tool.name} request failed.").strip()
            response_text = f"{detected_tool.name} error: {error_detail}"
            provider = f"tool_{detected_tool.id}_error"
            agent_type = "integration"
            logger.info(f"Integration tool {detected_tool.id} failed, returning error directly: {error_detail[:120]}")
        else:
            # False positive — discard and let LLM handle the message normally
            logger.info(f"Integration tool {detected_tool.id} failed but user didn't ask for it — discarding false positive")
            tool_result = None
            detected_tool = None
    if not execute_mode and response_text is None and time_tool_results and is_time_only_query(safe_user_message):
        tr = time_tool_results[0].result or {}
        local_str = tr.get("local") or tr.get("iso")
        tz = tr.get("timezone") or "America/Los_Angeles"
        response_text = f"The exact current time in San Francisco is {local_str} ({tz})."
        provider = "tool_time"
        agent_type = "time"

    # If we computed an authoritative time tool result but the query isn't time-only,
    # force the correct time into the assistant text to prevent contradictions.
    if (
        not execute_mode
        and response_text
        and provider != "tool_time"
        and time_tool_results
        and any(
            k in (safe_user_message or "").lower()
            for k in ["time", "current time", "time now", "what time", "exact time"]
        )
    ):
        tr = time_tool_results[0].result or {}
        local_str = tr.get("local") or tr.get("iso")
        tz = tr.get("timezone") or "America/Los_Angeles"
        response_text = (
            response_text.rstrip()
            + "\n\n"
            + f"Authoritative current time: {local_str} ({tz})."
        )
    
    # Use cached response if available
    if cached_response and not execute_mode and response_text is None:
        response_text = cached_response.content
        provider = f"cache_{cached_response.provider}"
        logger.info(f"📦 Using cached response from {cached_response.provider}")
    
    # IDE Execute Mode: Modify context for code-focused responses
    if execute_mode:
        logger.info("🔧 IDE Execute Mode enabled - code-focused response")
        # Add execute mode system prompt to context
        execute_system_prompt = {
            "role": "system",
            "content": (
                "EXECUTE MODE: You are in IDE execution mode. "
                "DO NOT explain or discuss. ONLY output executable code. "
                "Format: Return JSON with 'actions' array containing: "
                "{'type': 'create'|'modify'|'delete', 'path': 'file/path', 'content': 'code'} "
                "If no file changes needed, return {'actions': [], 'response': 'brief answer'}. "
                "Be direct. No markdown explanations. Code only."
            )
        }
        context_messages = [execute_system_prompt] + context_messages
        
        # Add project context if provided
        if request_body.project_context:
            project_prompt = {
                "role": "system",
                "content": f"Project context: {request_body.project_context}"
            }
            context_messages = [project_prompt] + context_messages
    
    # ============================================
    # HALLUCINATION GUARD: Code analysis topics
    # ============================================
    # If the user asks about code metrics, broken connections, pipelines, etc.
    # but no CV tool actually ran, inject a guard prompt so the LLM doesn't fabricate data.
    _code_analysis_keywords = [
        "broken connection", "broken import", "unresolved import",
        "pipeline", "dependency graph", "service count", "function count",
        "endpoint count", "connection count", "code metric", "code stat",
        "how many function", "how many endpoint", "how many service",
        "how many file", "node count", "broken dep",
    ]
    _user_asking_code_analysis = any(
        kw in (safe_user_message or "").lower() for kw in _code_analysis_keywords
    )
    _cv_tool_ran = (
        detected_tool and detected_tool.id == "code_visualizer" and tool_result and tool_result.get("success")
    )
    if _user_asking_code_analysis and not _cv_tool_ran and response_text is None:
        context_messages.append({
            "role": "system",
            "content": (
                "CRITICAL ACCURACY RULE: The user is asking about code analysis data "
                "(broken connections, pipelines, metrics, service counts, etc.). "
                "The Code Visualizer tool did NOT run for this query, so you have NO "
                "verified data. Do NOT fabricate, estimate, or invent any numbers, "
                "categories, or breakdowns. Instead, tell the user to run a Code "
                "Visualizer scan first (e.g. 'scan github <repo_url>') or ask them "
                "to say 'show broken connections' or 'list pipelines' to trigger "
                "the real analysis tool. NEVER generate fake code metrics."
            ),
        })
        logger.info("🛡️ HALLUCINATION GUARD: Code analysis question detected but no CV tool ran; injected accuracy prompt.")

    # ============================================
    # HALLUCINATION GUARD: Agent creation topics
    # ============================================
    # If the user asks about creating/building agents but the architect tool
    # didn't run, inject a guard so the LLM doesn't fabricate fake agent IDs,
    # webhook URLs, hashes, or endpoint URLs.
    _user_asking_agent_creation = _is_agent_intent(safe_user_message or "")
    _architect_tool_ran = (
        detected_tool and detected_tool.id == "agent_architect" and tool_result
    )
    if _user_asking_agent_creation and not _architect_tool_ran and response_text is None:
        # Force-execute agent_architect tool — classifier missed it
        logger.info("🛡️ AGENT GUARD: Agent management query detected but tool classifier missed agent_architect — force-executing")
        try:
            _forced_tool = tools_registry.get_tool("agent_architect")
            if _forced_tool:
                _forced_ctx = {
                    "analysis_id": (request_body.project_context or {}).get("projectId", ""),
                    "chat_id": chat_id,
                    "unlimited_credits": unlimited_credits,
                    "org_id": request.headers.get("x-org-id"),
                    "user_role": user_role,
                    "is_superuser": is_superuser,
                    "user_api_keys": user_api_keys or {},
                }
                if recent_messages:
                    _forced_ctx["recent_messages"] = [
                        {"role": (m.role if hasattr(m, "role") else m.get("role", "")),
                         "content": (m.content if hasattr(m, "content") else m.get("content", ""))[:1000]}
                        for m in recent_messages[-20:]
                    ]
                _forced_result = await tool_executor.execute(
                    tool=_forced_tool,
                    message=raw_user_message,
                    user_id=user_id,
                    user_role=user_role,
                    is_superuser=is_superuser,
                    context=_forced_ctx,
                )
                if _forced_result and _forced_result.get("success"):
                    _summary = (_forced_result.get("summary") or "").strip()
                    response_text = _summary or "Agent Architect operation completed successfully."
                    provider = "tool_agent_architect"
                    agent_type = "agents"
                    detected_tool = _forced_tool
                    tool_result = _forced_result
                    logger.info(f"🛡️ AGENT GUARD: Forced agent_architect succeeded: {len(response_text)} chars")
                else:
                    _err = (_forced_result.get("error") or "").strip() if _forced_result else ""
                    logger.warning(f"🛡️ AGENT GUARD: Forced agent_architect failed: {_err}")
        except Exception as _e:
            logger.warning(f"🛡️ AGENT GUARD: Forced agent_architect exception: {_e}")

        # Fallback: if force-execute also failed, inject accuracy prompt
        if _user_asking_agent_creation and response_text is None:
            context_messages.append({
                "role": "system",
                "content": (
                    "CRITICAL ACCURACY RULE: The user is asking about creating, managing, or configuring agents. "
                    "The Agent Architect tool did NOT run for this query, so you CANNOT create or manage agents. "
                    "Do NOT fabricate, invent, or hallucinate any Agent IDs, Agent Hashes, Endpoint URLs, "
                    "Webhook URLs, or any agent configuration details. These are ALL generated by the "
                    "Agent Architect via the Agent Engine API — you cannot produce them yourself. "
                    "Instead, tell the user you'll connect them with the Agent Architect — ask them "
                    "what kind of agent they want to build or what they want to do with their agents. "
                    "NEVER generate fake agent details."
                ),
            })
            logger.info("🛡️ AGENT GUARD: Force-execute failed; injected fallback accuracy prompt.")

    # Try team workflow first (Phase 1: Internal Teams)
    team_used = False
    team_name = None
    forced_agent_type = None
    allowed_forced_agents = {
        "reasoning",
        "code",
        "debug",
        "research",
        "summary",
        "planning",
        "math",
        "security",
        "architecture",
        "test",
        "review",
        "explain",
        "optimization",
        "documentation",
        "migration",
        "api",
        "database",
        "devops",
        "refactor",
        "accessibility",
        "i18n",
        "regex",
        "git",
        "css",
    }
    if request_body.agent_hash and request_body.agent_hash in allowed_forced_agents:
        forced_agent_type = request_body.agent_hash

    if not execute_mode and not response_text and not forced_agent_type:
        try:
            from ..domain.agent import maybe_run_team
            team_response, team_name, team_used = await maybe_run_team(
                message=message_with_images,
                context_messages=context_messages,
                preferred_provider=request_body.preferred_provider,
                user_id=user_id,
                user_api_keys=user_api_keys,
                images=request_body.images,
            )
            if team_used and team_response:
                response_text = team_response
                provider = f"team_{team_name.lower().replace(' ', '_')}" if team_name else "team"
                logger.info(f"👥 Used team: {team_name}")
        except Exception as e:
            logger.warning(f"Team engine failed: {e}")
    
    # Try multi-agent debate if no team (Patch #41) - skip in execute mode for speed
    if not response_text and not execute_mode and not forced_agent_type:
        try:
            debate_response, debate_used = await maybe_run_debate(
                message=message_with_images,
                context_messages=context_messages,
                preferred_provider=request_body.preferred_provider,
                images=request_body.images,
            )
            if debate_used and debate_response:
                response_text = debate_response
                provider = "debate_engine"
                logger.info("🧠 Used multi-agent debate")
        except Exception as e:
            logger.warning(f"Debate engine failed: {e}")
    
    # Try agent spawn if no debate/team (Patch #40)
    actual_llm_provider = None
    agent_type = None
    router_metadata = None  # model, fallback_chain, was_fallback, usage
    if not response_text:
        logger.info(f"🔍 Attempting agent spawn with preferred_provider={request_body.preferred_provider}...")
        logger.info(f"🔍 Context has {len(context_messages)} messages before agent spawn")
        try:
            agent_response, agent_type, actual_llm_provider, router_metadata = await maybe_spawn_agent(
                message=message_with_images,
                context_messages=context_messages,
                user_id=user_id,
                user_api_keys=user_api_keys,
                preferred_provider=request_body.preferred_provider,
                forced_agent_type=forced_agent_type,
                images=request_body.images,
            )
            logger.info(f"🔍 Agent spawn returned: agent_type={agent_type}, provider={actual_llm_provider}, model={router_metadata.get('model') if router_metadata else None}, has_response={bool(agent_response)}")
            if router_metadata and router_metadata.get("was_fallback"):
                logger.info(f"⚠️ FALLBACK occurred: chain={router_metadata.get('fallback_chain')}")
            if agent_type and agent_response:
                response_text = agent_response
                provider = f"agent_{agent_type}"
                logger.info(f"🤖 Used agent: {agent_type} via {actual_llm_provider}")
            else:
                logger.info(f"🔍 Agent spawn returned None/empty, will try fallback")
        except Exception as e:
            logger.warning(f"Agent spawn failed: {e}")
    
    # ============================================
    # STEP 9: FORCE Agent Response (Direct LLM blocked for quality)
    # ============================================
    # NOTE: Direct LLM calls are blocked. All responses must go through agents.
    # The agent_engine.should_spawn_agent() now ALWAYS returns an agent type.
    # This fallback only triggers if agent spawn completely failed.
    if not response_text:
        logger.info(f"⚠️ Agent/Debate failed, forcing reasoning agent as fallback")
        
        # Force reasoning agent instead of direct LLM
        try:
            from ..services.agent_engine import agent_engine
            from ..domain.provider import get_router_for_internal_use
            
            router = get_router_for_internal_use()
            if user_api_keys:
                router.set_user_api_keys(user_api_keys)
            agent_engine.set_router(router)
            
            result = await agent_engine.spawn(
                task=message_with_images,
                context=context_messages,
                agent_type="reasoning",  # Always use reasoning as final fallback
                model=request_body.preferred_provider,
                images=request_body.images,
            )
            response_text = result.get("content", "")
            actual_llm_provider = result.get("provider", None)
            provider = "agent_reasoning"
            agent_type = "reasoning"
            # Capture router_metadata from forced spawn too
            router_metadata = {
                "model": result.get("model"),
                "fallback_chain": result.get("fallback_chain"),
                "was_fallback": result.get("was_fallback", False),
                "preferred_provider": result.get("preferred_provider"),
                "usage": result.get("usage"),
            }
            logger.info(f"🤖 Forced reasoning agent response via {actual_llm_provider}, model={router_metadata.get('model')}")
            logger.info(f"🔍 Fallback result: provider={actual_llm_provider}, content_length={len(response_text) if response_text else 0}")
        except Exception as e:
            logger.error(f"Forced agent also failed: {e}")
            # Only as absolute last resort, use direct LLM
            ai_response = await route_query(
                message=message_with_images,
                context=context_messages,
                preferred_provider=request_body.preferred_provider,
                user_api_keys=user_api_keys,
                images=request_body.images,
            )
            response_text = ai_response.get("response", "")
            provider = ai_response.get("provider", "unknown")
            # Capture router_metadata from direct route_query fallback
            direct_meta = ai_response.get("metadata", {})
            router_metadata = {
                "model": direct_meta.get("model"),
                "fallback_chain": direct_meta.get("fallback_chain"),
                "was_fallback": direct_meta.get("was_fallback", False),
                "preferred_provider": direct_meta.get("preferred_provider"),
                "usage": direct_meta.get("usage"),
            }
    
    is_error = response_text.startswith("Error calling") if response_text else True
    
    if not response_text:
        response_text = "I apologize, but I couldn't generate a response. Please try again."
        is_error = True
    
    # Patch #48: Autonomous Error Correction (skip tool-grounded responses)
    _is_tool_response = provider and str(provider).startswith("tool_")
    if not is_error and not _is_tool_response and error_correction.detect_error(response_text):
        logger.info("⚠️ Error detected in response, attempting correction...")
        try:
            async def _correction_llm(prompt):
                return await route_query(prompt, context_messages, request_body.preferred_provider, user_api_keys)
            corrected = await error_correction.correct(
                llm_callable=_correction_llm,
                user_input=safe_user_message,
                last_output=response_text,
                context=context_messages
            )
            if corrected and corrected != response_text:
                response_text = corrected
                logger.info("✅ Response corrected")
        except Exception as e:
            logger.warning(f"Error correction failed: {e}")
    
    logger.info(f"✅ LLM response from {provider}: {len(response_text)} chars")
    
    # ============================================
    # STEP 9.5: SANITIZE RESPONSE (Remove Agent Prompt Leakage)
    # ============================================
    if response_text and not is_error:
        response_text = _sanitize_agent_response(response_text)
        logger.info(f"🧹 Response sanitized: {len(response_text)} chars")
    
    # ============================================
    # STEP 10: Hash and Store Assistant Message
    # ============================================
    try:
        assistant_hash = hasher.hash_text(response_text) if not is_error else _simple_hash(response_text)
        assistant_xyz = hasher.hash_to_coords(assistant_hash) if not is_error else _hash_to_xyz_simple(assistant_hash)
    except:
        assistant_hash = _simple_hash(response_text)
        assistant_xyz = _hash_to_xyz_simple(assistant_hash)
    
    # Calculate resonance score based on response quality
    resonance_score = 0.0 if is_error else _calculate_resonance_score(
        response_text=response_text,
        user_message=safe_user_message,
        memories=memories,
        context_messages=context_messages,
    )
    
    # ============================================
    # STEP 10.2: LAYER 8 - CONSISTENCY CHECK
    # C_k = cos(o_k, Ê*)
    # ============================================
    evidence_consistency = 0.0
    try:
        if evidence_vector is not None and assistant_xyz:
            evidence_consistency = evidence_graph.calculate_consistency(
                assistant_xyz, evidence_vector
            )
            logger.info(f"📊 Evidence consistency: {evidence_consistency:.3f}")
            # Boost resonance score if response is consistent with evidence
            if evidence_consistency > 0.5:
                resonance_score = min(1.0, resonance_score + (evidence_consistency - 0.5) * 0.2)
    except Exception as e:
        logger.warning(f"Evidence consistency check failed: {e}")
    
    # ============================================
    # STEP 10.2.5: LAYER 9 - OUTPUT CORRECTION
    # o_corrected = λ·o_k* + (1-λ)·Ê*
    # ============================================
    correction_result = None
    try:
        if evidence_vector is not None and assistant_xyz and not is_error:
            correction_result = output_correction.apply_correction(
                response_text=response_text,
                response_xyz=assistant_xyz,
                evidence_vector=evidence_vector,
                evidence_weight=evidence_weight,
                evidence_consistency=evidence_consistency,
                memories=memories
            )
            if correction_result.get("corrected"):
                # Update XYZ to corrected position
                corrected_xyz = correction_result["corrected_xyz"]
                assistant_xyz = corrected_xyz
                logger.info(f"📐 Layer 9 applied: {correction_result['reason']}")
                
                # Boost resonance score for corrected responses
                resonance_score = min(1.0, resonance_score + 0.05)
    except Exception as e:
        logger.warning(f"Layer 9 output correction failed: {e}")
    
    # Get provider metadata if available (ai_response only exists when LLM was called directly)
    provider_metadata = {}
    try:
        if ai_response:
            provider_metadata = ai_response.get("metadata", {})
    except NameError:
        pass  # ai_response not defined (debate/agent was used)
    
    # Include actual LLM provider in metadata if available from agent
    logger.info(f"🔍 Before storing message: actual_llm_provider={actual_llm_provider}, provider={provider}, agent_type={agent_type}")
    if actual_llm_provider:
        provider_metadata["provider"] = actual_llm_provider
        logger.info(f"🔍 Set provider_metadata['provider'] = {actual_llm_provider}")
    
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
        meta_data={
            "provider_metadata": provider_metadata,
            "is_error": is_error,
            "actual_llm_provider": actual_llm_provider,
            "agent_type": agent_type if 'agent_type' in locals() else None,
            "layer_9_correction": correction_result,
            "evidence_consistency": evidence_consistency,
            "model": (router_metadata or {}).get("model") if router_metadata else (provider_metadata.get("model") if provider_metadata else None),
            "fallback_chain": (router_metadata or {}).get("fallback_chain"),
            "was_fallback": (router_metadata or {}).get("was_fallback", False),
            "preferred_provider": request_body.preferred_provider,
            "usage": (router_metadata or {}).get("usage") or provider_metadata.get("usage"),
        }
    )
    session.add(assistant_message)
    await session.commit()
    await session.refresh(assistant_message)
    
    # ============================================
    # STEP 10.3: CREATE DSID FOR ASSISTANT MESSAGE (HSU-Spec Layer 1-2)
    # ============================================
    assistant_dsid = create_message_dsid(
        message_id=str(assistant_message.id),
        content=response_text,
        role="assistant",
        chat_id=chat_id,
        user_id=user_id,
        parent_message_id=str(user_message.id),  # Link to user message
        metadata={
            "hash": assistant_hash,
            "xyz": list(assistant_xyz),
            "provider": provider,
            "resonance_score": resonance_score,
        },
    )
    logger.info(f"🔗 DSID created for assistant message: {assistant_dsid.dsid[:32]}...")
    
    # ============================================
    # STEP 10.4: PERSIST DSID DATA TO MESSAGE META_DATA (Survives container restarts)
    # ============================================
    try:
        # Get lineage depth for T3 score calculation
        lineage = dsid_integration.get_message_lineage(str(assistant_message.id))
        lineage_depth = len(lineage)
        
        # Update message meta_data with DSID info for persistence
        updated_meta = assistant_message.meta_data or {}
        updated_meta["dsid"] = {
            "dsid_id": assistant_dsid.dsid,
            "content_hash": assistant_dsid.content_hash,
            "parent_dsid": assistant_dsid.parent_dsid,
            "root_dsid": assistant_dsid.root_dsid,
            "lineage_depth": lineage_depth,
            "created_at": assistant_dsid.created_at.isoformat() if assistant_dsid.created_at else None,
        }
        assistant_message.meta_data = updated_meta
        flag_modified(assistant_message, "meta_data")
        await session.commit()
        logger.info(f"💾 DSID data persisted to message meta_data")

        # Re-record memory usage with real message_id (was 'pending' earlier)
        if memories:
            enhanced_metrics_calculator.record_memory_usage(
                message_id=str(assistant_message.id),
                memories_retrieved=len(memories),
                memories_used=min(len(memories), 5),
                anchor_matches=sum(1 for m in memories if m.get("anchor_energy", 0) > 0),
                rag_queries=1,
                embedding_lookups=1 if any(m.get("proximity_score", 0) > 0 for m in memories) else 0,
                memory_tokens=sum(len(str(m.get("content", ""))) // 4 for m in memories[:5]),
            )
    except Exception as e:
        logger.warning(f"DSID persistence failed (non-critical): {e}")
    
    # ============================================
    # STEP 10.5: CACHE RESPONSE & RECORD FOR LEARNING (L4 Autonomy)
    # ============================================
    # Cache the response for future similar queries
    if not is_error and not cached_response and not execute_mode:
        try:
            context_summary = " ".join([m.content[:50] for m in recent_messages[-3:-1]])
            await cache_response(
                message=safe_user_message,
                response=response_text,
                provider=provider,
                model=provider_metadata.get("model", "unknown") if provider_metadata else "unknown",
                context_summary=context_summary,
                quality_score=resonance_score
            )
            logger.info(f"💾 Cached response for future use")
        except Exception as e:
            logger.warning(f"Response caching failed: {e}")
    
    # Record interaction for Self-Improving Agent learning
    if agent_type:
        try:
            await self_improving_agent.record_feedback(
                agent_id=agent_type,
                message=safe_user_message,
                response=response_text,
                feedback_type=FeedbackType.AUTO_SUCCESS,
                metadata={
                    "resonance_score": resonance_score
                }
            )
            logger.info(f"📝 Recorded interaction for agent learning: {agent_type}")
        except Exception as e:
            logger.warning(f"Learning record failed: {e}")
    
    # ============================================
    # STEP 11: Queue Background Tasks (Fire-and-forget) - PRODUCTION READY
    # ============================================
    try:
        # Ingest user message into memory service with circuit breaker
        await service_client.call_service(
            "memory_service",
            "POST",
            "http://memory_service:8000/memory/ingest",
            json={
                "user_id": user_id,
                "org_id": org_id,
                "chat_id": str(chat_id),
                "source": "resonant-chat",
                "content": safe_user_message,
                "metadata": {
                    "role": "user",
                    "hash": user_hash,
                    "xyz": list(user_xyz)
                }
            }
        )
        
        # Ingest assistant message into memory service with circuit breaker
        await service_client.call_service(
            "memory_service",
            "POST",
            "http://memory_service:8000/memory/ingest",
            json={
                "user_id": user_id,
                "org_id": org_id,
                "chat_id": str(chat_id),
                "source": "resonant-chat",
                "content": response_text,
                "metadata": {
                    "role": "assistant",
                    "hash": assistant_hash,
                    "xyz": list(assistant_xyz)
                }
            }
        )
        
        logger.info("✅ Background memory ingestion completed")
    except Exception as e:
        logger.warning(f"Background memory ingest failed (non-critical): {e}")
    
    # PMI Layer: Create blockchain memory events (Patch #49)
    try:
        pmi_manager.create_memory_event(
            user_id=user_id,
            org_id=org_id,
            chat_id=chat_id,
            session_id=chat_id,
            message_text=safe_user_message,
            event_type=pmi_manager.EVENT_PROMPT,
        )
        pmi_manager.create_memory_event(
            user_id=user_id,
            org_id=org_id,
            chat_id=chat_id,
            session_id=chat_id,
            message_text=response_text,
            event_type=pmi_manager.EVENT_RESPONSE,
        )
    except Exception as e:
        logger.warning(f"PMI Layer event creation failed (non-critical): {e}")
    
    # Insight Seed Generation (Patch #49)
    try:
        if insight_seed_engine.should_generate_seed(safe_user_message, response_text):
            seed = insight_seed_engine.generate_seed(
                user_msg=safe_user_message,
                assistant_msg=response_text,
                context=context_messages
            )
            logger.info(f"💡 Generated insight seed: {seed[:50]}...")
    except Exception as e:
        logger.warning(f"Insight seed generation failed (non-critical): {e}")
    
    # ============================================
    # STEP 12: Build Evidence Graph (Patch #45 + #50 IMPROVED)
    # ============================================
    intents = intent_engine.extract(safe_user_message)
    emotion = emotional_normalizer.detect(safe_user_message)
    
    # IMPROVEMENT: Pass actual XYZ coordinates and content for accurate graph
    evidence_graph_data = evidence_graph.build_graph(
        user_hash=user_hash,
        assistant_hash=assistant_hash,
        memories=memories,
        provider=provider,
        intents=intents,
        emotion=emotion,
        agent_type=agent_type,
        debate_used=debate_used,
        # NEW: Enhanced parameters for accurate Evidence Graph
        user_xyz=user_xyz,
        assistant_xyz=assistant_xyz,
        user_content=safe_user_message,
        assistant_content=response_text,
    )
    
    # ============================================
    # STEP 13: Build and Return Response
    # ============================================
    anchors = []
    if memories:
        anchors = [
            (mem.get("anchor_text", "") or mem.get("content", ""))[:50]
            for mem in memories[:5]  # Increased from 5 to 20
            if mem.get("anchor_text") or mem.get("content")
        ]
    
    # Store anchors and DSID data in message meta_data for metrics persistence
    current_meta = assistant_message.meta_data or {}
    current_meta["memory_count"] = len(memories) if memories else 0

    # Store memories as rag_sources so metrics calculator can see them
    if memories:
        current_meta["rag_sources"] = [
            {
                "content": (mem.get("content", "") or mem.get("anchor_text", ""))[:200],
                "score": float(mem.get("hybrid_score", 0) or mem.get("resonance_score", 0) or 0),
                "type": mem.get("type", "memory"),
                "hash": mem.get("hash", ""),
            }
            for mem in memories[:10]
            if mem.get("content") or mem.get("anchor_text")
        ]
    
    # IMPROVEMENT: Store memory contributors with XYZ for accurate Evidence Graph
    if memories:
        current_meta["memory_contributors"] = [
            {
                "text": (mem.get("content", "") or mem.get("anchor_text", ""))[:120],
                "xyz": mem.get("xyz"),
                "score": round(float(mem.get("hybrid_score", 0) or mem.get("resonance_score", 0) or 0), 3),
                "type": mem.get("type", "memory"),
            }
            for mem in memories[:10]
            if mem.get("content") or mem.get("anchor_text")
        ]
    
    # IMPROVEMENT: Store NLP-extracted anchors with confidence scores
    if response_text:
        extracted_anchors = evidence_graph._extract_keyphrases_nlp(response_text, max_phrases=10)
        current_meta["anchors"] = extracted_anchors
    elif anchors:
        # Fallback to simple anchor list
        current_meta["anchors"] = anchors
    
    # Store DSID data for T1/T3 trust scores (persists across container restarts)
    current_meta["dsid"] = {
        "dsid_id": assistant_dsid.dsid,
        "content_hash": assistant_dsid.content_hash,
        "parent_dsid": assistant_dsid.parent_dsid,
        "root_dsid": assistant_dsid.root_dsid,
        "lineage_depth": assistant_dsid.lineage_depth,
    }
    
    assistant_message.meta_data = current_meta
    flag_modified(assistant_message, "meta_data")
    session.add(assistant_message)
    await session.commit()

    
    message_data = MessageData(
        id=str(assistant_message.id),
        role="assistant",
        content=assistant_message.content,
        timestamp=assistant_message.created_at.isoformat() if assistant_message.created_at else datetime.utcnow().isoformat(),
        aiProvider=provider,
        llmProvider=actual_llm_provider,
        model=(router_metadata or {}).get("model") if router_metadata else None,
        preferredProvider=request_body.preferred_provider,
        wasFallback=(router_metadata or {}).get("was_fallback", False) if router_metadata else False,
        fallbackChain=(router_metadata or {}).get("fallback_chain") if router_metadata else None,
        tokenUsage=(router_metadata or {}).get("usage") if router_metadata else None,
        hash=assistant_hash,
        resonanceScore=resonance_score,
        xyz=[assistant_xyz[0], assistant_xyz[1], assistant_xyz[2]],
    )
    
    # Convert generated images to response format
    generated_images_data = None
    if generated_images:
        generated_images_data = [
            GeneratedImageData(
                url=img.url,
                base64_data=img.base64_data,
                revised_prompt=img.revised_prompt,
                model=img.model,
                size=img.size,
            )
            for img in generated_images
        ]
    
    # Convert web search results to response format
    web_search_data = None
    if web_search_results:
        web_search_data = [
            WebSearchResultData(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                source=result.source,
            )
            for result in web_search_results
        ]

    tool_results: List[ToolResultData] = []
    tool_results.extend(extract_navigation_tool_results(safe_user_message))
    tool_results.extend(time_tool_results)
    # Add skill results (success and failure) for frontend-grounded behavior.
    if tool_result:
        tool_success = bool(tool_result.get("success"))
        tool_payload: Dict[str, Any] = {
            "action": tool_result.get("action"),
            "tool_name": tool_result.get("tool_name"),
            "summary": (tool_result.get("summary") or "")[:500],
        }
        if "analysis_id" in tool_result:
            tool_payload["analysis_id"] = tool_result["analysis_id"]
        for key in ["panel_url", "operation", "created_agent_id", "created_agent_name", "created_agent_public_hash", "present_options", "pipeline_plan"]:
            if key in tool_result and tool_result.get(key) is not None:
                tool_payload[key] = tool_result.get(key)
        tool_results.append(
            ToolResultData(
                tool_name=f"tool_{tool_result.get('tool_id', 'unknown')}",
                success=tool_success,
                result=tool_payload,
                error=None if tool_success else str(tool_result.get("error") or "Tool execution failed"),
            )
        )
        # Emit a separate present_options entry so the frontend can find it
        if tool_result.get("present_options"):
            tool_results.append(
                ToolResultData(
                    tool_name="present_options",
                    success=True,
                    result=tool_result["present_options"],
                    error=None,
                )
            )

    current_meta = assistant_message.meta_data or {}
    if generated_images_data:
        current_meta["generatedImages"] = [img.model_dump() for img in generated_images_data]
    if web_search_data:
        current_meta["webSearchResults"] = [r.model_dump() for r in web_search_data]
    if tool_results:
        current_meta["toolResults"] = [t.model_dump() for t in tool_results]
    assistant_message.meta_data = current_meta
    flag_modified(assistant_message, "meta_data")
    session.add(assistant_message)
    await session.commit()
    
    return ResonantChatResponse(
        message=message_data,
        anchors=anchors,
        hash=user_hash,
        resonanceScore=resonance_score,
        aiProvider=provider,
        llmProvider=actual_llm_provider,
        memoryUpdated=len(memories) > 0,
        chatId=chat_id,
        evidenceGraph=evidence_graph_data,
        generatedImages=generated_images_data,
        webSearchResults=web_search_data,
        toolResults=tool_results or None,
    )


@router.get("/conversations")
async def list_conversations(
    request: Request,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    """Compatibility endpoint mapping /conversations -> /history."""
    return await get_chat_list(request, limit, session)


@router.post("/create", response_model=CreateChatResponse)
async def create_chat(
    request_body: CreateChatRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Create a new chat."""
    user_id = request.headers.get("x-user-id")
    org_id = request.headers.get("x-org-id") or user_id
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    chat = ResonantChat(
        user_id=user_id,
        org_id=org_id,
        title=request_body.title or "New Chat",
        status="active",
        agent_hash=request_body.agent_hash,
    )
    session.add(chat)
    await session.commit()
    await session.refresh(chat)
    
    return CreateChatResponse(
        chatId=str(chat.id),
        title=chat.title,
    )


@router.post("/conversations", response_model=CreateChatResponse)
async def create_conversation(
    request_body: CreateChatRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Compatibility endpoint mapping /conversations -> /create."""
    return await create_chat(request_body, request, session)


@router.post("/save-agentic")
async def save_agentic_message(
    body: SaveAgenticRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Save agentic-chat messages into the resonant chat pipeline.

    Allows AI Assistant (agentic-chat) messages to appear alongside regular
    resonant chat messages with full pipeline support:
      - Resonance hashing + XYZ coordinates
      - DSID creation (message lineage)
      - Memory ingestion to Hash Sphere
      - PMI Layer blockchain events
    Messages are tagged with ai_provider='agentic_assistant' so the UI can
    distinguish them, just like agent/team/debate messages.
    """
    # ── Auth ──
    if CRYPTO_IDENTITY_AVAILABLE:
        identity = get_crypto_identity(request)
        user_id = identity.user_id
        org_id = identity.org_id or user_id
    else:
        user_id = request.headers.get("x-user-id")
        org_id = request.headers.get("x-org-id") or user_id

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")

    safe_user_message = _sanitize_sensitive_tokens(body.user_message or "")
    safe_assistant = body.assistant_response or ""

    # ── Get or create resonant chat ──
    chat = None
    chat_id = body.chat_id
    if chat_id:
        try:
            result = await session.execute(
                select(ResonantChat).where(ResonantChat.id == UUID(chat_id))
            )
            chat = result.scalar_one_or_none()
        except (ValueError, Exception):
            chat = None

    if not chat:
        title = safe_user_message[:50] + ("..." if len(safe_user_message) > 50 else "")
        chat = ResonantChat(
            user_id=user_id,
            org_id=org_id,
            title=title or "AI Assistant Chat",
            status="active",
        )
        session.add(chat)
        await session.commit()
        await session.refresh(chat)

    chat_id = str(chat.id)

    # ── Resonance hashing ──
    try:
        hasher = ResonanceHasher()
        user_hash = hasher.hash_text(safe_user_message)
        user_xyz = hasher.hash_to_coords(user_hash)
        assistant_hash = hasher.hash_text(safe_assistant)
        assistant_xyz = hasher.hash_to_coords(assistant_hash)
    except Exception:
        user_hash = _simple_hash(safe_user_message)
        user_xyz = _hash_to_xyz_simple(user_hash)
        assistant_hash = _simple_hash(safe_assistant)
        assistant_xyz = _hash_to_xyz_simple(assistant_hash)

    # ── Store user message ──
    user_msg = ResonantChatMessage(
        chat_id=UUID(chat_id),
        role="user",
        content=safe_user_message,
        hash=user_hash,
        resonance_score=0.5,
        xyz_x=user_xyz[0],
        xyz_y=user_xyz[1],
        xyz_z=user_xyz[2],
    )
    session.add(user_msg)
    await session.commit()
    await session.refresh(user_msg)

    # ── Store assistant message ──
    resonance_score = _calculate_resonance_score(safe_assistant, safe_user_message, [], [])

    assistant_msg = ResonantChatMessage(
        chat_id=UUID(chat_id),
        role="assistant",
        content=safe_assistant,
        ai_provider="agentic_assistant",
        hash=assistant_hash,
        resonance_score=resonance_score,
        xyz_x=assistant_xyz[0],
        xyz_y=assistant_xyz[1],
        xyz_z=assistant_xyz[2],
        meta_data={
            "source": "agentic_assistant",
            "model": body.model,
            "tokens_used": body.tokens_used or 0,
            "loops": body.loops or 0,
            "tool_calls": body.tool_calls or [],
            "toolResults": body.tool_results or [],
        },
    )
    session.add(assistant_msg)
    await session.commit()
    await session.refresh(assistant_msg)

    # ── DSID creation (lineage tracking) ──
    try:
        user_dsid = create_message_dsid(
            message_id=str(user_msg.id),
            content=safe_user_message,
            role="user",
            chat_id=chat_id,
            user_id=user_id,
            parent_message_id=None,
            metadata={"hash": user_hash, "xyz": list(user_xyz), "source": "agentic_assistant"},
        )
        assistant_dsid = create_message_dsid(
            message_id=str(assistant_msg.id),
            content=safe_assistant,
            role="assistant",
            chat_id=chat_id,
            user_id=user_id,
            parent_message_id=str(user_msg.id),
            metadata={"hash": assistant_hash, "xyz": list(assistant_xyz), "source": "agentic_assistant"},
        )

        # Persist DSID to meta_data
        a_meta = assistant_msg.meta_data or {}
        a_meta["dsid"] = {
            "dsid_id": assistant_dsid.dsid,
            "content_hash": assistant_dsid.content_hash,
            "parent_dsid": assistant_dsid.parent_dsid,
            "root_dsid": assistant_dsid.root_dsid,
            "lineage_depth": assistant_dsid.lineage_depth,
        }
        assistant_msg.meta_data = a_meta
        flag_modified(assistant_msg, "meta_data")
        await session.commit()
    except Exception as e:
        logger.warning(f"[save-agentic] DSID creation failed (non-critical): {e}")

    # ── Memory ingestion to Hash Sphere ──
    try:
        await service_client.call_service(
            "memory_service", "POST",
            "http://memory_service:8000/memory/ingest",
            json={
                "user_id": user_id, "org_id": org_id, "chat_id": chat_id,
                "source": "agentic_assistant",
                "content": safe_user_message,
                "metadata": {"role": "user", "hash": user_hash, "xyz": list(user_xyz)},
            },
        )
        await service_client.call_service(
            "memory_service", "POST",
            "http://memory_service:8000/memory/ingest",
            json={
                "user_id": user_id, "org_id": org_id, "chat_id": chat_id,
                "source": "agentic_assistant",
                "content": safe_assistant,
                "metadata": {"role": "assistant", "hash": assistant_hash, "xyz": list(assistant_xyz)},
            },
        )
    except Exception as e:
        logger.warning(f"[save-agentic] Memory ingestion failed (non-critical): {e}")

    # ── PMI Layer blockchain events ──
    try:
        pmi_manager.create_memory_event(
            user_id=user_id, org_id=org_id, chat_id=chat_id,
            session_id=chat_id, message_text=safe_user_message,
            event_type=pmi_manager.EVENT_PROMPT,
        )
        pmi_manager.create_memory_event(
            user_id=user_id, org_id=org_id, chat_id=chat_id,
            session_id=chat_id, message_text=safe_assistant,
            event_type=pmi_manager.EVENT_RESPONSE,
        )
    except Exception as e:
        logger.warning(f"[save-agentic] PMI Layer failed (non-critical): {e}")

    logger.info(
        f"[save-agentic] Saved agentic messages to resonant pipeline: "
        f"chat_id={chat_id}, user_msg={str(user_msg.id)[:8]}, "
        f"assistant_msg={str(assistant_msg.id)[:8]}, model={body.model}"
    )

    return {
        "chat_id": chat_id,
        "user_message_id": str(user_msg.id),
        "assistant_message_id": str(assistant_msg.id),
        "resonance_score": resonance_score,
    }


@router.get("/conversations/{chat_id}")
async def get_conversation(
    chat_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Compatibility endpoint mapping /conversations/{chat_id} -> /history/{chat_id}."""
    return await get_chat_history(chat_id, request, session)


@router.put("/conversations/{chat_id}/archive")
async def archive_conversation(
    chat_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Archive a chat conversation (sets status='archived')."""
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")

    try:
        chat_uuid = UUID(chat_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chat ID format")

    result = await session.execute(select(ResonantChat).where(ResonantChat.id == chat_uuid))
    chat = result.scalar_one_or_none()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if str(chat.user_id) != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    chat.status = "archived"
    session.add(chat)
    await session.commit()

    return {"ok": True, "chat_id": str(chat.id), "status": chat.status}


@router.delete("/conversations/{chat_id}")
async def delete_conversation(
    chat_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Delete a chat conversation (sets status='deleted')."""
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")

    try:
        chat_uuid = UUID(chat_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chat ID format")

    result = await session.execute(select(ResonantChat).where(ResonantChat.id == chat_uuid))
    chat = result.scalar_one_or_none()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if str(chat.user_id) != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    chat.status = "deleted"
    session.add(chat)
    await session.commit()

    return {"ok": True, "chat_id": str(chat.id), "status": chat.status}


@router.delete("/conversations/{chat_id}/messages/{message_id}")
async def delete_conversation_message(
    chat_id: str,
    message_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Delete a message from a chat conversation."""
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")

    try:
        chat_uuid = UUID(chat_id)
        message_uuid = UUID(message_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chat/message ID format")

    result = await session.execute(select(ResonantChat).where(ResonantChat.id == chat_uuid))
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    if str(chat.user_id) != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    result = await session.execute(
        select(ResonantChatMessage).where(
            ResonantChatMessage.id == message_uuid,
            ResonantChatMessage.chat_id == chat_uuid,
        )
    )
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")

    await session.delete(msg)
    await session.commit()

    return {"ok": True, "chat_id": str(chat_uuid), "message_id": str(message_uuid)}


@router.post("/conversations/{chat_id}/messages", response_model=ResonantChatResponse)
async def add_conversation_message(
    chat_id: str,
    payload: ConversationMessageRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Compatibility endpoint to add a message to a conversation.

    Delegates to the main /message pipeline using the provided chat_id
    and message content.
    """
    request_body = SendMessageRequest(
        message=payload.content,
        chat_id=chat_id,
    )
    return await send_message(request_body, request, session)


@router.get("/history")
async def get_chat_list(
    request: Request,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    """Get list of user's chats with message counts."""
    user_id = request.headers.get("x-user-id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    # Get chats with message counts using a subquery
    from sqlalchemy import func as sql_func
    
    # Subquery to count messages per chat
    message_count_subq = (
        select(
            ResonantChatMessage.chat_id,
            sql_func.count(ResonantChatMessage.id).label('message_count'),
            sql_func.max(ResonantChatMessage.created_at).label('last_message_at')
        )
        .group_by(ResonantChatMessage.chat_id)
        .subquery()
    )
    
    # Join chats with message counts - show active chats (not archived)
    # Use != 'archived' to include NULL status and 'active' status
    from sqlalchemy import or_
    result = await session.execute(
        select(
            ResonantChat,
            message_count_subq.c.message_count,
            message_count_subq.c.last_message_at
        )
        .outerjoin(message_count_subq, ResonantChat.id == message_count_subq.c.chat_id)
        .where(ResonantChat.user_id == user_id)
        .where(or_(ResonantChat.status == "active", ResonantChat.status == None, ResonantChat.status == ""))  # Exclude only archived
        .order_by(ResonantChat.created_at.desc())
        .limit(limit)
    )
    rows = result.all()
    
    logger.info(f"[get_chat_list] Found {len(rows)} chats for user {user_id}")
    
    return [
        {
            "id": str(row[0].id),
            "title": row[0].title,
            "status": row[0].status,
            "agent_hash": row[0].agent_hash,
            "created_at": row[0].created_at.isoformat() if row[0].created_at else None,
            "message_count": row[1] or 0,
            "last_message_at": row[2].isoformat() if row[2] else None,
        }
        for row in rows
    ]


@router.get("/history/{chat_id}")
async def get_chat_history(
    chat_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Get chat history with messages."""
    user_id = request.headers.get("x-user-id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    try:
        result = await session.execute(
            select(ResonantChat).where(ResonantChat.id == UUID(chat_id))
        )
        chat = result.scalar_one_or_none()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chat ID format")
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    if str(chat.user_id) != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    result = await session.execute(
        select(ResonantChatMessage)
        .where(ResonantChatMessage.chat_id == UUID(chat_id))
        .order_by(ResonantChatMessage.created_at.asc())
    )
    messages = result.scalars().all()
    
    return {
        "chat_id": str(chat.id),
        "agent_hash": chat.agent_hash,
        "messages": [
            {
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                "aiProvider": msg.ai_provider,
                "llmProvider": (msg.meta_data or {}).get("actual_llm_provider") if isinstance(msg.meta_data, dict) else None,
                "hash": msg.hash,
                "resonanceScore": msg.resonance_score or 0.0,
                "xyz": [msg.xyz_x, msg.xyz_y, msg.xyz_z] if msg.xyz_x is not None else None,
                "generatedImages": (msg.meta_data or {}).get("generatedImages") if isinstance(msg.meta_data, dict) else None,
                "webSearchResults": (msg.meta_data or {}).get("webSearchResults") if isinstance(msg.meta_data, dict) else None,
                "toolResults": (msg.meta_data or {}).get("toolResults") if isinstance(msg.meta_data, dict) else None,
            }
            for msg in messages
        ]
    }


@router.get("/providers")
async def get_providers(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Get list of available AI providers - LIVE STATUS.
    
    This endpoint returns LIVE provider availability by checking actual API status.
    It uses the same status checking as the WebSocket endpoint for consistency.
    
    Features:
    - Live availability check (not hardcoded)
    - User BYOK keys automatically detected
    - Platform keys with credit system
    - Local LLM support
    """
    from .provider_status_ws import status_manager
    from ..services.user_api_keys import user_api_key_service
    
    # Get user info from request
    user_id = request.headers.get("x-user-id")
    user_role = request.headers.get("x-user-role", "user")
    user_plan = request.headers.get("x-user-plan", "free")
    is_superuser = request.headers.get("x-is-superuser", "false").lower() == "true"
    unlimited_credits = request.headers.get("x-unlimited-credits", "false").lower() == "true"
    
    # Free tier credit limit
    FREE_TIER_CREDITS = 1000
    
    # Check if user is on a paid plan (unlimited platform keys)
    is_paid_user = user_plan in ["plus", "enterprise"] or \
                   user_role in ["platform_dev", "system", "admin", "owner", "org_admin"] or \
                   is_superuser or unlimited_credits
    
    # Get user's credit balance for free users
    credits_remaining = FREE_TIER_CREDITS
    if user_id and not is_paid_user:
        try:
            credit_data = await service_client.call_service(
                "billing_service",
                "GET",
                f"http://billing_service:8000/billing/credits/balance/{user_id}"
            )
            if credit_data:
                credits_remaining = credit_data.get("balance", FREE_TIER_CREDITS)
        except Exception as e:
            logger.warning(f"Billing service failed: {e}")
    
    has_credits = is_paid_user or credits_remaining > 0
    
    # Get user's own API keys (BYOK)
    user_keys = {}
    if user_id:
        try:
            user_keys = await user_api_key_service.get_user_api_keys(user_id)
        except Exception:
            pass
    
    # Get LIVE provider status + real model lists in parallel
    import asyncio as _asyncio
    
    async def _fetch_llm_models():
        """Fetch real model lists from llm_service."""
        try:
            import os as _os
            llm_url = _os.getenv("LLM_SERVICE_URL", "http://llm_service:8000")
            headers = {}
            if user_id:
                headers["x-user-id"] = user_id
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{llm_url}/llm/providers", headers=headers)
                if r.status_code == 200:
                    return {p["id"]: p.get("models", []) for p in r.json().get("providers", [])}
        except Exception as e:
            logger.warning(f"Failed to fetch models from llm_service: {e}")
        return {}
    
    live_status, llm_models_map = await _asyncio.gather(
        status_manager.check_provider_status(),
        _fetch_llm_models(),
    )
    live_providers = live_status.get("providers", [])
    
    # Map chat_service provider IDs → llm_service provider IDs for model lookup
    # Core providers have ID aliases; all others identity-map
    provider_key_map = {
        "chatgpt": "openai",
        "gemini": "google",
    }
    
    # Enrich live providers with user-specific data + real model lists
    providers = []
    for p in live_providers:
        provider_id = p.get("id", "")
        provider_key = provider_key_map.get(provider_id, provider_id)
        
        # Check if user has their own key for this provider
        has_user_key = bool(user_keys.get(provider_key) or user_keys.get(provider_id))
        
        # Determine if this provider uses credits (platform key, not user key, not local)
        is_local = provider_id in ["local", "codellama"]
        is_byok_only = p.get("byok_only", False)
        uses_credits = not has_user_key and not is_local and not is_byok_only and has_credits
        
        # Provider is available if:
        # 1. User has their own key (always available), OR
        # 2. Platform has key AND (has credits or is local), OR
        # 3. Is local and live status says available
        live_available = p.get("available", False)
        can_use = has_user_key or (has_credits and live_available and not is_byok_only) or (is_local and live_available) or (is_byok_only and has_user_key)
        
        # Use the LONGER model list — status manager has the full 72-model
        # TokenRouter catalog, while llm_service may have a shorter subset
        status_models = p.get("models", [])
        llm_models = llm_models_map.get(provider_key, [])
        real_models = status_models if len(status_models) >= len(llm_models) else llm_models
        if not real_models:
            real_models = llm_models or status_models
        
        providers.append({
            "id": provider_id,
            "provider_key": provider_key,
            "name": p.get("name", provider_id),
            "available": can_use,
            "has_user_key": has_user_key,
            "uses_credits": uses_credits,
            "model": p.get("model", ""),
            "models": real_models,
            "description": f"{p.get('model', '')} - {p.get('status', 'unknown')}",
            "capabilities": p.get("capabilities", []),
            "latency": p.get("latency"),
            "status": p.get("status", "unknown"),
        })
    
    # Inject user BYOK providers NOT already in the status list
    # This ensures user-connected providers (e.g. bedrock, chatgpt) always appear
    existing_ids = {p["id"] for p in providers} | {p.get("provider_key") for p in providers}
    BYOK_PROVIDER_META = {
        "bedrock": {"name": "AWS Bedrock", "model": "anthropic.claude-3-5-sonnet-20241022-v2:0", "models": ["anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic.claude-3-haiku-20240307-v1:0", "amazon.nova-pro-v1:0", "amazon.nova-lite-v1:0"], "capabilities": ["chat", "coding", "vision"]},
        "chatgpt": {"name": "ChatGPT (Direct)", "model": "chatgpt-4o-latest", "models": ["chatgpt-4o-latest", "gpt-4o-mini"], "capabilities": ["chat", "coding", "vision"]},
        "replicate": {"name": "Replicate", "model": "meta/meta-llama-3-70b-instruct", "models": ["meta/meta-llama-3-70b-instruct"], "capabilities": ["chat"]},
        "stability": {"name": "Stability AI", "model": "stable-diffusion-xl", "models": ["stable-diffusion-xl"], "capabilities": ["image"]},
        "elevenlabs": {"name": "ElevenLabs", "model": "eleven_turbo_v2", "models": ["eleven_turbo_v2"], "capabilities": ["audio"]},
        "kimi": {"name": "Kimi (Moonshot)", "model": "moonshot-v1-128k", "models": ["moonshot-v1-128k"], "capabilities": ["chat"]},
        "meta": {"name": "Meta AI (Llama)", "model": "llama-3.2-90b", "models": ["llama-3.2-90b", "llama-3.2-11b"], "capabilities": ["chat", "coding"]},
        "copilot": {"name": "Microsoft Copilot", "model": "gpt-4o", "models": ["gpt-4o", "gpt-4-turbo"], "capabilities": ["chat", "coding"]},
        "glm": {"name": "GLM (Zhipu AI)", "model": "glm-4", "models": ["glm-4", "glm-4v", "glm-3-turbo"], "capabilities": ["chat"]},
        "tokenrouter": {"name": "TokenRouter (72 Models)", "model": "google/gemini-3-flash-preview", "models": [
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
            "openai/gpt-5.3-codex", "openai/gpt-5.2-codex", "openai/gpt-5.1-codex-max",
            "openai/gpt-5.1-codex-mini", "qwen/qwen3-coder-next",
            "openai/gpt-5.4-image-2", "openai/gpt-5-image", "openai/gpt-5-image-mini",
            "google/gemini-3.1-flash-image-preview", "google/gemini-3-pro-image-preview",
            "google/gemini-2.5-flash-image", "bytedance-seed/seedream-4.5",
            "happyhorse-1.0-i2v", "happyhorse-1.0-t2v", "dreamina-seedance-2-0-260128",
            "dreamina-seedance-2-0-fast-260128", "kling-v3", "kling-v2-6", "MiniMax-Hailuo-2.3",
            "openai/gpt-audio", "openai/gpt-audio-mini",
        ], "capabilities": ["chat", "coding", "vision", "tools", "image", "video", "audio"]},
    }
    for uk_provider, uk_key in (user_keys or {}).items():
        if uk_provider not in existing_ids and uk_key:
            meta = BYOK_PROVIDER_META.get(uk_provider, {})
            real_models = llm_models_map.get(uk_provider, []) or meta.get("models", [])
            providers.append({
                "id": uk_provider,
                "provider_key": uk_provider,
                "name": meta.get("name", uk_provider.replace("_", " ").title()),
                "available": True,
                "has_user_key": True,
                "uses_credits": False,
                "model": meta.get("model", real_models[0] if real_models else ""),
                "models": real_models,
                "description": f"Connected via your API key",
                "capabilities": meta.get("capabilities", ["chat"]),
                "latency": None,
                "status": "byok_connected",
            })

    # Build fallback chain from available providers (excluding local for now)
    fallback_chain = [p["id"] for p in providers if p["available"] and p["id"] not in ["local", "codellama"]]
    
    # Message based on user status
    message = None
    if not is_paid_user:
        if credits_remaining <= 0:
            message = "You've used all your free credits. Add your own API keys in Settings or upgrade to continue."
        elif credits_remaining < 100:
            message = f"You have {credits_remaining} credits remaining. Add your own API keys for unlimited usage."
    
    return {
        "providers": providers,
        "default": "auto",
        "fallback_chain": fallback_chain,
        "fallback_chain_provider_keys": [provider_key_map.get(pid, pid) for pid in fallback_chain],
        "can_use_platform_keys": has_credits,
        "user_plan": user_plan,
        "is_paid_user": is_paid_user,
        "credits": {
            "remaining": credits_remaining if not is_paid_user else None,
            "total": FREE_TIER_CREDITS if not is_paid_user else None,
            "unlimited": is_paid_user
        },
        "message": message
    }


@router.get("/evidence-graph/{message_id}")
async def get_evidence_graph(
    message_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Get evidence graph data for a specific message.
    
    IMPROVED VERSION (Patch #50):
    - Uses actual XYZ coordinates from Hash Sphere (deterministic)
    - NLP-based anchor extraction instead of simple regex
    - Tracks actual memory usage from meta_data
    - No random noise in positions
    
    Shows what influenced the response (relevance), not how it was computed (reasoning).
    """
    import math
    
    # User ID is optional for evidence graph - it's read-only visualization data
    user_id = request.headers.get("x-user-id") or "anonymous"
    
    # Try to find the message
    try:
        result = await session.execute(
            select(ResonantChatMessage).where(ResonantChatMessage.id == UUID(message_id))
        )
        message = result.scalar_one_or_none()
    except Exception:
        message = None
    
    if not message:
        return {
            "message_id": message_id,
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0,
        }
    
    # Get conversation context - find related messages
    result = await session.execute(
        select(ResonantChatMessage)
        .where(ResonantChatMessage.chat_id == message.chat_id)
        .order_by(ResonantChatMessage.created_at.desc())
        .limit(10)
    )
    context_messages = result.scalars().all()
    
    nodes = []
    edges = []
    
    # IMPROVEMENT: Helper to generate DETERMINISTIC 3D positions (no random)
    def get_deterministic_position(index: int, total: int, layer: float = 0.0) -> list:
        """Generate deterministic positions in a sphere layout - NO RANDOM."""
        if total <= 1:
            return [0.0, 0.0, layer]
        angle = (2 * math.pi * index) / total
        radius = 0.6 + (abs(layer) * 0.3)
        return [
            round(math.cos(angle) * radius, 3),
            round(math.sin(angle) * radius, 3),
            round(layer, 3)  # NO random noise
        ]
    
    # IMPROVEMENT: Use actual XYZ coordinates from database if available
    response_xyz = [0.0, 0.0, 0.0]
    if message.xyz_x is not None and message.xyz_y is not None and message.xyz_z is not None:
        # Use actual Hash Sphere coordinates (deterministic)
        response_xyz = [
            round(float(message.xyz_x), 3),
            round(float(message.xyz_y), 3),
            round(float(message.xyz_z), 3)
        ]
    
    # Add the current message (AI response) as central node
    response_id = str(message.id)
    nodes.append({
        "id": response_id,
        "type": "query",
        "role": "assistant",
        "label": message.content[:40] + "..." if len(message.content) > 40 else message.content,
        "xyz": response_xyz,  # ACTUAL coordinates from Hash Sphere
    })
    
    # Find the user query that prompted this response
    user_query = None
    for ctx_msg in context_messages:
        if ctx_msg.role == "user" and ctx_msg.created_at < message.created_at:
            user_query = ctx_msg
            break
    
    if user_query:
        query_id = str(user_query.id)
        # IMPROVEMENT: Use actual XYZ from user message
        user_xyz = [0.0, 0.8, 0.2]  # Default position
        if user_query.xyz_x is not None and user_query.xyz_y is not None and user_query.xyz_z is not None:
            user_xyz = [
                round(float(user_query.xyz_x), 3),
                round(float(user_query.xyz_y), 3),
                round(float(user_query.xyz_z), 3)
            ]
        
        nodes.append({
            "id": query_id,
            "type": "query",
            "role": "user",
            "label": user_query.content[:40] + "..." if len(user_query.content) > 40 else user_query.content,
            "xyz": user_xyz,  # ACTUAL coordinates from Hash Sphere
        })
        edges.append({
            "source": query_id,
            "target": response_id,
            "type": "evidence",
        })
    
    # IMPROVEMENT: Use NLP-based anchor extraction from evidence_graph service
    anchors = []
    anchor_data = []
    
    # First check meta_data for stored anchors with XYZ
    meta_data = message.meta_data or {}
    stored_anchors = meta_data.get("anchors", [])
    
    if stored_anchors:
        # Use stored anchors from message processing
        for i, anchor in enumerate(stored_anchors[:5]):
            if isinstance(anchor, dict):
                anchor_data.append({
                    "text": anchor.get("text", str(anchor))[:30],
                    "xyz": anchor.get("xyz"),
                    "confidence": anchor.get("confidence", 0.7)
                })
            else:
                anchor_data.append({
                    "text": str(anchor)[:30],
                    "xyz": None,
                    "confidence": 0.7
                })
    else:
        # IMPROVEMENT: Use NLP-based extraction from evidence_graph service
        extracted = evidence_graph._extract_keyphrases_nlp(message.content, max_phrases=5)
        for item in extracted:
            anchor_data.append({
                "text": item.get("text", "")[:30],
                "xyz": None,  # No XYZ for extracted anchors
                "confidence": item.get("confidence", 0.5)
            })
    
    # Add anchor nodes with deterministic positions
    for i, anchor in enumerate(anchor_data[:5]):
        anchor_id = f"anchor_{i}"
        # Use stored XYZ if available, otherwise deterministic position
        anchor_xyz = anchor.get("xyz")
        if not anchor_xyz:
            anchor_xyz = get_deterministic_position(i, len(anchor_data[:5]), layer=-0.3)
        
        nodes.append({
            "id": anchor_id,
            "type": "anchor",
            "label": anchor["text"],
            "xyz": anchor_xyz,
        })
        edges.append({
            "source": anchor_id,
            "target": response_id,
            "type": "anchor",
        })
    
    # IMPROVEMENT: Check actual memory usage from meta_data instead of keyword detection
    memory_count = meta_data.get("memory_count", 0)
    memory_contributors = meta_data.get("memory_contributors", [])
    
    # Add memory nodes based on ACTUAL memory usage
    if memory_count > 0 or memory_contributors:
        for i, mem in enumerate(memory_contributors[:3]):
            memory_id = f"memory_{i}"
            # Use actual XYZ from memory if available
            mem_xyz = mem.get("xyz") if isinstance(mem, dict) else None
            if not mem_xyz:
                mem_xyz = get_deterministic_position(i, min(memory_count, 3), layer=0.4)
            
            mem_label = "Prior Context"
            if isinstance(mem, dict):
                mem_label = mem.get("text", "Memory")[:25]
            
            nodes.append({
                "id": memory_id,
                "type": "memory",
                "label": mem_label,
                "xyz": mem_xyz,
            })
            edges.append({
                "source": memory_id,
                "target": response_id,
                "type": "memory",
            })
    elif memory_count == 0:
        # Fallback: Check for memory indicators in content (less accurate)
        memory_indicators = [
            "you mentioned", "earlier", "previously", "last time",
            "remember", "as we discussed", "you said", "your preference"
        ]
        has_memory = any(ind in message.content.lower() for ind in memory_indicators)
        
        if has_memory:
            memory_id = "memory_0"
            nodes.append({
                "id": memory_id,
                "type": "memory",
                "label": "Prior Context (inferred)",
                "xyz": get_deterministic_position(0, 1, layer=0.4),
            })
            edges.append({
                "source": memory_id,
                "target": response_id,
                "type": "memory",
            })
    
    # Add context from recent conversation with ACTUAL XYZ coordinates
    context_count = 0
    for ctx_msg in context_messages[1:4]:  # Skip first (current), take up to 3
        if ctx_msg.id != message.id and ctx_msg.role == "assistant":
            context_count += 1
            ctx_id = f"context_{context_count}"
            
            # IMPROVEMENT: Use actual XYZ from context message
            ctx_xyz = get_deterministic_position(context_count, 3, layer=0.5)
            if ctx_msg.xyz_x is not None and ctx_msg.xyz_y is not None and ctx_msg.xyz_z is not None:
                ctx_xyz = [
                    round(float(ctx_msg.xyz_x), 3),
                    round(float(ctx_msg.xyz_y), 3),
                    round(float(ctx_msg.xyz_z), 3)
                ]
            
            nodes.append({
                "id": ctx_id,
                "type": "memory",
                "label": f"Prior response {context_count}",
                "xyz": ctx_xyz,
            })
            edges.append({
                "source": ctx_id,
                "target": response_id,
                "type": "memory",
            })
            if context_count >= 2:
                break
    
    return {
        "message_id": message_id,
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        # NEW: Include accuracy metadata
        "meta": {
            "uses_actual_xyz": message.xyz_x is not None,
            "memory_count": memory_count,
            "anchor_count": len(anchor_data),
        }
    }


@router.get("/metrics/{chat_id}")
async def get_chat_metrics(
    chat_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Get comprehensive metrics for a chat conversation.
    
    Returns:
    - quality: Average resonance score (0-1, higher is better)
    - hallucination: Estimated hallucination rate (0-1, lower is better)
    - tokens: Estimated token count
    - message_count: Total messages in conversation
    - provider_breakdown: Usage by AI provider
    """
    # user_id is optional - if provided, we verify ownership
    user_id = request.headers.get("x-user-id")
    
    try:
        # Get chat
        result = await session.execute(
            select(ResonantChat).where(ResonantChat.id == UUID(chat_id))
        )
        chat = result.scalar_one_or_none()
        
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Only check ownership if user_id is provided
        if user_id and str(chat.user_id) != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get all messages
        result = await session.execute(
            select(ResonantChatMessage)
            .where(ResonantChatMessage.chat_id == UUID(chat_id))
            .order_by(ResonantChatMessage.created_at.asc())
        )
        messages = result.scalars().all()
        
        if not messages:
            return {
                "chat_id": chat_id,
                "quality": 0.0,
                "hallucination": 0.0,
                "tokens": 0,
                "message_count": 0,
                "provider_breakdown": {},
                "metrics_available": False,
            }
        
        # Calculate Quality (average resonance score)
        assistant_messages = [m for m in messages if m.role == "assistant"]
        quality_scores = [m.resonance_score for m in assistant_messages if m.resonance_score is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # Calculate Hallucination score
        # Based on: error indicators, uncertainty markers, unsupported claims
        hallucination_scores = []
        for msg in assistant_messages:
            score = _calculate_hallucination_score(msg.content)
            hallucination_scores.append(score)
        avg_hallucination = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0
        
        # Calculate Token count (estimate: ~4 chars per token)
        total_chars = sum(len(m.content) for m in messages)
        estimated_tokens = total_chars // 4
        
        # Provider breakdown
        provider_counts = {}
        for msg in assistant_messages:
            provider = msg.ai_provider or "unknown"
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        return {
            "chat_id": chat_id,
            "quality": round(avg_quality, 4),
            "hallucination": round(avg_hallucination, 4),
            "tokens": estimated_tokens,
            "message_count": len(messages),
            "assistant_message_count": len(assistant_messages),
            "provider_breakdown": provider_counts,
            "metrics_available": True,
            "quality_samples": len(quality_scores),
            "hallucination_samples": len(hallucination_scores),
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chat ID format")


@router.get("/message-metrics/{message_id}")
async def get_message_metrics(
    message_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Get detailed metrics for a specific message.
    
    Returns quality, hallucination, token count, and other metrics for a single message.
    """
    # user_id is optional for read-only metrics
    user_id = request.headers.get("x-user-id")
    
    try:
        result = await session.execute(
            select(ResonantChatMessage).where(ResonantChatMessage.id == UUID(message_id))
        )
        message = result.scalar_one_or_none()
        
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Get context: find related messages in the same chat to calculate anchor following
        chat_result = await session.execute(
            select(ResonantChatMessage)
            .where(ResonantChatMessage.chat_id == message.chat_id)
            .order_by(ResonantChatMessage.created_at.asc())
        )
        chat_messages = chat_result.scalars().all()
        
        # Calculate comprehensive metrics
        content = message.content or ""
        quality = message.resonance_score or 0.5
        tokens = len(content) // 4  # Estimate
        
        # Get anchors from message metadata if available
        anchors = []
        rag_sources = []
        if message.meta_data and isinstance(message.meta_data, dict):
            raw_anchors = message.meta_data.get("anchors", [])
            rag_sources = message.meta_data.get("rag_sources", [])
            # Normalize anchors: may be dicts from _extract_keyphrases_nlp or plain strings
            anchors = []
            for a in raw_anchors:
                if isinstance(a, dict):
                    anchors.append(a.get("text", a.get("content", str(a))))
                elif isinstance(a, str):
                    anchors.append(a)

        
        # Convert chat_messages to dict format for enhanced metrics
        chat_messages_dict = [
            {"role": m.role, "content": m.content or "", "id": str(m.id)}
            for m in chat_messages
        ]

        # Fetch real embeddings from memory_service for semantic similarity (best-effort)
        response_embedding = None
        recent_user_embedding = None
        all_user_embedding = None
        semantic_similarity_source = "fallback"
        try:
            recent_user_text = "\n".join(
                [m["content"] for m in chat_messages_dict[-5:] if m.get("role") == "user" and m.get("content")]
            )
            all_user_text = "\n".join(
                [m["content"] for m in chat_messages_dict if m.get("role") == "user" and m.get("content")]
            )

            embed_payload = {
                "texts": [content, recent_user_text, all_user_text],
                "task": "search_document",
            }
            # IMPORTANT: don't send x-user-id header here to avoid credit deduction
            embed_res = await service_client.call_service(
                "memory_service",
                "POST",
                "http://memory_service:8000/memory/embed",
                json=embed_payload,
                headers={},
                timeout=httpx.Timeout(1.5, connect=0.5),
            )
            embeddings = (embed_res or {}).get("embeddings") if isinstance(embed_res, dict) else None
            if isinstance(embeddings, list) and len(embeddings) >= 3:
                response_embedding = embeddings[0]
                recent_user_embedding = embeddings[1]
                all_user_embedding = embeddings[2]
                semantic_similarity_source = "embeddings"
        except Exception as e:
            logger.debug(f"Embedding fetch failed (non-critical): {e}")
        
        # Use Enhanced Metrics Calculator (NLP + Semantic Analysis)
        xyz_coords = (message.xyz_x, message.xyz_y, message.xyz_z) if message.xyz_x else None
        enhanced_result = enhanced_metrics_calculator.calculate_all_metrics(
            content=content,
            message_id=message_id,
            message_role=message.role,
            base_resonance_score=quality,
            chat_messages=chat_messages_dict,
            message_hash=message.hash,
            xyz_coords=xyz_coords,
            anchors=anchors,
            rag_sources=rag_sources,
            response_embedding=response_embedding,
            recent_user_embedding=recent_user_embedding,
            all_user_embedding=all_user_embedding,
        )
        
        # Extract enhanced metrics
        resonant_energy = enhanced_result.resonant_energy
        evidence_score = enhanced_result.evidence_score
        anchor_following = enhanced_result.anchor_following
        context_coherence = enhanced_result.context_coherence
        memory_utilization = enhanced_result.memory_utilization
        
        # Additional metrics from specialized services
        # Sentiment & Emotion Detection
        from ..services.sentiment_detection import detect_emotion, detect_sentiment
        emotion, emotion_confidence = detect_emotion(content)
        sentiment, sentiment_confidence = detect_sentiment(content)
        
        # Hallucination Detection (enhanced: system-prompt grounding + KB + LLM judge)
        from ..services.hallucination_detector import hallucination_detector, hallucination_settings, user_knowledge_base
        h_user_id = request.headers.get("x-user-id", "anonymous")
        
        # Load settings from DB if not in memory cache (e.g. after container restart)
        from ..models import HallucinationSettings as HSModel, KnowledgeBaseEntryDB
        if h_user_id not in hallucination_settings._configs:
            try:
                hs_result = await session.execute(
                    select(HSModel).where(HSModel.user_id == h_user_id)
                )
                hs_row = hs_result.scalar_one_or_none()
                if hs_row:
                    hallucination_settings.update_config(
                        h_user_id,
                        system_prompt_grounding=bool(hs_row.system_prompt_grounding),
                        llm_as_judge=bool(hs_row.llm_as_judge),
                        knowledge_base_check=bool(hs_row.knowledge_base_check),
                    )
                    logger.info(f"[HALLUCINATION] Loaded settings from DB for user {h_user_id[:8]}...")
            except Exception as e:
                logger.error(f"[HALLUCINATION] DB settings load failed: {e}")
        
        # Always load KB entries from DB if in-memory cache is empty for this user
        if not user_knowledge_base.get_entries(h_user_id):
            try:
                kb_result = await session.execute(
                    select(KnowledgeBaseEntryDB).where(KnowledgeBaseEntryDB.user_id == h_user_id)
                )
                kb_rows = kb_result.scalars().all()
                if kb_rows:
                    user_knowledge_base._entries[h_user_id] = []
                    for row in kb_rows:
                        user_knowledge_base.add_entry(h_user_id, str(row.id), row.title, row.content, row.entry_type)
                    logger.info(f"[HALLUCINATION] Loaded {len(kb_rows)} KB entries from DB for user {h_user_id[:8]}...")
            except Exception as e:
                logger.error(f"[HALLUCINATION] DB KB entries load failed: {e}")
        
        h_config = hallucination_settings.get_config(h_user_id)
        kb_entries_count = len(user_knowledge_base.get_entries(h_user_id))
        kb_corpus = user_knowledge_base.get_corpus(h_user_id) if h_config.knowledge_base_check else ""
        logger.info(f"[HALLUCINATION] user={h_user_id[:8]}... config: grounding={h_config.system_prompt_grounding}, kb_check={h_config.knowledge_base_check}, llm_judge={h_config.llm_as_judge} | KB entries={kb_entries_count}, corpus_len={len(kb_corpus)}")
        
        # Extract system prompt and user message from chat history
        system_prompt_text = ""
        last_user_message = ""
        for cm in chat_messages_dict:
            if cm.get("role") == "system":
                system_prompt_text = cm.get("content", "")
            if cm.get("role") == "user":
                last_user_message = cm.get("content", "")
        
        # Get router for LLM-as-judge if enabled
        judge_router = None
        if h_config.llm_as_judge:
            try:
                from ..domain.provider import get_router_for_internal_use
                judge_router = get_router_for_internal_use()
            except Exception:
                pass
        
        hallucination_report = await hallucination_detector.analyze_full(
            response=content,
            task="",
            config=h_config,
            system_prompt=system_prompt_text,
            user_message=last_user_message,
            kb_corpus=kb_corpus,
            rag_sources=rag_sources,
            anchors=anchors,
            router=judge_router,
        )
        
        # Calculate response time if we have metadata
        response_time_ms = None
        if message.meta_data and isinstance(message.meta_data, dict):
            response_time_ms = message.meta_data.get("response_time_ms")
        
        # Get agent feedback stats if available
        from ..services.user_feedback import user_feedback
        # Use agent_type from meta_data (e.g. "agent_summary") not ai_provider ("groq")
        _fb_agent_type = None
        if message.meta_data and isinstance(message.meta_data, dict):
            _fb_agent_type = message.meta_data.get("agent_type")
        if not _fb_agent_type:
            _fb_agent_type = message.ai_provider or "reasoning"
        if not user_feedback._db_initialized:
            await user_feedback.load_from_database(session)
        feedback_stats_obj = user_feedback.get_agent_stats(_fb_agent_type)
        if not feedback_stats_obj and not _fb_agent_type.startswith("agent_"):
            feedback_stats_obj = user_feedback.get_agent_stats("agent_" + _fb_agent_type)
        
        # Get DSID metrics for this message - try in-memory first, then fall back to meta_data
        from ..services.dsid_integration import dsid_integration
        dsid_data = dsid_integration.get_dsid_by_message(message_id)
        
        # Fall back to stored DSID data in meta_data (persists across container restarts)
        stored_dsid = None
        if message.meta_data and isinstance(message.meta_data, dict):
            stored_dsid = message.meta_data.get("dsid")
        
        # Calculate T1 (Trust Level 1) - Basic verification score
        t1_score = 0.0
        if dsid_data:
            # T1 based on content hash verification from in-memory cache
            is_valid, _ = dsid_integration.verify_message(message_id, message.content or "")
            t1_score = 1.0 if is_valid else 0.5
        elif stored_dsid:
            # Use stored DSID data - verify content hash matches
            import hashlib
            current_hash = hashlib.sha256((message.content or "").encode()).hexdigest()
            stored_hash = stored_dsid.get("content_hash", "")
            t1_score = 1.0 if current_hash == stored_hash else 0.5
        
        # Calculate T3 (Trust Level 3) - Full provenance chain
        t3_score = 0.0
        lineage_depth = 0
        if dsid_data:
            lineage = dsid_integration.get_message_lineage(message_id)
            lineage_depth = len(lineage)
            # T3 based on lineage completeness and merkle verification
            if lineage_depth > 0:
                t3_score = min(1.0, lineage_depth / 5)  # Full score at 5+ messages in chain
                # Boost if merkle root exists
                chat_lineage = dsid_integration.get_conversation_lineage(message.chat_id) if message.chat_id else None
                if chat_lineage and chat_lineage.merkle_root:
                    t3_score = min(1.0, t3_score + 0.2)
        elif stored_dsid:
            # Use stored lineage depth from meta_data
            lineage_depth = stored_dsid.get("lineage_depth", 0)
            if lineage_depth > 0:
                t3_score = min(1.0, lineage_depth / 5)
            # If we have parent_dsid, that's proof of provenance chain
            if stored_dsid.get("parent_dsid"):
                t3_score = max(t3_score, 0.4)  # Minimum 40% if we have parent link
            if stored_dsid.get("root_dsid"):
                t3_score = max(t3_score, 0.2)  # Minimum 20% if we have root link
        
        # Determine flagged status based on hallucination and quality
        is_flagged = hallucination_report.risk_level == "high" or quality < 0.3
        verification_status = "flagged" if is_flagged else "passed"
        
        # Extract actual LLM provider from metadata if available
        actual_llm_provider = None
        agent_type_from_meta = None
        model_used = None
        fallback_chain = None
        was_fallback = False
        preferred_provider_used = None
        token_usage = None
        if message.meta_data and isinstance(message.meta_data, dict):
            # Try to get actual LLM provider from meta_data
            actual_llm_provider = message.meta_data.get("actual_llm_provider")
            if not actual_llm_provider:
                provider_metadata = message.meta_data.get("provider_metadata", {})
                actual_llm_provider = provider_metadata.get("provider") or provider_metadata.get("model", "").split("/")[0] if provider_metadata.get("model") else None
            # Get agent type from meta_data
            agent_type_from_meta = message.meta_data.get("agent_type")
            # Get model, fallback chain, and usage from meta_data
            model_used = message.meta_data.get("model")
            if not model_used:
                pm = message.meta_data.get("provider_metadata", {})
                model_used = pm.get("model") if isinstance(pm, dict) else None
            fallback_chain = message.meta_data.get("fallback_chain")
            was_fallback = message.meta_data.get("was_fallback", False)
            preferred_provider_used = message.meta_data.get("preferred_provider")
            token_usage = message.meta_data.get("usage")
            if not token_usage:
                pm = message.meta_data.get("provider_metadata", {})
                token_usage = pm.get("usage") if isinstance(pm, dict) else None
        
        # Determine if this was a team response
        team_id = None
        if message.ai_provider and message.ai_provider.startswith("team_"):
            team_id = message.ai_provider
        
        # Use actual_llm_provider if available, otherwise fall back to ai_provider
        # But only if ai_provider doesn't start with "agent_" (which is just a wrapper)
        display_provider = actual_llm_provider
        if not display_provider:
            if message.ai_provider and not message.ai_provider.startswith("agent_"):
                display_provider = message.ai_provider
            else:
                display_provider = "unknown"
        
        return {
            "message_id": message_id,
            "role": message.role,
            "quality": round(quality, 4),
            "hallucination": round(hallucination_report.risk_score, 4),
            "tokens": tokens,
            "provider": display_provider,
            "model": model_used,
            "preferred_provider": preferred_provider_used,
            "was_fallback": was_fallback,
            "fallback_chain": fallback_chain,
            "token_usage": token_usage,
            "agent_id": agent_type_from_meta,
            "team_id": team_id,
            "hash": message.hash,
            "xyz": [message.xyz_x, message.xyz_y, message.xyz_z] if message.xyz_x else None,
            "created_at": message.created_at.isoformat() if message.created_at else None,
            "anchors": anchors,
            "metrics": {
                "resonant_energy": round(resonant_energy, 4),
                "hallucination": round(hallucination_report.risk_score, 4),
                "evidence": round(evidence_score, 4),
                "anchor_following": round(anchor_following, 4),
                "context_coherence": round(context_coherence, 4),
                "memory_utilization": round(memory_utilization, 4),
                "sentiment": sentiment.value,
                "sentiment_confidence": round(sentiment_confidence, 4),
                "emotion": emotion.value,
                "emotion_confidence": round(emotion_confidence, 4),
                "hallucination_risk_level": hallucination_report.risk_level,
                "hallucination_flags": len(hallucination_report.flags),
                "hallucination_details": [
                    {"type": f.type, "content": f.content[:100], "confidence": f.confidence}
                    for f in hallucination_report.flags[:5]
                ],
                "response_time_ms": response_time_ms,
                # Agent feedback metrics
                "agent_feedback": {
                    "positive_count": feedback_stats_obj.positive_count if feedback_stats_obj else 0,
                    "negative_count": feedback_stats_obj.negative_count if feedback_stats_obj else 0,
                    "satisfaction_rate": feedback_stats_obj.satisfaction_rate if feedback_stats_obj else 0,
                    "total_feedback": feedback_stats_obj.total_count if feedback_stats_obj else 0,
                    "trend": feedback_stats_obj.recent_trend if feedback_stats_obj else "stable",
                },
                # DSID Trust metrics (use stored data as fallback)
                "dsid": {
                    "dsid_id": dsid_data.dsid if dsid_data else (stored_dsid.get("dsid_id") if stored_dsid else None),
                    "t1_score": round(t1_score, 4),
                    "t3_score": round(t3_score, 4),
                    "lineage_depth": lineage_depth,
                    "verification_status": verification_status,
                    "content_hash": dsid_data.content_hash if dsid_data else (stored_dsid.get("content_hash") if stored_dsid else None),
                    "parent_dsid": dsid_data.parent_dsid if dsid_data else (stored_dsid.get("parent_dsid") if stored_dsid else None),
                },
                # Enhanced metrics breakdown (transparency)
                "enhanced_metrics": {
                    "calculation_method": "nlp_semantic_analysis",
                    "confidence": round(enhanced_result.confidence, 4),
                    "semantic_similarity_source": semantic_similarity_source,
                    "resonant_energy_breakdown": {k: round(v, 4) for k, v in enhanced_result.resonant_energy_breakdown.items()},
                    "evidence_breakdown": {k: round(v, 4) for k, v in enhanced_result.evidence_breakdown.items()},
                    "anchor_breakdown": {k: round(v, 4) for k, v in enhanced_result.anchor_breakdown.items()},
                    "coherence_breakdown": {k: round(v, 4) for k, v in enhanced_result.coherence_breakdown.items()},
                    "memory_breakdown": {k: round(v, 4) for k, v in enhanced_result.memory_breakdown.items()},
                },
                # RAG verification results (hallucination grounding)
                "rag_verification": hallucination_report.rag_verification if hallucination_report.rag_verification else {
                    "verified": False,
                    "reason": "no_rag_sources",
                    "support_score": 0.0,
                },
                # Enhanced verification results (system-prompt grounding, KB, LLM judge)
                "claim_verification": hallucination_report.claim_verification if hallucination_report.claim_verification else None,
            }
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid message ID format")


def _calculate_resonant_energy(message: ResonantChatMessage, chat_messages: List[ResonantChatMessage]) -> float:
    """
    Calculate Resonant Energy - measures how well the response resonates with the conversation.
    
    Factors:
    - Base resonance score from Hash Sphere
    - Response length and completeness
    - Structural quality (code blocks, lists, headers)
    - Coherence with previous messages
    """
    content = message.content or ""
    score = message.resonance_score or 0.5
    
    # Boost for well-structured responses
    if "```" in content:
        score += 0.08  # Code blocks indicate technical depth
    if any(marker in content for marker in ["1.", "2.", "- ", "* ", "##"]):
        score += 0.05  # Lists and headers indicate organization
    
    # Boost for appropriate length
    if 200 < len(content) < 2000:
        score += 0.05  # Good length range
    elif len(content) > 2000 and "```" in content:
        score += 0.03  # Long but has code
    elif len(content) < 50:
        score -= 0.1  # Too short
    
    # Boost for hash presence (indicates proper processing)
    if message.hash:
        score += 0.05
    
    # Boost for XYZ coordinates (indicates Hash Sphere integration)
    if message.xyz_x is not None:
        score += 0.03
    
    return max(0.1, min(1.0, score))


def _calculate_evidence_score(content: str, message: ResonantChatMessage) -> float:
    """
    Calculate Evidence Score - measures how well-grounded the response is.
    
    Factors:
    - Presence of citations or references
    - Use of specific examples
    - Avoidance of vague claims
    - Hash Sphere memory integration
    """
    if not content:
        return 0.5
    
    score = 0.5  # Base score
    content_lower = content.lower()
    
    # Evidence indicators (increase score)
    evidence_markers = [
        "according to", "based on", "for example", "specifically",
        "in this case", "as shown", "the code", "the error",
        "from your", "you mentioned", "earlier you", "in our conversation",
    ]
    evidence_count = sum(1 for marker in evidence_markers if marker in content_lower)
    score += 0.08 * min(evidence_count, 4)
    
    # Code blocks are strong evidence
    code_blocks = content.count("```")
    if code_blocks > 0:
        score += 0.15 * min(code_blocks, 2)
    
    # Specific numbers/data
    import re
    numbers = len(re.findall(r'\b\d+\b', content))
    if numbers > 2:
        score += 0.05
    
    # URLs/links indicate external references
    if "http" in content_lower or "www." in content_lower:
        score += 0.1
    
    # Hash presence indicates memory grounding
    if message.hash:
        score += 0.1
    
    # Vague language reduces score
    vague_markers = ["maybe", "perhaps", "i think", "probably", "might"]
    vague_count = sum(1 for marker in vague_markers if marker in content_lower)
    score -= 0.05 * min(vague_count, 2)
    
    return max(0.1, min(1.0, score))


def _calculate_anchor_following(message: ResonantChatMessage, chat_messages: List[ResonantChatMessage]) -> float:
    """
    Calculate Anchor Following - measures how well the response follows conversation context.
    
    Factors:
    - References to previous messages
    - Topic continuity
    - User question addressing
    - Memory anchor utilization
    """
    content = message.content or ""
    content_lower = content.lower()
    
    if message.role != "assistant" or not chat_messages:
        return 0.5
    
    score = 0.4  # Base score
    
    # Find the user message this is responding to
    msg_index = next((i for i, m in enumerate(chat_messages) if m.id == message.id), -1)
    if msg_index > 0:
        prev_messages = chat_messages[max(0, msg_index-3):msg_index]
        
        # Check for keyword overlap with recent user messages
        user_keywords = set()
        for prev_msg in prev_messages:
            if prev_msg.role == "user":
                words = prev_msg.content.lower().split()
                # Filter out common words
                significant_words = [w for w in words if len(w) > 4]
                user_keywords.update(significant_words[:10])
        
        # Count keyword matches
        matches = sum(1 for kw in user_keywords if kw in content_lower)
        if matches > 0:
            score += 0.1 * min(matches / max(len(user_keywords), 1), 1.0)
    
    # Direct reference indicators
    reference_markers = [
        "you asked", "your question", "you mentioned", "as you said",
        "regarding your", "to answer", "here's", "let me",
    ]
    ref_count = sum(1 for marker in reference_markers if marker in content_lower)
    score += 0.08 * min(ref_count, 3)
    
    # Task continuation indicators
    task_markers = [
        "continuing", "next step", "as discussed", "following up",
        "here are the", "option", "choice",
    ]
    task_count = sum(1 for marker in task_markers if marker in content_lower)
    score += 0.06 * min(task_count, 2)
    
    # Resonance score correlation
    if message.resonance_score and message.resonance_score > 0.6:
        score += 0.1
    
    return max(0.1, min(1.0, score))


def _calculate_context_coherence(message: ResonantChatMessage, chat_messages: List[ResonantChatMessage]) -> float:
    """
    Calculate Context Coherence - measures how well the response relates to the overall conversation.
    
    Factors:
    - Topic consistency across messages
    - Semantic similarity with user questions
    - Logical flow and progression
    """
    content = message.content or ""
    content_lower = content.lower()
    
    if message.role != "assistant" or not chat_messages or len(chat_messages) < 2:
        return 0.5
    
    score = 0.5  # Base score
    
    # Find all user messages in the conversation
    user_messages = [m for m in chat_messages if m.role == "user"]
    
    if not user_messages:
        return 0.5
    
    # Extract key topics from user messages
    all_user_words = set()
    for msg in user_messages:
        words = msg.content.lower().split()
        significant = [w for w in words if len(w) > 4 and w.isalpha()]
        all_user_words.update(significant[:15])
    
    # Check topic overlap
    response_words = set(w for w in content_lower.split() if len(w) > 4 and w.isalpha())
    overlap = len(all_user_words & response_words)
    
    if overlap > 5:
        score += 0.2
    elif overlap > 2:
        score += 0.1
    
    # Check for conversation flow indicators
    flow_markers = [
        "as mentioned", "building on", "to continue", "following",
        "regarding", "about your", "for your", "based on",
    ]
    flow_count = sum(1 for marker in flow_markers if marker in content_lower)
    score += 0.05 * min(flow_count, 3)
    
    # Check for question answering patterns
    if any(m.content.strip().endswith("?") for m in user_messages[-3:]):
        # Recent question was asked
        answer_indicators = ["here", "this", "the answer", "you can", "to do this"]
        if any(ind in content_lower for ind in answer_indicators):
            score += 0.1
    
    # Penalize off-topic indicators
    off_topic = ["by the way", "unrelated", "different topic", "changing subject"]
    if any(ind in content_lower for ind in off_topic):
        score -= 0.15
    
    return max(0.1, min(1.0, score))


def _calculate_hallucination_score(content: str) -> float:
    """
    Calculate hallucination score for a message.
    
    Factors that increase hallucination score:
    - Vague or uncertain language
    - Claims without evidence markers
    - Contradictions or inconsistencies
    - Overly confident assertions about unknowable things
    
    Returns a score between 0.0 (no hallucination) and 1.0 (high hallucination risk)
    """
    if not content:
        return 0.0
    
    score = 0.0
    content_lower = content.lower()
    
    # Uncertainty markers (reduce hallucination score - these are honest)
    honest_markers = [
        "i'm not sure", "i don't know", "i cannot", "i'm unable",
        "according to", "based on", "it appears", "it seems",
        "may be", "might be", "could be", "possibly",
    ]
    honest_count = sum(1 for marker in honest_markers if marker in content_lower)
    if honest_count > 0:
        score -= 0.05 * min(honest_count, 3)  # Honesty reduces hallucination risk
    
    # Overconfidence markers (increase hallucination score)
    overconfident_markers = [
        "definitely", "absolutely", "certainly", "always", "never",
        "everyone knows", "it's obvious", "clearly", "undoubtedly",
        "without a doubt", "100%", "guaranteed",
    ]
    overconfident_count = sum(1 for marker in overconfident_markers if marker in content_lower)
    if overconfident_count > 0:
        score += 0.1 * min(overconfident_count, 3)
    
    # Fabrication indicators (increase hallucination score)
    fabrication_markers = [
        "studies show", "research indicates", "experts say",
        "according to statistics", "data shows",
    ]
    # These are risky if not backed by actual sources
    fabrication_count = sum(1 for marker in fabrication_markers if marker in content_lower)
    if fabrication_count > 0 and "source" not in content_lower and "http" not in content_lower:
        score += 0.15 * min(fabrication_count, 2)
    
    # Very short responses might be evasive
    if len(content) < 50:
        score += 0.05
    
    # Very long responses without structure might ramble
    if len(content) > 2000 and "```" not in content and "\n\n" not in content:
        score += 0.1
    
    # Clamp to valid range
    return max(0.0, min(1.0, score))


# ============================================
# HALLUCINATION DETECTION SETTINGS & KNOWLEDGE BASE
# ============================================

@router.get("/hallucination-settings")
async def get_hallucination_settings(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Get current hallucination detection settings for the user (DB-persisted)."""
    user_id = request.headers.get("x-user-id", "anonymous")
    from ..models import HallucinationSettings, KnowledgeBaseEntryDB
    
    # Get or create settings from DB
    result = await session.execute(
        select(HallucinationSettings).where(HallucinationSettings.user_id == user_id)
    )
    settings_row = result.scalar_one_or_none()
    
    spg = True
    laj = False
    kbc = False
    if settings_row:
        spg = bool(settings_row.system_prompt_grounding)
        laj = bool(settings_row.llm_as_judge)
        kbc = bool(settings_row.knowledge_base_check)
    
    # Also sync to in-memory cache for metrics endpoint
    from ..services.hallucination_detector import hallucination_settings
    hallucination_settings.update_config(user_id, system_prompt_grounding=spg, llm_as_judge=laj, knowledge_base_check=kbc)
    
    # Get KB entries from DB
    kb_result = await session.execute(
        select(KnowledgeBaseEntryDB).where(KnowledgeBaseEntryDB.user_id == user_id).order_by(KnowledgeBaseEntryDB.created_at.desc())
    )
    kb_rows = kb_result.scalars().all()
    
    # Sync KB to in-memory cache
    from ..services.hallucination_detector import user_knowledge_base
    user_knowledge_base._entries[user_id] = []
    for row in kb_rows:
        user_knowledge_base.add_entry(user_id, str(row.id), row.title, row.content, row.entry_type)
    
    return {
        "settings": {
            "system_prompt_grounding": spg,
            "llm_as_judge": laj,
            "knowledge_base_check": kbc,
        },
        "knowledge_base": {
            "entry_count": len(kb_rows),
            "entries": [
                {"id": str(row.id), "title": row.title, "type": row.entry_type, "length": len(row.content or ""), "file_name": row.file_name}
                for row in kb_rows
            ],
        },
    }


@router.patch("/hallucination-settings")
async def update_hallucination_settings(
    body: HallucinationSettingsRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Update hallucination detection settings for the user (DB-persisted)."""
    user_id = request.headers.get("x-user-id", "anonymous")
    from ..models import HallucinationSettings
    
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No settings provided")
    
    # Upsert in DB
    result = await session.execute(
        select(HallucinationSettings).where(HallucinationSettings.user_id == user_id)
    )
    settings_row = result.scalar_one_or_none()
    
    if not settings_row:
        settings_row = HallucinationSettings(user_id=user_id)
        session.add(settings_row)
    
    if "system_prompt_grounding" in updates:
        settings_row.system_prompt_grounding = 1.0 if updates["system_prompt_grounding"] else 0.0
    if "llm_as_judge" in updates:
        settings_row.llm_as_judge = 1.0 if updates["llm_as_judge"] else 0.0
    if "knowledge_base_check" in updates:
        settings_row.knowledge_base_check = 1.0 if updates["knowledge_base_check"] else 0.0
    
    await session.commit()
    
    spg = bool(settings_row.system_prompt_grounding)
    laj = bool(settings_row.llm_as_judge)
    kbc = bool(settings_row.knowledge_base_check)
    
    # Sync to in-memory cache
    from ..services.hallucination_detector import hallucination_settings
    hallucination_settings.update_config(user_id, system_prompt_grounding=spg, llm_as_judge=laj, knowledge_base_check=kbc)
    
    return {
        "settings": {
            "system_prompt_grounding": spg,
            "llm_as_judge": laj,
            "knowledge_base_check": kbc,
        },
    }


@router.post("/knowledge-base")
async def add_knowledge_base_entry(
    body: KnowledgeBaseAddRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Add a knowledge base entry for hallucination cross-referencing (DB-persisted)."""
    user_id = request.headers.get("x-user-id", "anonymous")
    from ..models import KnowledgeBaseEntryDB
    
    entry = KnowledgeBaseEntryDB(
        user_id=user_id,
        title=body.title,
        content=body.content,
        entry_type=body.entry_type,
    )
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    
    # Sync to in-memory cache
    from ..services.hallucination_detector import user_knowledge_base
    user_knowledge_base.add_entry(user_id, str(entry.id), entry.title, entry.content, entry.entry_type)
    
    return {
        "id": str(entry.id),
        "title": entry.title,
        "type": entry.entry_type,
        "length": len(entry.content),
    }


@router.post("/knowledge-base/upload")
async def upload_knowledge_base_file(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Upload a file (.txt, .md, .csv, .json) as a knowledge base entry."""
    user_id = request.headers.get("x-user-id", "anonymous")
    from ..models import KnowledgeBaseEntryDB
    
    form = await request.form()
    file = form.get("file")
    title = form.get("title", "")
    entry_type = form.get("entry_type", "document")
    
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_name = getattr(file, "filename", "unknown")
    file_bytes = await file.read()
    
    # Decode text content
    try:
        content = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content = file_bytes.decode("latin-1")
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode file. Only text files (.txt, .md, .csv, .json) are supported.")
    
    if not content.strip():
        raise HTTPException(status_code=400, detail="File is empty")
    
    # Limit content size (500KB max)
    if len(content) > 500_000:
        content = content[:500_000]
    
    if not title:
        title = file_name
    
    entry = KnowledgeBaseEntryDB(
        user_id=user_id,
        title=title,
        content=content,
        entry_type=entry_type,
        file_name=file_name,
    )
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    
    # Sync to in-memory cache
    from ..services.hallucination_detector import user_knowledge_base
    user_knowledge_base.add_entry(user_id, str(entry.id), entry.title, entry.content, entry.entry_type)
    
    return {
        "id": str(entry.id),
        "title": entry.title,
        "type": entry.entry_type,
        "length": len(entry.content),
        "file_name": file_name,
    }


@router.get("/knowledge-base")
async def list_knowledge_base(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """List all knowledge base entries for the user (DB-persisted)."""
    user_id = request.headers.get("x-user-id", "anonymous")
    from ..models import KnowledgeBaseEntryDB
    
    result = await session.execute(
        select(KnowledgeBaseEntryDB).where(KnowledgeBaseEntryDB.user_id == user_id).order_by(KnowledgeBaseEntryDB.created_at.desc())
    )
    rows = result.scalars().all()
    
    return {
        "entries": [
            {"id": str(row.id), "title": row.title, "type": row.entry_type, "length": len(row.content or ""), "file_name": row.file_name}
            for row in rows
        ],
    }


@router.delete("/knowledge-base/{entry_id}")
async def delete_knowledge_base_entry(
    entry_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Delete a knowledge base entry (DB-persisted)."""
    user_id = request.headers.get("x-user-id", "anonymous")
    from ..models import KnowledgeBaseEntryDB
    
    result = await session.execute(
        select(KnowledgeBaseEntryDB).where(
            KnowledgeBaseEntryDB.id == UUID(entry_id),
            KnowledgeBaseEntryDB.user_id == user_id,
        )
    )
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    await session.delete(entry)
    await session.commit()
    
    # Sync to in-memory cache
    from ..services.hallucination_detector import user_knowledge_base
    user_knowledge_base.delete_entry(user_id, entry_id)
    
    return {"deleted": True, "id": entry_id}


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"service": "resonant-chat", "status": "ok"}


# ============================================
# ARCHITECT SESSION POLL ENDPOINT
# ============================================

@router.get("/architect-session/{chat_id}")
async def poll_architect_session(chat_id: str, request: Request, since_event: int = 0):
    """
    Poll for an active/completed architect session.
    Used by the frontend to resume after page refresh.
    Returns current status, accumulated text, and new events since `since_event` index.
    """
    sess = _get_architect_session(chat_id)
    if not sess:
        return {"active": False}

    events_since = sess.get("events", [])[since_event:]
    return {
        "active": True,
        "status": sess.get("status", "unknown"),
        "accumulated_text": sess.get("accumulated_text", ""),
        "options": sess.get("options"),
        "event_count": len(sess.get("events", [])),
        "new_events": events_since[:100],  # cap to avoid huge payloads
        "message_id": sess.get("message_id"),
        "started_at": sess.get("started_at"),
        "completed_at": sess.get("completed_at"),
    }


@router.get("/architect-session/{chat_id}/stream")
async def stream_architect_session(chat_id: str, request: Request, since_event: int = 0):
    """
    Reconnect to an active architect session via SSE.
    Replays events from `since_event` then streams new ones live.
    """
    sess = _get_architect_session(chat_id)
    if not sess:
        return JSONResponse({"error": "No active architect session"}, status_code=404)

    async def _resume_stream():
        import time as _t
        idx = since_event
        # First replay any existing events the client missed
        events = sess.get("events", [])
        for ev in events[idx:]:
            idx += 1
            etype = ev.get("type", "")
            edata = ev.get("data", {})
            if etype == "text":
                yield _sse_event({"event": "chunk", "content": edata.get("content", "")})
            elif etype == "stream_token":
                yield _sse_event({"event": "chunk", "content": edata.get("content", "")})
            elif etype == "options":
                yield _sse_event({"event": "options", "options": tool_executor._map_architect_options(edata)})
            elif etype == "complete":
                rd = edata.get("response", edata)
                ft = rd.get("text", "")
                if ft:
                    yield _sse_event({"event": "chunk", "content": ft})
                opts = rd.get("options")
                if opts:
                    yield _sse_event({"event": "options", "options": tool_executor._map_architect_options(opts)})
            elif etype in ("tool_call", "tool_result", "thinking", "phase",
                           "build_progress", "build_step", "build_complete",
                           "test_step", "test_result", "verify_step",
                           "verify_complete", "prompt_step", "prompt_ready",
                           "research_complete", "plan_ready", "offers_ready",
                           "self_heal", "warning"):
                yield _sse_event({"event": "step", "step": etype, **edata})
            elif etype == "error":
                yield _sse_event({"event": "error", "error": edata.get("error", "Unknown")})

        # If still running, keep tailing live
        while sess.get("status") == "running":
            events = sess.get("events", [])
            while idx < len(events):
                ev = events[idx]
                idx += 1
                etype = ev.get("type", "")
                edata = ev.get("data", {})
                if etype == "text":
                    yield _sse_event({"event": "chunk", "content": edata.get("content", "")})
                elif etype == "stream_token":
                    yield _sse_event({"event": "chunk", "content": edata.get("content", "")})
                elif etype == "options":
                    yield _sse_event({"event": "options", "options": tool_executor._map_architect_options(edata)})
                elif etype in ("tool_call", "tool_result", "thinking", "phase",
                               "build_progress", "build_step", "build_complete",
                               "test_step", "test_result", "verify_step",
                               "verify_complete", "prompt_step", "prompt_ready",
                               "research_complete", "plan_ready", "offers_ready",
                               "self_heal", "warning"):
                    yield _sse_event({"event": "step", "step": etype, **edata})
                elif etype == "error":
                    yield _sse_event({"event": "error", "error": edata.get("error", "Unknown")})
            await asyncio.sleep(0.3)

        # Emit done
        yield _sse_event({"event": "done", "content": sess.get("accumulated_text", ""),
                          "message_id": sess.get("message_id"), "provider": "tool_agent_architect"})

    return StreamingResponse(_resume_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ============================================
# INTERNAL ROUTE-QUERY ENDPOINT (for Agent Engine)
# ============================================

@router.post("/internal/route-query")
async def internal_route_query(request_body: InternalRouteQueryRequest, request: Request):
    """
    Internal endpoint for agent_engine_service to use the unified multi-provider system.
    
    Uses the same provider system as Resonant Chat:
    - Supports: Groq, Gemini, Claude, ChatGPT
    - Automatic fallback chain: Groq → Gemini → Claude → ChatGPT
    - BYOK (Bring Your Own Key) support
    """
    try:
        # Get user API keys if provided
        user_api_keys = request_body.user_api_keys or {}
        
        # Call the unified route_query function
        result = await route_query(
            message=request_body.message,
            context=request_body.context,
            preferred_provider=request_body.preferred_provider,
            user_api_keys=user_api_keys,
        )
        
        return {
            "response": result.get("response", ""),
            "provider": result.get("provider", "unknown"),
            "metadata": result.get("metadata", {}),
        }
    except Exception as e:
        logger.error(f"Internal route-query failed: {e}")
        return {
            "response": f"Error: {str(e)}",
            "provider": "error",
            "metadata": {"error": str(e)},
        }


# ============================================
# PROVIDER STATS ENDPOINT
# ============================================

@router.get("/provider/stats")
async def get_provider_stats():
    """Get provider health and latency stats."""
    try:
        # Return mock stats for now - can be enhanced with real monitoring
        providers = {
            "openai": {"health": "healthy", "latency": 150, "available": True},
            "anthropic": {"health": "healthy", "latency": 200, "available": True},
            "google": {"health": "healthy", "latency": 180, "available": True},
            "groq": {"health": "healthy", "latency": 80, "available": True},
            "mistral": {"health": "healthy", "latency": 160, "available": True},
        }
        return {"providers": providers, "status": "ok"}
    except Exception as e:
        logger.error(f"Failed to get provider stats: {e}")
        return {"providers": {}, "status": "error", "error": str(e)}


# ============================================
# MEMORY ANCHORS & CLUSTERS ENDPOINTS
# ============================================

@router.get("/anchors")
async def get_memory_anchors(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Get memory anchors for the user from recent messages."""
    # Extract user_id and crypto identity from headers (set by gateway)
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user_id header")
    
    # Extract crypto identity headers for enhanced security
    crypto_hash = request.headers.get("x-crypto-hash")
    user_hash = request.headers.get("x-user-hash")
    universe_id = request.headers.get("x-universe-id")
    
    logger.info(f"Chat request with crypto identity: user_hash={user_hash[:16] if user_hash else 'None'}..., universe_id={universe_id[:16] if universe_id else 'None'}...")
    
    try:
        # Get user's chats first
        chats_result = await session.execute(
            select(ResonantChat.id)
            .where(ResonantChat.user_id == user_id)
        )
        chat_ids = [row[0] for row in chats_result.fetchall()]
        
        if not chat_ids:
            return {"anchors": [], "count": 0}
        
        # Get recent messages from user's chats
        result = await session.execute(
            select(ResonantChatMessage)
            .where(ResonantChatMessage.chat_id.in_(chat_ids))
            .where(ResonantChatMessage.role == "assistant")
            .order_by(ResonantChatMessage.created_at.desc())
            .limit(50)
        )
        messages = result.scalars().all()
        
        # Extract anchors from message metadata
        all_anchors = []
        for msg in messages:
            if msg.meta_data and isinstance(msg.meta_data, dict):
                anchors = msg.meta_data.get("anchors", [])
                all_anchors.extend(anchors)
        
        # Deduplicate and limit — anchors may be dicts (unhashable), so normalise first
        seen = set()
        unique_anchors = []
        for a in all_anchors:
            # If anchor is a dict, use its JSON repr as dedup key; keep the dict as-is
            if isinstance(a, dict):
                key = json.dumps(a, sort_keys=True, default=str)
            else:
                try:
                    key = a
                    hash(key)  # ensure hashable
                except TypeError:
                    key = str(a)
            if key not in seen:
                seen.add(key)
                unique_anchors.append(a)
            if len(unique_anchors) >= 100:
                break
        
        return {"anchors": unique_anchors, "count": len(unique_anchors)}
    except Exception as e:
        logger.error(f"Failed to get memory anchors: {e}")
        return {"anchors": [], "count": 0, "error": str(e)}


@router.get("/clusters")
async def get_resonance_clusters(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Get resonance clusters from user's conversation history."""
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    try:
        # Get user's chats with their messages
        chats_result = await session.execute(
            select(ResonantChat)
            .where(ResonantChat.user_id == user_id)
            .order_by(ResonantChat.updated_at.desc())
            .limit(20)
        )
        chats = chats_result.scalars().all()
        
        # Build clusters from chats
        cluster_list = []
        for chat in chats:
            # Get message count for this chat
            msg_result = await session.execute(
                select(ResonantChatMessage)
                .where(ResonantChatMessage.chat_id == chat.id)
                .order_by(ResonantChatMessage.created_at.desc())
                .limit(5)
            )
            messages = msg_result.scalars().all()
            
            cluster_list.append({
                "id": str(chat.id),
                "title": chat.title or "Untitled",
                "message_count": len(messages),
                "messages": [
                    {
                        "id": str(msg.id),
                        "role": msg.role,
                        "preview": msg.content[:100] if msg.content else "",
                    }
                    for msg in messages
                ],
            })
        
        return {"clusters": cluster_list, "count": len(cluster_list)}
    except Exception as e:
        logger.error(f"Failed to get resonance clusters: {e}")
        return {"clusters": [], "count": 0, "error": str(e)}


# ============================================
# AGENT & TEAM ENDPOINTS (Phase 4)
# ============================================

# Note: /agents/stats endpoint is defined below in AGENT METRICS ENDPOINTS section
# This duplicate was removed to avoid conflicts


@router.get("/teams")
async def list_available_teams():
    """List all available internal teams."""
    try:
        from ..domain.agent import get_team_list
        teams = get_team_list()
        return {
            "status": "ok",
            "teams": teams,
        }
    except Exception as e:
        logger.error(f"Failed to list teams: {e}")
        return {
            "status": "error",
            "error": str(e),
            "teams": [],
        }


@router.get("/agents/list")
async def list_available_agents():
    """List all available agents with their trigger keywords."""
    agents = [
        {"id": "reasoning", "name": "Reasoning Agent", "keywords": ["analyze", "analysis", "explain why", "how does"]},
        {"id": "code", "name": "Code Agent", "keywords": ["write code", "generate code", "implement", "script"]},
        {"id": "debug", "name": "Debug Agent", "keywords": ["fix", "debug", "error", "bug", "broken"]},
        {"id": "research", "name": "Research Agent", "keywords": ["research", "find information", "investigate"]},
        {"id": "summary", "name": "Summary Agent", "keywords": ["summarize", "summary", "tl;dr"]},
        {"id": "planning", "name": "Planning Agent", "keywords": ["plan", "strategy", "roadmap"]},
        {"id": "math", "name": "Math Agent", "keywords": ["calculate", "math", "equation", "solve"]},
        {"id": "security", "name": "Security Agent", "keywords": ["security", "vulnerability", "hack"]},
        {"id": "architecture", "name": "Architecture Agent", "keywords": ["architecture", "design pattern", "structure"]},
        {"id": "test", "name": "Test Agent", "keywords": ["test", "unit test", "coverage"]},
        {"id": "review", "name": "Review Agent", "keywords": ["review", "critique", "feedback"]},
        {"id": "explain", "name": "Explain Agent", "keywords": ["eli5", "simple terms", "beginner"]},
        {"id": "optimization", "name": "Optimization Agent", "keywords": ["optimize", "performance", "speed up"]},
        {"id": "documentation", "name": "Documentation Agent", "keywords": ["document", "readme", "jsdoc"]},
        {"id": "migration", "name": "Migration Agent", "keywords": ["migrate", "upgrade", "convert"]},
        {"id": "api", "name": "API Agent", "keywords": ["api", "endpoint", "rest", "graphql"]},
        {"id": "database", "name": "Database Agent", "keywords": ["database", "sql", "query", "schema"]},
        {"id": "devops", "name": "DevOps Agent", "keywords": ["deploy", "ci/cd", "docker", "kubernetes"]},
        {"id": "refactor", "name": "Refactor Agent", "keywords": ["refactor", "restructure", "clean up"]},
        {"id": "accessibility", "name": "Accessibility Agent", "keywords": ["accessibility", "a11y", "wcag"]},
        {"id": "i18n", "name": "i18n Agent", "keywords": ["translate", "i18n", "localize"]},
        {"id": "regex", "name": "Regex Agent", "keywords": ["regex", "regular expression", "pattern"]},
        {"id": "git", "name": "Git Agent", "keywords": ["git", "merge", "branch", "commit"]},
        {"id": "css", "name": "CSS Agent", "keywords": ["css", "style", "flexbox", "tailwind"]},
    ]
    return {
        "status": "ok",
        "agents": agents,
        "total": len(agents),
    }


# ============================================
# PHASE 5 ENDPOINTS
# ============================================

@router.post("/feedback")
async def submit_user_feedback(
    request: Request,
    body: FeedbackRequest,
    session: AsyncSession = Depends(get_session),
):
    """Submit thumbs up/down feedback for an agent response.
    
    Persists to database for agent quality scoring.
    """
    try:
        from ..services.user_feedback import user_feedback
        # Get user_id from headers or state
        user_id = request.headers.get("x-user-id") or (request.state.user_id if hasattr(request.state, 'user_id') else "anonymous")
        
        # Use agent_hash for custom agents, otherwise use agent_type
        # This allows tracking feedback per custom agent (by hash) or per agent type (reasoning, code, etc.)
        agent_identifier = body.agent_hash or body.agent_type or "reasoning"
        
        # Use async version with database persistence
        entry = await user_feedback.submit_feedback_async(
            session=session,
            message_id=body.message_id,
            user_id=user_id,
            agent_type=agent_identifier,  # Can be hash (custom agent) or type (built-in)
            is_positive=body.is_positive,
            task=body.task,
            response=body.response,
            comment=body.comment,
        )
        
        is_custom = bool(body.agent_hash)
        result = {
            "id": entry.id,
            "agent_type": entry.agent_type,
            "feedback_type": entry.feedback_type,
            "timestamp": entry.timestamp,
        }
        logger.info(f"Feedback submitted & persisted: {agent_identifier[:16]}{'...' if len(agent_identifier) > 16 else ''} ({'custom' if is_custom else 'built-in'}) - {'👍' if body.is_positive else '👎'}")
        return {"status": "ok", "data": result, "is_custom_agent": is_custom, "persisted": True}
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@router.get("/feedback/stats")
async def get_feedback_statistics(
    session: AsyncSession = Depends(get_session),
):
    """Get feedback statistics for all agents.
    
    Now loads from database for persistence across restarts.
    """
    try:
        from ..services.user_feedback import user_feedback
        from ..models import AgentPerformanceScore
        from sqlalchemy import select
        
        # Load from database if not already loaded
        if not user_feedback._db_initialized:
            await user_feedback.load_from_database(session)
        
        # Get stats from database for accurate counts
        result = await session.execute(select(AgentPerformanceScore))
        db_scores = result.scalars().all()
        
        # Build stats from database
        all_stats = {}
        best_agents = []
        needs_improvement = []
        
        for score in db_scores:
            all_stats[score.agent_type] = {
                "agent_type": score.agent_type,
                "positive_count": int(score.positive_count),
                "negative_count": int(score.negative_count),
                "total_count": int(score.total_count),
                "satisfaction_rate": score.satisfaction_rate,
                "quality_score": score.quality_score,
            }
            
            best_agents.append({
                "agent_type": score.agent_type,
                "satisfaction_rate": score.satisfaction_rate,
                "total_feedback": int(score.total_count),
                "trend": "stable",  # TODO: Calculate from recent feedback
            })
            
            if score.quality_score < 0.5:
                needs_improvement.append(score.agent_type)
        
        # Sort best agents by satisfaction rate
        best_agents.sort(key=lambda x: x["satisfaction_rate"], reverse=True)
        
        return {
            "status": "ok",
            "data": {
                "all_stats": all_stats,
                "best_agents": best_agents[:5],
                "needs_improvement": needs_improvement,
            }
        }
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        # Fall back to in-memory stats
        from ..domain.agent import get_feedback_stats
        stats = get_feedback_stats()
        return {"status": "ok", "data": stats}


@router.get("/chains")
async def list_agent_chains(user_id: Optional[str] = None):
    """List available agent chains/pipelines."""
    try:
        from ..domain.agent import get_chain_list
        chains = get_chain_list(user_id)
        return {"status": "ok", "chains": chains}
    except Exception as e:
        logger.error(f"Failed to list chains: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/chains")
async def create_agent_chain(
    request: Request,
    body: CreateChainRequest,
):
    """Create a custom agent chain."""
    try:
        from ..domain.agent import create_chain
        user_id = request.state.user_id if hasattr(request.state, 'user_id') else "anonymous"
        
        result = create_chain(
            user_id=user_id,
            name=body.name,
            description=body.description,
            steps=body.steps,
        )
        return {"status": "ok", "chain": result}
    except Exception as e:
        logger.error(f"Failed to create chain: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/chains/execute")
async def execute_agent_chain(body: ExecuteChainRequest):
    """Execute an agent chain."""
    try:
        from ..domain.agent import run_chain
        result = await run_chain(
            chain_id=body.chain_id,
            task=body.task,
            context_messages=body.context,
        )
        return {"status": "ok", "data": result}
    except Exception as e:
        logger.error(f"Chain execution failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/sandbox/execute")
async def execute_code_sandbox(body: ExecuteCodeRequest):
    """Execute code in sandbox environment."""
    try:
        from ..domain.agent import execute_code
        result = await execute_code(
            code=body.code,
            language=body.language,
            test_input=body.test_input,
        )
        return {"status": "ok", "data": result}
    except Exception as e:
        logger.error(f"Code execution failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/analyze/confidence")
async def analyze_response_confidence(body: AnalyzeRequest):
    """Analyze confidence level of a response."""
    try:
        from ..domain.agent import analyze_confidence
        result = analyze_confidence(body.response, body.task)
        return {"status": "ok", "data": result}
    except Exception as e:
        logger.error(f"Confidence analysis failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/analyze/hallucinations")
async def analyze_hallucinations(body: AnalyzeRequest):
    """Detect potential hallucinations in a response."""
    try:
        from ..domain.agent import detect_hallucinations
        result = detect_hallucinations(body.response, body.task)
        return {"status": "ok", "data": result}
    except Exception as e:
        logger.error(f"Hallucination detection failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/analyze/citations")
async def add_response_citations(body: AnalyzeRequest):
    """Add citations to a response."""
    try:
        from ..domain.agent import add_citations
        result = add_citations(body.response, body.task, body.agent_type)
        return {"status": "ok", "data": result}
    except Exception as e:
        logger.error(f"Citation addition failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/validate")
async def cross_validate_response(body: ValidateRequest):
    """Cross-validate an agent response with another agent."""
    try:
        from ..domain.agent import validate_response
        result = await validate_response(
            response=body.response,
            task=body.task,
            agent_type=body.agent_type,
            context_messages=body.context,
        )
        return {"status": "ok", "data": result}
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/voting")
async def run_agent_voting(body: VotingRequest):
    """Run agent voting on a task."""
    try:
        from ..domain.agent import run_voting
        result = await run_voting(
            task=body.task,
            context_messages=body.context,
            candidate_agents=body.candidate_agents,
            voter_agents=body.voter_agents,
        )
        return {"status": "ok", "data": result}
    except Exception as e:
        logger.error(f"Voting failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/chunking/info")
async def get_chunking_info(
    request: Request,
    body: ChunkingInfoRequest,
):
    """Get information about how text would be chunked for processing."""
    try:
        from ..services.multi_provider_chunking import multi_provider_chunker
        info = multi_provider_chunker.get_chunking_info(body.text)
        return {"status": "ok", "data": info}
    except Exception as e:
        logger.error(f"Chunking info failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/chunking/process")
async def process_chunked_text(
    request: Request,
    body: ProcessChunkedRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Process large text by chunking across multiple AI providers.
    
    This endpoint splits large documents into chunks and processes them
    in parallel across available providers, then combines the results.
    """
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    try:
        from ..services.multi_provider_chunking import multi_provider_chunker
        from ..domain.provider import get_router_for_internal_use
        
        # Get router and set up chunker
        router_instance = get_router_for_internal_use()
        multi_provider_chunker.set_router(router_instance)
        
        # Get user API keys
        user_api_keys = await _get_user_api_keys(session, user_id)
        
        # Get available providers
        available_providers = router_instance.get_available_providers()
        if not available_providers:
            return {
                "status": "error",
                "error": "No AI providers available. Please configure API keys."
            }
        
        # Process with chunking
        combined_result, chunk_results = await multi_provider_chunker.process_with_chunking(
            text=body.text,
            task_prompt=body.task_prompt,
            context=[],
            available_providers=available_providers,
            user_api_keys=user_api_keys,
        )
        
        return {
            "status": "ok",
            "response": combined_result,
            "chunks_processed": len(chunk_results),
            "successful_chunks": sum(1 for r in chunk_results if r.success),
            "providers_used": list(set(r.provider for r in chunk_results)),
        }
    except Exception as e:
        logger.error(f"Chunked processing failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/context/project")
async def get_or_create_project_context(
    request: Request,
    body: ProjectContextRequest,
):
    """Get or create project context for persistent memory."""
    try:
        from ..domain.agent import get_project_context
        user_id = request.state.user_id if hasattr(request.state, 'user_id') else "anonymous"
        
        result = get_project_context(user_id, body.project_name)
        return {"status": "ok", "data": result}
    except Exception as e:
        logger.error(f"Project context failed: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/agents/list")
async def list_agents():
    """Get list of available agents with their trigger keywords."""
    agents = [
        {"id": "reasoning", "name": "Reasoning Agent", "keywords": ["analyze", "explain", "why", "how"]},
        {"id": "code", "name": "Code Agent", "keywords": ["write code", "generate", "implement", "function"]},
        {"id": "debug", "name": "Debug Agent", "keywords": ["fix", "debug", "error", "bug"]},
        {"id": "research", "name": "Research Agent", "keywords": ["research", "find", "compare", "information"]},
        {"id": "summary", "name": "Summary Agent", "keywords": ["summarize", "tldr", "brief", "overview"]},
        {"id": "planning", "name": "Planning Agent", "keywords": ["plan", "strategy", "roadmap", "steps"]},
        {"id": "creative", "name": "Creative Agent", "keywords": ["creative", "brainstorm", "ideas", "innovate"]},
        {"id": "review", "name": "Review Agent", "keywords": ["review", "critique", "feedback", "evaluate"]},
        {"id": "refactor", "name": "Refactor Agent", "keywords": ["refactor", "optimize", "improve", "clean"]},
        {"id": "test", "name": "Test Agent", "keywords": ["test", "unit test", "testing", "coverage"]},
        {"id": "docs", "name": "Documentation Agent", "keywords": ["document", "docs", "readme", "comments"]},
        {"id": "security", "name": "Security Agent", "keywords": ["security", "vulnerability", "secure", "audit"]},
        {"id": "data", "name": "Data Agent", "keywords": ["data", "database", "sql", "query"]},
        {"id": "api", "name": "API Agent", "keywords": ["api", "endpoint", "rest", "graphql"]},
        {"id": "devops", "name": "DevOps Agent", "keywords": ["deploy", "docker", "ci/cd", "kubernetes"]},
        {"id": "frontend", "name": "Frontend Agent", "keywords": ["frontend", "ui", "react", "css"]},
        {"id": "backend", "name": "Backend Agent", "keywords": ["backend", "server", "api", "database"]},
        {"id": "architect", "name": "Architect Agent", "keywords": ["architecture", "design", "system", "structure"]},
    ]
    return {"agents": agents}


@router.get("/teams")
async def list_teams():
    """Get list of available teams with their configurations."""
    teams = [
        {
            "id": "code_review",
            "name": "Code Review Team",
            "agents": ["review", "security", "refactor"],
            "workflow": "sequential",
            "description": "Reviews code for quality, security, and suggests improvements"
        },
        {
            "id": "full_stack",
            "name": "Full Stack Team",
            "agents": ["frontend", "backend", "api"],
            "workflow": "parallel",
            "description": "Handles full-stack development tasks"
        },
        {
            "id": "research_analysis",
            "name": "Research & Analysis Team",
            "agents": ["research", "reasoning", "summary"],
            "workflow": "sequential",
            "description": "Deep research and analysis with summarization"
        },
        {
            "id": "development",
            "name": "Development Team",
            "agents": ["code", "test", "docs"],
            "workflow": "sequential",
            "description": "Code generation with testing and documentation"
        },
        {
            "id": "architecture",
            "name": "Architecture Team",
            "agents": ["architect", "planning", "review"],
            "workflow": "debate",
            "description": "System design and architecture planning"
        },
    ]
    return {"teams": teams}


# ============================================
# SENTIMENT/EMOTION DETECTION ENDPOINT
# ============================================
@router.post("/sentiment/analyze")
async def analyze_sentiment(request: Request):
    """
    Analyze sentiment and emotion in a message.
    Returns emotion, sentiment, confidence scores, and tone recommendations.
    """
    try:
        body = await request.json()
        message = body.get("message", "")
        
        if not message:
            return {"error": "Message is required"}
        
        from ..services.sentiment_detection import analyze_message
        analysis = analyze_message(message)
        
        return {
            "status": "ok",
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return {"error": str(e)}


# ============================================
# CODE EXECUTION SANDBOX ENDPOINT
# ============================================
@router.post("/code/execute")
async def execute_code(request: Request):
    """
    Execute code in a sandboxed environment.
    Supports Python, JavaScript, and shell commands.
    """
    try:
        body = await request.json()
        code = body.get("code", "")
        language = body.get("language", "python").lower()
        timeout = min(body.get("timeout", 30), 60)  # Max 60 seconds
        
        if not code:
            return {"error": "Code is required"}
        
        from ..services.code_sandbox import execute_code_safely
        result = await execute_code_safely(code, language, timeout)
        
        return {
            "status": "ok",
            "result": result
        }
    except Exception as e:
        logger.error(f"Code execution error: {e}")
        return {"error": str(e)}


# ============================================
# AUTONOMOUS SERVICES ENDPOINTS (L3-L5)
# ============================================

@router.get("/autonomous/stats")
async def get_autonomous_stats(request: Request):
    """
    Get statistics for all autonomous services.
    Returns routing stats, cache stats, learning stats, and planning stats.
    """
    try:
        return {
            "status": "ok",
            "cache": response_cache.get_stats(),
            "learning": self_improving_agent.get_all_stats(),
            "planning": autonomous_planner.get_planning_stats()
        }
    except Exception as e:
        logger.error(f"Autonomous stats error: {e}")
        return {"error": str(e)}


@router.get("/autonomous/cache/stats")
async def get_cache_stats(request: Request):
    """Get Response Cache statistics."""
    try:
        return {
            "status": "ok",
            "stats": response_cache.get_stats()
        }
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return {"error": str(e)}


@router.post("/autonomous/cache/clear")
async def clear_cache(request: Request):
    """Clear the response cache."""
    try:
        response_cache.clear()
        return {"status": "ok", "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return {"error": str(e)}


@router.get("/autonomous/learning/stats")
async def get_learning_stats(request: Request):
    """Get Self-Improving Agent learning statistics."""
    try:
        return {
            "status": "ok",
            "stats": self_improving_agent.get_all_stats()
        }
    except Exception as e:
        logger.error(f"Learning stats error: {e}")
        return {"error": str(e)}


@router.get("/autonomous/learning/agent/{agent_id}")
async def get_agent_learning_stats(agent_id: str, request: Request):
    """Get learning statistics for a specific agent."""
    try:
        return {
            "status": "ok",
            "stats": self_improving_agent.get_agent_stats(agent_id)
        }
    except Exception as e:
        logger.error(f"Agent learning stats error: {e}")
        return {"error": str(e)}


@router.get("/autonomous/learning/agent/{agent_id}/suggestions")
async def get_agent_improvement_suggestions(agent_id: str, request: Request):
    """Get improvement suggestions for a specific agent."""
    try:
        suggestions = self_improving_agent.get_improvement_suggestions(agent_id)
        return {
            "status": "ok",
            "agent_id": agent_id,
            "suggestions": suggestions
        }
    except Exception as e:
        logger.error(f"Agent suggestions error: {e}")
        return {"error": str(e)}


@router.post("/autonomous/feedback")
async def record_feedback(request: Request):
    """
    Record user feedback for agent learning.
    Feedback types: thumbs_up, thumbs_down, regenerate, edit, copy, apply_code
    """
    try:
        body = await request.json()
        agent_id = body.get("agent_id", "default")
        message = body.get("message", "")
        response = body.get("response", "")
        feedback_type = body.get("feedback_type", "thumbs_up")
        value = body.get("value")
        
        # Map string to FeedbackType enum
        feedback_map = {
            "thumbs_up": FeedbackType.THUMBS_UP,
            "thumbs_down": FeedbackType.THUMBS_DOWN,
            "regenerate": FeedbackType.REGENERATE,
            "edit": FeedbackType.EDIT,
            "copy": FeedbackType.COPY,
            "apply_code": FeedbackType.APPLY_CODE,
            "quality_score": FeedbackType.QUALITY_SCORE
        }
        
        ft = feedback_map.get(feedback_type, FeedbackType.THUMBS_UP)
        
        await self_improving_agent.record_feedback(
            agent_id=agent_id,
            message=message,
            response=response,
            feedback_type=ft,
            value=value
        )
        
        return {"status": "ok", "message": "Feedback recorded"}
    except Exception as e:
        logger.error(f"Feedback recording error: {e}")
        return {"error": str(e)}


@router.get("/autonomous/classifiers/stats")
async def get_classifier_stats(request: Request):
    """Get statistics for both neural classifiers (tool + agent)."""
    try:
        from ..services.agent_classifier import agent_classifier
        tool_stats = await tool_classifier.get_stats()
        agent_stats = await agent_classifier.get_stats()
        return {
            "status": "ok",
            "tool_classifier": tool_stats,
            "agent_classifier": agent_stats,
        }
    except Exception as e:
        logger.error(f"Classifier stats error: {e}")
        return {"error": str(e)}


@router.post("/autonomous/classifiers/retrain")
async def retrain_classifiers(request: Request):
    """Retrain both neural classifiers with accumulated active learning data."""
    try:
        from ..services.agent_classifier import agent_classifier
        tool_stats = await tool_classifier.retrain()
        agent_stats = await agent_classifier.retrain()
        return {
            "status": "ok",
            "tool_classifier": tool_stats,
            "agent_classifier": agent_stats,
        }
    except Exception as e:
        logger.error(f"Classifier retrain error: {e}")
        return {"error": str(e)}


@router.post("/autonomous/classifiers/test")
async def test_classifiers(request: Request):
    """Test both classifiers on a message without executing."""
    try:
        body = await request.json()
        message = body.get("message", "")
        if not message:
            return {"error": "Message is required"}
        from ..services.agent_classifier import agent_classifier
        from ..services.tool_classifier import ALL_TOOLS
        enabled = {s for s in ALL_TOOLS if s is not None}
        tool_pred = await tool_classifier.predict(message, enabled)
        agent_pred = await agent_classifier.predict(message)
        return {
            "status": "ok",
            "message": message,
            "tool": {
                "tool_id": tool_pred.tool_id,
                "confidence": tool_pred.confidence,
                "method": tool_pred.method,
                "top_5": dict(sorted(tool_pred.probabilities.items(), key=lambda x: -x[1])[:5]),
            },
            "agent": {
                "agent_type": agent_pred.agent_type,
                "confidence": agent_pred.confidence,
                "method": agent_pred.method,
                "top_5": dict(sorted(agent_pred.probabilities.items(), key=lambda x: -x[1])[:5]),
            },
        }
    except Exception as e:
        logger.error(f"Classifier test error: {e}")
        return {"error": str(e)}


@router.get("/autonomous/planning/stats")
async def get_planning_stats(request: Request):
    """Get Autonomous Planner statistics."""
    try:
        return {
            "status": "ok",
            "stats": autonomous_planner.get_planning_stats()
        }
    except Exception as e:
        logger.error(f"Planning stats error: {e}")
        return {"error": str(e)}


@router.post("/autonomous/planning/create")
async def create_plan(request: Request):
    """
    Create an execution plan for a goal.
    Returns the plan with decomposed steps.
    """
    try:
        body = await request.json()
        goal = body.get("goal", "")
        
        if not goal:
            return {"error": "Goal is required"}
        
        plan = autonomous_planner.create_plan(goal)
        
        return {
            "status": "ok",
            "plan": {
                "id": plan.id,
                "goal": plan.goal,
                "status": plan.status.value,
                "steps": [
                    {
                        "id": s.id,
                        "description": s.description,
                        "action_type": s.action_type,
                        "status": s.status.value,
                        "dependencies": s.dependencies
                    }
                    for s in plan.steps
                ]
            }
        }
    except Exception as e:
        logger.error(f"Plan creation error: {e}")
        return {"error": str(e)}


@router.get("/autonomous/planning/{plan_id}")
async def get_plan_status_endpoint(plan_id: str, request: Request):
    """Get status of a specific plan."""
    try:
        status = autonomous_planner.get_plan_status(plan_id)
        if not status:
            return {"error": "Plan not found"}
        return {"status": "ok", "plan": status}
    except Exception as e:
        logger.error(f"Plan status error: {e}")
        return {"error": str(e)}


# ============================================
# DSID-P ENDPOINTS (HSU-Spec Layer 1-2)
# ============================================

@router.get("/dsid/stats")
async def get_dsid_stats(request: Request):
    """Get DSID integration statistics."""
    try:
        return {
            "status": "ok",
            "stats": dsid_integration.get_stats()
        }
    except Exception as e:
        logger.error(f"DSID stats error: {e}")
        return {"error": str(e)}


@router.get("/dsid/message/{message_id}")
async def get_message_dsid(message_id: str, request: Request):
    """Get DSID for a specific message."""
    try:
        dsid = dsid_integration.get_dsid_by_message(message_id)
        if not dsid:
            return {"error": "DSID not found for message"}
        
        return {
            "status": "ok",
            "dsid": {
                "dsid": dsid.dsid,
                "entity_type": dsid.entity_type,
                "entity_id": dsid.entity_id,
                "content_hash": dsid.content_hash,
                "parent_dsid": dsid.parent_dsid,
                "root_dsid": dsid.root_dsid,
                "lineage_depth": dsid.lineage_depth,
                "chat_id": dsid.chat_id,
                "created_at": dsid.created_at.isoformat(),
            }
        }
    except Exception as e:
        logger.error(f"Get message DSID error: {e}")
        return {"error": str(e)}


@router.get("/dsid/lineage/{message_id}")
async def get_message_lineage(message_id: str, request: Request):
    """Get the full lineage chain for a message."""
    try:
        lineage = dsid_integration.get_message_lineage(message_id)
        
        return {
            "status": "ok",
            "lineage": [
                {
                    "dsid": d.dsid,
                    "entity_type": d.entity_type,
                    "content_hash": d.content_hash,
                    "lineage_depth": d.lineage_depth,
                }
                for d in lineage
            ],
            "total_depth": len(lineage),
        }
    except Exception as e:
        logger.error(f"Get lineage error: {e}")
        return {"error": str(e)}


@router.get("/dsid/conversation/{chat_id}")
async def get_conversation_lineage(chat_id: str, request: Request):
    """Get the full lineage for a conversation."""
    try:
        lineage = dsid_integration.get_conversation_lineage(chat_id)
        if not lineage:
            return {"error": "Conversation not found"}
        
        return {
            "status": "ok",
            "conversation": {
                "chat_id": lineage.chat_id,
                "root_dsid": lineage.root_dsid,
                "merkle_root": lineage.merkle_root,
                "message_count": len(lineage.messages),
                "messages": [
                    {
                        "dsid": m.dsid,
                        "entity_type": m.entity_type,
                        "content_hash": m.content_hash[:16] + "...",
                        "lineage_depth": m.lineage_depth,
                    }
                    for m in lineage.messages
                ]
            }
        }
    except Exception as e:
        logger.error(f"Get conversation lineage error: {e}")
        return {"error": str(e)}


@router.post("/dsid/verify/{message_id}")
async def verify_message_content(message_id: str, request: Request):
    """
    Verify a message's content against its DSID.
    
    Body: {"content": "message content to verify"}
    """
    try:
        body = await request.json()
        content = body.get("content", "")
        
        if not content:
            return {"error": "Content is required"}
        
        is_valid, reason = dsid_integration.verify_message(message_id, content)
        
        return {
            "status": "ok",
            "verified": is_valid,
            "reason": reason,
        }
    except Exception as e:
        logger.error(f"Verify message error: {e}")
        return {"error": str(e)}


@router.get("/dsid/proof/{message_id}")
async def get_merkle_proof(message_id: str, request: Request):
    """Get Merkle proof for a message."""
    try:
        proof = dsid_integration.compute_merkle_proof(message_id)
        dsid = dsid_integration.get_dsid_by_message(message_id)
        
        if not dsid:
            return {"error": "DSID not found for message"}
        
        # Get conversation merkle root
        lineage = dsid_integration.get_conversation_lineage(dsid.chat_id)
        merkle_root = lineage.merkle_root if lineage else None
        
        return {
            "status": "ok",
            "message_id": message_id,
            "content_hash": dsid.content_hash,
            "merkle_root": merkle_root,
            "proof": proof,
            "proof_length": len(proof),
        }
    except Exception as e:
        logger.error(f"Get Merkle proof error: {e}")
        return {"error": str(e)}


# ============================================
# AGENT METRICS ENDPOINTS
# ============================================

@router.get("/agents/stats")
async def get_agent_stats(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Get agent execution statistics, metrics, and memory data.
    
    Memory Architecture:
    - Layer 1: Episodic Memory (short-term, decays over 30 days)
    - Layer 2: Semantic Memory (long-term crystallized facts)
    - 3-Level Hierarchy: User Memory -> Team Memory -> Agent Memory
    """
    user_id = request.headers.get("x-user-id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    try:
        from ..services.agent_metrics import agent_metrics as metrics_collector
        from ..services.agent_memory import agent_memory_store
        from ..services.user_feedback import user_feedback
        
        # Get real metrics from the metrics collector
        all_stats = metrics_collector.get_all_stats()
        
        # Format metrics for frontend
        metrics = {}
        for agent_type, stats in all_stats.items():
            if stats:
                metrics[agent_type] = {
                    "agent_type": agent_type,
                    "total_executions": stats.total_executions,
                    "successful_executions": stats.successful_executions,
                    "failed_executions": stats.failed_executions,
                    "success_rate": stats.success_rate,
                    "avg_execution_time_ms": stats.avg_execution_time_ms,
                    "avg_quality_score": stats.avg_quality_score,
                }
        
        # Get top agents by success rate
        top_agents = metrics_collector.get_top_agents(metric="success_rate", limit=5)
        
        # ============================================
        # FETCH MEMORIES FROM HASH SPHERE (PERSISTENT)
        # Uses 3-level hierarchical memory architecture
        # ============================================
        recent_memories = []
        total_memory_count = 0
        avg_relevance = 0.0
        
        try:
            # Fetch from Hash Sphere memory service (persistent storage)
            hash_sphere_result = await service_client.call_service(
                "memory_service",
                "POST",
                "http://memory_service:8000/memory/hash-sphere/extract",
                json={
                    "query": "",  # Empty query to get all recent memories
                    "user_id": user_id,
                    "org_id": request.headers.get("x-org-id", "default"),
                    "agent_hash": None,  # Get user-level memories
                    "limit": 20,
                    "use_anchors": True,
                    "use_proximity": False,
                    "use_resonance": True,
                    "include_coordinates": True,
                },
                timeout=httpx.Timeout(5.0, connect=2.0),
            )
            
            if hash_sphere_result and hash_sphere_result.get("memories"):
                memories_data = hash_sphere_result.get("memories", [])
                total_memory_count = hash_sphere_result.get("total_count", len(memories_data))
                
                total_relevance = 0.0
                for mem in memories_data:
                    relevance = mem.get("hybrid_score") or mem.get("resonance_score") or 0.5
                    total_relevance += relevance
                    recent_memories.append({
                        "id": mem.get("id") or mem.get("hash", "")[:16],
                        "task": mem.get("content", "")[:100],
                        "response_summary": mem.get("content", "")[:200],
                        "timestamp": mem.get("created_at") or mem.get("timestamp", ""),
                        "relevance_score": relevance,
                        "agent_type": mem.get("agent_hash") or "user",
                        "hash": mem.get("hash"),
                        "xyz": mem.get("xyz"),
                        "type": mem.get("type", "memory"),
                    })
                
                if recent_memories:
                    avg_relevance = total_relevance / len(recent_memories)
                
                logger.info(f"✅ Loaded {len(recent_memories)} memories from Hash Sphere for user {user_id[:8]}...")
        except Exception as mem_err:
            logger.warning(f"Hash Sphere memory fetch failed, falling back to in-memory: {mem_err}")
            
            # Fallback to in-memory agent_memory_store
            memory_stats = agent_memory_store.get_stats(user_id)
            total_memory_count = memory_stats.get("total_memories", 0)
            
            if user_id in agent_memory_store.memories:
                for agent_type, memories in agent_memory_store.memories[user_id].items():
                    for mem in memories[-10:]:
                        recent_memories.append({
                            "id": mem.id,
                            "task": mem.task,
                            "response_summary": mem.response[:200] + "..." if len(mem.response) > 200 else mem.response,
                            "timestamp": mem.created_at,
                            "relevance_score": mem.relevance_score,
                            "agent_type": agent_type,
                        })
        
        # Sort by timestamp descending
        recent_memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        recent_memories = recent_memories[:5]
        
        # Get feedback stats
        feedback_stats = user_feedback.get_all_stats()
        
        return {
            "status": "ok",
            "data": {
                "metrics": metrics,
                "top_agents": top_agents,
                "memory": {
                    "total_count": total_memory_count,
                    "avg_relevance": round(avg_relevance * 100, 1),  # As percentage
                    "agents": {},  # Agent-specific breakdown
                    "recent_memories": recent_memories,
                },
                "feedback": feedback_stats,
                "projects": [],  # Project context - can be extended later
            }
        }
    except Exception as e:
        logger.error(f"Get agent stats error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e), "data": {"metrics": {}, "top_agents": [], "memory": {"total_count": 0, "recent_memories": []}}}


@router.get("/feedback/stats")
async def get_feedback_stats(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Get feedback statistics for agents."""
    user_id = request.headers.get("x-user-id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    try:
        from ..services.user_feedback import user_feedback
        
        # Get real feedback stats from the user_feedback service
        all_stats = user_feedback.get_all_stats()
        
        # Format for frontend - ensure each agent has the expected fields
        formatted_stats = {}
        for agent_type, stats in all_stats.items():
            formatted_stats[agent_type] = {
                "agent_type": agent_type,
                "positive_count": stats.get("positive_count", 0),
                "negative_count": stats.get("negative_count", 0),
                "satisfaction_rate": stats.get("satisfaction_rate", 0),
                "recent_trend": stats.get("recent_trend", "stable"),
            }
        
        return {
            "status": "ok",
            "data": {
                "all_stats": formatted_stats,
            }
        }
    except Exception as e:
        logger.error(f"Get feedback stats error: {e}")
        return {"status": "error", "error": str(e), "data": {"all_stats": {}}}


@router.post("/extract-memories")
async def extract_memories_from_conversation(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Extract memorable facts from a conversation exchange and save to memory service.
    Called after each AI response. Extracts: user preferences, facts about user,
    key decisions, important context — all per-user (cross-chat).
    
    Body: { "user_message": str, "assistant_message": str, "chat_id": str }
    Returns: { "memories_extracted": [...], "count": int }
    """
    user_id = request.headers.get("x-user-id", "anonymous")
    if user_id == "anonymous":
        return {"memories_extracted": [], "count": 0}
    
    try:
        body = await request.json()
        user_msg = body.get("user_message", "")
        assistant_msg = body.get("assistant_message", "")
        chat_id = body.get("chat_id", "")
        
        if not user_msg or len(user_msg) < 5:
            return {"memories_extracted": [], "count": 0}
        
        extracted = []
        user_lower = user_msg.lower().strip()
        
        # 1. Explicit "remember" requests
        remember_triggers = ["remember that", "remember this", "don't forget", "keep in mind", "note that", "save this", "my name is", "i am called"]
        for trigger in remember_triggers:
            if trigger in user_lower:
                idx = user_lower.index(trigger)
                fact_after = user_msg[idx + len(trigger):].strip()
                
                # If user said just "remember this" with no extra text,
                # summarize the exchange (user question + assistant answer key points)
                if len(fact_after) < 10 and assistant_msg and len(assistant_msg) > 10:
                    # Find the previous user question (look at the context)
                    # Create a concise summary: "Topic: key answer"
                    # Take first 2 sentences of assistant response as summary
                    sentences = assistant_msg.replace('\n', ' ').replace('  ', ' ').split('.')
                    summary_parts = [s.strip() for s in sentences if len(s.strip()) > 15][:3]
                    content = '. '.join(summary_parts)
                    if len(content) > 400:
                        content = content[:400] + '...'
                    if not content:
                        content = assistant_msg[:400].strip()
                    extracted.append({"content": content, "type": "explicit", "source": "user_request"})
                elif len(fact_after) >= 10:
                    extracted.append({"content": fact_after, "type": "explicit", "source": "user_request"})
                else:
                    # Fallback: save the full user message
                    extracted.append({"content": user_msg.strip(), "type": "explicit", "source": "user_request"})
                break
        
        # 2. Personal preferences ("I prefer", "I like", "I use", "I work with")
        if not extracted:
            pref_patterns = [
                ("i prefer ", "preference"), ("i like ", "preference"), ("i love ", "preference"),
                ("i hate ", "preference"), ("i don't like ", "preference"), ("i dislike ", "preference"),
                ("i use ", "tool"), ("i work with ", "tool"), ("i'm using ", "tool"),
                ("my favorite ", "preference"), ("i always ", "habit"), ("i never ", "habit"),
                ("i usually ", "habit"), ("i'm a ", "identity"), ("i am a ", "identity"),
                ("my job is ", "identity"), ("i work as ", "identity"), ("i work at ", "identity"),
                ("my name is ", "identity"), ("call me ", "identity"),
                ("i live in ", "location"), ("i'm from ", "location"), ("i'm based in ", "location"),
                ("my project ", "project"), ("i'm building ", "project"), ("i'm working on ", "project"),
            ]
            
            for pattern, mem_type in pref_patterns:
                if pattern in user_lower:
                    sentences = user_msg.replace("!", ".").replace("?", ".").split(".")
                    for sent in sentences:
                        if pattern in sent.lower() and len(sent.strip()) > 10:
                            extracted.append({"content": sent.strip(), "type": mem_type, "source": "auto_extract"})
                            break
                    break
        
        # 3. Tech stack detection from longer messages
        if not extracted and assistant_msg and len(user_msg) > 50:
            tech_indicators = ["python", "javascript", "typescript", "react", "vue", "angular", 
                             "node", "django", "flask", "fastapi", "postgres", "mongodb",
                             "docker", "kubernetes", "aws", "gcp", "azure"]
            mentioned_tech = [t for t in tech_indicators if t in user_lower]
            if len(mentioned_tech) >= 2:
                tech_str = ", ".join(mentioned_tech[:5])
                extracted.append({
                    "content": f"User works with: {tech_str}",
                    "type": "tech_stack",
                    "source": "auto_extract"
                })
        
        if not extracted:
            return {"memories_extracted": [], "count": 0}
        
        # Save extracted memories via memory_service HTTP API
        # Path is /memory/rag/memories (memory_service mounts rag under /memory prefix)
        import httpx
        saved = []
        
        # Build proxy headers for memory_service billing checks
        extract_headers = {"x-user-id": user_id}
        for h in ("x-user-role", "x-user-plan", "x-unlimited-credits", "x-is-superuser", "x-org-id"):
            val = request.headers.get(h)
            if val:
                extract_headers[h] = val
        
        for mem in extracted[:3]:
            try:
                content = mem["content"]
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        "http://memory_service:8000/memory/rag/memories",
                        json={
                            "content": content,
                            "metadata": {
                                "type": mem["type"],
                                "source": mem["source"],
                                "chat_id": chat_id,
                                "auto_extracted": True,
                            }
                        },
                        headers=extract_headers,
                    )
                    logger.info(f"[MEMORY] memory_service response: status={resp.status_code} body={resp.text[:200]}")
                    if resp.status_code in (200, 201):
                        data = resp.json()
                        saved.append({
                            "id": data.get("id"),
                            "content": content[:100],
                            "type": mem["type"],
                        })
                        logger.info(f"[MEMORY] Saved memory for user {user_id[:8]}...: {content[:60]}")
                    else:
                        logger.error(f"[MEMORY] memory_service returned {resp.status_code}: {resp.text[:300]}")
            except Exception as e:
                logger.error(f"[MEMORY] Failed to save memory via HTTP: {e}")
        
        return {"memories_extracted": saved, "count": len(saved)}
    
    except Exception as e:
        logger.error(f"[MEMORY] Extract memories error: {e}")
        return {"memories_extracted": [], "count": 0, "error": str(e)}


@router.post("/memories/save")
async def save_memory_proxy(request: Request):
    """
    Proxy endpoint to save a memory via memory_service.
    Frontend calls this instead of /rag/memories directly to avoid auth issues.
    Body: { "content": str, "metadata"?: dict, "summarize"?: bool }
    
    When summarize=true (default for long content), creates a concise summary
    instead of saving raw text. This prevents 15 raw messages from being dumped.
    """
    user_id = request.headers.get("x-user-id", "anonymous")
    if user_id == "anonymous":
        return {"error": "Not authenticated"}, 401
    
    try:
        body = await request.json()
        content = body.get("content", "")
        metadata = body.get("metadata", {})
        should_summarize = body.get("summarize", True)
        
        if not content or len(content.strip()) < 2:
            return {"error": "Content too short"}
        
        # Auto-summarize long content
        save_content = content.strip()
        if should_summarize and len(save_content) > 300:
            save_content = _summarize_for_memory(save_content)
        elif len(save_content) > 500:
            save_content = save_content[:500] + "..."
        
        # Forward auth headers so memory_service billing checks work correctly
        proxy_headers = {"x-user-id": user_id}
        for h in ("x-user-role", "x-user-plan", "x-unlimited-credits", "x-is-superuser", "x-org-id"):
            val = request.headers.get(h)
            if val:
                proxy_headers[h] = val
        
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "http://memory_service:8000/memory/rag/memories",
                json={"content": save_content, "metadata": metadata},
                headers=proxy_headers,
            )
            logger.info(f"[MEMORY-PROXY] Save memory: status={resp.status_code} (summarized={len(save_content) != len(content.strip())})")
            if resp.status_code in (200, 201):
                return resp.json()
            else:
                logger.error(f"[MEMORY-PROXY] memory_service returned {resp.status_code}: {resp.text[:300]}")
                return {"error": f"Memory service error: {resp.status_code}"}
    except Exception as e:
        logger.error(f"[MEMORY-PROXY] Save memory error: {e}")
        return {"error": str(e)}


def _summarize_for_memory(text: str) -> str:
    """Create a concise memory summary from long text.
    Extracts key facts, strips markdown formatting, and limits to ~300 chars."""
    import re
    
    # Strip markdown formatting
    clean = text
    clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)  # Bold
    clean = re.sub(r'\*(.+?)\*', r'\1', clean)  # Italic
    clean = re.sub(r'#+\s+', '', clean)  # Headings
    clean = re.sub(r'```[\s\S]*?```', '[code block]', clean)  # Code blocks
    clean = re.sub(r'`([^`]+)`', r'\1', clean)  # Inline code
    clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean)  # Links
    clean = re.sub(r'^\s*[-*]\s+', '• ', clean, flags=re.MULTILINE)  # Bullets
    clean = re.sub(r'\n{3,}', '\n\n', clean)  # Extra newlines
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', clean.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    # Take first 3-4 meaningful sentences as summary
    summary_parts = []
    char_count = 0
    for s in sentences:
        if char_count + len(s) > 350:
            break
        summary_parts.append(s)
        char_count += len(s)
    
    result = ' '.join(summary_parts)
    if not result or len(result) < 20:
        # Fallback: take first 300 chars
        result = clean[:300].strip()
    
    if len(result) > 400:
        result = result[:400] + "..."
    
    return result


@router.get("/memories/list")
async def list_memories_proxy(request: Request):
    """
    Proxy endpoint to list memories via memory_service.
    Frontend calls this instead of /rag/memories directly to avoid auth issues.
    """
    user_id = request.headers.get("x-user-id", "anonymous")
    if user_id == "anonymous":
        return []
    
    try:
        limit = request.query_params.get("limit", "50")
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"http://memory_service:8000/memory/rag/memories?limit={limit}",
                headers={"x-user-id": user_id},
            )
            logger.info(f"[MEMORY-PROXY] List memories: status={resp.status_code}, count={len(resp.json()) if resp.status_code == 200 else 'N/A'}")
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"[MEMORY-PROXY] memory_service returned {resp.status_code}: {resp.text[:300]}")
                return []
    except Exception as e:
        logger.error(f"[MEMORY-PROXY] List memories error: {e}")
        return []


@router.delete("/memories/{memory_id}")
async def delete_memory_proxy(memory_id: str, request: Request):
    """
    Proxy endpoint to delete a memory via memory_service.
    """
    user_id = request.headers.get("x-user-id", "anonymous")
    if user_id == "anonymous":
        return {"error": "Not authenticated"}
    
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(
                f"http://memory_service:8000/memory/rag/memories/{memory_id}",
                headers={"x-user-id": user_id},
            )
            logger.info(f"[MEMORY-PROXY] Delete memory {memory_id}: status={resp.status_code}")
            if resp.status_code in (200, 204):
                return {"success": True}
            else:
                return {"error": f"Memory service error: {resp.status_code}"}
    except Exception as e:
        logger.error(f"[MEMORY-PROXY] Delete memory error: {e}")
        return {"error": str(e)}


@router.post("/conversations/categorize")
async def categorize_conversations(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Smart-group conversations into topic categories.
    Returns conversations organized by auto-detected topics.
    """
    user_id = request.headers.get("x-user-id", "anonymous")
    if user_id == "anonymous":
        return {"groups": []}
    
    try:
        from sqlalchemy import func as sqlfunc
        
        # Get conversations with message counts via subquery
        msg_count_sub = (
            select(
                ResonantChatMessage.chat_id,
                sqlfunc.count(ResonantChatMessage.id).label("msg_count"),
                sqlfunc.max(ResonantChatMessage.created_at).label("last_msg_at"),
            )
            .group_by(ResonantChatMessage.chat_id)
            .subquery()
        )
        
        result = await session.execute(
            select(
                ResonantChat,
                msg_count_sub.c.msg_count,
                msg_count_sub.c.last_msg_at,
            )
            .outerjoin(msg_count_sub, ResonantChat.id == msg_count_sub.c.chat_id)
            .where(ResonantChat.user_id == user_id)
            .where(ResonantChat.status != "deleted")
            .order_by(ResonantChat.updated_at.desc())
        )
        rows = result.all()
        
        if not rows:
            return {"groups": []}
        
        # Broader topic keywords to match real conversation titles
        topic_keywords = {
            "Coding & Development": ["code", "coding", "python", "javascript", "react", "api", "bug", "error", "function", "class", "database", "sql", "html", "css", "deploy", "git", "debug", "build", "compile", "server", "frontend", "backend", "dev", "programming", "script", "app", "software", "github", "docker", "component", "endpoint"],
            "Shopping & Local": ["shop", "store", "buy", "price", "open", "close", "hours", "target", "walmart", "amazon", "restaurant", "food", "order", "delivery", "near", "location", "address", "directions", "map", "where is", "san francisco", "new york", "los angeles"],
            "Writing & Content": ["write", "writing", "essay", "article", "blog", "story", "email", "letter", "draft", "edit", "grammar", "content", "copy", "text", "document", "report", "summary", "summarize", "translate", "translation"],
            "Research & Learning": ["research", "learn", "study", "explain", "how does", "what is", "why", "understand", "tutorial", "course", "book", "science", "history", "math", "compare", "difference", "between", "tell me about", "check now", "find out"],
            "Business & Work": ["business", "meeting", "project", "plan", "strategy", "marketing", "sales", "revenue", "startup", "company", "client", "proposal", "budget", "schedule", "market", "competitor", "investment", "profit", "team"],
            "Creative & Design": ["design", "creative", "image", "art", "color", "logo", "ui", "ux", "layout", "graphic", "photo", "video", "music", "animation", "draw", "illustration", "brand"],
            "Data & Analysis": ["data", "analysis", "chart", "graph", "statistics", "metrics", "dashboard", "csv", "excel", "insight", "trend", "numbers", "calculate", "formula"],
            "AI & Agents": ["agent", "ai", "model", "prompt", "gpt", "llm", "neural", "training", "fine-tune", "hallucination", "embedding", "vector", "create.*agent", "new agent", "autonomous"],
        }
        
        groups = {}
        uncategorized = []
        
        def _build_conv(chat, msg_count, last_msg_at):
            return {
                "id": str(chat.id),
                "title": chat.title or "Untitled Chat",
                "created_at": chat.created_at.isoformat() if chat.created_at else None,
                "updated_at": chat.updated_at.isoformat() if chat.updated_at else None,
                "message_count": msg_count or 0,
                "last_message_at": last_msg_at.isoformat() if last_msg_at else None,
            }
        
        # Pre-fetch first user message for each chat to use as fallback for categorization
        chat_ids = [chat.id for chat, _, _ in rows]
        first_msgs = {}
        if chat_ids:
            # Get first user message per chat for keyword matching fallback
            from sqlalchemy import distinct
            first_msg_sub = (
                select(
                    ResonantChatMessage.chat_id,
                    sqlfunc.min(ResonantChatMessage.created_at).label("first_at"),
                )
                .where(ResonantChatMessage.chat_id.in_(chat_ids))
                .where(ResonantChatMessage.role == "user")
                .group_by(ResonantChatMessage.chat_id)
                .subquery()
            )
            fm_result = await session.execute(
                select(ResonantChatMessage.chat_id, ResonantChatMessage.content)
                .join(first_msg_sub, (ResonantChatMessage.chat_id == first_msg_sub.c.chat_id) & (ResonantChatMessage.created_at == first_msg_sub.c.first_at))
                .where(ResonantChatMessage.role == "user")
            )
            for fm_chat_id, fm_content in fm_result.all():
                first_msgs[fm_chat_id] = (fm_content or "")[:200].lower()
        
        def _match_topic(text):
            """Match text against topic keywords, return topic name or None."""
            for topic, keywords in topic_keywords.items():
                if any(kw in text for kw in keywords):
                    return topic
            return None
        
        for chat, msg_count, last_msg_at in rows:
            title = (chat.title or "").lower()
            
            # Try title first, then first message content as fallback
            matched_topic = _match_topic(title)
            if not matched_topic and chat.id in first_msgs:
                matched_topic = _match_topic(first_msgs[chat.id])
            
            if matched_topic:
                if matched_topic not in groups:
                    groups[matched_topic] = []
                groups[matched_topic].append(_build_conv(chat, msg_count, last_msg_at))
            else:
                uncategorized.append(_build_conv(chat, msg_count, last_msg_at))
        
        # Build sorted groups list (most conversations first)
        result_groups = []
        for topic, convs in sorted(groups.items(), key=lambda x: -len(x[1])):
            result_groups.append({
                "topic": topic,
                "count": len(convs),
                "conversations": sorted(convs, key=lambda c: c.get("updated_at") or "", reverse=True),
            })
        
        if uncategorized:
            result_groups.append({
                "topic": "Other",
                "count": len(uncategorized),
                "conversations": sorted(uncategorized, key=lambda c: c.get("updated_at") or "", reverse=True),
            })
        
        return {"groups": result_groups, "total": len(rows)}
    
    except Exception as e:
        logger.error(f"Categorize conversations error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"groups": [], "error": str(e)}


@router.post("/dsid/verify-proof")
async def verify_merkle_proof_endpoint(request: Request):
    """
    Verify a Merkle proof.
    
    Body: {
        "target_hash": "hash to verify",
        "merkle_root": "expected root",
        "proof": [{"hash": "...", "position": "left|right"}, ...]
    }
    """
    try:
        body = await request.json()
        target_hash = body.get("target_hash", "")
        merkle_root = body.get("merkle_root", "")
        proof = body.get("proof", [])
        
        if not target_hash or not merkle_root:
            return {"error": "target_hash and merkle_root are required"}
        
        is_valid = dsid_integration.verify_merkle_proof(target_hash, merkle_root, proof)
        
        return {
            "status": "ok",
            "verified": is_valid,
        }
    except Exception as e:
        logger.error(f"Verify Merkle proof error: {e}")
        return {"error": str(e)}
