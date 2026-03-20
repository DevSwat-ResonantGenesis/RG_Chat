"""
Conversation Search Router
===========================

Provides full-text and semantic search across conversations.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, or_, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..models import ResonantChat, ResonantChatMessage
from ..services.resonance_hashing import ResonanceHasher

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resonant-chat", tags=["search"])


class SearchResult(BaseModel):
    message_id: str
    chat_id: str
    chat_title: str
    role: str
    content: str
    content_snippet: str
    timestamp: str
    relevance_score: float
    match_type: str  # "exact", "fuzzy", "semantic"
    highlights: List[str] = []


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResult]
    search_time_ms: float


def _create_snippet(content: str, query: str, max_length: int = 200) -> str:
    """Create a snippet around the query match."""
    content_lower = content.lower()
    query_lower = query.lower()
    
    # Find query position
    pos = content_lower.find(query_lower)
    
    if pos == -1:
        # No exact match, return start of content
        return content[:max_length] + ("..." if len(content) > max_length else "")
    
    # Calculate snippet boundaries
    start = max(0, pos - 50)
    end = min(len(content), pos + len(query) + 150)
    
    snippet = content[start:end]
    
    # Add ellipsis if truncated
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    
    return snippet


def _extract_highlights(content: str, query: str, max_highlights: int = 3) -> List[str]:
    """Extract highlighted portions containing the query."""
    highlights = []
    content_lower = content.lower()
    query_lower = query.lower()
    
    # Split query into words for partial matching
    query_words = query_lower.split()
    
    # Find sentences containing query words
    sentences = content.replace("\n", ". ").split(". ")
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(word in sentence_lower for word in query_words):
            highlight = sentence.strip()
            if len(highlight) > 150:
                highlight = highlight[:150] + "..."
            if highlight and highlight not in highlights:
                highlights.append(highlight)
                if len(highlights) >= max_highlights:
                    break
    
    return highlights


def _calculate_relevance(content: str, query: str) -> float:
    """Calculate relevance score for a search result."""
    content_lower = content.lower()
    query_lower = query.lower()
    
    score = 0.0
    
    # Exact match bonus
    if query_lower in content_lower:
        score += 0.5
        # Multiple occurrences bonus
        occurrences = content_lower.count(query_lower)
        score += min(0.2, occurrences * 0.05)
    
    # Word match scoring
    query_words = set(query_lower.split())
    content_words = set(content_lower.split())
    
    matching_words = query_words.intersection(content_words)
    if query_words:
        word_match_ratio = len(matching_words) / len(query_words)
        score += word_match_ratio * 0.3
    
    # Length penalty (prefer concise matches)
    if len(content) > 1000:
        score *= 0.9
    
    return min(1.0, score)


@router.get("/search", response_model=SearchResponse)
async def search_conversations(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    role: Optional[str] = Query(None, regex="^(user|assistant)$"),
    chat_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """
    Search across all conversations.
    
    Supports:
    - Full-text search in message content
    - Filter by role (user/assistant)
    - Filter by specific chat
    - Date range filtering
    """
    import time
    start_time = time.time()
    
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    # Build base query
    query_stmt = (
        select(ResonantChatMessage, ResonantChat)
        .join(ResonantChat, ResonantChatMessage.chat_id == ResonantChat.id)
        .where(ResonantChat.user_id == UUID(user_id))
    )
    
    # Apply filters
    if role:
        query_stmt = query_stmt.where(ResonantChatMessage.role == role)
    
    if chat_id:
        try:
            query_stmt = query_stmt.where(ResonantChatMessage.chat_id == UUID(chat_id))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid chat ID format")
    
    if date_from:
        try:
            from_date = datetime.fromisoformat(date_from)
            query_stmt = query_stmt.where(ResonantChatMessage.created_at >= from_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_from format")
    
    if date_to:
        try:
            to_date = datetime.fromisoformat(date_to)
            query_stmt = query_stmt.where(ResonantChatMessage.created_at <= to_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_to format")
    
    # Text search filter (case-insensitive)
    query_stmt = query_stmt.where(
        func.lower(ResonantChatMessage.content).contains(q.lower())
    )
    
    # Order by recency
    query_stmt = query_stmt.order_by(ResonantChatMessage.created_at.desc())
    
    # Execute query
    result = await session.execute(query_stmt)
    rows = result.all()
    
    # Process results
    results = []
    for msg, chat in rows:
        relevance = _calculate_relevance(msg.content, q)
        
        results.append(SearchResult(
            message_id=str(msg.id),
            chat_id=str(msg.chat_id),
            chat_title=chat.title or "Untitled",
            role=msg.role,
            content=msg.content,
            content_snippet=_create_snippet(msg.content, q),
            timestamp=msg.created_at.isoformat() if msg.created_at else "",
            relevance_score=round(relevance, 4),
            match_type="exact" if q.lower() in msg.content.lower() else "fuzzy",
            highlights=_extract_highlights(msg.content, q),
        ))
    
    # Sort by relevance
    results.sort(key=lambda x: x.relevance_score, reverse=True)
    
    # Apply pagination
    total_results = len(results)
    results = results[offset:offset + limit]
    
    search_time = (time.time() - start_time) * 1000
    
    return SearchResponse(
        query=q,
        total_results=total_results,
        results=results,
        search_time_ms=round(search_time, 2),
    )


@router.get("/search/semantic")
async def semantic_search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(10, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
):
    """
    Semantic search using Hash Sphere coordinates.
    
    Finds messages that are semantically similar to the query
    based on their 3D position in the Hash Sphere.
    """
    import time
    start_time = time.time()
    
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    # Hash the query to get XYZ coordinates
    try:
        hasher = ResonanceHasher()
        query_hash = hasher.hash_text(q)
        query_xyz = hasher.hash_to_coords(query_hash)
    except Exception as e:
        logger.warning(f"Hashing failed: {e}")
        # Fallback to simple hash
        import hashlib
        query_hash = hashlib.sha256(q.encode()).hexdigest()[:32]
        query_xyz = (
            int(query_hash[:8], 16) / 0xFFFFFFFF,
            int(query_hash[8:16], 16) / 0xFFFFFFFF,
            int(query_hash[16:24], 16) / 0xFFFFFFFF,
        )
    
    # Get all messages with XYZ coordinates for this user
    query_stmt = (
        select(ResonantChatMessage, ResonantChat)
        .join(ResonantChat, ResonantChatMessage.chat_id == ResonantChat.id)
        .where(
            and_(
                ResonantChat.user_id == UUID(user_id),
                ResonantChatMessage.xyz_x.isnot(None),
            )
        )
    )
    
    result = await session.execute(query_stmt)
    rows = result.all()
    
    # Calculate distances and sort
    import math
    
    results_with_distance = []
    for msg, chat in rows:
        # Calculate Euclidean distance in 3D space
        dx = query_xyz[0] - (msg.xyz_x or 0.5)
        dy = query_xyz[1] - (msg.xyz_y or 0.5)
        dz = query_xyz[2] - (msg.xyz_z or 0.5)
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Convert distance to similarity (closer = higher score)
        similarity = math.exp(-distance * 2)  # Exponential decay
        
        results_with_distance.append({
            "message_id": str(msg.id),
            "chat_id": str(msg.chat_id),
            "chat_title": chat.title or "Untitled",
            "role": msg.role,
            "content": msg.content[:500] + ("..." if len(msg.content) > 500 else ""),
            "timestamp": msg.created_at.isoformat() if msg.created_at else "",
            "distance": round(distance, 4),
            "similarity": round(similarity, 4),
            "xyz": [msg.xyz_x, msg.xyz_y, msg.xyz_z],
            "hash": msg.hash[:16] + "..." if msg.hash else None,
        })
    
    # Sort by similarity (descending)
    results_with_distance.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Limit results
    results = results_with_distance[:limit]
    
    search_time = (time.time() - start_time) * 1000
    
    return {
        "query": q,
        "query_hash": query_hash[:16] + "...",
        "query_xyz": list(query_xyz),
        "total_searched": len(rows),
        "results": results,
        "search_time_ms": round(search_time, 2),
    }


@router.get("/search/suggestions")
async def search_suggestions(
    request: Request,
    q: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(5, ge=1, le=10),
    session: AsyncSession = Depends(get_session),
):
    """
    Get search suggestions based on partial query.
    
    Returns recent messages and chat titles that match the query prefix.
    """
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    suggestions = []
    
    # Search chat titles
    chat_result = await session.execute(
        select(ResonantChat)
        .where(
            and_(
                ResonantChat.user_id == UUID(user_id),
                func.lower(ResonantChat.title).contains(q.lower())
            )
        )
        .order_by(ResonantChat.created_at.desc())
        .limit(limit)
    )
    chats = chat_result.scalars().all()
    
    for chat in chats:
        suggestions.append({
            "type": "chat",
            "text": chat.title,
            "chat_id": str(chat.id),
        })
    
    # Search recent messages
    if len(suggestions) < limit:
        remaining = limit - len(suggestions)
        msg_result = await session.execute(
            select(ResonantChatMessage, ResonantChat)
            .join(ResonantChat, ResonantChatMessage.chat_id == ResonantChat.id)
            .where(
                and_(
                    ResonantChat.user_id == UUID(user_id),
                    func.lower(ResonantChatMessage.content).contains(q.lower())
                )
            )
            .order_by(ResonantChatMessage.created_at.desc())
            .limit(remaining * 2)  # Get more to deduplicate
        )
        rows = msg_result.all()
        
        seen_snippets = set()
        for msg, chat in rows:
            snippet = _create_snippet(msg.content, q, max_length=100)
            snippet_key = snippet[:50].lower()
            
            if snippet_key not in seen_snippets:
                seen_snippets.add(snippet_key)
                suggestions.append({
                    "type": "message",
                    "text": snippet,
                    "chat_id": str(chat.id),
                    "message_id": str(msg.id),
                })
                
                if len(suggestions) >= limit:
                    break
    
    return {
        "query": q,
        "suggestions": suggestions[:limit],
    }
