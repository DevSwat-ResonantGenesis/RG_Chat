"""
Message Feedback Router
========================

Implements feedback collection for chat messages to enable self-improving agents.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..models import ResonantChatMessage
from ..services.self_improving_agent import self_improving_agent, FeedbackType
from ..services.adaptive_weights import adaptive_tuner
from ..services.ab_testing import ab_tester

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resonant-chat", tags=["feedback"])


class FeedbackRating(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    NEUTRAL = "neutral"
    POOR = "poor"
    TERRIBLE = "terrible"


class FeedbackCategory(str, Enum):
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    TONE = "tone"
    SPEED = "speed"
    OTHER = "other"


class SubmitFeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (worst) to 5 (best)")
    category: Optional[FeedbackCategory] = None
    comment: Optional[str] = Field(None, max_length=1000)
    is_helpful: Optional[bool] = None
    suggested_response: Optional[str] = Field(None, max_length=5000)
    tags: Optional[List[str]] = None


class FeedbackResponse(BaseModel):
    message_id: str
    feedback_id: str
    rating: int
    category: Optional[str]
    recorded: bool
    learning_applied: bool


class FeedbackStats(BaseModel):
    total_feedback: int
    average_rating: float
    rating_distribution: Dict[int, int]
    category_breakdown: Dict[str, int]
    helpful_count: int
    not_helpful_count: int


# In-memory feedback storage (in production, use database)
_feedback_store: Dict[str, Dict[str, Any]] = {}


def _rating_to_score(rating: int) -> float:
    """Convert 1-5 rating to 0-1 score."""
    return (rating - 1) / 4.0


@router.post("/message/{message_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    message_id: str,
    feedback: SubmitFeedbackRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Submit feedback for a specific message.
    
    This feedback is used to:
    1. Improve agent responses over time (self-improving agent)
    2. Track quality metrics
    3. Identify areas for improvement
    """
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    # Verify message exists
    try:
        result = await session.execute(
            select(ResonantChatMessage).where(ResonantChatMessage.id == UUID(message_id))
        )
        message = result.scalar_one_or_none()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid message ID format")
    
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Only allow feedback on assistant messages
    if message.role != "assistant":
        raise HTTPException(status_code=400, detail="Feedback can only be submitted for assistant messages")
    
    # Generate feedback ID
    import uuid
    feedback_id = str(uuid.uuid4())
    
    # Store feedback
    feedback_data = {
        "id": feedback_id,
        "message_id": message_id,
        "user_id": user_id,
        "rating": feedback.rating,
        "category": feedback.category.value if feedback.category else None,
        "comment": feedback.comment,
        "is_helpful": feedback.is_helpful,
        "suggested_response": feedback.suggested_response,
        "tags": feedback.tags or [],
        "provider": message.ai_provider,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    if message_id not in _feedback_store:
        _feedback_store[message_id] = {}
    _feedback_store[message_id][feedback_id] = feedback_data
    
    # Apply learning to self-improving agent
    learning_applied = False
    try:
        # Get agent ID from provider
        agent_id = message.ai_provider or "default"
        
        # Record quality score
        score = _rating_to_score(feedback.rating)
        await self_improving_agent.record_feedback(
            agent_id=agent_id,
            message=message.content[:500],  # Truncate for storage
            response=message.content,
            feedback_type=FeedbackType.QUALITY_SCORE,
            value=score,
        )
        
        # Record helpfulness if provided
        if feedback.is_helpful is not None:
            await self_improving_agent.record_feedback(
                agent_id=agent_id,
                message=message.content[:500],
                response=message.content,
                feedback_type=FeedbackType.HELPFUL if feedback.is_helpful else FeedbackType.NOT_HELPFUL,
                value=1.0 if feedback.is_helpful else 0.0,
            )
        
        # Record correction if suggested response provided
        if feedback.suggested_response:
            await self_improving_agent.record_feedback(
                agent_id=agent_id,
                message=message.content[:500],
                response=message.content,
                feedback_type=FeedbackType.CORRECTION,
                value=0.0,
                correction=feedback.suggested_response,
            )
        
        learning_applied = True
        logger.info(f"Applied learning from feedback for agent {agent_id}")
        
        # Record feedback for adaptive weight tuning
        # Get memory scores from message metadata if available
        memory_scores = {}
        if message.meta_data and isinstance(message.meta_data, dict):
            memory_scores = message.meta_data.get("top_memory_scores", {})
        
        if memory_scores:
            adaptive_tuner.record_feedback(
                user_id=user_id,
                query=message.content[:200],  # Use message content as query proxy
                top_memory=memory_scores,
                feedback=feedback.rating >= 4,  # 4-5 = positive, 1-3 = negative
                clicked_index=0,
            )
            logger.info(f"Recorded adaptive weight feedback for user {user_id[:8]}...")
        
    except Exception as e:
        logger.warning(f"Failed to apply learning: {e}")
    
    return FeedbackResponse(
        message_id=message_id,
        feedback_id=feedback_id,
        rating=feedback.rating,
        category=feedback.category.value if feedback.category else None,
        recorded=True,
        learning_applied=learning_applied,
    )


@router.get("/message/{message_id}/feedback")
async def get_message_feedback(
    message_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Get all feedback for a specific message."""
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    feedback_list = list(_feedback_store.get(message_id, {}).values())
    
    # Calculate summary
    if feedback_list:
        ratings = [f["rating"] for f in feedback_list]
        avg_rating = sum(ratings) / len(ratings)
        helpful_count = sum(1 for f in feedback_list if f.get("is_helpful") is True)
        not_helpful_count = sum(1 for f in feedback_list if f.get("is_helpful") is False)
    else:
        avg_rating = 0.0
        helpful_count = 0
        not_helpful_count = 0
    
    return {
        "message_id": message_id,
        "feedback_count": len(feedback_list),
        "average_rating": round(avg_rating, 2),
        "helpful_count": helpful_count,
        "not_helpful_count": not_helpful_count,
        "feedback": feedback_list,
    }


@router.post("/message/{message_id}/thumbs-up")
async def thumbs_up(
    message_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Quick thumbs up feedback for a message."""
    feedback_request = SubmitFeedbackRequest(
        rating=5,
        is_helpful=True,
    )
    return await submit_feedback(message_id, feedback_request, request, session)


@router.post("/message/{message_id}/thumbs-down")
async def thumbs_down(
    message_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Quick thumbs down feedback for a message."""
    feedback_request = SubmitFeedbackRequest(
        rating=1,
        is_helpful=False,
    )
    return await submit_feedback(message_id, feedback_request, request, session)


@router.get("/feedback/stats")
async def get_feedback_stats(
    request: Request,
    time_range: str = "7d",
):
    """
    Get aggregated feedback statistics.
    
    Args:
        time_range: Time range for stats (1d, 7d, 30d, all)
    """
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    # Collect all feedback
    all_feedback = []
    for message_feedback in _feedback_store.values():
        all_feedback.extend(message_feedback.values())
    
    # Filter by user if not admin
    user_role = request.headers.get("x-user-role", "user")
    if user_role not in ["admin", "owner", "platform_dev"]:
        all_feedback = [f for f in all_feedback if f.get("user_id") == user_id]
    
    if not all_feedback:
        return FeedbackStats(
            total_feedback=0,
            average_rating=0.0,
            rating_distribution={1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            category_breakdown={},
            helpful_count=0,
            not_helpful_count=0,
        )
    
    # Calculate stats
    ratings = [f["rating"] for f in all_feedback]
    rating_dist = {i: ratings.count(i) for i in range(1, 6)}
    
    categories = [f.get("category") for f in all_feedback if f.get("category")]
    category_breakdown = {}
    for cat in categories:
        category_breakdown[cat] = category_breakdown.get(cat, 0) + 1
    
    helpful_count = sum(1 for f in all_feedback if f.get("is_helpful") is True)
    not_helpful_count = sum(1 for f in all_feedback if f.get("is_helpful") is False)
    
    return FeedbackStats(
        total_feedback=len(all_feedback),
        average_rating=round(sum(ratings) / len(ratings), 2),
        rating_distribution=rating_dist,
        category_breakdown=category_breakdown,
        helpful_count=helpful_count,
        not_helpful_count=not_helpful_count,
    )
