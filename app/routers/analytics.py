"""
Analytics Dashboard Router
===========================

Provides analytics and metrics endpoints for chat usage, quality, and performance.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from pydantic import BaseModel
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..models import ResonantChat, ResonantChatMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


class TimeRange:
    """Helper for time range calculations."""
    
    @staticmethod
    def get_start_date(range_str: str) -> datetime:
        now = datetime.utcnow()
        if range_str == "1d":
            return now - timedelta(days=1)
        elif range_str == "7d":
            return now - timedelta(days=7)
        elif range_str == "30d":
            return now - timedelta(days=30)
        elif range_str == "90d":
            return now - timedelta(days=90)
        elif range_str == "1y":
            return now - timedelta(days=365)
        else:  # "all"
            return datetime(2020, 1, 1)


class UsageStats(BaseModel):
    total_messages: int
    total_conversations: int
    user_messages: int
    assistant_messages: int
    avg_messages_per_conversation: float
    active_days: int


class QualityStats(BaseModel):
    avg_resonance_score: float
    high_quality_responses: int  # resonance > 0.7
    low_quality_responses: int   # resonance < 0.3
    quality_trend: str  # "improving", "stable", "declining"


class ProviderStats(BaseModel):
    provider: str
    message_count: int
    avg_resonance: float
    percentage: float


class TopicStats(BaseModel):
    topic: str
    count: int
    avg_resonance: float


class DailyStats(BaseModel):
    date: str
    messages: int
    conversations: int
    avg_resonance: float


class AnalyticsDashboard(BaseModel):
    time_range: str
    usage: UsageStats
    quality: QualityStats
    providers: List[ProviderStats]
    daily_activity: List[DailyStats]
    top_topics: List[TopicStats]


def _extract_topics(content: str) -> List[str]:
    """Extract potential topics from message content."""
    import re
    
    # Common topic indicators
    topic_patterns = [
        r'\b(code|coding|programming|developer)\b',
        r'\b(bug|error|fix|debug)\b',
        r'\b(api|endpoint|request|response)\b',
        r'\b(database|sql|query|data)\b',
        r'\b(design|ui|ux|frontend|backend)\b',
        r'\b(test|testing|unit|integration)\b',
        r'\b(deploy|deployment|production|staging)\b',
        r'\b(security|auth|authentication|authorization)\b',
        r'\b(performance|optimize|speed|cache)\b',
        r'\b(ai|machine learning|ml|model)\b',
        r'\b(business|strategy|plan|goal)\b',
        r'\b(writing|content|blog|article)\b',
        r'\b(analysis|analyze|research|study)\b',
    ]
    
    topics = []
    content_lower = content.lower()
    
    topic_names = [
        "coding", "debugging", "api", "database", "design",
        "testing", "deployment", "security", "performance",
        "ai/ml", "business", "writing", "analysis"
    ]
    
    for i, pattern in enumerate(topic_patterns):
        if re.search(pattern, content_lower):
            topics.append(topic_names[i])
    
    return topics if topics else ["general"]


async def _get_analytics_impl(
    request: Request,
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y|all)$"),
    session: AsyncSession = Depends(get_session),
) -> AnalyticsDashboard:
    """
    Get comprehensive analytics dashboard.
    
    Args:
        time_range: Time range for analytics (1d, 7d, 30d, 90d, 1y, all)
    
    Returns:
        Complete analytics dashboard with usage, quality, and activity metrics
    """
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    start_date = TimeRange.get_start_date(time_range)
    
    # Get all conversations for user in time range
    conv_result = await session.execute(
        select(ResonantChat)
        .where(
            and_(
                ResonantChat.user_id == UUID(user_id),
                ResonantChat.created_at >= start_date
            )
        )
    )
    conversations = conv_result.scalars().all()
    conversation_ids = [c.id for c in conversations]
    
    # Get all messages for these conversations
    if conversation_ids:
        msg_result = await session.execute(
            select(ResonantChatMessage)
            .where(
                and_(
                    ResonantChatMessage.chat_id.in_(conversation_ids),
                    ResonantChatMessage.created_at >= start_date
                )
            )
            .order_by(ResonantChatMessage.created_at)
        )
        messages = msg_result.scalars().all()
    else:
        messages = []
    
    # Calculate usage stats
    user_messages = [m for m in messages if m.role == "user"]
    assistant_messages = [m for m in messages if m.role == "assistant"]
    
    active_dates = set()
    for m in messages:
        if m.created_at:
            active_dates.add(m.created_at.date())
    
    usage = UsageStats(
        total_messages=len(messages),
        total_conversations=len(conversations),
        user_messages=len(user_messages),
        assistant_messages=len(assistant_messages),
        avg_messages_per_conversation=len(messages) / max(len(conversations), 1),
        active_days=len(active_dates),
    )
    
    # Calculate quality stats
    resonance_scores = [
        m.resonance_score for m in assistant_messages 
        if m.resonance_score is not None
    ]
    
    avg_resonance = sum(resonance_scores) / max(len(resonance_scores), 1)
    high_quality = sum(1 for s in resonance_scores if s > 0.7)
    low_quality = sum(1 for s in resonance_scores if s < 0.3)
    
    # Calculate quality trend (compare first half vs second half)
    if len(resonance_scores) >= 10:
        mid = len(resonance_scores) // 2
        first_half_avg = sum(resonance_scores[:mid]) / mid
        second_half_avg = sum(resonance_scores[mid:]) / (len(resonance_scores) - mid)
        
        if second_half_avg > first_half_avg + 0.05:
            trend = "improving"
        elif second_half_avg < first_half_avg - 0.05:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "stable"
    
    quality = QualityStats(
        avg_resonance_score=round(avg_resonance, 4),
        high_quality_responses=high_quality,
        low_quality_responses=low_quality,
        quality_trend=trend,
    )
    
    # Calculate provider stats
    provider_counts = defaultdict(lambda: {"count": 0, "resonance_sum": 0.0})
    for m in assistant_messages:
        provider = m.ai_provider or "unknown"
        provider_counts[provider]["count"] += 1
        if m.resonance_score:
            provider_counts[provider]["resonance_sum"] += m.resonance_score
    
    total_assistant = len(assistant_messages)
    providers = []
    for provider, data in provider_counts.items():
        count = data["count"]
        avg_res = data["resonance_sum"] / max(count, 1)
        providers.append(ProviderStats(
            provider=provider,
            message_count=count,
            avg_resonance=round(avg_res, 4),
            percentage=round(count / max(total_assistant, 1) * 100, 2),
        ))
    providers.sort(key=lambda x: x.message_count, reverse=True)
    
    # Calculate daily activity
    daily_data = defaultdict(lambda: {"messages": 0, "conversations": set(), "resonance_sum": 0.0, "resonance_count": 0})
    
    for m in messages:
        if m.created_at:
            date_str = m.created_at.strftime("%Y-%m-%d")
            daily_data[date_str]["messages"] += 1
            daily_data[date_str]["conversations"].add(str(m.chat_id))
            if m.resonance_score and m.role == "assistant":
                daily_data[date_str]["resonance_sum"] += m.resonance_score
                daily_data[date_str]["resonance_count"] += 1
    
    daily_activity = []
    for date_str in sorted(daily_data.keys())[-30:]:  # Last 30 days max
        data = daily_data[date_str]
        avg_res = data["resonance_sum"] / max(data["resonance_count"], 1)
        daily_activity.append(DailyStats(
            date=date_str,
            messages=data["messages"],
            conversations=len(data["conversations"]),
            avg_resonance=round(avg_res, 4),
        ))
    
    # Calculate top topics
    topic_data = defaultdict(lambda: {"count": 0, "resonance_sum": 0.0})
    
    for m in user_messages:
        topics = _extract_topics(m.content)
        for topic in topics:
            topic_data[topic]["count"] += 1
    
    # Add resonance from assistant responses
    for i, m in enumerate(assistant_messages):
        if i < len(user_messages):
            topics = _extract_topics(user_messages[i].content)
            for topic in topics:
                if m.resonance_score:
                    topic_data[topic]["resonance_sum"] += m.resonance_score
    
    top_topics = []
    for topic, data in sorted(topic_data.items(), key=lambda x: x[1]["count"], reverse=True)[:10]:
        avg_res = data["resonance_sum"] / max(data["count"], 1)
        top_topics.append(TopicStats(
            topic=topic,
            count=data["count"],
            avg_resonance=round(avg_res, 4),
        ))
    
    return AnalyticsDashboard(
        time_range=time_range,
        usage=usage,
        quality=quality,
        providers=providers,
        daily_activity=daily_activity,
        top_topics=top_topics,
    )


@router.get("")
async def get_analytics_root(
    request: Request,
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y|all)$"),
    session: AsyncSession = Depends(get_session),
) -> AnalyticsDashboard:
    return await _get_analytics_impl(request=request, time_range=time_range, session=session)


@router.get("/analytics")
async def get_analytics(
    request: Request,
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y|all)$"),
    session: AsyncSession = Depends(get_session),
) -> AnalyticsDashboard:
    return await _get_analytics_impl(request=request, time_range=time_range, session=session)


@router.get("/analytics/usage")
async def get_usage_analytics(
    request: Request,
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y|all)$"),
    session: AsyncSession = Depends(get_session),
):
    """Get detailed usage analytics."""
    dashboard = await get_analytics(request, time_range, session)
    return {
        "time_range": time_range,
        "usage": dashboard.usage,
        "daily_activity": dashboard.daily_activity,
    }


@router.get("/analytics/quality")
async def get_quality_analytics(
    request: Request,
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y|all)$"),
    session: AsyncSession = Depends(get_session),
):
    """Get detailed quality analytics."""
    dashboard = await get_analytics(request, time_range, session)
    return {
        "time_range": time_range,
        "quality": dashboard.quality,
        "providers": dashboard.providers,
    }


@router.get("/analytics/topics")
async def get_topic_analytics(
    request: Request,
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y|all)$"),
    session: AsyncSession = Depends(get_session),
):
    """Get topic distribution analytics."""
    dashboard = await get_analytics(request, time_range, session)
    return {
        "time_range": time_range,
        "top_topics": dashboard.top_topics,
    }


@router.get("/analytics/memory")
async def get_memory_analytics(
    request: Request,
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y|all)$"),
    session: AsyncSession = Depends(get_session),
):
    """Get memory usage analytics."""
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    start_date = TimeRange.get_start_date(time_range)
    
    # Get messages with hash data
    result = await session.execute(
        select(ResonantChatMessage)
        .join(ResonantChat, ResonantChatMessage.chat_id == ResonantChat.id)
        .where(
            and_(
                ResonantChat.user_id == UUID(user_id),
                ResonantChatMessage.created_at >= start_date
            )
        )
    )
    messages = result.scalars().all()
    
    # Calculate memory metrics
    messages_with_hash = sum(1 for m in messages if m.hash)
    messages_with_xyz = sum(1 for m in messages if m.xyz_x is not None)
    
    # Calculate hash sphere distribution
    xyz_points = []
    for m in messages:
        if m.xyz_x is not None:
            xyz_points.append({
                "x": m.xyz_x,
                "y": m.xyz_y,
                "z": m.xyz_z,
                "resonance": m.resonance_score or 0.5,
            })
    
    # Calculate centroid
    if xyz_points:
        centroid = {
            "x": sum(p["x"] for p in xyz_points) / len(xyz_points),
            "y": sum(p["y"] for p in xyz_points) / len(xyz_points),
            "z": sum(p["z"] for p in xyz_points) / len(xyz_points),
        }
    else:
        centroid = {"x": 0.5, "y": 0.5, "z": 0.5}
    
    return {
        "time_range": time_range,
        "total_messages": len(messages),
        "messages_with_hash": messages_with_hash,
        "messages_with_xyz": messages_with_xyz,
        "hash_coverage": round(messages_with_hash / max(len(messages), 1) * 100, 2),
        "xyz_coverage": round(messages_with_xyz / max(len(messages), 1) * 100, 2),
        "sphere_centroid": centroid,
        "sample_points": xyz_points[:100],  # Limit for visualization
    }


@router.get("/admin/stats")
async def get_admin_stats(
    session: AsyncSession = Depends(get_session),
):
    """Get platform-wide chat statistics (admin only).
    
    Returns global metrics across ALL users - for owner dashboard.
    No user filtering applied.
    """
    # Get total conversations count
    conv_result = await session.execute(
        select(func.count(ResonantChat.id))
    )
    total_conversations = conv_result.scalar() or 0
    
    # Get total messages count
    msg_result = await session.execute(
        select(func.count(ResonantChatMessage.id))
    )
    total_messages = msg_result.scalar() or 0
    
    # Get total tokens used (sum of all token counts)
    tokens_result = await session.execute(
        select(func.sum(ResonantChatMessage.tokens))
    )
    total_tokens = tokens_result.scalar() or 0
    
    # Get unique users count
    users_result = await session.execute(
        select(func.count(func.distinct(ResonantChat.user_id)))
    )
    unique_users = users_result.scalar() or 0
    
    # Get messages in last 24 hours
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_result = await session.execute(
        select(func.count(ResonantChatMessage.id))
        .where(ResonantChatMessage.created_at >= yesterday)
    )
    messages_24h = recent_result.scalar() or 0
    
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "total_tokens": total_tokens,
        "unique_users": unique_users,
        "messages_24h": messages_24h,
        "total_api_calls": total_messages,  # Each message is an API call
    }
