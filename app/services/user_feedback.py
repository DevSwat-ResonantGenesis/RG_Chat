"""
User Feedback Loop System (UFLS)
=================================

Phase 5.5: Collect and process user feedback (thumbs up/down) to improve agents.

Features:
- Collect thumbs up/down feedback
- Store feedback with context (PERSISTED TO DATABASE)
- Adjust agent quality scores
- Train agent preferences
- Connect to agent_router for biased agent selection
"""
from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

# Avoid circular imports
if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    id: str
    message_id: str
    user_id: str
    agent_type: str
    feedback_type: str  # 'positive' or 'negative'
    task: str
    response: str
    timestamp: str
    comment: Optional[str] = None


@dataclass
class AgentFeedbackStats:
    """Aggregated feedback stats for an agent."""
    agent_type: str
    positive_count: int
    negative_count: int
    total_count: int
    satisfaction_rate: float  # 0-1
    recent_trend: str  # 'improving', 'stable', 'declining'


class UserFeedbackEngine:
    """
    Manages user feedback collection and processing.
    
    Now with:
    - Database persistence (survives restarts)
    - Connection to agent_router for biased selection
    - Automatic score propagation
    """
    
    def __init__(self, max_entries: int = 10000):
        self.feedback: Dict[str, List[FeedbackEntry]] = defaultdict(list)  # agent_type -> entries (in-memory cache)
        self.max_entries = max_entries
        self.agent_scores: Dict[str, float] = {}  # agent_type -> quality score
        self._db_initialized = False
        self._db_stats = {}  # DB-backed fallback
        self._agent_router = None  # Will be set lazily to avoid circular imports
    
    def _get_agent_router(self):
        """Lazily get agent_router to avoid circular imports."""
        if self._agent_router is None:
            try:
                from .agent_router import agent_router
                self._agent_router = agent_router
            except ImportError:
                logger.warning("Could not import agent_router - feedback won't affect routing")
        return self._agent_router
    
    def _sync_to_agent_router(self, agent_type: str, score: float):
        """Sync agent score to agent_router for biased selection."""
        router = self._get_agent_router()
        if router:
            router.update_performance(agent_type, score)
            logger.info(f"📊 Synced {agent_type} score {score:.2f} to agent_router")
    
    def submit_feedback(
        self,
        message_id: str,
        user_id: str,
        agent_type: str,
        is_positive: bool,
        task: str = "",
        response: str = "",
        comment: Optional[str] = None,
    ) -> FeedbackEntry:
        """Submit feedback for an agent response."""
        import hashlib
        
        feedback_id = hashlib.sha256(
            f"{message_id}:{user_id}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        entry = FeedbackEntry(
            id=feedback_id,
            message_id=message_id,
            user_id=user_id,
            agent_type=agent_type,
            feedback_type="positive" if is_positive else "negative",
            task=task[:500],
            response=response[:1000],
            timestamp=datetime.now().isoformat(),
            comment=comment[:500] if comment else None,
        )
        
        self.feedback[agent_type].append(entry)
        
        # Prune old entries
        if len(self.feedback[agent_type]) > self.max_entries:
            self.feedback[agent_type] = self.feedback[agent_type][-self.max_entries:]
        
        # Update agent score
        self._update_agent_score(agent_type)
        
        logger.info(f"{'👍' if is_positive else '👎'} Feedback for {agent_type}: {entry.id}")
        return entry
    
    async def submit_feedback_async(
        self,
        session: "AsyncSession",
        message_id: str,
        user_id: str,
        agent_type: str,
        is_positive: bool,
        task: str = "",
        response: str = "",
        comment: Optional[str] = None,
    ) -> FeedbackEntry:
        """Submit feedback with database persistence."""
        # First, do the in-memory update
        entry = self.submit_feedback(
            message_id=message_id,
            user_id=user_id,
            agent_type=agent_type,
            is_positive=is_positive,
            task=task,
            response=response,
            comment=comment,
        )
        
        # Then persist to database
        try:
            from ..models import AgentFeedback, AgentPerformanceScore
            from sqlalchemy import select
            
            # Create feedback record
            db_feedback = AgentFeedback(
                id=uuid.uuid4(),
                message_id=uuid.UUID(message_id) if len(message_id) == 36 else uuid.uuid4(),
                user_id=uuid.UUID(user_id) if len(user_id) == 36 else uuid.uuid4(),
                agent_type=agent_type,
                feedback_type="positive" if is_positive else "negative",
                task_preview=task[:500] if task else None,
                response_preview=response[:1000] if response else None,
                comment=comment[:500] if comment else None,
            )
            session.add(db_feedback)
            
            # Update or create performance score record
            result = await session.execute(
                select(AgentPerformanceScore).where(AgentPerformanceScore.agent_type == agent_type)
            )
            perf_score = result.scalar_one_or_none()
            
            if perf_score:
                # Update existing
                if is_positive:
                    perf_score.positive_count += 1
                else:
                    perf_score.negative_count += 1
                perf_score.total_count += 1
                perf_score.satisfaction_rate = perf_score.positive_count / perf_score.total_count if perf_score.total_count > 0 else 0.5
                # Bayesian smoothed score
                prior_positive = 5
                prior_total = 10
                perf_score.quality_score = (perf_score.positive_count + prior_positive) / (perf_score.total_count + prior_total)
            else:
                # Create new
                perf_score = AgentPerformanceScore(
                    id=uuid.uuid4(),
                    agent_type=agent_type,
                    positive_count=1 if is_positive else 0,
                    negative_count=0 if is_positive else 1,
                    total_count=1,
                    satisfaction_rate=1.0 if is_positive else 0.0,
                    quality_score=(1 + 5) / (1 + 10) if is_positive else (0 + 5) / (1 + 10),  # Bayesian
                )
                session.add(perf_score)
            
            await session.commit()
            
            # Sync to agent_router for biased selection
            self._sync_to_agent_router(agent_type, perf_score.quality_score)
            
            logger.info(f"💾 Persisted feedback to database for {agent_type}")
            
        except Exception as e:
            logger.error(f"Failed to persist feedback to database: {e}")
            # Don't fail the request - in-memory feedback still works
        
        return entry
    
    async def load_from_database(self, session: "AsyncSession"):
        """Load feedback scores from database on startup."""
        try:
            from ..models import AgentPerformanceScore
            from sqlalchemy import select
            
            result = await session.execute(select(AgentPerformanceScore))
            scores = result.scalars().all()
            
            for score in scores:
                self.agent_scores[score.agent_type] = score.quality_score
                self._db_stats[score.agent_type] = AgentFeedbackStats(
                    agent_type=score.agent_type,
                    positive_count=int(score.positive_count or 0),
                    negative_count=int(score.negative_count or 0),
                    total_count=int(score.total_count or 0),
                    satisfaction_rate=float(score.satisfaction_rate or 0),
                    recent_trend="stable",
                )
                # Also sync to agent_router
                self._sync_to_agent_router(score.agent_type, score.quality_score)
            
            self._db_initialized = True
            logger.info(f"📥 Loaded {len(scores)} agent performance scores from database")
            
        except Exception as e:
            logger.error(f"Failed to load feedback from database: {e}")
    
    def _update_agent_score(self, agent_type: str):
        """Update quality score for an agent based on feedback."""
        entries = self.feedback.get(agent_type, [])
        if not entries:
            self.agent_scores[agent_type] = 0.5  # Default
            return
        
        positive = sum(1 for e in entries if e.feedback_type == "positive")
        total = len(entries)
        
        # Calculate satisfaction rate with smoothing
        # Use Bayesian average to handle low sample sizes
        prior_positive = 5  # Assume 5 positive
        prior_total = 10  # Out of 10 total
        
        smoothed_rate = (positive + prior_positive) / (total + prior_total)
        self.agent_scores[agent_type] = smoothed_rate
        
        # Sync to agent_router for biased selection
        self._sync_to_agent_router(agent_type, smoothed_rate)
    
    def get_agent_stats(self, agent_type: str) -> Optional[AgentFeedbackStats]:
        """Get feedback statistics for an agent."""
        entries = self.feedback.get(agent_type, [])
        if not entries:
            return self._db_stats.get(agent_type)
        
        positive = sum(1 for e in entries if e.feedback_type == "positive")
        negative = len(entries) - positive
        
        # Calculate recent trend (last 20 vs previous 20)
        if len(entries) >= 40:
            recent = entries[-20:]
            previous = entries[-40:-20]
            recent_rate = sum(1 for e in recent if e.feedback_type == "positive") / 20
            previous_rate = sum(1 for e in previous if e.feedback_type == "positive") / 20
            
            if recent_rate > previous_rate + 0.1:
                trend = "improving"
            elif recent_rate < previous_rate - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return AgentFeedbackStats(
            agent_type=agent_type,
            positive_count=positive,
            negative_count=negative,
            total_count=len(entries),
            satisfaction_rate=positive / len(entries) if entries else 0,
            recent_trend=trend,
        )
    
    def get_all_stats(self) -> Dict[str, AgentFeedbackStats]:
        """Get feedback statistics for all agents."""
        return {
            agent_type: self.get_agent_stats(agent_type)
            for agent_type in self.feedback.keys()
            if self.get_agent_stats(agent_type) is not None
        }
    
    def get_agent_quality_score(self, agent_type: str) -> float:
        """Get quality score for an agent (0-1)."""
        return self.agent_scores.get(agent_type, 0.5)
    
    def get_best_agents(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing agents by satisfaction rate."""
        all_stats = self.get_all_stats()
        
        sorted_agents = sorted(
            all_stats.items(),
            key=lambda x: x[1].satisfaction_rate,
            reverse=True
        )
        
        return [
            {
                "agent_type": agent_type,
                "satisfaction_rate": stats.satisfaction_rate,
                "total_feedback": stats.total_count,
                "trend": stats.recent_trend,
            }
            for agent_type, stats in sorted_agents[:limit]
        ]
    
    def get_agents_needing_improvement(self, threshold: float = 0.5) -> List[str]:
        """Get agents with satisfaction rate below threshold."""
        return [
            agent_type
            for agent_type, score in self.agent_scores.items()
            if score < threshold
        ]
    
    def get_user_feedback_history(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get feedback history for a specific user."""
        all_entries = []
        for entries in self.feedback.values():
            for entry in entries:
                if entry.user_id == user_id:
                    all_entries.append({
                        "id": entry.id,
                        "agent_type": entry.agent_type,
                        "feedback_type": entry.feedback_type,
                        "timestamp": entry.timestamp,
                        "comment": entry.comment,
                    })
        
        # Sort by timestamp descending
        all_entries.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_entries[:limit]
    
    def export_feedback(self, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export feedback data for analysis."""
        if agent_type:
            entries = self.feedback.get(agent_type, [])
        else:
            entries = []
            for agent_entries in self.feedback.values():
                entries.extend(agent_entries)
        
        return [
            {
                "id": e.id,
                "agent_type": e.agent_type,
                "feedback_type": e.feedback_type,
                "timestamp": e.timestamp,
                "task_length": len(e.task),
                "response_length": len(e.response),
                "has_comment": bool(e.comment),
            }
            for e in entries
        ]


# Global instance
user_feedback = UserFeedbackEngine()
