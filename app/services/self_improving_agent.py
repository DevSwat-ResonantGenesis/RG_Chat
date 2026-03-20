"""
Self-Improving Agent - Agent that learns and improves from feedback.

STATUS: DENIED CAPABILITY
CREATED: 2025-12-21
GOVERNANCE: This capability is DENIED until proper governance framework exists.
            All methods return 501 NOT_IMPLEMENTED.
            Reason: Self-improvement requires governance for self-modification.
            Risk: Unbounded behavior without proper constraints.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Governance: This capability is PERMANENTLY DENIED
_IS_STUB = True
_CAPABILITY_DENIED = True
_DENIAL_REASON = "Self-improvement requires governance framework for self-modification"


class CapabilityDeniedError(Exception):
    """Raised when a denied capability is invoked."""
    def __init__(self, capability: str, reason: str):
        self.capability = capability
        self.reason = reason
        super().__init__(f"501 NOT_IMPLEMENTED: {capability} - {reason}")


class FeedbackType(Enum):
    """Types of feedback."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTION = "correction"


@dataclass
class FeedbackEntry:
    """A feedback entry."""
    feedback_type: FeedbackType
    message: str
    response: str
    correction: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPattern:
    """A learned pattern from feedback."""
    pattern_id: str
    trigger: str
    learned_response: str
    confidence: float
    usage_count: int = 0
    success_rate: float = 0.5


class SelfImprovingAgent:
    """
    An agent that learns and improves from user feedback.
    
    Tracks feedback, identifies patterns, and adjusts responses
    based on historical performance.
    """
    
    def __init__(self, agent_id: str = "self_improving"):
        self.agent_id = agent_id
        self.feedback_history: List[FeedbackEntry] = []
        self.learned_patterns: Dict[str, LearningPattern] = {}
        self.performance_score: float = 0.5
        self.total_interactions: int = 0
        self.positive_feedback_count: int = 0
        
    def record_feedback(
        self,
        feedback_type: FeedbackType,
        message: str,
        response: str,
        correction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record feedback for a response. DENIED - returns 501."""
        logger.warning(
            f"GOVERNANCE_DENIED: record_feedback called on denied capability. "
            f"Reason: {_DENIAL_REASON}"
        )
        raise CapabilityDeniedError("SelfImprovingAgent.record_feedback", _DENIAL_REASON)
    
    def _reinforce_pattern(self, message: str, response: str) -> None:
        """Reinforce a successful pattern."""
        pattern_id = self._generate_pattern_id(message)
        if pattern_id in self.learned_patterns:
            pattern = self.learned_patterns[pattern_id]
            pattern.usage_count += 1
            pattern.success_rate = min(1.0, pattern.success_rate + 0.1)
            pattern.confidence = min(1.0, pattern.confidence + 0.05)
        else:
            self.learned_patterns[pattern_id] = LearningPattern(
                pattern_id=pattern_id,
                trigger=message[:100],
                learned_response=response[:500],
                confidence=0.6,
                usage_count=1,
                success_rate=0.7
            )
    
    def _weaken_pattern(self, message: str, response: str) -> None:
        """Weaken an unsuccessful pattern."""
        pattern_id = self._generate_pattern_id(message)
        if pattern_id in self.learned_patterns:
            pattern = self.learned_patterns[pattern_id]
            pattern.success_rate = max(0.0, pattern.success_rate - 0.15)
            pattern.confidence = max(0.0, pattern.confidence - 0.1)
            
            # Remove pattern if confidence drops too low
            if pattern.confidence < 0.2:
                del self.learned_patterns[pattern_id]
    
    def _learn_correction(self, message: str, old_response: str, correction: str) -> None:
        """Learn from a correction."""
        pattern_id = self._generate_pattern_id(message)
        self.learned_patterns[pattern_id] = LearningPattern(
            pattern_id=pattern_id,
            trigger=message[:100],
            learned_response=correction[:500],
            confidence=0.8,
            usage_count=1,
            success_rate=0.8
        )
    
    def _generate_pattern_id(self, message: str) -> str:
        """Generate a pattern ID from a message."""
        # Simple hash-based ID
        import hashlib
        return hashlib.md5(message.lower().strip()[:100].encode()).hexdigest()[:16]
    
    def _update_performance_score(self) -> None:
        """Update the overall performance score."""
        if self.total_interactions > 0:
            self.performance_score = self.positive_feedback_count / self.total_interactions
    
    def get_learned_response(self, message: str) -> Optional[str]:
        """Get a learned response. DENIED - returns 501."""
        logger.warning(
            f"GOVERNANCE_DENIED: get_learned_response called on denied capability. "
            f"Reason: {_DENIAL_REASON}"
        )
        raise CapabilityDeniedError("SelfImprovingAgent.get_learned_response", _DENIAL_REASON)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "total_interactions": self.total_interactions,
            "positive_feedback_count": self.positive_feedback_count,
            "performance_score": self.performance_score,
            "learned_patterns_count": len(self.learned_patterns),
            "feedback_history_count": len(self.feedback_history)
        }
    
    def get_top_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing patterns."""
        sorted_patterns = sorted(
            self.learned_patterns.values(),
            key=lambda p: (p.success_rate, p.confidence),
            reverse=True
        )
        return [
            {
                "pattern_id": p.pattern_id,
                "trigger": p.trigger,
                "confidence": p.confidence,
                "success_rate": p.success_rate,
                "usage_count": p.usage_count
            }
            for p in sorted_patterns[:limit]
        ]


# Global instance
self_improving_agent = SelfImprovingAgent()
