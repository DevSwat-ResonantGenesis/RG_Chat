"""
Adaptive Weight Tuning Service
Adjusts memory scoring weights based on user feedback.

Uses a simple gradient-like approach to personalize relevance scoring.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from threading import Lock
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Record of user feedback on a query result."""
    query: str
    top_memory_scores: Dict[str, float]  # component -> score
    feedback: bool  # True = positive, False = negative
    clicked_index: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class UserWeights:
    """Personalized weights for a user."""
    rag: float = 0.30
    resonance: float = 0.25
    proximity: float = 0.20
    recency: float = 0.15
    anchor: float = 0.10
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "UserWeights":
        return cls(
            rag=data.get("rag", 0.30),
            resonance=data.get("resonance", 0.25),
            proximity=data.get("proximity", 0.20),
            recency=data.get("recency", 0.15),
            anchor=data.get("anchor", 0.10),
        )
    
    def normalize(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = self.rag + self.resonance + self.proximity + self.recency + self.anchor
        if total > 0:
            self.rag /= total
            self.resonance /= total
            self.proximity /= total
            self.recency /= total
            self.anchor /= total


class AdaptiveWeightTuner:
    """
    Adjust scoring weights based on user feedback.
    
    Uses a simple gradient-like approach:
    - Positive feedback: increase weights for high-scoring components
    - Negative feedback: decrease weights for high-scoring components
    """
    
    DEFAULT_WEIGHTS = UserWeights()
    
    def __init__(
        self,
        learning_rate: float = 0.05,
        min_feedback_for_update: int = 5,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize the adaptive weight tuner.
        
        Args:
            learning_rate: How much to adjust weights per feedback batch
            min_feedback_for_update: Minimum feedback count before updating weights
            persistence_path: Optional path to persist weights
        """
        self.learning_rate = learning_rate
        self.min_feedback_for_update = min_feedback_for_update
        self.persistence_path = persistence_path
        
        self._lock = Lock()
        self._user_weights: Dict[str, UserWeights] = {}
        self._feedback_history: Dict[str, List[FeedbackRecord]] = {}
        
        # Stats
        self._total_feedbacks = 0
        self._weight_updates = 0
        
        # Load persisted weights
        if persistence_path:
            self._load_weights()
    
    def get_weights(self, user_id: str) -> Dict[str, float]:
        """
        Get personalized weights for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary of weight name -> value
        """
        with self._lock:
            if user_id in self._user_weights:
                return self._user_weights[user_id].to_dict()
            return self.DEFAULT_WEIGHTS.to_dict()
    
    def record_feedback(
        self,
        user_id: str,
        query: str,
        top_memory: Dict[str, Any],
        feedback: bool,
        clicked_index: int = 0
    ) -> None:
        """
        Record user feedback for weight adjustment.
        
        Args:
            user_id: User ID
            query: Search query
            top_memory: Top memory result with scores
            feedback: True = positive (thumbs up), False = negative
            clicked_index: Index of clicked result (0 = first)
        """
        # Extract component scores from memory
        scores = {
            "rag": float(top_memory.get("rag_score") or top_memory.get("similarity_score") or 0),
            "resonance": float(top_memory.get("resonance_score") or 0),
            "proximity": float(top_memory.get("proximity_score") or 0),
            "recency": float(top_memory.get("recency_score") or 0),
            "anchor": float(top_memory.get("anchor_score") or 0),
        }
        
        record = FeedbackRecord(
            query=query,
            top_memory_scores=scores,
            feedback=feedback,
            clicked_index=clicked_index,
        )
        
        with self._lock:
            if user_id not in self._feedback_history:
                self._feedback_history[user_id] = []
            
            self._feedback_history[user_id].append(record)
            self._total_feedbacks += 1
            
            # Check if we should update weights
            if len(self._feedback_history[user_id]) >= self.min_feedback_for_update:
                self._update_weights(user_id)
    
    def _update_weights(self, user_id: str) -> None:
        """Update weights based on accumulated feedback."""
        history = self._feedback_history.get(user_id, [])
        if not history:
            return
        
        # Get current weights
        if user_id in self._user_weights:
            weights = self._user_weights[user_id]
        else:
            weights = UserWeights()
        
        # Separate positive and negative feedback
        positive = [h for h in history if h.feedback]
        negative = [h for h in history if not h.feedback]
        
        if not positive and not negative:
            return
        
        # Calculate average scores for each component
        components = ["rag", "resonance", "proximity", "recency", "anchor"]
        
        for component in components:
            pos_avg = 0.0
            neg_avg = 0.0
            
            if positive:
                pos_avg = sum(h.top_memory_scores.get(component, 0) for h in positive) / len(positive)
            if negative:
                neg_avg = sum(h.top_memory_scores.get(component, 0) for h in negative) / len(negative)
            
            # Adjust weight: increase if component is higher in positive feedback
            adjustment = self.learning_rate * (pos_avg - neg_avg)
            
            current_weight = getattr(weights, component)
            new_weight = max(0.05, min(0.50, current_weight + adjustment))  # Clamp to [0.05, 0.50]
            setattr(weights, component, new_weight)
        
        # Normalize weights to sum to 1
        weights.normalize()
        
        # Store updated weights
        self._user_weights[user_id] = weights
        self._weight_updates += 1
        
        # Keep only recent feedback
        self._feedback_history[user_id] = history[-5:]
        
        logger.info(f"[AdaptiveWeights] Updated weights for user {user_id[:8]}...: {weights.to_dict()}")
        
        # Persist if configured
        if self.persistence_path:
            self._save_weights()
    
    def reset_user_weights(self, user_id: str) -> None:
        """Reset a user's weights to default."""
        with self._lock:
            if user_id in self._user_weights:
                del self._user_weights[user_id]
            if user_id in self._feedback_history:
                del self._feedback_history[user_id]
        
        logger.info(f"[AdaptiveWeights] Reset weights for user {user_id[:8]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tuner statistics."""
        with self._lock:
            return {
                "total_feedbacks": self._total_feedbacks,
                "weight_updates": self._weight_updates,
                "users_with_custom_weights": len(self._user_weights),
                "pending_feedback_users": len(self._feedback_history),
                "learning_rate": self.learning_rate,
                "min_feedback_for_update": self.min_feedback_for_update,
            }
    
    def _save_weights(self) -> None:
        """Persist weights to file."""
        if not self.persistence_path:
            return
        
        try:
            path = Path(self.persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                user_id: weights.to_dict()
                for user_id, weights in self._user_weights.items()
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"[AdaptiveWeights] Saved weights to {path}")
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")
    
    def _load_weights(self) -> None:
        """Load weights from file."""
        if not self.persistence_path:
            return
        
        try:
            path = Path(self.persistence_path)
            if not path.exists():
                return
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            for user_id, weights_dict in data.items():
                self._user_weights[user_id] = UserWeights.from_dict(weights_dict)
            
            logger.info(f"[AdaptiveWeights] Loaded weights for {len(self._user_weights)} users")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"AdaptiveWeightTuner(users={stats['users_with_custom_weights']}, updates={stats['weight_updates']})"


# Global singleton
adaptive_tuner = AdaptiveWeightTuner(
    learning_rate=0.05,
    min_feedback_for_update=5,
    persistence_path=None  # Set from config if needed
)
