"""
Resonance-Based Token Optimization (RBTO)
==========================================

Patch #56: Reduces token usage without losing meaning by compressing context.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/resonance_token_optimizer.py
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ResonanceTokenOptimizer:
    """
    Resonance-Based Token Optimizer
    
    Prunes and optimizes context based on resonance scores,
    semantic proximity, and memory importance to reduce token usage
    while maintaining quality.
    """
    
    def __init__(self, min_score_threshold: float = 0.3):
        self.min_score_threshold = min_score_threshold
    
    def prune(
        self,
        memories: List[Dict[str, Any]],
        anchors: List[Any],
        limit: int = 4,
        max_length_per_item: int = 200
    ) -> List[str]:
        """Reduce context size using resonance-based scoring."""
        try:
            scored: List[Tuple[float, str, str]] = []
            
            for mem in memories:
                content = mem.get("content") or mem.get("anchor_text") or mem.get("text")
                if not content:
                    continue
                
                resonance_score = mem.get("resonance_score", 0.5) or 0.5
                proximity_score = mem.get("proximity_score", 0.3) or 0.3
                recency_score = mem.get("recency_score", 0.1) or 0.1
                combined_score = mem.get("combined_score", 0.0) or 0.0
                
                if combined_score > 0:
                    score = combined_score
                else:
                    score = (
                        resonance_score * 0.6 +
                        proximity_score * 0.3 +
                        recency_score * 0.1
                    )
                
                if score >= self.min_score_threshold:
                    scored.append((score, content[:max_length_per_item], "memory"))
            
            for anchor in anchors:
                anchor_text = None
                if hasattr(anchor, 'anchor_text'):
                    anchor_text = anchor.anchor_text
                elif hasattr(anchor, 'context'):
                    anchor_text = anchor.context
                elif isinstance(anchor, dict):
                    anchor_text = anchor.get("anchor_text") or anchor.get("content") or anchor.get("context")
                
                if not anchor_text:
                    continue
                
                importance_score = 0.7
                if hasattr(anchor, 'importance_score'):
                    importance_score = anchor.importance_score or 0.7
                elif isinstance(anchor, dict):
                    importance_score = anchor.get("importance_score") or anchor.get("score") or 0.7
                
                score = importance_score * 1.2
                
                if score >= self.min_score_threshold:
                    scored.append((score, anchor_text[:max_length_per_item], "anchor"))
            
            scored.sort(reverse=True, key=lambda x: x[0])
            top = [text for _, text, _ in scored[:limit]]
            
            logger.info(f"🔧 Token optimization: {len(memories)} memories + {len(anchors)} anchors → {len(top)} optimized items")
            
            return top
            
        except Exception as e:
            logger.error(f"Error in token optimization: {e}", exc_info=True)
            fallback = []
            for mem in memories[:limit]:
                content = mem.get("content") or mem.get("anchor_text") or ""
                if content:
                    fallback.append(content[:200])
            return fallback
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        return len(text) // 4
    
    def optimize_context(
        self,
        context_messages: List[Dict[str, Any]],
        max_tokens: int = 2000
    ) -> List[Dict[str, Any]]:
        """Optimize entire context message list to fit within token limit."""
        try:
            optimized = []
            current_tokens = 0
            
            for msg in context_messages:
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    tokens = self.estimate_tokens(content)
                    if current_tokens + tokens <= max_tokens:
                        optimized.append(msg)
                        current_tokens += tokens
            
            for msg in reversed(context_messages):
                if msg.get("role") != "system":
                    content = msg.get("content", "")
                    tokens = self.estimate_tokens(content)
                    if current_tokens + tokens <= max_tokens:
                        optimized.insert(0, msg)
                        current_tokens += tokens
                    else:
                        max_chars = (max_tokens - current_tokens) * 4
                        if max_chars > 100:
                            truncated_msg = msg.copy()
                            truncated_msg["content"] = content[:max_chars] + "..."
                            optimized.insert(0, truncated_msg)
                        break
            
            logger.info(f"🔧 Context optimization: {len(context_messages)} messages → {len(optimized)} messages ({current_tokens} tokens)")
            
            return optimized
            
        except Exception as e:
            logger.warning(f"Context optimization failed: {e}")
            return context_messages


# Global instance
token_optimizer = ResonanceTokenOptimizer()
