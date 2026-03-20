"""
Dual-Layer Long-Term Memory (DLLM)
===================================

Patch #44: Introduces two layers of long-term memory:
- Layer 1: Episodic Memory (short-term, with decay)
- Layer 2: Semantic Memory (long-term crystallized facts)

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/dual_memory_engine.py
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DualMemoryEngine:
    """
    Dual-Layer Long-Term Memory Engine
    
    Combines episodic (decaying) and semantic (stable) memory layers
    to create a brain-like memory system.
    """
    
    def __init__(self, episodic_decay_days: int = 30):
        self.EPISODIC_DECAY_DAYS = episodic_decay_days
    
    def compute_decay(self, timestamp: datetime) -> float:
        """Reduce importance of episodic memory as it gets older."""
        if not timestamp:
            return 0.5
        
        try:
            days_old = (datetime.utcnow() - timestamp).days
            decay = max(0.1, 1.0 - (days_old / self.EPISODIC_DECAY_DAYS))
            return decay
        except Exception as e:
            logger.warning(f"Error computing decay: {e}")
            return 0.5
    
    def build_dual_memory(
        self,
        episodic_msgs: List[Any],
        semantic_anchors: List[Any]
    ) -> List[Dict[str, Any]]:
        """Combine episodic chat messages + long-term anchors."""
        episodic_context = []
        
        for msg in episodic_msgs:
            timestamp = None
            if hasattr(msg, 'created_at'):
                timestamp = msg.created_at
            elif hasattr(msg, 'timestamp'):
                timestamp = msg.timestamp
            elif isinstance(msg, dict):
                timestamp_str = msg.get('timestamp') or msg.get('created_at')
                if timestamp_str:
                    try:
                        if isinstance(timestamp_str, str):
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            timestamp = timestamp_str
                    except Exception:
                        timestamp = None
            
            content = None
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content') or msg.get('text')
            
            if content:
                decay = self.compute_decay(timestamp) if timestamp else 0.5
                episodic_context.append({
                    "content": content,
                    "weight": decay,
                    "type": "episodic",
                    "timestamp": timestamp.isoformat() if timestamp else None
                })
        
        semantic_context = []
        for anchor in semantic_anchors:
            anchor_text = None
            if hasattr(anchor, 'anchor_text'):
                anchor_text = anchor.anchor_text
            elif hasattr(anchor, 'context'):
                anchor_text = anchor.context
            elif isinstance(anchor, dict):
                anchor_text = anchor.get('anchor_text') or anchor.get('content') or anchor.get('context')
            
            importance_score = 0.5
            if hasattr(anchor, 'importance_score'):
                importance_score = anchor.importance_score or 0.5
            elif isinstance(anchor, dict):
                importance_score = anchor.get('importance_score') or anchor.get('score') or 0.5
            
            if anchor_text:
                semantic_context.append({
                    "content": anchor_text,
                    "weight": importance_score * 1.2,
                    "type": "semantic",
                    "importance_score": importance_score
                })
        
        combined = episodic_context + semantic_context
        combined = sorted(combined, key=lambda x: x["weight"], reverse=True)
        
        return combined[:10]
    
    def get_memory_summary(self, memories: List[Dict[str, Any]]) -> str:
        """Generate a human-readable summary of dual-layer memories."""
        if not memories:
            return "No memories available."
        
        parts = []
        for i, mem in enumerate(memories, 1):
            mem_type = mem.get("type", "unknown")
            weight = mem.get("weight", 0.0)
            content = mem.get("content", "")[:200]
            parts.append(f"{i}. [{mem_type}] (w={weight:.2f}): {content}")
        
        return "\n".join(parts)
    
    def get_system_prompt(self, episodic_msgs: List[Any], semantic_anchors: List[Any]) -> str:
        """Generate system prompt with dual-layer memory context."""
        memories = self.build_dual_memory(episodic_msgs, semantic_anchors)
        if memories:
            summary = self.get_memory_summary(memories[:5])
            return f"DUAL-LAYER MEMORY (episodic + semantic):\n{summary}"
        return ""


# Global instance
dual_memory_engine = DualMemoryEngine()
