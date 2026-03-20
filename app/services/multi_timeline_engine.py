"""
Multi-Timeline Reasoning Engine (MTRE)
========================================

Patch #36: Advanced cognitive patch that enables multiverse-style reasoning.
Transforms Resonant Chat from "smart conversational LLM" into a predictive,
temporal, self-aware simulation system.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/multi_timeline_engine.py
"""
from __future__ import annotations

import uuid
import hashlib
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MultiTimelineEngine:
    """
    Multi-Timeline Reasoning Engine
    
    Simulates multiple possible futures in parallel, branches timelines based on
    different user choices, and scores each timeline for stability/risk/outcome.
    """
    
    def __init__(
        self,
        causal_graph: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        resonance_fn: Optional[Callable[[str, str], float]] = None
    ):
        self.causal_graph = causal_graph or {}
        self.resonance_fn = resonance_fn or self._default_resonance
        self.max_depth = 3
        self.max_branches_per_node = 5
        
        logger.debug("MultiTimelineEngine initialized")
    
    def _default_resonance(self, hash1: str, hash2: str) -> float:
        """Default resonance calculation (simple similarity)"""
        if not hash1 or not hash2:
            return 0.5
        
        if hash1 == hash2:
            return 1.0
        
        matches = sum(1 for a, b in zip(hash1[:16], hash2[:16]) if a == b)
        return matches / 16.0
    
    def make_node(
        self,
        cause: Dict[str, Any],
        effect: Dict[str, Any],
        depth: int
    ) -> Dict[str, Any]:
        """Create a timeline node representing a future branch."""
        cause_hash = cause.get("hash", "")
        effect_hash = effect.get("hash", "")
        
        node = {
            "id": str(uuid.uuid4()),
            "cause": cause,
            "effect": effect,
            "depth": depth,
            "probability": effect.get("strength", 0.5),
            "resonance": self.resonance_fn(cause_hash, effect_hash),
            "created_at": datetime.utcnow().isoformat(),
            "anchor_text": effect.get("text", effect.get("content", "")),
            "projection_hash": self._generate_projection_hash(cause_hash, effect_hash),
        }
        
        return node
    
    def _generate_projection_hash(self, cause_hash: str, effect_hash: str) -> str:
        """Generate a unique hash for this timeline projection"""
        combined = f"{cause_hash}:{effect_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def build_timelines(
        self,
        root_message: Dict[str, Any],
        depth: int = 3,
        max_branches: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate divergent timelines from a root message."""
        timelines = []
        queue: List[Tuple[Dict[str, Any], int]] = [(root_message, 0)]
        visited = set()
        
        while queue:
            current, d = queue.pop(0)
            
            if d >= depth:
                continue
            
            current_id = current.get("id") or current.get("hash", "")
            if current_id in visited:
                continue
            visited.add(current_id)
            
            cause_key = current.get("cause") or current.get("hash", "")
            future_paths = self.causal_graph.get(cause_key, [])
            future_paths = future_paths[:max_branches]
            
            for future in future_paths:
                node = self.make_node(current, future, d + 1)
                timelines.append(node)
                
                if d + 1 < depth:
                    queue.append((node, d + 1))
        
        logger.debug(f"Built {len(timelines)} timeline nodes from depth {depth}")
        return timelines
    
    def score_timeline(self, node: Dict[str, Any]) -> float:
        """Score a timeline node for quality."""
        probability = node.get("probability", 0.5)
        resonance = node.get("resonance", 0.5)
        depth = node.get("depth", 0)
        
        depth_score = 1.0 - (depth * 0.1)
        depth_score = max(0.0, min(1.0, depth_score))
        
        score = (
            probability * 0.4 +
            resonance * 0.4 +
            depth_score * 0.2
        )
        
        return round(score, 3)
    
    def calculate_stability_score(self, node: Dict[str, Any]) -> float:
        """Calculate stability score for a timeline."""
        probability = node.get("probability", 0.5)
        resonance = node.get("resonance", 0.5)
        risk = 1.0 - probability
        stability = (probability * 0.5) + (resonance * 0.3) + ((1.0 - risk) * 0.2)
        return round(stability, 3)
    
    def calculate_risk_score(self, node: Dict[str, Any]) -> float:
        """Calculate risk score for a timeline."""
        probability = node.get("probability", 0.5)
        resonance = node.get("resonance", 0.5)
        depth = node.get("depth", 0)
        
        low_probability_risk = 1.0 - probability
        low_resonance_risk = 1.0 - resonance
        depth_risk = min(1.0, depth * 0.2)
        
        risk = (low_probability_risk * 0.4) + (low_resonance_risk * 0.3) + (depth_risk * 0.3)
        return round(risk, 3)
    
    def simulate(
        self,
        root_message: Dict[str, Any],
        depth: int = 3,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Simulate multiple timelines from a root message."""
        timelines = self.build_timelines(root_message, depth=depth)
        
        for timeline in timelines:
            timeline["score"] = self.score_timeline(timeline)
            timeline["stability"] = self.calculate_stability_score(timeline)
            timeline["risk"] = self.calculate_risk_score(timeline)
        
        sorted_timelines = sorted(timelines, key=lambda x: x["score"], reverse=True)
        top_timelines = sorted_timelines[:top_k]
        
        stable_timelines = [t for t in sorted_timelines if t["stability"] >= 0.7]
        risky_timelines = [t for t in sorted_timelines if t["risk"] >= 0.7]
        
        result = {
            "timelines": timelines,
            "top_timelines": top_timelines,
            "stable_timelines": stable_timelines[:top_k],
            "risky_timelines": risky_timelines[:top_k],
            "total_timelines": len(timelines),
            "simulation_depth": depth,
        }
        
        logger.info(f"Simulated {len(timelines)} timelines, top {len(top_timelines)} selected")
        return result
    
    def get_timeline_context(self, simulation_result: Dict[str, Any]) -> str:
        """Generate context string for LLM from simulation results."""
        if not simulation_result or not simulation_result.get("top_timelines"):
            return ""
        
        top_timelines = simulation_result["top_timelines"]
        
        context_parts = ["MULTI-TIMELINE FORECAST:"]
        context_parts.append(f"Based on causal analysis, here are the top {len(top_timelines)} possible futures:\n")
        
        for i, timeline in enumerate(top_timelines, 1):
            effect_text = timeline.get("effect", {}).get("text", timeline.get("effect", {}).get("content", "Unknown outcome"))
            probability = timeline.get("probability", 0.5)
            stability = timeline.get("stability", 0.5)
            risk = timeline.get("risk", 0.5)
            
            context_parts.append(f"Timeline {i}:")
            context_parts.append(f"  Outcome: {effect_text}")
            context_parts.append(f"  Probability: {probability:.2f}")
            context_parts.append(f"  Stability: {stability:.2f}")
            context_parts.append(f"  Risk: {risk:.2f}\n")
        
        context_parts.append("Consider these timelines when generating your answer.")
        context_parts.append("If certain actions lead to high-risk timelines, warn the user.")
        context_parts.append("Recommend the most stable timeline when appropriate.")
        
        return "\n".join(context_parts)


# Global instance
multi_timeline_engine = MultiTimelineEngine()
