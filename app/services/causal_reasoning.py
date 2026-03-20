"""
Causal Reasoning Layer (CRL) Service
=====================================

Patch #34: Adds causal graph inference, detects cause-effect relationships,
predicts consequences, and suggests interventions.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/causal_reasoning.py
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CausalLink:
    """Represents a causal relationship between two concepts."""
    cause: str
    effect: str
    strength: float  # 0.0 - 1.0
    source_node_id: Optional[str] = None


class CausalReasoner:
    """
    Causal Reasoning Layer
    
    Enables the AI to understand cause → effect relationships,
    predict consequences, and suggest interventions.
    """
    
    def __init__(self):
        self.links: List[CausalLink] = []
        self.causal_graph: Dict[str, List[Dict[str, Any]]] = {}
    
    def extract_causal_candidates(self, texts: List[str]) -> List[CausalLink]:
        """Extract causal candidates from text patterns."""
        links = []
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            node_id = str(i)
            
            # Pattern 1: "X because Y" -> Y causes X
            if "because" in text_lower:
                parts = text_lower.split("because", 1)
                if len(parts) == 2:
                    effect = parts[0].strip()
                    cause = parts[1].strip()
                    cause = re.sub(r'\s+(so|then|therefore|thus|hence).*$', '', cause)
                    if cause and effect:
                        links.append(CausalLink(
                            cause=cause[:100],
                            effect=effect[:100],
                            strength=0.7,
                            source_node_id=node_id
                        ))
            
            # Pattern 2: "X leads to Y" -> X causes Y
            if "leads to" in text_lower:
                parts = text_lower.split("leads to", 1)
                if len(parts) == 2:
                    cause = parts[0].strip()
                    effect = parts[1].strip()
                    if cause and effect:
                        links.append(CausalLink(
                            cause=cause[:100],
                            effect=effect[:100],
                            strength=0.8,
                            source_node_id=node_id
                        ))
            
            # Pattern 3: "X causes Y" -> X causes Y
            if "causes" in text_lower:
                parts = text_lower.split("causes", 1)
                if len(parts) == 2:
                    cause = parts[0].strip()
                    effect = parts[1].strip()
                    if cause and effect:
                        links.append(CausalLink(
                            cause=cause[:100],
                            effect=effect[:100],
                            strength=0.8,
                            source_node_id=node_id
                        ))
            
            # Pattern 4: "If X then Y" -> X causes Y
            if "if" in text_lower and "then" in text_lower:
                match = re.search(r'if\s+(.+?)\s+then\s+(.+)', text_lower)
                if match:
                    cause = match.group(1).strip()
                    effect = match.group(2).strip()
                    if cause and effect:
                        links.append(CausalLink(
                            cause=cause[:100],
                            effect=effect[:100],
                            strength=0.75,
                            source_node_id=node_id
                        ))
        
        return links
    
    def build_causal_graph(self, links: List[CausalLink]) -> Dict[str, List[Dict[str, Any]]]:
        """Build causal graph from links."""
        g: Dict[str, List[Dict[str, Any]]] = {}
        
        for link in links:
            if link.cause not in g:
                g[link.cause] = []
            
            g[link.cause].append({
                "effect": link.effect,
                "strength": link.strength,
                "source": link.source_node_id
            })
        
        return g
    
    def predict_effects(self, cause: str) -> List[Dict[str, Any]]:
        """Predict effects given a cause."""
        predictions = []
        cause_lower = cause.lower()
        
        for c, effects in self.causal_graph.items():
            if cause_lower in c.lower() or c.lower() in cause_lower:
                for eff in effects:
                    predictions.append(eff)
        
        cause_words = set(cause_lower.split())
        for c, effects in self.causal_graph.items():
            c_words = set(c.lower().split())
            if len(cause_words & c_words) >= 2:
                for eff in effects:
                    predictions.append(eff)
        
        seen = set()
        unique_predictions = []
        for pred in predictions:
            pred_key = (pred["effect"], pred.get("source"))
            if pred_key not in seen:
                seen.add(pred_key)
                unique_predictions.append(pred)
        
        return unique_predictions
    
    def suggest_interventions(self, cause: str) -> List[str]:
        """Suggest interventions for a cause."""
        fixes = []
        
        preds = self.predict_effects(cause)
        for p in preds:
            effect = p.get("effect", "")
            strength = p.get("strength", 0.5)
            
            if strength >= 0.6:
                fixes.append(
                    f"To prevent '{effect}', avoid or reverse: {cause}"
                )
        
        return fixes
    
    def find_causal_chain(self, start: str, end: str, max_depth: int = 3) -> Optional[List[str]]:
        """Find a chain of causality from start to end."""
        def dfs(current: str, target: str, path: List[str], depth: int) -> Optional[List[str]]:
            if depth > max_depth:
                return None
            
            if target.lower() in current.lower() or current.lower() in target.lower():
                return path + [current]
            
            if current in self.causal_graph:
                for effect_data in self.causal_graph[current]:
                    next_effect = effect_data["effect"]
                    if next_effect not in path:
                        result = dfs(next_effect, target, path + [current], depth + 1)
                        if result:
                            return result
            
            return None
        
        return dfs(start, end, [], 0)
    
    def update(self, texts: List[str]) -> Dict[str, Any]:
        """Extract causal links and build graph from texts."""
        self.links = self.extract_causal_candidates(texts)
        self.causal_graph = self.build_causal_graph(self.links)
        
        return {
            "causal_links": len(self.links),
            "graph_size": len(self.causal_graph)
        }
    
    def get_system_prompt(self, query: str) -> str:
        """Generate system prompt with causal reasoning context."""
        predictions = self.predict_effects(query)
        if predictions:
            effects = [p["effect"][:50] for p in predictions[:3]]
            return f"CAUSAL REASONING: Query may lead to: {', '.join(effects)}"
        return ""


# Global instance
causal_reasoner = CausalReasoner()
