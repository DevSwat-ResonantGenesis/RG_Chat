"""
Self-Evolving Knowledge Graph (SEKG)
=====================================

Patch #53: Turns all memories + anchors + chat history into a continuously evolving knowledge graph.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/self_evolving_knowledge_graph.py
"""
from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class SelfEvolvingKnowledgeGraph:
    """
    Self-Evolving Knowledge Graph
    
    Builds and maintains a knowledge graph from memories, anchors, and chat history.
    Nodes represent concepts, edges represent relationships.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.node_text_to_id: Dict[str, str] = {}
    
    def _create_node(self, text: str, node_type: str = "concept", weight: float = 1.0) -> str:
        """Create a new node in the graph."""
        text_normalized = text.lower().strip()[:200]
        if text_normalized in self.node_text_to_id:
            existing_id = self.node_text_to_id[text_normalized]
            if existing_id in self.nodes:
                self.nodes[existing_id]["weight"] = max(self.nodes[existing_id]["weight"], weight)
            return existing_id
        
        node_id = str(uuid.uuid4())
        self.nodes[node_id] = {
            "content": text[:500],
            "type": node_type,
            "timestamp": datetime.utcnow().isoformat(),
            "weight": weight
        }
        self.node_text_to_id[text_normalized] = node_id
        return node_id
    
    def _link(self, n1: str, n2: str, weight: float = 1.0, bidirectional: bool = True):
        """Create an edge between two nodes."""
        if n1 not in self.nodes or n2 not in self.nodes:
            return
        
        existing_weights = [w for node_id, w in self.edges[n1] if node_id == n2]
        if existing_weights:
            max_weight = max(existing_weights[0], weight)
            self.edges[n1] = [(node_id, w) for node_id, w in self.edges[n1] if node_id != n2]
            self.edges[n1].append((n2, max_weight))
        else:
            self.edges[n1].append((n2, weight))
        
        if bidirectional:
            if n1 not in self.edges[n2]:
                self.edges[n2].append((n1, weight))
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple lexical similarity between two texts."""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            if not union:
                return 0.0
            
            return len(intersection) / len(union)
        except Exception:
            return 0.0
    
    def update_graph(
        self,
        memories: List[Dict[str, Any]],
        anchors: List[Any],
        min_similarity: float = 0.2
    ) -> List[str]:
        """Build or update graph from new memories & anchors."""
        try:
            node_ids = []
            
            for mem in memories:
                content = mem.get("content") or mem.get("anchor_text") or mem.get("text")
                if not content:
                    continue
                
                weight = mem.get("combined_score") or mem.get("resonance_score") or mem.get("weight") or 0.5
                node_id = self._create_node(content, node_type="memory", weight=float(weight))
                node_ids.append(node_id)
            
            for anchor in anchors:
                anchor_text = None
                if hasattr(anchor, 'anchor_text'):
                    anchor_text = anchor.anchor_text
                elif hasattr(anchor, 'context'):
                    anchor_text = anchor.context
                elif isinstance(anchor, dict):
                    anchor_text = anchor.get("anchor_text") or anchor.get("content") or anchor.get("context")
                
                if anchor_text:
                    importance = 0.7
                    if hasattr(anchor, 'importance_score'):
                        importance = anchor.importance_score or 0.7
                    elif isinstance(anchor, dict):
                        importance = anchor.get("importance_score") or anchor.get("score") or 0.7
                    
                    node_id = self._create_node(anchor_text, node_type="anchor", weight=float(importance))
                    node_ids.append(node_id)
            
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    n1 = node_ids[i]
                    n2 = node_ids[j]
                    
                    if n1 not in self.nodes or n2 not in self.nodes:
                        continue
                    
                    text1 = self.nodes[n1]["content"]
                    text2 = self.nodes[n2]["content"]
                    
                    similarity = self._calculate_similarity(text1, text2)
                    
                    if similarity >= min_similarity:
                        link_weight = similarity * 0.5
                        self._link(n1, n2, weight=link_weight)
            
            return node_ids
            
        except Exception as e:
            logger.error(f"Error updating knowledge graph: {e}", exc_info=True)
            return []
    
    def get_connected_nodes(self, node_id: str, max_depth: int = 2) -> List[str]:
        """Get all nodes connected to a given node (BFS traversal)."""
        if node_id not in self.nodes:
            return []
        
        visited = set()
        queue = [(node_id, 0)]
        connected = []
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            connected.append(current_id)
            
            for neighbor_id, weight in self.edges.get(current_id, []):
                if neighbor_id not in visited:
                    queue.append((neighbor_id, depth + 1))
        
        return connected
    
    def get_graph_summary(self, limit: int = 10) -> str:
        """Generate a human-readable summary of the knowledge graph."""
        try:
            if not self.nodes:
                return "Knowledge graph is empty."
            
            top_nodes = sorted(
                self.nodes.items(),
                key=lambda x: x[1].get("weight", 0.0),
                reverse=True
            )[:limit]
            
            parts = []
            for node_id, node_data in top_nodes:
                content = node_data.get("content", "")[:100]
                node_type = node_data.get("type", "unknown")
                weight = node_data.get("weight", 0.0)
                edge_count = len(self.edges.get(node_id, []))
                
                parts.append(
                    f"- [{node_type}] (w={weight:.2f}, edges={edge_count}): {content}"
                )
            
            return "\n".join(parts) if parts else "No nodes in graph."
            
        except Exception as e:
            logger.warning(f"Error generating graph summary: {e}")
            return "Error generating graph summary."
    
    def get_system_prompt(self, limit: int = 5) -> str:
        """Generate system prompt with knowledge graph context."""
        summary = self.get_graph_summary(limit=limit)
        if summary == "Knowledge graph is empty." or summary.startswith("Error"):
            return ""
        
        return f"KNOWLEDGE GRAPH CONTEXT:\nYou have an internal knowledge graph linking concepts:\n{summary}\n\nUse connected nodes to form deeper reasoning and maintain consistency across related topics."


# Global instance
knowledge_graph = SelfEvolvingKnowledgeGraph()
