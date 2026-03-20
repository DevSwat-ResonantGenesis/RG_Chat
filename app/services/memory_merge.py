"""
Memory Merge Service
Merges and ranks RAG + Hash Sphere memories using combined scoring.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/memory_merge.py
"""
from __future__ import annotations

from typing import List, Dict, Optional

from .adaptive_weights import adaptive_tuner
from .ab_testing import ab_tester


def merge_and_rank_memories(
    rag_memories: List[Dict],
    sphere_memories: List[Dict],
    limit: int = 5,
    user_id: Optional[str] = None,
    use_adaptive_weights: bool = True,
    experiment: Optional[str] = None
) -> List[Dict]:
    """
    Merge RAG results + Hash Sphere results.
    Then re-rank by combined score.

    Each memory should contain:
    - 'content'
    - 'resonance_score' (sphere)
    - 'proximity_score' (sphere)
    - 'anchor_score' (sphere)
    - 'recency_score' (sphere)
    
    Args:
        rag_memories: List of RAG memory dicts
        sphere_memories: List of Hash Sphere memory dicts
        limit: Maximum number of memories to return
    
    Returns:
        Merged and ranked list of memories
    """
    # Normalize RAG memories (assign default scores for fair ranking)
    normalized_rag = []
    for mem in rag_memories:
        content = mem.get("content", "") or mem.get("anchor_text", "")
        if not content:
            continue
        
        normalized_rag.append({
            "content": content,
            "id": mem.get("id", ""),
            "hash": mem.get("hash", ""),
            "type": "rag",
            "resonance_score": 0.3,
            "proximity_score": 0.3,
            "anchor_score": 0.3,
            "recency_score": 0.1,
            "similarity_score": mem.get("similarity_score", 0.5),
            "rag_score": mem.get("score", mem.get("similarity_score", 0.5)),
        })
    
    # Normalize Hash Sphere memories
    normalized_sphere = []
    for mem in sphere_memories:
        content = mem.get("content", "") or mem.get("anchor_text", "")
        if not content:
            continue
        
        normalized_sphere.append({
            "content": content,
            "id": mem.get("id", ""),
            "hash": mem.get("hash", ""),
            "type": mem.get("type", "hash_sphere"),
            "resonance_score": mem.get("resonance_score", 0.0),
            "proximity_score": mem.get("proximity_score", 0.0),
            "anchor_score": mem.get("anchor_score", 0.0),
            "recency_score": mem.get("recency_score", 0.0),
            "cluster_score": mem.get("cluster_score", 0.0),
        })
    
    # Merge both sets
    merged = normalized_rag + normalized_sphere
    
    # Get weights: A/B test > Adaptive > Default
    weights = {"rag": 0.30, "resonance": 0.25, "proximity": 0.20, "recency": 0.15, "anchor": 0.10}
    
    if user_id:
        if experiment:
            # Use A/B test weights
            ab_weights = ab_tester.get_weights(user_id, experiment)
            if ab_weights:
                weights.update(ab_weights)
        elif use_adaptive_weights:
            # Use adaptive (personalized) weights
            adaptive_weights = adaptive_tuner.get_weights(user_id)
            if adaptive_weights:
                weights.update(adaptive_weights)
    
    # Scoring function with dynamic weights
    def score(m):
        def safe(val):
            return float(val) if val is not None else 0.0
        
        rag_score = safe(m.get("rag_score") or m.get("similarity_score"))
        resonance = safe(m.get("resonance_score"))
        proximity = safe(m.get("proximity_score"))
        recency = safe(m.get("recency_score"))
        anchor = safe(m.get("anchor_score"))
        
        return (
            rag_score * weights.get("rag", 0.30) +
            resonance * weights.get("resonance", 0.25) +
            proximity * weights.get("proximity", 0.20) +
            recency * weights.get("recency", 0.15) +
            anchor * weights.get("anchor", 0.10)
        )
    
    # Add combined_score to each memory
    for m in merged:
        m["combined_score"] = score(m)
        m["weights_used"] = weights  # Track which weights were used
    
    # Rank by combined score
    ranked = sorted(merged, key=lambda x: x["combined_score"], reverse=True)
    
    return ranked[:limit]


def compute_hybrid_score(mem: Dict) -> float:
    """
    Compute hybrid memory score using 5-factor formula.
    
    Weights:
    - RAG score: 0.25
    - Resonance score: 0.25
    - Proximity score: 0.20
    - Recency score: 0.15
    - Anchor score: 0.15
    """
    W_RAG = 0.25
    W_RESONANCE = 0.25
    W_PROXIMITY = 0.20
    W_RECENCY = 0.15
    W_ANCHOR = 0.15
    
    def safe(val):
        return float(val) if val is not None else 0.0
    
    rag_score = safe(mem.get("rag_score") or mem.get("similarity_score") or mem.get("semantic_score"))
    resonance_score = safe(mem.get("resonance_score"))
    proximity_score = safe(mem.get("proximity_score"))
    recency_score = safe(mem.get("recency_score"))
    anchor_score = safe(mem.get("anchor_score") or mem.get("importance_score"))
    
    final = (
        rag_score * W_RAG +
        resonance_score * W_RESONANCE +
        proximity_score * W_PROXIMITY +
        recency_score * W_RECENCY +
        anchor_score * W_ANCHOR
    )
    
    return final
