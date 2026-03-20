"""
Evidence Graph Enhancer (EGE) - IMPROVED VERSION
=================================================

Patch #45: Lets the AI keep a lightweight reasoning trace of why it chose its answer.

IMPROVEMENTS (Patch #50):
1. Real embedding-based similarity using cosine similarity on actual embeddings
2. Track actual memory retrievals from Hash Sphere extraction
3. Deterministic positions using actual XYZ coordinates from Hash Sphere
4. NLP-based anchor extraction using entity recognition and keyphrase extraction

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/evidence_graph.py
"""
from __future__ import annotations

import re
import math
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

# Stopwords for NLP processing
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "what", "which", "who", "whom", "where", "when", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "also", "now", "here", "there", "then", "if", "because", "about",
}


class EvidenceGraph:
    """
    Evidence Graph Enhancer - IMPROVED VERSION
    
    Creates a lightweight reasoning trace that shows why the AI
    chose its answer, without exposing chain-of-thought.
    
    Now uses:
    - Real embedding-based semantic similarity
    - Actual memory retrieval tracking
    - Deterministic XYZ positions from Hash Sphere
    - NLP-based anchor/keyphrase extraction
    """
    
    def __init__(self):
        self._embedding_cache: Dict[str, List[float]] = {}
    
    def build_graph(
        self,
        user_hash: str,
        assistant_hash: str,
        memories: List[Dict[str, Any]],
        provider: str,
        intents: Optional[List[str]] = None,
        emotion: Optional[str] = None,
        agent_type: Optional[str] = None,
        debate_used: bool = False,
        # NEW: Enhanced parameters for accurate graph
        user_embedding: Optional[List[float]] = None,
        assistant_embedding: Optional[List[float]] = None,
        user_xyz: Optional[Tuple[float, float, float]] = None,
        assistant_xyz: Optional[Tuple[float, float, float]] = None,
        user_content: Optional[str] = None,
        assistant_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build accurate evidence graph using real embeddings and XYZ coordinates.
        
        IMPROVEMENTS:
        1. Uses cosine similarity on embeddings instead of hash character matching
        2. Stores actual XYZ coordinates from Hash Sphere
        3. Extracts meaningful anchors using NLP
        4. Tracks actual memory retrieval scores
        """
        try:
            # IMPROVEMENT 1: Real embedding-based similarity
            if user_embedding and assistant_embedding:
                semantic_similarity = self._cosine_similarity(user_embedding, assistant_embedding)
                similarity_method = "embedding_cosine"
            else:
                # Fallback to improved hash-based similarity using XYZ proximity
                if user_xyz and assistant_xyz:
                    semantic_similarity = self._xyz_proximity_score(user_xyz, assistant_xyz)
                    similarity_method = "xyz_proximity"
                else:
                    # Last resort: hash character matching (least accurate)
                    semantic_similarity = self._calculate_hash_similarity(user_hash, assistant_hash)
                    similarity_method = "hash_fallback"
            
            # IMPROVEMENT 2: Track actual memory retrievals with real scores
            memory_contributors = []
            for mem in memories[:10]:  # Increased from 5 to 10
                content = mem.get("content", "") or mem.get("anchor_text", "")
                # Use actual scores from Hash Sphere extraction
                hybrid_score = mem.get("hybrid_score", 0.0)
                resonance_score = mem.get("resonance_score", 0.0)
                proximity_score = mem.get("proximity_score", 0.0)
                anchor_energy = mem.get("anchor_energy", 0.0)
                
                # Combined score from actual retrieval
                combined = hybrid_score or resonance_score or mem.get("combined_score", 0.0)
                
                if content:
                    memory_contributors.append({
                        "text": content[:120],
                        "score": round(float(combined), 3),
                        "type": mem.get("type", "memory"),
                        # IMPROVEMENT 3: Include actual XYZ coordinates
                        "xyz": mem.get("xyz"),
                        # Detailed score breakdown for transparency
                        "score_breakdown": {
                            "hybrid": round(float(hybrid_score), 3),
                            "resonance": round(float(resonance_score), 3),
                            "proximity": round(float(proximity_score), 3),
                            "anchor_energy": round(float(anchor_energy), 3),
                        }
                    })
            
            # IMPROVEMENT 4: NLP-based anchor extraction
            extracted_anchors = []
            if assistant_content:
                extracted_anchors = self._extract_keyphrases_nlp(assistant_content)
            
            graph = {
                "timestamp": datetime.utcnow().isoformat(),
                "provider": provider,
                "resonance": {
                    "between_user_and_assistant": round(semantic_similarity, 3),
                    "similarity_method": similarity_method,
                    "user_hash": user_hash[:16] + "..." if user_hash and len(user_hash) > 16 else user_hash,
                    "assistant_hash": assistant_hash[:16] + "..." if assistant_hash and len(assistant_hash) > 16 else assistant_hash,
                    # IMPROVEMENT 3: Include actual XYZ coordinates
                    "user_xyz": list(user_xyz) if user_xyz else None,
                    "assistant_xyz": list(assistant_xyz) if assistant_xyz else None,
                },
                "memory_contributors": memory_contributors,
                "extracted_anchors": extracted_anchors,  # NEW: NLP-extracted keyphrases
                "reasoning": {
                    "intents": intents or [],
                    "emotion": emotion or "neutral",
                    "agent_type": agent_type,
                    "debate_used": debate_used
                },
                "meta": {
                    "memory_count": len(memories),
                    "memories_with_xyz": sum(1 for m in memories if m.get("xyz")),
                    "provider_used": provider,
                    "top_memory_score": round(memory_contributors[0]["score"], 3) if memory_contributors else 0.0,
                    "anchor_count": len(extracted_anchors),
                    "accuracy_level": "high" if similarity_method == "embedding_cosine" else (
                        "medium" if similarity_method == "xyz_proximity" else "low"
                    ),
                }
            }
            
            return graph
            
        except Exception as e:
            logger.error(f"Error building evidence graph: {e}", exc_info=True)
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "provider": provider,
                "error": str(e)
            }
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.
        
        This is the ACCURATE method for semantic similarity.
        Returns value between 0 and 1 (normalized from -1 to 1).
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Cosine similarity ranges from -1 to 1
        cosine = dot_product / (magnitude1 * magnitude2)
        
        # Normalize to 0-1 range for consistency
        return (cosine + 1) / 2
    
    def _xyz_proximity_score(
        self, 
        xyz1: Tuple[float, float, float], 
        xyz2: Tuple[float, float, float]
    ) -> float:
        """
        Calculate proximity score based on Euclidean distance in 3D Hash Sphere space.
        
        Uses exponential decay: score = e^(-distance)
        This gives 1.0 for same point, ~0.37 for distance 1, ~0.14 for distance 2.
        """
        if not xyz1 or not xyz2:
            return 0.0
        
        distance = math.sqrt(
            (xyz1[0] - xyz2[0]) ** 2 +
            (xyz1[1] - xyz2[1]) ** 2 +
            (xyz1[2] - xyz2[2]) ** 2
        )
        
        # Exponential decay for proximity score
        return math.exp(-distance)
    
    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """
        FALLBACK: Calculate similarity between two hashes using character matching.
        
        NOTE: This is the LEAST accurate method and should only be used
        when embeddings and XYZ coordinates are not available.
        """
        if not hash1 or not hash2:
            return 0.0
        
        if hash1 == hash2:
            return 1.0
        
        min_len = min(len(hash1), len(hash2))
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for a, b in zip(hash1[:min_len], hash2[:min_len]) if a == b)
        return min(1.0, max(0.0, matches / min_len))
    
    def _extract_keyphrases_nlp(self, text: str, max_phrases: int = 10) -> List[Dict[str, Any]]:
        """
        Extract meaningful keyphrases using NLP techniques.
        
        Methods used:
        1. Named Entity Recognition (capitalized phrases)
        2. Technical term detection (camelCase, snake_case, code patterns)
        3. TF-IDF-like keyword extraction
        4. N-gram extraction for multi-word concepts
        """
        if not text:
            return []
        
        keyphrases = []
        seen = set()
        
        # Method 1: Extract named entities (capitalized multi-word phrases)
        # Pattern: "Machine Learning", "Hash Sphere", "Evidence Graph"
        named_entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
        for entity in named_entities[:5]:
            if entity.lower() not in seen and len(entity) > 3:
                keyphrases.append({
                    "text": entity,
                    "type": "named_entity",
                    "confidence": 0.9
                })
                seen.add(entity.lower())
        
        # Method 2: Extract technical terms (code patterns)
        # camelCase: getUserData, hashSphere
        camel_case = re.findall(r'\b([a-z]+[A-Z][a-zA-Z]*)\b', text)
        for term in camel_case[:3]:
            if term.lower() not in seen:
                keyphrases.append({
                    "text": term,
                    "type": "technical_term",
                    "confidence": 0.85
                })
                seen.add(term.lower())
        
        # snake_case: user_data, hash_sphere
        snake_case = re.findall(r'\b([a-z]+_[a-z_]+)\b', text)
        for term in snake_case[:3]:
            if term not in seen:
                keyphrases.append({
                    "text": term,
                    "type": "technical_term",
                    "confidence": 0.85
                })
                seen.add(term)
        
        # Method 3: Extract quoted terms (explicit emphasis)
        quoted = re.findall(r'"([^"]+)"', text)
        for term in quoted[:3]:
            if term.lower() not in seen and len(term) > 2:
                keyphrases.append({
                    "text": term,
                    "type": "quoted",
                    "confidence": 0.95
                })
                seen.add(term.lower())
        
        # Method 4: TF-IDF-like keyword extraction
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        filtered_words = [w for w in words if w not in STOPWORDS]
        
        # Count frequencies
        word_freq = Counter(filtered_words)
        
        # Get top keywords by frequency (TF-like)
        for word, freq in word_freq.most_common(5):
            if word not in seen:
                # Higher confidence for more frequent terms
                confidence = min(0.8, 0.5 + (freq * 0.1))
                keyphrases.append({
                    "text": word,
                    "type": "keyword",
                    "confidence": round(confidence, 2),
                    "frequency": freq
                })
                seen.add(word)
        
        # Method 5: Extract bigrams (two-word phrases)
        words_list = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()) if w not in STOPWORDS]
        bigrams = [f"{words_list[i]} {words_list[i+1]}" for i in range(len(words_list) - 1)]
        bigram_freq = Counter(bigrams)
        
        for bigram, freq in bigram_freq.most_common(3):
            if bigram not in seen and freq > 1:
                keyphrases.append({
                    "text": bigram,
                    "type": "bigram",
                    "confidence": round(0.6 + (freq * 0.1), 2),
                    "frequency": freq
                })
                seen.add(bigram)
        
        # Sort by confidence and return top phrases
        keyphrases.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return keyphrases[:max_phrases]
    
    def get_evidence_summary(self, graph: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the evidence graph."""
        if not graph:
            return "No evidence available."
        
        parts = []
        parts.append(f"Provider: {graph.get('provider', 'unknown')}")
        parts.append(f"Resonance: {graph.get('resonance', {}).get('between_user_and_assistant', 0.0):.3f}")
        parts.append(f"Memories used: {graph.get('meta', {}).get('memory_count', 0)}")
        
        intents = graph.get('reasoning', {}).get('intents', [])
        if intents:
            parts.append(f"Intents: {', '.join(intents)}")
        
        emotion = graph.get('reasoning', {}).get('emotion')
        if emotion and emotion != "neutral":
            parts.append(f"Emotion: {emotion}")
        
        if graph.get('reasoning', {}).get('debate_used'):
            parts.append("Multi-agent debate: Yes")
        
        return " | ".join(parts)
    
    # ============================================
    # LAYER 7: EVIDENCE AGGREGATION
    # E* = Σ_{i∈R} w_i · s_i
    # Ê* = E* / ||E*||
    # ============================================
    
    def aggregate_evidence(
        self,
        memories: List[Dict[str, Any]],
        weight_key: str = "combined_score"
    ) -> Tuple[np.ndarray, float]:
        """
        Aggregate evidence from memories into a single evidence vector.
        
        Mathematical definition (Layer 7):
        E* = Σ_{i∈R} w_i · s_i  (weighted sum of memory positions)
        Ê* = E* / ||E*||        (normalized evidence vector)
        
        Args:
            memories: List of memory dictionaries with xyz coordinates and scores
            weight_key: Key to use for weights (default: combined_score)
        
        Returns:
            Tuple of (normalized_evidence_vector, total_weight)
        """
        evidence = np.zeros(3)  # 3D evidence vector
        total_weight = 0.0
        
        for mem in memories:
            xyz = mem.get("xyz")
            if xyz and all(x is not None for x in xyz):
                weight = float(mem.get(weight_key, 0.5) or 0.5)
                point = np.array(xyz)
                evidence += weight * point
                total_weight += weight
        
        # Normalize: Ê* = E* / ||E*||
        norm = np.linalg.norm(evidence)
        if norm > 0:
            evidence = evidence / norm
        
        return evidence, total_weight
    
    def calculate_consistency(
        self,
        response_xyz: Tuple[float, float, float],
        evidence_vector: np.ndarray
    ) -> float:
        """
        Calculate consistency between response and evidence.
        
        Mathematical definition (Layer 8):
        C_k = cos(o_k, Ê*)  (cosine similarity)
        
        Args:
            response_xyz: XYZ coordinates of response
            evidence_vector: Normalized evidence vector from aggregate_evidence()
        
        Returns:
            Consistency score (0-1, higher = more consistent with evidence)
        """
        if response_xyz is None or evidence_vector is None:
            return 0.0
        
        response_vec = np.array(response_xyz)
        
        # Normalize response vector
        response_norm = np.linalg.norm(response_vec)
        if response_norm > 0:
            response_vec = response_vec / response_norm
        
        # Cosine similarity: C_k = cos(o_k, Ê*)
        # For normalized vectors: cos = dot product
        consistency = float(np.dot(response_vec, evidence_vector))
        
        # Clamp to [0, 1] (cosine can be negative)
        return max(0.0, min(1.0, (consistency + 1) / 2))


# Global instance
evidence_graph = EvidenceGraph()
