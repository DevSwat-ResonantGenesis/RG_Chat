"""
Hash Sphere Magnetic Pull System (HS-MPS)
==========================================

Patch #39: Non-linear boost to strong memories. Weak memories stay weak,
strong memories get amplified, creating a "magnetic field" effect
that pulls responses toward core stable meaning.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/memory_extraction.py
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class MagneticPullSystem:
    """
    Hash Sphere Magnetic Pull System
    
    Creates a non-linear boost to strong memories, amplifying
    high-resonance memories while weakening low-resonance ones.
    """
    
    def __init__(self, boost_factor: float = 1.5):
        self.boost_factor = boost_factor
    
    def magnetic_pull(self, resonance_score: float) -> float:
        """
        Apply non-linear magnetic pull to resonance score.
        
        Non-linear boost: square the score and multiply by boost_factor.
        - Low resonance (0.3) -> 0.3^2 * 1.5 = 0.135 (weaker)
        - Medium resonance (0.6) -> 0.6^2 * 1.5 = 0.54 (moderate)
        - High resonance (0.9) -> 0.9^2 * 1.5 = 1.215 (capped at 1.0)
        
        Args:
            resonance_score: Original resonance score (0.0 to 1.0)
        
        Returns:
            Magnetic-pulled score (amplified for high resonance)
        """
        magnetic = (resonance_score ** 2) * self.boost_factor
        return min(magnetic, 1.0)
    
    def apply_to_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply magnetic pull to a list of memories.
        
        Args:
            memories: List of memory dictionaries with resonance_score
        
        Returns:
            Memories with magnetic_score added
        """
        # Create a new list to avoid modifying during iteration
        updated_memories = []
        for mem in memories:
            # Handle both string and dictionary memory objects
            if isinstance(mem, str):
                # Convert string to dictionary with default resonance
                updated_memories.append({
                    "content": mem,
                    "resonance_score": 0.5,  # Default resonance for string memories
                    "magnetic_score": self.magnetic_pull(0.5)
                })
            elif isinstance(mem, dict):
                # Handle dictionary memories
                resonance = mem.get("resonance_score", 0.0) or 0.0
                mem_copy = mem.copy()
                mem_copy["magnetic_score"] = self.magnetic_pull(resonance)
                updated_memories.append(mem_copy)
            else:
                # Handle other types (fallback)
                updated_memories.append({
                    "content": str(mem),
                    "resonance_score": 0.3,  # Lower resonance for unknown types
                    "magnetic_score": self.magnetic_pull(0.3)
                })
        
        return updated_memories
    
    def rank_with_magnetic_pull(
        self,
        memories: List[Dict[str, Any]],
        weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank memories using magnetic pull scoring.
        
        Args:
            memories: List of memory dictionaries
            weights: Optional custom weights for scoring
        
        Returns:
            Sorted memories by combined score
        """
        if weights is None:
            weights = {
                "magnetic": 0.4,
                "proximity": 0.25,
                "anchor": 0.15,
                "recency": 0.10,
                "gravity": 0.10
            }
        
        for mem in memories:
            # Handle both string and dictionary memory objects
            if isinstance(mem, str):
                resonance = 0.5  # Default resonance for string memories
                proximity = 0.0
                anchor = 0.0
                recency = 0.0
                gravity = 0.0
            elif isinstance(mem, dict):
                resonance = mem.get("resonance_score", 0.0) or 0.0
                proximity = mem.get("proximity_score", 0.0) or 0.0
                anchor = mem.get("anchor_score", 0.0) or 0.0
                recency = mem.get("recency_score", 0.0) or 0.0
                gravity = mem.get("gravity_force", 0.0) or 0.0
            else:
                resonance = 0.3  # Lower resonance for unknown types
                proximity = 0.0
                anchor = 0.0
                recency = 0.0
                gravity = 0.0
            
            magnetic = self.magnetic_pull(resonance)
            
            combined = (
                magnetic * weights.get("magnetic", 0.4) +
                proximity * weights.get("proximity", 0.25) +
                anchor * weights.get("anchor", 0.15) +
                recency * weights.get("recency", 0.10) +
                gravity * weights.get("gravity", 0.10)
            )
            
            # Ensure mem is a dictionary for scoring
            if isinstance(mem, str):
                mem = {"content": mem}
            elif not isinstance(mem, dict):
                mem = {"content": str(mem)}
                
            mem["magnetic_score"] = magnetic
            mem["combined_score"] = combined
        
        memories.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        return memories


# Global instance
magnetic_pull_system = MagneticPullSystem()
