"""
Probabilistic Thought Branching (PTB)
======================================

Patch #57: Enables the agent to create multiple internal reasoning paths,
then merge them into one final answer.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/thought_branching.py
"""
from __future__ import annotations

import logging
import random
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ProbabilisticThoughtBranching:
    """
    Probabilistic Thought Branching
    
    Creates multiple internal interpretations of queries and merges them
    into unified reasoning hints.
    """
    
    def __init__(self, branch_count: int = 3):
        self.branch_count = branch_count
    
    def generate_branches(
        self,
        message: str,
        memories: List[Dict[str, Any]],
        count: Optional[int] = None
    ) -> List[str]:
        """Create multiple internal interpretations of the query."""
        try:
            if count is None:
                count = self.branch_count
            
            branches = []
            
            if not memories:
                for i in range(count):
                    branch = (
                        f"Interpretation {i+1}:\n"
                        f"- User said: {message}\n"
                        f"- Focus: General understanding"
                    )
                    branches.append(branch)
                return branches
            
            for i in range(count):
                if len(memories) >= 2:
                    sampled = random.sample(memories, min(2, len(memories)))
                elif len(memories) == 1:
                    sampled = memories
                else:
                    sampled = []
                
                memory_texts = []
                for mem in sampled:
                    content = mem.get("content") or mem.get("anchor_text") or ""
                    if content:
                        memory_texts.append(content[:80])
                
                branch = (
                    f"Interpretation {i+1}:\n"
                    f"- User said: {message}\n"
                )
                
                if memory_texts:
                    branch += f"- Related memories: {', '.join(memory_texts)}\n"
                    branch += f"- Reasoning angle: Focus on {' and '.join([m[:30] for m in memory_texts[:2]])}"
                else:
                    branch += "- Reasoning angle: Direct interpretation without specific memory context"
                
                branches.append(branch)
            
            return branches
            
        except Exception as e:
            logger.warning(f"Error generating thought branches: {e}")
            return [
                f"Interpretation:\n- User said: {message}\n- Reasoning angle: Direct interpretation"
            ]
    
    def merge(self, branches: List[str]) -> str:
        """Merge branches into a unified reasoning hint."""
        try:
            if not branches:
                return "Internal reasoning: Direct interpretation."
            
            merged = "Internal reasoning synthesis (multiple perspectives considered):\n"
            
            for i, branch in enumerate(branches, 1):
                branch_preview = branch[:200] if len(branch) > 200 else branch
                merged += f"\n{branch_preview}"
            
            merged += "\n\nSynthesize these perspectives into a coherent, accurate response."
            
            return merged
            
        except Exception as e:
            logger.warning(f"Error merging branches: {e}")
            return "Internal reasoning: Multiple perspectives considered."
    
    def generate_and_merge(
        self,
        message: str,
        memories: List[Dict[str, Any]],
        count: Optional[int] = None
    ) -> str:
        """Generate branches and merge them in one call."""
        branches = self.generate_branches(message, memories, count)
        return self.merge(branches)


# Global instance
thought_branching = ProbabilisticThoughtBranching()
