"""
Agent Memory Persistence System (AMPS)
=======================================

Phase 4: Enables agents to remember previous interactions and learn from them.

Features:
- Per-agent memory storage
- Context retrieval for agent continuity
- Memory pruning and relevance scoring
- Cross-session agent memory
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class AgentMemoryEntry:
    """A single memory entry for an agent."""
    id: str
    agent_type: str
    user_id: str
    task: str
    response: str
    context_hash: str
    relevance_score: float
    created_at: str
    last_accessed: str
    access_count: int = 1
    feedback_score: Optional[float] = None  # User feedback if provided
    tags: List[str] = field(default_factory=list)


class AgentMemoryStore:
    """
    In-memory agent memory store.
    
    In production, this would be backed by Redis or a database.
    """
    
    def __init__(self, max_memories_per_agent: int = 100, memory_ttl_days: int = 30):
        self.memories: Dict[str, Dict[str, List[AgentMemoryEntry]]] = {}  # user_id -> agent_type -> memories
        self.max_memories_per_agent = max_memories_per_agent
        self.memory_ttl_days = memory_ttl_days
    
    def _generate_id(self, agent_type: str, task: str, user_id: str) -> str:
        """Generate unique memory ID."""
        content = f"{agent_type}:{user_id}:{task}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _compute_context_hash(self, context: List[Dict[str, Any]]) -> str:
        """Compute hash of context for similarity matching."""
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()[:8]
    
    def store(
        self,
        agent_type: str,
        user_id: str,
        task: str,
        response: str,
        context: List[Dict[str, Any]],
        tags: Optional[List[str]] = None,
    ) -> AgentMemoryEntry:
        """Store a new memory for an agent."""
        if user_id not in self.memories:
            self.memories[user_id] = {}
        
        if agent_type not in self.memories[user_id]:
            self.memories[user_id][agent_type] = []
        
        memory = AgentMemoryEntry(
            id=self._generate_id(agent_type, task, user_id),
            agent_type=agent_type,
            user_id=user_id,
            task=task[:500],  # Truncate for storage
            response=response[:2000],  # Truncate for storage
            context_hash=self._compute_context_hash(context),
            relevance_score=1.0,  # New memories start with high relevance
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            access_count=1,
            tags=tags or [],
        )
        
        self.memories[user_id][agent_type].append(memory)
        
        # Prune old memories if over limit
        self._prune_memories(user_id, agent_type)
        
        logger.debug(f"💾 Stored memory for {agent_type} agent (user: {user_id[:8]}...)")
        return memory
    
    def retrieve(
        self,
        agent_type: str,
        user_id: str,
        current_task: str,
        limit: int = 5,
    ) -> List[AgentMemoryEntry]:
        """Retrieve relevant memories for an agent."""
        if user_id not in self.memories:
            return []
        
        if agent_type not in self.memories[user_id]:
            return []
        
        memories = self.memories[user_id][agent_type]
        
        # Score memories by relevance to current task
        scored_memories = []
        task_words = set(current_task.lower().split())
        
        for memory in memories:
            # Calculate relevance based on word overlap
            memory_words = set(memory.task.lower().split())
            overlap = len(task_words & memory_words)
            word_score = overlap / max(len(task_words), 1)
            
            # Factor in recency
            try:
                created = datetime.fromisoformat(memory.created_at)
                age_days = (datetime.now() - created).days
                recency_score = max(0, 1 - (age_days / self.memory_ttl_days))
            except:
                recency_score = 0.5
            
            # Factor in access frequency
            frequency_score = min(memory.access_count / 10, 1.0)
            
            # Factor in user feedback if available
            feedback_score = memory.feedback_score if memory.feedback_score else 0.5
            
            # Combined score
            total_score = (
                word_score * 0.4 +
                recency_score * 0.2 +
                frequency_score * 0.1 +
                feedback_score * 0.3
            )
            
            scored_memories.append((memory, total_score))
        
        # Sort by score and return top N
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for memory, score in scored_memories[:limit]:
            # Update access stats
            memory.last_accessed = datetime.now().isoformat()
            memory.access_count += 1
            memory.relevance_score = score
            results.append(memory)
        
        logger.debug(f"📖 Retrieved {len(results)} memories for {agent_type} agent")
        return results
    
    def _prune_memories(self, user_id: str, agent_type: str):
        """Remove old or low-relevance memories."""
        if user_id not in self.memories or agent_type not in self.memories[user_id]:
            return
        
        memories = self.memories[user_id][agent_type]
        
        # Remove expired memories
        cutoff = datetime.now() - timedelta(days=self.memory_ttl_days)
        memories = [
            m for m in memories
            if datetime.fromisoformat(m.created_at) > cutoff
        ]
        
        # Keep only top N by relevance if over limit
        if len(memories) > self.max_memories_per_agent:
            memories.sort(key=lambda m: m.relevance_score, reverse=True)
            memories = memories[:self.max_memories_per_agent]
        
        self.memories[user_id][agent_type] = memories
    
    def add_feedback(
        self,
        memory_id: str,
        user_id: str,
        agent_type: str,
        feedback_score: float,
    ) -> bool:
        """Add user feedback to a memory."""
        if user_id not in self.memories or agent_type not in self.memories[user_id]:
            return False
        
        for memory in self.memories[user_id][agent_type]:
            if memory.id == memory_id:
                memory.feedback_score = max(0, min(1, feedback_score))
                logger.debug(f"📝 Updated feedback for memory {memory_id}: {feedback_score}")
                return True
        
        return False
    
    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics."""
        if user_id:
            if user_id not in self.memories:
                return {"total_memories": 0, "agents": {}}
            
            user_memories = self.memories[user_id]
            return {
                "total_memories": sum(len(m) for m in user_memories.values()),
                "agents": {
                    agent: len(memories)
                    for agent, memories in user_memories.items()
                }
            }
        
        # Global stats
        total = 0
        agent_counts = {}
        for user_memories in self.memories.values():
            for agent, memories in user_memories.items():
                total += len(memories)
                agent_counts[agent] = agent_counts.get(agent, 0) + len(memories)
        
        return {
            "total_memories": total,
            "total_users": len(self.memories),
            "agents": agent_counts,
        }
    
    def clear_user_memories(self, user_id: str, agent_type: Optional[str] = None):
        """Clear memories for a user."""
        if user_id not in self.memories:
            return
        
        if agent_type:
            if agent_type in self.memories[user_id]:
                del self.memories[user_id][agent_type]
        else:
            del self.memories[user_id]
        
        logger.info(f"🗑️ Cleared memories for user {user_id[:8]}...")
    
    def export_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Export all memories for a user."""
        if user_id not in self.memories:
            return []
        
        result = []
        for agent_type, memories in self.memories[user_id].items():
            for memory in memories:
                result.append(asdict(memory))
        
        return result


# Global instance
agent_memory_store = AgentMemoryStore()
