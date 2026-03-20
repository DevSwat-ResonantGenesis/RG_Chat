"""
Agent Knowledge Base
====================

Phase 3 of Agent Autonomy Enhancement - Knowledge Management and Sharing.

Captures and stores agent learnings to build a comprehensive knowledge base.
Enables knowledge reuse and sharing across agents.

Expected Impact: 50-70% total reduction in LLM calls (combined with Phases 1-2).

Author: Resonant Chat Systems Team
Date: December 26, 2025
"""
from __future__ import annotations

import logging
import hashlib
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """A single knowledge entry in the knowledge base."""
    id: str
    agent_type: str
    topic: str
    question: str
    answer: str
    confidence: float  # 0.0 - 1.0
    created_at: datetime
    used_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    tags: List[str] = field(default_factory=list)
    related_entries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_used: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failed_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def is_stale(self) -> bool:
        """Check if entry is stale (not used in 60 days)."""
        if not self.last_used:
            return (datetime.now() - self.created_at) > timedelta(days=60)
        return (datetime.now() - self.last_used) > timedelta(days=60)
    
    @property
    def quality_score(self) -> float:
        """Calculate overall quality score."""
        # Factors: confidence, success rate, usage frequency
        usage_score = min(self.used_count / 10, 1.0)  # Cap at 10 uses
        return (
            self.confidence * 0.4 +
            self.success_rate * 0.4 +
            usage_score * 0.2
        )


class AgentKnowledgeBase:
    """
    Knowledge base for storing and retrieving agent learnings.
    
    Features:
    - Semantic search (simple keyword-based for now)
    - Success rate tracking
    - Automatic cleanup of stale entries
    - Cross-agent knowledge sharing
    """
    
    def __init__(self, agent_type: Optional[str] = None):
        self.agent_type = agent_type  # None = shared knowledge base
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.topic_index: Dict[str, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        
        logger.info(
            f"AgentKnowledgeBase initialized "
            f"(agent_type={agent_type or 'shared'})"
        )
    
    def add_entry(
        self,
        question: str,
        answer: str,
        topic: str,
        agent_type: str,
        confidence: float = 0.8,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeEntry:
        """Add a new knowledge entry."""
        # Generate unique ID
        entry_id = self._generate_id(question, answer, agent_type)
        
        # Check if entry already exists
        if entry_id in self.entries:
            # Update existing entry
            entry = self.entries[entry_id]
            entry.used_count += 1
            entry.last_used = datetime.now()
            logger.debug(f"Updated existing entry: {entry_id}")
            return entry
        
        # Create new entry
        entry = KnowledgeEntry(
            id=entry_id,
            agent_type=agent_type,
            topic=topic,
            question=question,
            answer=answer,
            confidence=confidence,
            created_at=datetime.now(),
            tags=tags or [],
            metadata=metadata or {},
        )
        
        # Store entry
        self.entries[entry_id] = entry
        
        # Update indices
        self.topic_index[topic].append(entry_id)
        for tag in entry.tags:
            self.tag_index[tag].append(entry_id)
        
        logger.info(
            f"Added knowledge entry: {topic} "
            f"(agent: {agent_type}, confidence: {confidence:.2%})"
        )
        
        return entry
    
    def search(
        self,
        query: str,
        agent_type: Optional[str] = None,
        min_confidence: float = 0.7,
        min_success_rate: float = 0.6,
        top_k: int = 5
    ) -> List[KnowledgeEntry]:
        """
        Search knowledge base for relevant entries.
        
        Args:
            query: Search query
            agent_type: Filter by agent type (None = all agents)
            min_confidence: Minimum confidence threshold
            min_success_rate: Minimum success rate threshold
            top_k: Maximum number of results
            
        Returns:
            List of relevant knowledge entries, sorted by relevance
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score all entries
        scored_entries = []
        
        for entry in self.entries.values():
            # Filter by agent type
            if agent_type and entry.agent_type != agent_type and entry.agent_type != "shared":
                continue
            
            # Filter by confidence
            if entry.confidence < min_confidence:
                continue
            
            # Filter by success rate (if entry has been used)
            if entry.used_count > 0 and entry.success_rate < min_success_rate:
                continue
            
            # Filter out stale entries
            if entry.is_stale:
                continue
            
            # Calculate relevance score
            score = self._calculate_relevance(query_words, entry)
            
            if score > 0:
                scored_entries.append((score, entry))
        
        # Sort by score (descending)
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        
        # Return top K
        results = [entry for score, entry in scored_entries[:top_k]]
        
        logger.debug(
            f"Search for '{query[:50]}...' returned {len(results)} results"
        )
        
        return results
    
    def _calculate_relevance(
        self,
        query_words: set,
        entry: KnowledgeEntry
    ) -> float:
        """Calculate relevance score between query and entry."""
        score = 0.0
        
        # Check question match
        question_words = set(entry.question.lower().split())
        question_overlap = len(query_words & question_words)
        score += question_overlap * 2.0  # Question match is important
        
        # Check topic match
        topic_words = set(entry.topic.lower().split())
        topic_overlap = len(query_words & topic_words)
        score += topic_overlap * 1.5
        
        # Check tag match
        for tag in entry.tags:
            tag_words = set(tag.lower().split())
            tag_overlap = len(query_words & tag_words)
            score += tag_overlap * 1.0
        
        # Boost by quality score
        score *= entry.quality_score
        
        # Boost by usage frequency (popular entries)
        usage_boost = min(entry.used_count / 20, 1.5)
        score *= usage_boost
        
        return score
    
    def update_success(self, entry_id: str, success: bool):
        """Update success rate of an entry."""
        entry = self.entries.get(entry_id)
        if not entry:
            logger.warning(f"Entry not found: {entry_id}")
            return
        
        entry.used_count += 1
        entry.last_used = datetime.now()
        
        if success:
            entry.success_count += 1
        else:
            entry.failed_count += 1
        
        logger.debug(
            f"Updated entry success: {entry_id} "
            f"(success_rate: {entry.success_rate:.2%})"
        )
    
    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get entry by ID."""
        return self.entries.get(entry_id)
    
    def get_entries_by_topic(self, topic: str) -> List[KnowledgeEntry]:
        """Get all entries for a topic."""
        entry_ids = self.topic_index.get(topic, [])
        return [self.entries[eid] for eid in entry_ids if eid in self.entries]
    
    def get_entries_by_tag(self, tag: str) -> List[KnowledgeEntry]:
        """Get all entries with a tag."""
        entry_ids = self.tag_index.get(tag, [])
        return [self.entries[eid] for eid in entry_ids if eid in self.entries]
    
    def get_top_entries(
        self,
        top_k: int = 10,
        agent_type: Optional[str] = None
    ) -> List[KnowledgeEntry]:
        """Get top entries by quality score."""
        entries = list(self.entries.values())
        
        # Filter by agent type
        if agent_type:
            entries = [
                e for e in entries
                if e.agent_type == agent_type or e.agent_type == "shared"
            ]
        
        # Sort by quality score
        entries.sort(key=lambda e: e.quality_score, reverse=True)
        
        return entries[:top_k]
    
    def cleanup_stale(self) -> int:
        """Remove stale entries. Returns count of removed entries."""
        before_count = len(self.entries)
        
        # Find stale entries
        stale_ids = [
            eid for eid, entry in self.entries.items()
            if entry.is_stale and entry.used_count < 3  # Keep if used frequently
        ]
        
        # Remove stale entries
        for entry_id in stale_ids:
            entry = self.entries[entry_id]
            
            # Remove from entries
            del self.entries[entry_id]
            
            # Remove from indices
            if entry.topic in self.topic_index:
                self.topic_index[entry.topic] = [
                    eid for eid in self.topic_index[entry.topic]
                    if eid != entry_id
                ]
            
            for tag in entry.tags:
                if tag in self.tag_index:
                    self.tag_index[tag] = [
                        eid for eid in self.tag_index[tag]
                        if eid != entry_id
                    ]
        
        removed_count = before_count - len(self.entries)
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} stale entries")
        
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if not self.entries:
            return {
                "agent_type": self.agent_type or "shared",
                "total_entries": 0,
                "total_uses": 0,
                "avg_success_rate": 0.0,
                "avg_quality_score": 0.0,
            }
        
        total_uses = sum(e.used_count for e in self.entries.values())
        total_successes = sum(e.success_count for e in self.entries.values())
        total_failures = sum(e.failed_count for e in self.entries.values())
        
        avg_success_rate = (
            total_successes / (total_successes + total_failures)
            if (total_successes + total_failures) > 0 else 0.0
        )
        
        avg_quality_score = sum(
            e.quality_score for e in self.entries.values()
        ) / len(self.entries)
        
        # Count by topic
        topics = defaultdict(int)
        for entry in self.entries.values():
            topics[entry.topic] += 1
        
        return {
            "agent_type": self.agent_type or "shared",
            "total_entries": len(self.entries),
            "total_uses": total_uses,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "avg_success_rate": avg_success_rate,
            "avg_quality_score": avg_quality_score,
            "unique_topics": len(topics),
            "top_topics": sorted(
                topics.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
        }
    
    def _generate_id(self, question: str, answer: str, agent_type: str) -> str:
        """Generate unique ID for entry."""
        content = f"{agent_type}:{question}:{answer[:100]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def export_shareable_knowledge(
        self,
        min_quality_score: float = 0.8,
        min_uses: int = 5
    ) -> List[KnowledgeEntry]:
        """
        Export high-quality knowledge suitable for sharing with other agents.
        
        Criteria:
        - High quality score (>0.8)
        - Used multiple times (>5)
        - High success rate (>0.8)
        """
        shareable = []
        
        for entry in self.entries.values():
            if (entry.quality_score >= min_quality_score and
                entry.used_count >= min_uses and
                entry.success_rate >= 0.8):
                shareable.append(entry)
        
        logger.info(
            f"Exported {len(shareable)} shareable knowledge entries "
            f"from {self.agent_type or 'shared'}"
        )
        
        return shareable


# Global knowledge bases
_knowledge_bases: Dict[str, AgentKnowledgeBase] = {}
_shared_knowledge_base: Optional[AgentKnowledgeBase] = None


def get_knowledge_base(agent_type: Optional[str] = None) -> AgentKnowledgeBase:
    """
    Get knowledge base for agent type.
    
    Args:
        agent_type: Agent type (None = shared knowledge base)
        
    Returns:
        AgentKnowledgeBase instance
    """
    global _shared_knowledge_base
    
    if agent_type is None:
        # Return shared knowledge base
        if _shared_knowledge_base is None:
            _shared_knowledge_base = AgentKnowledgeBase(agent_type=None)
        return _shared_knowledge_base
    
    # Return agent-specific knowledge base
    if agent_type not in _knowledge_bases:
        _knowledge_bases[agent_type] = AgentKnowledgeBase(agent_type=agent_type)
    
    return _knowledge_bases[agent_type]


def get_all_knowledge_bases() -> Dict[str, AgentKnowledgeBase]:
    """Get all knowledge bases (agent-specific + shared)."""
    all_kbs = _knowledge_bases.copy()
    if _shared_knowledge_base:
        all_kbs["shared"] = _shared_knowledge_base
    return all_kbs


def get_aggregate_stats() -> Dict[str, Any]:
    """Get aggregate statistics across all knowledge bases."""
    all_kbs = get_all_knowledge_bases()
    
    if not all_kbs:
        return {
            "total_knowledge_bases": 0,
            "total_entries": 0,
            "total_uses": 0,
            "avg_success_rate": 0.0,
        }
    
    total_entries = sum(len(kb.entries) for kb in all_kbs.values())
    total_uses = sum(
        sum(e.used_count for e in kb.entries.values())
        for kb in all_kbs.values()
    )
    
    all_successes = sum(
        sum(e.success_count for e in kb.entries.values())
        for kb in all_kbs.values()
    )
    all_failures = sum(
        sum(e.failed_count for e in kb.entries.values())
        for kb in all_kbs.values()
    )
    
    avg_success_rate = (
        all_successes / (all_successes + all_failures)
        if (all_successes + all_failures) > 0 else 0.0
    )
    
    return {
        "total_knowledge_bases": len(all_kbs),
        "total_entries": total_entries,
        "total_uses": total_uses,
        "total_successes": all_successes,
        "total_failures": all_failures,
        "avg_success_rate": avg_success_rate,
        "knowledge_bases": {
            name: kb.get_stats()
            for name, kb in all_kbs.items()
        },
    }
