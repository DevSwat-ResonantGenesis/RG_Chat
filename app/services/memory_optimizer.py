"""
Memory Optimizer Service
Optimizes context window usage and summarizes long conversations.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MemoryChunk:
    """A chunk of memory/context"""
    content: str
    role: str
    timestamp: float
    importance: float = 0.5
    token_estimate: int = 0
    summary: Optional[str] = None


@dataclass
class OptimizedContext:
    """Optimized context ready for LLM"""
    messages: List[Dict[str, str]]
    total_tokens: int
    was_summarized: bool
    dropped_count: int
    summary_text: Optional[str] = None


class MemoryOptimizer:
    """
    Memory Optimizer for context window management.
    
    Features:
    - Token estimation for messages
    - Importance-based memory ranking
    - Automatic summarization of old messages
    - Context window fitting
    - Deduplication of similar memories
    """
    
    # Approximate tokens per character (conservative estimate)
    TOKENS_PER_CHAR = 0.3
    
    # Default context limits
    DEFAULT_MAX_TOKENS = 8000
    SUMMARY_THRESHOLD = 10  # Summarize after this many messages
    
    # Importance keywords
    IMPORTANCE_KEYWORDS = {
        "high": ["important", "critical", "must", "required", "error", "bug", "fix", 
                 "urgent", "deadline", "key", "essential", "remember"],
        "medium": ["should", "need", "want", "please", "help", "create", "build"],
        "low": ["maybe", "perhaps", "optional", "just", "simple", "quick"]
    }
    
    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.max_tokens = max_tokens
        self.conversation_summaries: Dict[str, str] = {}
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return int(len(text) * self.TOKENS_PER_CHAR)
    
    def calculate_importance(self, message: str, role: str) -> float:
        """Calculate importance score for a message"""
        importance = 0.5  # Base score
        message_lower = message.lower()
        
        # Role-based adjustment
        if role == "system":
            importance += 0.3  # System messages are important
        elif role == "user":
            importance += 0.1  # User messages slightly more important
        
        # Keyword-based adjustment
        for kw in self.IMPORTANCE_KEYWORDS["high"]:
            if kw in message_lower:
                importance += 0.15
                break
        
        for kw in self.IMPORTANCE_KEYWORDS["medium"]:
            if kw in message_lower:
                importance += 0.05
                break
        
        for kw in self.IMPORTANCE_KEYWORDS["low"]:
            if kw in message_lower:
                importance -= 0.05
                break
        
        # Length-based adjustment (longer messages often more important)
        if len(message) > 500:
            importance += 0.1
        elif len(message) < 50:
            importance -= 0.1
        
        # Code blocks are important
        if "```" in message:
            importance += 0.2
        
        # Questions are important
        if "?" in message:
            importance += 0.1
        
        return max(0.1, min(1.0, importance))
    
    def summarize_messages(self, messages: List[Dict[str, str]], max_length: int = 500) -> str:
        """
        Create a summary of multiple messages.
        Uses extractive summarization (key sentences).
        """
        if not messages:
            return ""
        
        # Extract key information from each message
        summaries = []
        
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            # Skip very short messages
            if len(content) < 20:
                continue
            
            # Extract first meaningful sentence
            sentences = re.split(r'[.!?]\s+', content)
            if sentences:
                first_sentence = sentences[0][:150]
                if len(first_sentence) > 30:
                    summaries.append(f"[{role}] {first_sentence}")
        
        # Combine summaries
        combined = " | ".join(summaries[:5])  # Limit to 5 key points
        
        if len(combined) > max_length:
            combined = combined[:max_length] + "..."
        
        return combined
    
    def deduplicate_memories(self, memories: List[Dict[str, Any]], 
                            similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Remove duplicate or very similar memories.
        Uses simple word overlap for similarity.
        """
        if len(memories) <= 1:
            return memories
        
        unique_memories = []
        seen_hashes = set()
        
        for mem in memories:
            content = mem.get("content", "")
            
            # Create a simple hash of key words
            words = set(re.findall(r'\b\w{4,}\b', content.lower()))
            word_hash = hashlib.md5("".join(sorted(words)[:20]).encode()).hexdigest()[:8]
            
            # Check for duplicates
            if word_hash not in seen_hashes:
                seen_hashes.add(word_hash)
                unique_memories.append(mem)
        
        if len(unique_memories) < len(memories):
            logger.info(f"🧹 Deduplicated memories: {len(memories)} -> {len(unique_memories)}")
        
        return unique_memories
    
    def optimize_context(self, 
                        messages: List[Dict[str, str]],
                        memories: Optional[List[Dict[str, Any]]] = None,
                        max_tokens: Optional[int] = None) -> OptimizedContext:
        """
        Optimize context to fit within token limit.
        
        Args:
            messages: Conversation messages
            memories: Retrieved memories
            max_tokens: Maximum tokens (uses default if not specified)
        
        Returns:
            OptimizedContext with fitted messages
        """
        max_tokens = max_tokens or self.max_tokens
        
        # Convert to MemoryChunks for processing
        chunks: List[MemoryChunk] = []
        
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            chunk = MemoryChunk(
                content=content,
                role=role,
                timestamp=time.time() - (len(messages) - i),  # Older messages have lower timestamp
                importance=self.calculate_importance(content, role),
                token_estimate=self.estimate_tokens(content)
            )
            chunks.append(chunk)
        
        # Calculate total tokens
        total_tokens = sum(c.token_estimate for c in chunks)
        
        # If within limit, return as-is
        if total_tokens <= max_tokens:
            return OptimizedContext(
                messages=messages,
                total_tokens=total_tokens,
                was_summarized=False,
                dropped_count=0
            )
        
        # Need to optimize
        logger.info(f"⚡ Optimizing context: {total_tokens} tokens -> {max_tokens} max")
        
        # Strategy 1: Keep system messages and recent messages, summarize old ones
        system_messages = [c for c in chunks if c.role == "system"]
        non_system = [c for c in chunks if c.role != "system"]
        
        # Always keep last 5 messages
        recent_count = min(5, len(non_system))
        recent_messages = non_system[-recent_count:] if recent_count > 0 else []
        old_messages = non_system[:-recent_count] if recent_count > 0 else non_system
        
        # Summarize old messages if there are many
        summary_text = None
        if len(old_messages) > 3:
            old_dicts = [{"content": c.content, "role": c.role} for c in old_messages]
            summary_text = self.summarize_messages(old_dicts)
            logger.info(f"📝 Summarized {len(old_messages)} old messages")
        
        # Build optimized message list
        optimized_messages = []
        
        # Add system messages first
        for chunk in system_messages:
            optimized_messages.append({"role": chunk.role, "content": chunk.content})
        
        # Add summary if created
        if summary_text:
            optimized_messages.append({
                "role": "system",
                "content": f"CONVERSATION SUMMARY (earlier messages):\n{summary_text}"
            })
        
        # Add recent messages
        for chunk in recent_messages:
            optimized_messages.append({"role": chunk.role, "content": chunk.content})
        
        # Calculate new total
        new_total = sum(self.estimate_tokens(m.get("content", "")) for m in optimized_messages)
        
        # If still over limit, truncate long messages
        if new_total > max_tokens:
            for msg in optimized_messages:
                content = msg.get("content", "")
                if len(content) > 2000:
                    msg["content"] = content[:2000] + "... [truncated]"
            new_total = sum(self.estimate_tokens(m.get("content", "")) for m in optimized_messages)
        
        dropped_count = len(messages) - len(optimized_messages) + (1 if summary_text else 0)
        
        return OptimizedContext(
            messages=optimized_messages,
            total_tokens=new_total,
            was_summarized=summary_text is not None,
            dropped_count=max(0, dropped_count),
            summary_text=summary_text
        )
    
    def rank_memories_by_relevance(self, 
                                   memories: List[Dict[str, Any]], 
                                   query: str,
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Rank memories by relevance to query.
        Uses keyword overlap and recency.
        """
        if not memories:
            return []
        
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        scored_memories = []
        for mem in memories:
            content = mem.get("content", "")
            mem_words = set(re.findall(r'\b\w{3,}\b', content.lower()))
            
            # Calculate overlap score
            overlap = len(query_words & mem_words)
            overlap_score = overlap / max(len(query_words), 1)
            
            # Get existing relevance score if any
            existing_score = mem.get("relevance_score", 0.5)
            
            # Combine scores
            final_score = (overlap_score * 0.4) + (existing_score * 0.6)
            
            scored_memories.append((final_score, mem))
        
        # Sort by score descending
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # Return top memories
        return [mem for _, mem in scored_memories[:limit]]
    
    def get_optimization_stats(self, 
                              original_messages: List[Dict[str, str]],
                              optimized: OptimizedContext) -> Dict[str, Any]:
        """Get statistics about the optimization"""
        original_tokens = sum(self.estimate_tokens(m.get("content", "")) for m in original_messages)
        
        return {
            "original_message_count": len(original_messages),
            "optimized_message_count": len(optimized.messages),
            "original_tokens": original_tokens,
            "optimized_tokens": optimized.total_tokens,
            "tokens_saved": original_tokens - optimized.total_tokens,
            "compression_ratio": optimized.total_tokens / original_tokens if original_tokens > 0 else 1.0,
            "was_summarized": optimized.was_summarized,
            "dropped_count": optimized.dropped_count
        }


# Global instance
memory_optimizer = MemoryOptimizer()


# Convenience functions
def optimize_context(messages: List[Dict[str, str]], 
                    max_tokens: int = 8000) -> OptimizedContext:
    """Optimize context to fit within token limit"""
    return memory_optimizer.optimize_context(messages, max_tokens=max_tokens)


def deduplicate_memories(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate memories"""
    return memory_optimizer.deduplicate_memories(memories)


def rank_memories(memories: List[Dict[str, Any]], 
                 query: str, 
                 limit: int = 10) -> List[Dict[str, Any]]:
    """Rank memories by relevance"""
    return memory_optimizer.rank_memories_by_relevance(memories, query, limit)
