"""
Narrative Continuity Engine (NCE)
==================================

Patch #55: Gives the AI story-like continuity across the entire chat.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/narrative_continuity_engine.py
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class NarrativeContinuityEngine:
    """
    Narrative Continuity Engine
    
    Tracks ongoing topics, unresolved questions, and narrative threads
    to maintain conversation continuity.
    """
    
    TOPIC_KEYWORDS = [
        "why", "how", "fix", "build", "hash", "memory", "RAG",
        "error", "bug", "issue", "problem", "solution", "approach",
        "design", "architecture", "implementation", "deployment",
        "test", "debug", "optimize", "refactor", "migrate"
    ]
    
    QUESTION_INDICATORS = [
        "?", "how", "why", "what", "when", "where", "which", "who"
    ]
    
    UNRESOLVED_INDICATORS = [
        "still", "yet", "not fixed", "not resolved", "pending",
        "waiting", "need to", "should", "must", "have to"
    ]
    
    def __init__(self):
        pass
    
    def extract_threads(self, history: List[Dict[str, Any]], max_threads: int = 3) -> List[Dict[str, Any]]:
        """Detect ongoing topics within conversation history."""
        try:
            threads = []
            seen_topics = set()
            
            for msg in reversed(history):
                if not isinstance(msg, dict):
                    continue
                
                content = msg.get("content") or msg.get("text") or ""
                if not content:
                    continue
                
                text_lower = content.lower()
                
                found_keywords = []
                for keyword in self.TOPIC_KEYWORDS:
                    if keyword in text_lower:
                        found_keywords.append(keyword)
                
                if found_keywords:
                    topic = self._extract_topic_context(content, found_keywords)
                    topic_normalized = topic.lower().strip()[:100]
                    
                    if topic_normalized not in seen_topics:
                        seen_topics.add(topic_normalized)
                        threads.append({
                            "topic": topic,
                            "keywords": found_keywords,
                            "content": content[:200],
                            "timestamp": msg.get("timestamp") or msg.get("created_at")
                        })
                        
                        if len(threads) >= max_threads:
                            break
            
            return list(reversed(threads))
            
        except Exception as e:
            logger.warning(f"Error extracting threads: {e}")
            return []
    
    def _extract_topic_context(self, content: str, keywords: List[str]) -> str:
        """Extract topic context around keywords."""
        try:
            sentences = content.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(kw in sentence_lower for kw in keywords):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                return '. '.join(relevant_sentences[:2])
            else:
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(kw in sentence_lower for kw in keywords):
                        return sentence.strip()[:150]
            
            return content[:100]
            
        except Exception:
            return content[:100]
    
    def extract_unresolved_questions(self, history: List[Dict[str, Any]]) -> List[str]:
        """Extract unresolved questions from conversation history."""
        try:
            unresolved = []
            
            for msg in history:
                if not isinstance(msg, dict):
                    continue
                
                content = msg.get("content") or msg.get("text") or ""
                if not content:
                    continue
                
                is_question = any(indicator in content.lower() for indicator in self.QUESTION_INDICATORS)
                has_unresolved = any(indicator in content.lower() for indicator in self.UNRESOLVED_INDICATORS)
                
                if is_question or has_unresolved:
                    if has_unresolved:
                        unresolved.append(content[:200])
            
            return unresolved[:5]
            
        except Exception as e:
            logger.warning(f"Error extracting unresolved questions: {e}")
            return []
    
    def build_continuity_hint(
        self,
        threads: List[Dict[str, Any]],
        unresolved: Optional[List[str]] = None
    ) -> Optional[str]:
        """Build continuity hint for system prompt."""
        try:
            if not threads and not unresolved:
                return None
            
            parts = []
            
            if threads:
                parts.append("Ongoing topics you must keep consistent:")
                for i, thread in enumerate(threads, 1):
                    topic = thread.get("topic", thread.get("content", ""))[:120]
                    parts.append(f"{i}. {topic}")
            
            if unresolved:
                parts.append("\nUnresolved questions that need attention:")
                for i, question in enumerate(unresolved, 1):
                    parts.append(f"{i}. {question[:120]}")
            
            if parts:
                return "\n".join(parts)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error building continuity hint: {e}")
            return None
    
    def track_narrative_arc(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track narrative arc across conversation."""
        try:
            if not history:
                return {
                    "arcs": [],
                    "current_focus": None,
                    "evolution": []
                }
            
            topic_groups = defaultdict(list)
            
            for msg in history:
                if not isinstance(msg, dict):
                    continue
                
                content = msg.get("content") or msg.get("text") or ""
                if not content:
                    continue
                
                text_lower = content.lower()
                for keyword in self.TOPIC_KEYWORDS:
                    if keyword in text_lower:
                        topic_groups[keyword].append(content[:100])
                        break
            
            arcs = []
            for topic, messages in topic_groups.items():
                arcs.append({
                    "topic": topic,
                    "message_count": len(messages),
                    "sample": messages[0] if messages else ""
                })
            
            current_focus = None
            if arcs:
                current_focus = arcs[-1]["topic"]
            
            return {
                "arcs": arcs,
                "current_focus": current_focus,
                "evolution": [arc["topic"] for arc in arcs]
            }
            
        except Exception as e:
            logger.warning(f"Error tracking narrative arc: {e}")
            return {
                "arcs": [],
                "current_focus": None,
                "evolution": []
            }
    
    def get_system_prompt(self, history: List[Dict[str, Any]]) -> str:
        """Generate system prompt with narrative continuity context."""
        threads = self.extract_threads(history)
        unresolved = self.extract_unresolved_questions(history)
        hint = self.build_continuity_hint(threads, unresolved)
        
        if hint:
            return f"NARRATIVE CONTINUITY:\n{hint}"
        return ""


# Global instance
narrative_continuity_engine = NarrativeContinuityEngine()
