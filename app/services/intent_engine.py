"""
Intent Decomposition Engine (IDE)
==================================

Patch #43: Extracts the real intent behind the user message, even if the
message is emotional, unclear, short, or chaotic.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/intent_engine.py
"""
from __future__ import annotations

import logging
from typing import List, Dict, Set

logger = logging.getLogger(__name__)


class IntentEngine:
    """
    Intent Decomposition Engine
    
    Extracts the real intent behind user messages, even if they are
    emotional, unclear, short, or chaotic.
    """
    
    INTENT_PATTERNS: Dict[str, List[str]] = {
        "debug": ["fix", "error", "bug", "issue", "problem", "broken", "not working", "failing"],
        "memory": ["remember", "what did we talk", "recall", "memory", "we discussed", "you said", "i told you"],
        "coding": ["code", "write", "patch", "function", "refactor", "implement", "create function", "generate code"],
        "analysis": ["analyze", "compare", "break down", "explain", "evaluate", "assess"],
        "planning": ["plan", "roadmap", "strategy", "steps", "how to", "approach", "method"],
        "emotion": ["i feel", "i'm mad", "i'm sad", "i'm stressed", "frustrated", "worried", "excited"],
        "research": ["research", "find information", "look up", "investigate", "search"],
        "summary": ["summarize", "summary", "brief overview", "tl;dr", "recap"],
        "question": ["what", "why", "how", "when", "where", "who", "which", "?"],
        "action": ["do", "make", "create", "build", "generate", "run", "execute", "perform"]
    }
    
    def extract(self, text: str) -> List[str]:
        """
        Extract intent categories from text.
        
        Args:
            text: User message text
        
        Returns:
            List of intent categories (primary first, then secondary)
        """
        if not text:
            return ["general_query"]
        
        t = text.lower()
        intents: Set[str] = set()
        
        for intent_name, keywords in self.INTENT_PATTERNS.items():
            for keyword in keywords:
                if keyword in t:
                    intents.add(intent_name)
                    break
        
        if not intents:
            intents.add("general_query")
        
        priority_order = [
            "debug", "coding", "analysis", "planning", "memory",
            "research", "summary", "action", "question", "emotion", "general_query"
        ]
        
        sorted_intents = []
        for intent in priority_order:
            if intent in intents:
                sorted_intents.append(intent)
        
        for intent in intents:
            if intent not in sorted_intents:
                sorted_intents.append(intent)
        
        logger.debug(f"Extracted intents from '{text[:50]}...': {sorted_intents}")
        
        return sorted_intents
    
    def get_primary_intent(self, text: str) -> str:
        """Get the primary intent from text."""
        intents = self.extract(text)
        return intents[0] if intents else "general_query"
    
    def get_intent_description(self, intent: str) -> str:
        """Get human-readable description of intent."""
        descriptions = {
            "debug": "User wants to fix or resolve an issue",
            "memory": "User is asking about past conversations or memories",
            "coding": "User wants code generation or programming help",
            "analysis": "User wants analysis or evaluation",
            "planning": "User wants planning or strategic guidance",
            "emotion": "User is expressing emotional state",
            "research": "User wants information or research",
            "summary": "User wants a summary or overview",
            "question": "User is asking a question",
            "action": "User wants to perform an action",
            "general_query": "General query or conversation"
        }
        return descriptions.get(intent, "Unknown intent")
    
    def get_intent_system_prompt(self, intents: List[str]) -> str:
        """Generate system prompt based on detected intents."""
        if not intents:
            return ""
        
        primary = intents[0]
        prompt_parts = [f"Primary intent detected: {self.get_intent_description(primary)}."]
        
        if len(intents) > 1:
            secondary = ", ".join([self.get_intent_description(i) for i in intents[1:3]])
            prompt_parts.append(f"Secondary intents: {secondary}.")
        
        # Add intent-specific guidance
        if primary == "debug":
            prompt_parts.append("Focus on identifying root cause and providing specific solutions with code examples.")
        elif primary == "memory":
            prompt_parts.append("Search memories carefully and reference specific past conversations.")
        elif primary == "coding":
            prompt_parts.append("Provide clean, working code with proper structure and comments.")
        elif primary == "planning":
            prompt_parts.append("Break down into clear, actionable steps with priorities.")
        
        return " ".join(prompt_parts)


# Global instance
intent_engine = IntentEngine()
