"""
Insight Seed Generator (ISG)
==============================

Patch #49: Generates insight seeds - compressed micro-summaries, patterns, and predictions.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/insight_seed_engine.py
"""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class InsightSeedEngine:
    """
    Insight Seed Generator
    
    Generates compressed insight seeds from conversations
    that can be stored as synthetic memory anchors.
    """
    
    def generate_seed(
        self,
        user_msg: str,
        assistant_msg: str,
        context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generate a compressed insight seed (safe)."""
        try:
            user_preview = user_msg[:80].strip()
            assistant_preview = assistant_msg[:80].strip()
            
            seed = (
                f"Core insight: Based on '{user_preview}', "
                f"the assistant concluded '{assistant_preview}'. "
                f"Key relationship preserved."
            )
            
            if context:
                topics = []
                for msg in context[-3:]:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")[:50]
                        if content:
                            topics.append(content)
                
                if topics:
                    seed += f" Context: {' | '.join(topics)}"
            
            return seed
            
        except Exception as e:
            logger.warning(f"Error generating insight seed: {e}")
            return f"Insight: {user_msg[:50]} → {assistant_msg[:50]}"
    
    def generate_pattern_seed(
        self,
        messages: List[Dict[str, Any]],
        pattern_type: str = "general"
    ) -> Optional[str]:
        """Generate a pattern-based insight seed."""
        try:
            if not messages or len(messages) < 2:
                return None
            
            user_phrases = []
            assistant_phrases = []
            
            for msg in messages[-5:]:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")[:100]
                    
                    if role == "user" and content:
                        user_phrases.append(content)
                    elif role == "assistant" and content:
                        assistant_phrases.append(content)
            
            if not user_phrases or not assistant_phrases:
                return None
            
            pattern = (
                f"Pattern ({pattern_type}): "
                f"User topics: {', '.join(user_phrases[:2])} | "
                f"Assistant approach: {', '.join(assistant_phrases[:2])}"
            )
            
            return pattern
            
        except Exception as e:
            logger.warning(f"Error generating pattern seed: {e}")
            return None
    
    def generate_prediction_seed(
        self,
        user_msg: str,
        assistant_msg: str,
        prediction_type: str = "outcome"
    ) -> Optional[str]:
        """Generate a prediction-based insight seed."""
        try:
            prediction_keywords = [
                "will", "should", "likely", "probably", "expected",
                "future", "next", "then", "after", "result"
            ]
            
            msg_low = assistant_msg.lower()
            has_prediction = any(kw in msg_low for kw in prediction_keywords)
            
            if not has_prediction:
                return None
            
            seed = (
                f"Prediction ({prediction_type}): "
                f"Based on '{user_msg[:60]}', "
                f"predicted: '{assistant_msg[:100]}'"
            )
            
            return seed
            
        except Exception as e:
            logger.warning(f"Error generating prediction seed: {e}")
            return None
    
    def generate_fact_seed(
        self,
        user_msg: str,
        assistant_msg: str
    ) -> Optional[str]:
        """Generate a fact-based insight seed."""
        try:
            fact_indicators = [
                "is", "are", "has", "have", "contains", "includes",
                "means", "refers to", "defined as", "consists of"
            ]
            
            msg_low = assistant_msg.lower()
            has_fact = any(indicator in msg_low for indicator in fact_indicators)
            
            if not has_fact:
                return None
            
            seed = (
                f"Fact: '{user_msg[:60]}' → "
                f"'{assistant_msg[:100]}'"
            )
            
            return seed
            
        except Exception as e:
            logger.warning(f"Error generating fact seed: {e}")
            return None
    
    def generate_concept_seed(
        self,
        user_msg: str,
        assistant_msg: str,
        concept_name: Optional[str] = None
    ) -> str:
        """Generate a concept-based insight seed."""
        try:
            concept = concept_name or "emerging_concept"
            
            seed = (
                f"Concept ({concept}): "
                f"User query: '{user_msg[:60]}' | "
                f"Assistant explanation: '{assistant_msg[:100]}'"
            )
            
            return seed
            
        except Exception as e:
            logger.warning(f"Error generating concept seed: {e}")
            return f"Concept: {user_msg[:50]} → {assistant_msg[:50]}"
    
    def should_generate_seed(
        self,
        user_msg: str,
        assistant_msg: str,
        min_length: int = 20
    ) -> bool:
        """Determine if an insight seed should be generated."""
        try:
            if len(user_msg) < min_length or len(assistant_msg) < min_length:
                return False
            
            greetings = ["hi", "hello", "hey", "thanks", "thank you"]
            user_low = user_msg.lower().strip()
            
            if any(greeting in user_low for greeting in greetings) and len(user_low) < 30:
                return False
            
            return True
            
        except Exception:
            return True


# Global instance
insight_seed_engine = InsightSeedEngine()
