"""
Emotional Context Normalizer (ECN)
===================================

Patch #38: Makes the LLM emotionally consistent, aware of tone, and able to 
normalize emotional spikes in user input.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/emotional_normalizer.py
"""
from __future__ import annotations

import re
import logging
from collections import Counter
from typing import Dict, List

logger = logging.getLogger(__name__)


class EmotionalContextNormalizer:
    """
    Emotional Context Normalizer
    
    Detects and normalizes emotional context from user messages to help
    the LLM respond with appropriate emotional awareness and tone.
    """
    
    EMOTION_KEYWORDS: Dict[str, List[str]] = {
        "anger": ["angry", "mad", "fuck", "hate", "annoyed", "frustrated", "pissed", "irritated", "rage"],
        "stress": ["stress", "overwhelmed", "pressure", "panic", "stressed", "tension", "deadline"],
        "sadness": ["sad", "depressed", "hurt", "cry", "upset", "disappointed", "down", "unhappy"],
        "excitement": ["excited", "wow", "amazing", "can't believe", "awesome", "fantastic", "great", "love it"],
        "fear": ["scared", "afraid", "worried", "anxious", "nervous", "concerned", "fear", "doubt"]
    }
    
    def detect(self, text: str) -> str:
        """
        Detect the primary emotion in the given text.
        
        Returns:
            Detected emotion category (anger, stress, sadness, excitement, fear, or neutral)
        """
        if not text:
            return "neutral"
        
        scores = Counter()
        t = text.lower()
        
        for emotion, words in self.EMOTION_KEYWORDS.items():
            for word in words:
                pattern = r'\b' + re.escape(word) + r'\b'
                matches = len(re.findall(pattern, t))
                scores[emotion] += matches
        
        if not scores:
            return "neutral"
        
        return scores.most_common(1)[0][0]
    
    def normalize(self, emotion: str) -> str:
        """Normalize detected emotion into a context string for the LLM."""
        mapping: Dict[str, str] = {
            "anger": "The user is emotionally heated. Respond with calmness, understanding, and patience. Acknowledge their frustration without escalating.",
            "stress": "The user is under stress. Provide clear, actionable solutions. Be supportive and help reduce their cognitive load.",
            "sadness": "The user feels emotionally low. Respond with empathy, warmth, and encouragement. Focus on positive, constructive solutions.",
            "excitement": "The user is highly excited. Match their enthusiasm appropriately while maintaining professionalism and accuracy.",
            "fear": "The user feels uncertain or scared. Provide reassurance, clear information, and step-by-step guidance to build confidence.",
            "neutral": "No special emotional signal detected. Respond with standard professional tone."
        }
        return mapping.get(emotion, "No special emotional signal detected.")
    
    def get_emotional_anchor(self, emotion: str, text: str) -> Dict[str, any]:
        """Create an emotional anchor for memory storage."""
        return {
            "emotion": emotion,
            "normalized": self.normalize(emotion),
            "intensity": "high" if emotion != "neutral" else "low",
            "source": "emotional_normalizer"
        }
    
    def get_system_prompt(self, text: str) -> str:
        """Generate system prompt based on detected emotion."""
        emotion = self.detect(text)
        if emotion == "neutral":
            return ""
        return f"EMOTIONAL CONTEXT: {self.normalize(emotion)}"


# Global instance
emotional_normalizer = EmotionalContextNormalizer()
