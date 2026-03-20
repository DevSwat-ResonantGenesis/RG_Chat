"""
Latent Intent Predictor (LIP)
================================

Patch #54: LLM understands the true meaning behind a user message, even if not explicitly stated.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/latent_intent_predictor.py
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class LatentIntentPredictor:
    """
    Latent Intent Predictor
    
    Extracts hidden intents, emotional states, and implicit requests
    from user messages.
    """
    
    INTENT_PATTERNS = {
        "seeking_explanation": [
            "why", "how", "what is", "explain", "tell me about",
            "can you explain", "what does", "how does"
        ],
        "request_debugging": [
            "fix", "error", "bug", "issue", "problem", "broken",
            "doesn't work", "not working", "failed", "crash"
        ],
        "emotional_state": [
            "don't trust", "angry", "mad", "frustrated", "annoyed",
            "worried", "scared", "confused", "lost", "stuck"
        ],
        "guidance_request": [
            "help me", "what should I do", "how should I", "what do you recommend",
            "advice", "suggest", "guide", "direction"
        ],
        "confirmation_seeking": [
            "is this correct", "am I right", "should I", "can I",
            "is it okay", "does this make sense"
        ],
        "comparison_request": [
            "compare", "difference", "better", "versus", "vs",
            "which is", "prefer", "choose"
        ],
        "planning_request": [
            "plan", "roadmap", "strategy", "steps", "approach",
            "how to proceed", "next steps"
        ],
        "validation_request": [
            "check", "verify", "validate", "test", "confirm",
            "make sure", "ensure"
        ]
    }
    
    EMOTIONAL_INDICATORS = {
        "frustration": ["can't", "won't", "doesn't work", "stuck", "frustrated"],
        "uncertainty": ["not sure", "confused", "don't know", "unclear", "unsure"],
        "urgency": ["asap", "urgent", "quickly", "fast", "immediately", "now"],
        "satisfaction": ["great", "perfect", "thanks", "awesome", "excellent"],
        "concern": ["worried", "concerned", "afraid", "scared", "anxious"]
    }
    
    def __init__(self):
        pass
    
    def predict(self, message: str) -> Dict[str, Any]:
        """Predict latent intents from user message."""
        if not message:
            return {
                "intents": ["general_query"],
                "emotions": [],
                "hidden_tasks": [],
                "confidence": 0.0
            }
        
        try:
            m = message.lower()
            intents = []
            emotions = []
            hidden_tasks = []
            
            for intent_name, patterns in self.INTENT_PATTERNS.items():
                for pattern in patterns:
                    if pattern in m:
                        intents.append(intent_name)
                        break
            
            for emotion_name, indicators in self.EMOTIONAL_INDICATORS.items():
                for indicator in indicators:
                    if indicator in m:
                        emotions.append(emotion_name)
                        break
            
            if "request_debugging" in intents:
                hidden_tasks.append("needs_step_by_step_guidance")
            if "emotional_state" in intents or "frustration" in emotions:
                hidden_tasks.append("needs_reassurance_and_stability")
            if "seeking_explanation" in intents:
                hidden_tasks.append("needs_clear_detailed_explanation")
            if "guidance_request" in intents:
                hidden_tasks.append("needs_actionable_advice")
            
            if not intents:
                intents.append("general_query")
            
            total_matches = len(intents) + len(emotions) + len(hidden_tasks)
            confidence = min(1.0, total_matches / 5.0)
            
            return {
                "intents": list(set(intents)),
                "emotions": list(set(emotions)),
                "hidden_tasks": list(set(hidden_tasks)),
                "confidence": confidence
            }
            
        except Exception as e:
            logger.warning(f"Error predicting latent intent: {e}")
            return {
                "intents": ["general_query"],
                "emotions": [],
                "hidden_tasks": [],
                "confidence": 0.0
            }
    
    def get_intent_description(self, intent: str) -> str:
        """Get human-readable description of an intent."""
        descriptions = {
            "seeking_explanation": "User wants explanation or understanding",
            "request_debugging": "User needs help fixing/debugging",
            "emotional_state": "User is expressing emotional state",
            "guidance_request": "User is asking for guidance or advice",
            "confirmation_seeking": "User wants confirmation or validation",
            "comparison_request": "User wants to compare options",
            "planning_request": "User wants planning or strategy",
            "validation_request": "User wants to verify or test something",
            "general_query": "General question or query"
        }
        return descriptions.get(intent, "Unknown intent")
    
    def format_for_prompt(self, prediction: Dict[str, Any]) -> str:
        """Format prediction for injection into system prompt."""
        parts = []
        
        intents = prediction.get("intents", [])
        if intents:
            intent_descriptions = [self.get_intent_description(i) for i in intents]
            parts.append(f"Latent intents: {', '.join(intents)} ({', '.join(intent_descriptions)})")
        
        emotions = prediction.get("emotions", [])
        if emotions:
            parts.append(f"Emotional signals: {', '.join(emotions)}")
        
        hidden_tasks = prediction.get("hidden_tasks", [])
        if hidden_tasks:
            parts.append(f"Hidden tasks: {', '.join(hidden_tasks)}")
        
        if not parts:
            return ""
        
        return ". ".join(parts) + ". Respond with awareness of these intents and emotional signals."
    
    def get_system_prompt(self, message: str) -> str:
        """Generate system prompt with latent intent context."""
        prediction = self.predict(message)
        if prediction["confidence"] > 0.2:
            return f"LATENT INTENT ANALYSIS:\n{self.format_for_prompt(prediction)}"
        return ""


# Global instance
latent_intent_predictor = LatentIntentPredictor()
