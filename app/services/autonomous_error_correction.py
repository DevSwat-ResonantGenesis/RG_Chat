"""
Autonomous Error Correction Module (AECM)
==========================================

Patch #48: Enables the AI to detect and correct errors in output.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/autonomous_error_correction.py
"""
from __future__ import annotations

import logging
from typing import Optional, Callable, Dict, Any, List

logger = logging.getLogger(__name__)


class AutonomousErrorCorrection:
    """
    Autonomous Error Correction Module
    
    Detects errors in AI output and triggers self-correction.
    """
    
    KEYWORDS_BAD = [
        "incorrect",
        "false",
        "not true",
        "doesn't make sense",
        "contradiction",
        "you said earlier",
        "that conflicts",
        "wrong",
        "mistake",
        "error",
        "sorry, i was wrong",
        "i apologize, that's not correct",
        "actually, let me correct",
        "i need to correct myself"
    ]
    
    KEYWORDS_CONTRADICTION = [
        "contradict",
        "conflict",
        "but earlier",
        "however, you said",
        "that doesn't match",
        "inconsistent"
    ]
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    def detect_error(self, message: str) -> bool:
        """Detect if AI output may be flawed."""
        if not message:
            return False
        
        try:
            msg_low = message.lower()
            
            for keyword in self.KEYWORDS_BAD:
                if keyword in msg_low:
                    logger.info(f"⚠️ Error detected: keyword '{keyword}' found in output")
                    return True
            
            for pattern in self.KEYWORDS_CONTRADICTION:
                if pattern in msg_low:
                    logger.info(f"⚠️ Contradiction detected: pattern '{pattern}' found in output")
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error in error detection: {e}")
            return False
    
    async def correct(
        self,
        llm_callable,  # async callable: (str) -> Dict[str, Any]
        user_input: str,
        last_output: str,
        context: Optional[List[Dict]] = None
    ) -> str:
        """Trigger correction cycle."""
        try:
            correction_prompt = (
                "Your previous answer may be incorrect or contradictory.\n"
                "Please provide a corrected and more accurate version.\n"
                "Be direct and factual. Do not repeat the user's question.\n"
                "\nUser message: " + user_input[:500] + "\n"
                "Previous answer: " + last_output[:500] + "\n"
                "\nProvide the corrected answer:"
            )
            
            if context:
                correction_prompt += "\n\nContext from conversation:\n"
                for msg in context[-3:]:
                    if isinstance(msg, dict):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")[:200]
                        correction_prompt += f"{role}: {content}\n"
            
            result = await llm_callable(correction_prompt)
            
            if isinstance(result, dict):
                corrected = result.get("response") or result.get("content") or ""
            else:
                corrected = str(result)
            
            if corrected and corrected != last_output:
                logger.info(f"✅ Error corrected: {len(corrected)} chars")
                return corrected
            else:
                logger.warning("Correction returned same or empty output")
                return last_output
                
        except Exception as e:
            logger.error(f"Error in correction cycle: {e}", exc_info=True)
            return last_output
    
    def should_suppress_memory(self, content: str) -> bool:
        """Determine if a memory should be suppressed (contains errors)."""
        if not content:
            return False
        
        try:
            content_low = content.lower()
            
            for keyword in self.KEYWORDS_BAD[:5]:
                if keyword in content_low:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def get_correction_confidence(self, original: str, corrected: str) -> float:
        """Calculate confidence that correction is better."""
        if not original or not corrected:
            return 0.5
        
        try:
            if corrected != original and len(corrected) > len(original) * 0.8:
                return 0.8
            elif corrected != original:
                return 0.6
            else:
                return 0.3
                
        except Exception:
            return 0.5


# Global instance
error_correction = AutonomousErrorCorrection()
