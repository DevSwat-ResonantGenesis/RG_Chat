"""
Agent Confidence Scoring System (ACSS)
=======================================

Phase 5.4: Calculate and display confidence levels in agent responses.

Features:
- Analyze response quality indicators
- Calculate confidence scores
- Threshold-based escalation to debate
- Confidence display in responses
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    """Result of confidence analysis."""
    score: float  # 0.0 to 1.0
    level: str  # 'low', 'medium', 'high', 'very_high'
    factors: Dict[str, float]  # Individual factor scores
    should_escalate: bool  # Whether to escalate to debate
    explanation: str


class AgentConfidenceAnalyzer:
    """
    Analyzes agent responses to calculate confidence scores.
    """
    
    def __init__(self, escalation_threshold: float = 0.4):
        self.escalation_threshold = escalation_threshold
        
        # Confidence indicators (positive)
        self.high_confidence_markers = [
            "definitely", "certainly", "clearly", "obviously",
            "the answer is", "the solution is", "here's how",
            "you should", "the best approach", "recommended",
        ]
        
        # Uncertainty indicators (negative)
        self.low_confidence_markers = [
            "i'm not sure", "i don't know", "might be", "could be",
            "possibly", "perhaps", "maybe", "it depends",
            "i think", "i believe", "it seems", "arguably",
            "one option", "you could try", "not certain",
        ]
        
        # Quality indicators (positive)
        self.quality_markers = [
            "```",  # Code blocks
            "1.", "2.", "3.",  # Numbered lists
            "- ",  # Bullet points
            "example:", "for instance",  # Examples
            "because", "therefore", "since",  # Reasoning
        ]
        
        # Vagueness indicators (negative)
        self.vague_markers = [
            "etc", "and so on", "things like that",
            "something like", "kind of", "sort of",
            "basically", "essentially", "generally",
        ]
    
    def analyze(self, response: str, task: str = "") -> ConfidenceResult:
        """Analyze a response and calculate confidence score."""
        if not response:
            return ConfidenceResult(
                score=0.0,
                level="low",
                factors={},
                should_escalate=True,
                explanation="Empty response"
            )
        
        response_lower = response.lower()
        factors = {}
        
        # Factor 1: Certainty language (0-1)
        high_count = sum(1 for m in self.high_confidence_markers if m in response_lower)
        low_count = sum(1 for m in self.low_confidence_markers if m in response_lower)
        certainty_score = min(1.0, max(0.0, 0.5 + (high_count * 0.1) - (low_count * 0.15)))
        factors["certainty"] = certainty_score
        
        # Factor 2: Response quality (0-1)
        quality_count = sum(1 for m in self.quality_markers if m in response_lower)
        quality_score = min(1.0, quality_count * 0.15 + 0.3)
        factors["quality"] = quality_score
        
        # Factor 3: Specificity (0-1)
        vague_count = sum(1 for m in self.vague_markers if m in response_lower)
        specificity_score = max(0.0, 1.0 - (vague_count * 0.15))
        factors["specificity"] = specificity_score
        
        # Factor 4: Response length appropriateness (0-1)
        length = len(response)
        if length < 50:
            length_score = 0.3  # Too short
        elif length < 200:
            length_score = 0.6  # Brief
        elif length < 1000:
            length_score = 0.9  # Good length
        elif length < 3000:
            length_score = 0.8  # Detailed
        else:
            length_score = 0.6  # Possibly too verbose
        factors["length"] = length_score
        
        # Factor 5: Structure (0-1)
        has_code = "```" in response
        has_lists = bool(re.search(r'^\s*[-*\d]+[.)]\s', response, re.MULTILINE))
        has_headers = bool(re.search(r'^#+\s', response, re.MULTILINE))
        structure_score = 0.4 + (0.2 if has_code else 0) + (0.2 if has_lists else 0) + (0.2 if has_headers else 0)
        factors["structure"] = structure_score
        
        # Factor 6: Task relevance (0-1)
        if task:
            task_words = set(task.lower().split())
            response_words = set(response_lower.split())
            overlap = len(task_words & response_words)
            relevance_score = min(1.0, overlap / max(len(task_words), 1) + 0.3)
        else:
            relevance_score = 0.7  # Default if no task provided
        factors["relevance"] = relevance_score
        
        # Calculate weighted average
        weights = {
            "certainty": 0.20,
            "quality": 0.20,
            "specificity": 0.15,
            "length": 0.10,
            "structure": 0.15,
            "relevance": 0.20,
        }
        
        total_score = sum(factors[k] * weights[k] for k in factors)
        
        # Determine level
        if total_score >= 0.8:
            level = "very_high"
        elif total_score >= 0.6:
            level = "high"
        elif total_score >= 0.4:
            level = "medium"
        else:
            level = "low"
        
        # Determine if escalation needed
        should_escalate = total_score < self.escalation_threshold
        
        # Generate explanation
        explanation = self._generate_explanation(factors, total_score, level)
        
        return ConfidenceResult(
            score=total_score,
            level=level,
            factors=factors,
            should_escalate=should_escalate,
            explanation=explanation,
        )
    
    def _generate_explanation(
        self,
        factors: Dict[str, float],
        score: float,
        level: str,
    ) -> str:
        """Generate human-readable explanation of confidence."""
        # Find strongest and weakest factors
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_factors[0]
        weakest = sorted_factors[-1]
        
        explanations = {
            "very_high": f"High confidence ({score:.0%}). Strong {strongest[0]}.",
            "high": f"Good confidence ({score:.0%}). {strongest[0].title()} is solid.",
            "medium": f"Moderate confidence ({score:.0%}). {weakest[0].title()} could be improved.",
            "low": f"Low confidence ({score:.0%}). Consider verifying with additional sources.",
        }
        
        return explanations.get(level, f"Confidence: {score:.0%}")
    
    def get_confidence_badge(self, score: float) -> str:
        """Get a visual badge for confidence level."""
        if score >= 0.8:
            return "🟢 Very High"
        elif score >= 0.6:
            return "🟡 High"
        elif score >= 0.4:
            return "🟠 Medium"
        else:
            return "🔴 Low"
    
    def should_use_debate(self, score: float) -> bool:
        """Determine if debate should be used based on confidence."""
        return score < self.escalation_threshold


# Global instance
confidence_analyzer = AgentConfidenceAnalyzer()
