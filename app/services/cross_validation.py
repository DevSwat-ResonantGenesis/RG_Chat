"""
Cross-Validation System (CVS)
==============================

Phase 5.15: Second agent verifies first agent's output.

Features:
- Automatic verification of agent responses
- Error detection and correction
- Confidence boosting through validation
- Hallucination flagging
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of cross-validation."""
    is_valid: bool
    confidence_boost: float  # -0.3 to +0.3
    issues_found: List[str]
    corrections: List[str]
    hallucination_flags: List[str]
    validator_agent: str
    validation_summary: str


class CrossValidationEngine:
    """
    Validates agent responses using a second agent.
    """
    
    def __init__(self, agent_engine=None):
        self.agent_engine = agent_engine
        
        # Validator agent preferences by primary agent type
        self.validator_map = {
            "code": "review",
            "debug": "test",
            "security": "review",
            "architecture": "review",
            "api": "security",
            "database": "review",
            "math": "reasoning",
            "explain": "research",
            "documentation": "review",
            "refactor": "test",
        }
        
        # Default validator
        self.default_validator = "review"
    
    def set_agent_engine(self, agent_engine):
        """Set the agent engine for spawning agents."""
        self.agent_engine = agent_engine
    
    def get_validator_for(self, agent_type: str) -> str:
        """Get the best validator agent for a given agent type."""
        return self.validator_map.get(agent_type, self.default_validator)
    
    async def validate(
        self,
        original_response: str,
        task: str,
        primary_agent: str,
        context: List[Dict[str, Any]] = None,
        preferred_provider: Optional[str] = None,
    ) -> ValidationResult:
        """Validate an agent response using a second agent."""
        if not self.agent_engine:
            return ValidationResult(
                is_valid=True,
                confidence_boost=0.0,
                issues_found=[],
                corrections=[],
                hallucination_flags=[],
                validator_agent="none",
                validation_summary="Validation skipped: no agent engine",
            )
        
        validator_agent = self.get_validator_for(primary_agent)
        
        logger.info(f"🔍 Cross-validating {primary_agent} response with {validator_agent}")
        
        # Build validation prompt
        validation_prompt = f"""You are validating another AI agent's response. Be critical but fair.

ORIGINAL TASK: {task}

RESPONSE TO VALIDATE (from {primary_agent} agent):
{original_response[:3000]}

Analyze this response for:
1. CORRECTNESS - Are there any factual errors or bugs?
2. COMPLETENESS - Does it fully address the task?
3. HALLUCINATIONS - Are there any made-up facts, fake libraries, or non-existent APIs?
4. QUALITY - Is it well-structured and clear?

Respond in this EXACT format:
VALID: [YES/NO]
ISSUES: [comma-separated list of issues, or "none"]
CORRECTIONS: [comma-separated list of corrections needed, or "none"]
HALLUCINATIONS: [comma-separated list of potential hallucinations, or "none"]
SUMMARY: [one sentence summary]"""

        try:
            result = await self.agent_engine.spawn(
                task=validation_prompt,
                context=context or [],
                agent_type=validator_agent,
                model=preferred_provider,
            )
            
            validation_response = result.get("content", "")
            
            # Parse validation response
            parsed = self._parse_validation(validation_response)
            
            return ValidationResult(
                is_valid=parsed["is_valid"],
                confidence_boost=parsed["confidence_boost"],
                issues_found=parsed["issues"],
                corrections=parsed["corrections"],
                hallucination_flags=parsed["hallucinations"],
                validator_agent=validator_agent,
                validation_summary=parsed["summary"],
            )
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return ValidationResult(
                is_valid=True,  # Assume valid if validation fails
                confidence_boost=0.0,
                issues_found=[],
                corrections=[],
                hallucination_flags=[],
                validator_agent=validator_agent,
                validation_summary=f"Validation error: {str(e)}",
            )
    
    def _parse_validation(self, response: str) -> Dict[str, Any]:
        """Parse validation response."""
        result = {
            "is_valid": True,
            "confidence_boost": 0.0,
            "issues": [],
            "corrections": [],
            "hallucinations": [],
            "summary": "",
        }
        
        response_upper = response.upper()
        
        # Parse VALID
        valid_match = re.search(r'VALID:\s*(YES|NO)', response_upper)
        if valid_match:
            result["is_valid"] = valid_match.group(1) == "YES"
        
        # Parse ISSUES
        issues_match = re.search(r'ISSUES:\s*(.+?)(?=\n[A-Z]+:|$)', response, re.IGNORECASE | re.DOTALL)
        if issues_match:
            issues_text = issues_match.group(1).strip()
            if issues_text.lower() != "none":
                result["issues"] = [i.strip() for i in issues_text.split(",") if i.strip()]
        
        # Parse CORRECTIONS
        corrections_match = re.search(r'CORRECTIONS:\s*(.+?)(?=\n[A-Z]+:|$)', response, re.IGNORECASE | re.DOTALL)
        if corrections_match:
            corrections_text = corrections_match.group(1).strip()
            if corrections_text.lower() != "none":
                result["corrections"] = [c.strip() for c in corrections_text.split(",") if c.strip()]
        
        # Parse HALLUCINATIONS
        halluc_match = re.search(r'HALLUCINATIONS:\s*(.+?)(?=\n[A-Z]+:|$)', response, re.IGNORECASE | re.DOTALL)
        if halluc_match:
            halluc_text = halluc_match.group(1).strip()
            if halluc_text.lower() != "none":
                result["hallucinations"] = [h.strip() for h in halluc_text.split(",") if h.strip()]
        
        # Parse SUMMARY
        summary_match = re.search(r'SUMMARY:\s*(.+?)(?=\n|$)', response, re.IGNORECASE)
        if summary_match:
            result["summary"] = summary_match.group(1).strip()
        
        # Calculate confidence boost
        if result["is_valid"] and not result["issues"] and not result["hallucinations"]:
            result["confidence_boost"] = 0.2  # Boost for clean validation
        elif result["hallucinations"]:
            result["confidence_boost"] = -0.3  # Penalty for hallucinations
        elif result["issues"]:
            result["confidence_boost"] = -0.1 * min(len(result["issues"]), 3)
        
        return result
    
    async def validate_and_correct(
        self,
        original_response: str,
        task: str,
        primary_agent: str,
        context: List[Dict[str, Any]] = None,
        preferred_provider: Optional[str] = None,
    ) -> Tuple[str, ValidationResult]:
        """Validate and optionally correct a response."""
        validation = await self.validate(
            original_response=original_response,
            task=task,
            primary_agent=primary_agent,
            context=context,
            preferred_provider=preferred_provider,
        )
        
        # If valid with no major issues, return original
        if validation.is_valid and not validation.hallucination_flags:
            return original_response, validation
        
        # If issues found, attempt correction
        if validation.corrections or validation.hallucination_flags:
            logger.info("🔧 Attempting to correct response based on validation")
            
            correction_prompt = f"""The following response has issues that need correction:

ORIGINAL TASK: {task}

ORIGINAL RESPONSE:
{original_response[:2000]}

ISSUES FOUND:
{chr(10).join(['- ' + i for i in validation.issues_found])}

CORRECTIONS NEEDED:
{chr(10).join(['- ' + c for c in validation.corrections])}

HALLUCINATIONS TO REMOVE:
{chr(10).join(['- ' + h for h in validation.hallucination_flags])}

Please provide a CORRECTED response that fixes these issues."""

            try:
                result = await self.agent_engine.spawn(
                    task=correction_prompt,
                    context=context or [],
                    agent_type=primary_agent,
                    model=preferred_provider,
                )
                
                corrected = result.get("content", "")
                if corrected:
                    return corrected, validation
            except Exception as e:
                logger.error(f"Correction failed: {e}")
        
        return original_response, validation


# Global instance
cross_validation = CrossValidationEngine()
