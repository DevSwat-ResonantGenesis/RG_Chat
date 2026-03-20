"""
Output Correction Service (Layer 9)
===================================

Implements evidence-based output correction:
o_corrected = λ·o_k* + (1-λ)·Ê*

This blends the LLM response with aggregated evidence from memories
to reduce hallucinations and improve consistency.

Part of the Hash Sphere 9-Layer Architecture.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OutputCorrectionService:
    """
    Layer 9: Output Correction
    
    Corrects LLM output by blending with evidence vector.
    
    Mathematical definition:
    o_corrected = λ·o_k* + (1-λ)·Ê*
    
    Where:
    - λ ∈ [0,1] controls model vs evidence weight (default 0.85)
    - o_k* = normalized response position in sphere
    - Ê* = normalized evidence vector from Layer 7
    """
    
    def __init__(self, lambda_weight: float = 0.85):
        """
        Initialize output correction service.
        
        Args:
            lambda_weight: Weight for model output (default 0.85 = 85% model, 15% evidence)
        """
        self.lambda_weight = lambda_weight
    
    def correct_output_vector(
        self,
        response_xyz: Tuple[float, float, float],
        evidence_vector: np.ndarray,
        lambda_weight: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Correct response position using evidence.
        
        Mathematical definition (Layer 9):
        o_corrected = λ·o_k* + (1-λ)·Ê*
        
        Args:
            response_xyz: XYZ coordinates of response
            evidence_vector: Normalized evidence vector from Layer 7
            lambda_weight: Optional override for lambda (default: self.lambda_weight)
        
        Returns:
            Corrected XYZ coordinates
        """
        if response_xyz is None or evidence_vector is None:
            return response_xyz
        
        lw = lambda_weight if lambda_weight is not None else self.lambda_weight
        
        # Convert to numpy
        response_vec = np.array(response_xyz, dtype=np.float64)
        evidence_vec = np.array(evidence_vector, dtype=np.float64)
        
        # Ensure 3D vectors (take first 3 components if larger)
        if len(evidence_vec) > 3:
            evidence_vec = evidence_vec[:3]
        elif len(evidence_vec) < 3:
            # Pad with zeros if smaller
            evidence_vec = np.pad(evidence_vec, (0, 3 - len(evidence_vec)))
        
        # Normalize response vector
        response_norm = np.linalg.norm(response_vec)
        if response_norm > 0:
            response_vec = response_vec / response_norm
        
        # Ensure evidence is normalized (should already be from Layer 7)
        evidence_norm = np.linalg.norm(evidence_vec)
        if evidence_norm > 0:
            evidence_vec = evidence_vec / evidence_norm
        
        # Apply correction: o_corrected = λ·o_k* + (1-λ)·Ê*
        corrected = lw * response_vec + (1 - lw) * evidence_vec
        
        # Normalize result to stay on unit sphere
        corrected_norm = np.linalg.norm(corrected)
        if corrected_norm > 0:
            corrected = corrected / corrected_norm
        
        return tuple(corrected.tolist())
    
    def should_apply_correction(
        self,
        evidence_weight: float,
        evidence_consistency: float,
        min_evidence_weight: float = 0.5,
        max_consistency: float = 0.8
    ) -> bool:
        """
        Determine if output correction should be applied.
        
        Only apply correction when:
        1. We have enough evidence (weight > threshold)
        2. Response is not already highly consistent with evidence
        
        Args:
            evidence_weight: Total weight from evidence aggregation
            evidence_consistency: Consistency score from Layer 8
            min_evidence_weight: Minimum evidence weight to apply correction
            max_consistency: If consistency is above this, skip correction
        
        Returns:
            True if correction should be applied
        """
        # Don't correct if not enough evidence
        if evidence_weight < min_evidence_weight:
            logger.debug(f"Skipping correction: evidence_weight={evidence_weight:.3f} < {min_evidence_weight}")
            return False
        
        # Don't correct if already highly consistent
        if evidence_consistency > max_consistency:
            logger.debug(f"Skipping correction: already consistent ({evidence_consistency:.3f})")
            return False
        
        return True
    
    def calculate_correction_strength(
        self,
        evidence_consistency: float,
        evidence_weight: float
    ) -> float:
        """
        Calculate dynamic lambda based on evidence quality.
        
        Higher evidence weight + lower consistency = more correction needed
        
        Args:
            evidence_consistency: How consistent response is with evidence (0-1)
            evidence_weight: Total evidence weight
        
        Returns:
            Adjusted lambda weight (0.7 to 0.95)
        """
        # Base lambda
        base_lambda = self.lambda_weight
        
        # If response is inconsistent with strong evidence, apply more correction
        if evidence_weight > 1.0 and evidence_consistency < 0.5:
            # More correction needed (lower lambda = more evidence influence)
            adjusted = base_lambda - 0.1 * (1 - evidence_consistency)
            return max(0.7, adjusted)
        
        # If response is somewhat consistent, apply less correction
        if evidence_consistency > 0.6:
            adjusted = base_lambda + 0.05 * evidence_consistency
            return min(0.95, adjusted)
        
        return base_lambda
    
    def apply_correction(
        self,
        response_text: str,
        response_xyz: Tuple[float, float, float],
        evidence_vector: np.ndarray,
        evidence_weight: float,
        evidence_consistency: float,
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Full output correction pipeline.
        
        Args:
            response_text: Original LLM response
            response_xyz: Response XYZ coordinates
            evidence_vector: Aggregated evidence vector
            evidence_weight: Total evidence weight
            evidence_consistency: Consistency score
            memories: List of memories used for context
        
        Returns:
            Dictionary with correction results
        """
        result = {
            "corrected": False,
            "original_xyz": response_xyz,
            "corrected_xyz": response_xyz,
            "lambda_used": self.lambda_weight,
            "correction_applied": False,
            "reason": None,
            "evidence_weight": evidence_weight,
            "evidence_consistency": evidence_consistency
        }
        
        # Check if correction should be applied
        if not self.should_apply_correction(evidence_weight, evidence_consistency):
            result["reason"] = "Correction not needed"
            return result
        
        # Calculate dynamic lambda
        dynamic_lambda = self.calculate_correction_strength(
            evidence_consistency, evidence_weight
        )
        result["lambda_used"] = dynamic_lambda
        
        # Apply correction
        corrected_xyz = self.correct_output_vector(
            response_xyz, evidence_vector, dynamic_lambda
        )
        
        if corrected_xyz and corrected_xyz != response_xyz:
            result["corrected"] = True
            result["corrected_xyz"] = corrected_xyz
            result["correction_applied"] = True
            result["reason"] = f"Applied correction with λ={dynamic_lambda:.3f}"
            logger.info(f"📐 Layer 9 Output Correction: λ={dynamic_lambda:.3f}, "
                       f"original={response_xyz}, corrected={corrected_xyz}")
        else:
            result["reason"] = "No correction needed after calculation"
        
        return result


# Global instance
output_correction = OutputCorrectionService(lambda_weight=0.85)
