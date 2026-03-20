"""
Hash Sphere Neural Gravity Engine (NG-Engine)
==============================================

Patch #47: Introduces dynamic gravitational forces in the Hash Sphere.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/neural_gravity_engine.py
"""
from __future__ import annotations

import logging
import math
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class NeuralGravityEngine:
    """
    Neural Gravity Engine
    
    Computes gravitational forces between query points and anchors
    in the Hash Sphere, creating semantic physics-like behavior.
    """
    
    def compute_gravity(
        self,
        query_xyz: Tuple[float, float, float],
        anchor_xyz: Tuple[float, float, float],
        strength: float = 1.0
    ) -> float:
        """Compute gravitational pull between query and anchor."""
        try:
            dx = anchor_xyz[0] - query_xyz[0]
            dy = anchor_xyz[1] - query_xyz[1]
            dz = anchor_xyz[2] - query_xyz[2]
            
            distance = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6
            force = strength / (distance * distance)
            
            return force
            
        except Exception as e:
            logger.warning(f"Error computing gravity: {e}")
            return 0.0
    
    def apply_gravity(
        self,
        query_xyz: Tuple[float, float, float],
        anchors: List[Any],
        max_force: float = 1.0
    ) -> float:
        """Modify retrieval scores based on gravity forces."""
        try:
            total_force = 0.0
            
            for anchor in anchors:
                anchor_xyz = None
                if hasattr(anchor, 'xyz_x') and hasattr(anchor, 'xyz_y') and hasattr(anchor, 'xyz_z'):
                    anchor_xyz = (anchor.xyz_x, anchor.xyz_y, anchor.xyz_z)
                elif isinstance(anchor, dict):
                    anchor_xyz = (
                        anchor.get('xyz_x') or anchor.get('x') or 0.0,
                        anchor.get('xyz_y') or anchor.get('y') or 0.0,
                        anchor.get('xyz_z') or anchor.get('z') or 0.0
                    )
                elif isinstance(anchor, (list, tuple)) and len(anchor) >= 3:
                    anchor_xyz = (float(anchor[0]), float(anchor[1]), float(anchor[2]))
                
                if anchor_xyz is None:
                    continue
                
                importance_score = 1.0
                if hasattr(anchor, 'importance_score'):
                    importance_score = anchor.importance_score or 1.0
                elif isinstance(anchor, dict):
                    importance_score = anchor.get('importance_score') or anchor.get('score') or 1.0
                
                force = self.compute_gravity(
                    query_xyz,
                    anchor_xyz,
                    strength=importance_score
                )
                
                total_force += force
            
            return min(max_force, total_force)
            
        except Exception as e:
            logger.warning(f"Error applying gravity: {e}")
            return 0.0
    
    def compute_repulsion(
        self,
        query_xyz: Tuple[float, float, float],
        anchor_xyz: Tuple[float, float, float],
        strength: float = 1.0
    ) -> float:
        """Compute repulsive force (for contradictory anchors)."""
        try:
            dx = anchor_xyz[0] - query_xyz[0]
            dy = anchor_xyz[1] - query_xyz[1]
            dz = anchor_xyz[2] - query_xyz[2]
            
            distance = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6
            force = -strength / (distance * distance)
            
            return force
            
        except Exception as e:
            logger.warning(f"Error computing repulsion: {e}")
            return 0.0
    
    def get_gravity_wells(
        self,
        anchors: List[Any],
        min_strength: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Identify gravity wells (clusters of high-importance anchors)."""
        try:
            wells = []
            
            for anchor in anchors:
                importance = 0.0
                if hasattr(anchor, 'importance_score'):
                    importance = anchor.importance_score or 0.0
                elif isinstance(anchor, dict):
                    importance = anchor.get('importance_score') or anchor.get('score') or 0.0
                
                if importance >= min_strength:
                    xyz = None
                    if hasattr(anchor, 'xyz_x'):
                        xyz = (anchor.xyz_x, anchor.xyz_y, anchor.xyz_z)
                    elif isinstance(anchor, dict):
                        xyz = (
                            anchor.get('xyz_x') or anchor.get('x') or 0.0,
                            anchor.get('xyz_y') or anchor.get('y') or 0.0,
                            anchor.get('xyz_z') or anchor.get('z') or 0.0
                        )
                    
                    if xyz:
                        wells.append({
                            "xyz": xyz,
                            "strength": importance,
                            "anchor": anchor
                        })
            
            return wells
            
        except Exception as e:
            logger.warning(f"Error getting gravity wells: {e}")
            return []


# Global instance
neural_gravity_engine = NeuralGravityEngine()
