"""
Unit Tests for Evidence Graph Math Layers
=========================================

Tests for the mathematical functions integrated in Week 2:
- Layer 7: Evidence Aggregation (aggregate_evidence)
- Layer 8: Consistency Check (calculate_consistency)

Run with: pytest chat_service/tests/test_evidence_graph_math.py -v
"""

import pytest
import numpy as np
from typing import Dict, List, Any

# Import the functions we're testing
import sys
sys.path.insert(0, '/Users/devswat/resonantgenesis_backend/chat_service')

from app.services.evidence_graph import EvidenceGraph, evidence_graph


class TestEvidenceAggregation:
    """Tests for Layer 7: Evidence Aggregation - E* = Σ w_i · s_i"""
    
    def test_aggregation_output_is_unit_vector(self):
        """Verify aggregated evidence has unit norm: ||Ê*|| = 1"""
        eg = EvidenceGraph()
        
        memories = [
            {"xyz": [0.5, 0.3, 0.2], "combined_score": 0.8},
            {"xyz": [0.1, 0.7, 0.4], "combined_score": 0.6},
            {"xyz": [-0.3, 0.2, 0.9], "combined_score": 0.9},
        ]
        
        evidence_vector, total_weight = eg.aggregate_evidence(memories)
        
        # Check unit norm
        norm = np.linalg.norm(evidence_vector)
        assert abs(norm - 1.0) < 1e-6, f"Evidence vector norm should be 1.0, got {norm}"
    
    def test_aggregation_weights_memories(self):
        """Verify higher-weighted memories have more influence"""
        eg = EvidenceGraph()
        
        # Memory 1 has high weight, Memory 2 has low weight
        memories = [
            {"xyz": [1.0, 0.0, 0.0], "combined_score": 0.99},
            {"xyz": [0.0, 1.0, 0.0], "combined_score": 0.01},
        ]
        
        evidence_vector, _ = eg.aggregate_evidence(memories)
        
        # Evidence should be closer to [1, 0, 0] direction
        assert evidence_vector[0] > evidence_vector[1], "High-weight memory should dominate"
    
    def test_aggregation_returns_total_weight(self):
        """Verify total weight is sum of all memory weights"""
        eg = EvidenceGraph()
        
        memories = [
            {"xyz": [0.5, 0.3, 0.2], "combined_score": 0.8},
            {"xyz": [0.1, 0.7, 0.4], "combined_score": 0.6},
        ]
        
        _, total_weight = eg.aggregate_evidence(memories)
        
        expected = 0.8 + 0.6
        assert abs(total_weight - expected) < 1e-6, f"Total weight should be {expected}, got {total_weight}"
    
    def test_aggregation_handles_empty_list(self):
        """Verify graceful handling of empty memory list"""
        eg = EvidenceGraph()
        
        evidence_vector, total_weight = eg.aggregate_evidence([])
        
        assert total_weight == 0.0, "Total weight should be 0 for empty list"
        assert np.allclose(evidence_vector, np.zeros(3)), "Evidence should be zero vector"
    
    def test_aggregation_handles_missing_xyz(self):
        """Verify memories without xyz are skipped"""
        eg = EvidenceGraph()
        
        memories = [
            {"xyz": [0.5, 0.3, 0.2], "combined_score": 0.8},
            {"content": "no xyz", "combined_score": 0.9},  # Missing xyz
            {"xyz": None, "combined_score": 0.7},  # None xyz
        ]
        
        evidence_vector, total_weight = eg.aggregate_evidence(memories)
        
        # Only first memory should contribute
        assert abs(total_weight - 0.8) < 1e-6, "Only memory with xyz should contribute"
    
    def test_aggregation_custom_weight_key(self):
        """Verify custom weight key works"""
        eg = EvidenceGraph()
        
        memories = [
            {"xyz": [1.0, 0.0, 0.0], "custom_score": 0.9, "combined_score": 0.1},
            {"xyz": [0.0, 1.0, 0.0], "custom_score": 0.1, "combined_score": 0.9},
        ]
        
        # Using custom_score as weight
        evidence_vector, _ = eg.aggregate_evidence(memories, weight_key="custom_score")
        
        # With custom_score, first memory dominates
        assert evidence_vector[0] > evidence_vector[1], "Custom weight key should be used"


class TestConsistencyCheck:
    """Tests for Layer 8: Consistency Check - C_k = cos(o_k, Ê*)"""
    
    def test_consistency_identical_vectors(self):
        """Verify consistency is 1.0 for identical directions"""
        eg = EvidenceGraph()
        
        response_xyz = (0.5, 0.5, 0.5)
        evidence_vector = np.array([0.5, 0.5, 0.5])
        evidence_vector = evidence_vector / np.linalg.norm(evidence_vector)
        
        consistency = eg.calculate_consistency(response_xyz, evidence_vector)
        
        assert abs(consistency - 1.0) < 1e-6, f"Identical vectors should have consistency 1.0, got {consistency}"
    
    def test_consistency_opposite_vectors(self):
        """Verify consistency is 0.0 for opposite directions"""
        eg = EvidenceGraph()
        
        response_xyz = (1.0, 0.0, 0.0)
        evidence_vector = np.array([-1.0, 0.0, 0.0])
        
        consistency = eg.calculate_consistency(response_xyz, evidence_vector)
        
        # Opposite vectors have cosine -1, mapped to 0
        assert consistency == 0.0, f"Opposite vectors should have consistency 0.0, got {consistency}"
    
    def test_consistency_orthogonal_vectors(self):
        """Verify consistency is 0.5 for orthogonal vectors"""
        eg = EvidenceGraph()
        
        response_xyz = (1.0, 0.0, 0.0)
        evidence_vector = np.array([0.0, 1.0, 0.0])
        
        consistency = eg.calculate_consistency(response_xyz, evidence_vector)
        
        # Orthogonal vectors have cosine 0, mapped to 0.5
        assert abs(consistency - 0.5) < 1e-6, f"Orthogonal vectors should have consistency 0.5, got {consistency}"
    
    def test_consistency_range(self):
        """Verify consistency is always in [0, 1]"""
        eg = EvidenceGraph()
        
        for _ in range(100):
            response_xyz = tuple(np.random.randn(3))
            evidence_vector = np.random.randn(3)
            evidence_vector = evidence_vector / np.linalg.norm(evidence_vector)
            
            consistency = eg.calculate_consistency(response_xyz, evidence_vector)
            
            assert 0.0 <= consistency <= 1.0, f"Consistency {consistency} should be in [0, 1]"
    
    def test_consistency_handles_none_inputs(self):
        """Verify graceful handling of None inputs"""
        eg = EvidenceGraph()
        
        # None response
        consistency = eg.calculate_consistency(None, np.array([1, 0, 0]))
        assert consistency == 0.0, "None response should return 0.0"
        
        # None evidence
        consistency = eg.calculate_consistency((1, 0, 0), None)
        assert consistency == 0.0, "None evidence should return 0.0"


class TestEvidenceGraphIntegration:
    """Integration tests for evidence graph with build_graph"""
    
    def test_build_graph_with_memories(self):
        """Verify build_graph works with memory data"""
        eg = EvidenceGraph()
        
        memories = [
            {"content": "Test memory 1", "combined_score": 0.8, "type": "chat"},
            {"content": "Test memory 2", "combined_score": 0.6, "type": "document"},
        ]
        
        graph = eg.build_graph(
            user_hash="abc123",
            assistant_hash="def456",
            memories=memories,
            provider="test_provider"
        )
        
        assert "timestamp" in graph
        assert "provider" in graph
        assert graph["provider"] == "test_provider"
        assert "memory_contributors" in graph
        assert len(graph["memory_contributors"]) == 2
    
    def test_get_evidence_summary(self):
        """Verify evidence summary generation"""
        eg = EvidenceGraph()
        
        graph = {
            "provider": "openai",
            "resonance": {"between_user_and_assistant": 0.85},
            "meta": {"memory_count": 5},
            "reasoning": {"intents": ["question", "clarification"], "emotion": "curious"}
        }
        
        summary = eg.get_evidence_summary(graph)
        
        assert "openai" in summary
        assert "0.85" in summary or "0.850" in summary
        assert "5" in summary


class TestGlobalInstance:
    """Tests for the global evidence_graph instance"""
    
    def test_global_instance_exists(self):
        """Verify global instance is available"""
        assert evidence_graph is not None
        assert isinstance(evidence_graph, EvidenceGraph)
    
    def test_global_instance_has_new_methods(self):
        """Verify global instance has the new Layer 7 & 8 methods"""
        assert hasattr(evidence_graph, 'aggregate_evidence')
        assert hasattr(evidence_graph, 'calculate_consistency')
        assert callable(evidence_graph.aggregate_evidence)
        assert callable(evidence_graph.calculate_consistency)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
