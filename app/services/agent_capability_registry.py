"""
Agent Capability Registry
==========================

Phase 1 of Agent Autonomy Enhancement.

Tracks capabilities, strengths, weaknesses, and performance metrics for all agents.
Enables intelligent agent selection based on task requirements and agent capabilities.

Author: Resonant Chat Systems Team
Date: December 26, 2025
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Capability profile for an agent."""
    agent_type: str
    strengths: List[str]
    weaknesses: List[str]
    success_rate: float  # 0.0 - 1.0
    avg_response_time_ms: float
    current_workload: int
    specializations: Dict[str, float]  # topic → confidence score (0.0 - 1.0)
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    last_updated: Optional[datetime] = None


# Agent capability definitions
AGENT_CAPABILITIES: Dict[str, AgentCapability] = {
    "reasoning": AgentCapability(
        agent_type="reasoning",
        strengths=["analysis", "logic", "problem_solving", "critical_thinking", "deduction"],
        weaknesses=["creative_writing", "visual_design", "artistic_tasks"],
        success_rate=0.92,
        avg_response_time_ms=1200,
        current_workload=0,
        specializations={
            "code_analysis": 0.95,
            "system_design": 0.88,
            "debugging": 0.90,
            "logical_reasoning": 0.96,
            "problem_decomposition": 0.93,
        }
    ),
    "explain": AgentCapability(
        agent_type="explain",
        strengths=["simplification", "teaching", "clarity", "beginner_friendly", "eli5"],
        weaknesses=["advanced_math", "low_level_systems", "complex_algorithms"],
        success_rate=0.89,
        avg_response_time_ms=1000,
        current_workload=0,
        specializations={
            "beginner_tutorials": 0.98,
            "concept_explanation": 0.94,
            "eli5": 0.96,
            "teaching": 0.92,
            "simplification": 0.95,
        }
    ),
    "code": AgentCapability(
        agent_type="code",
        strengths=["code_generation", "implementation", "syntax", "best_practices"],
        weaknesses=["theoretical_concepts", "high_level_design"],
        success_rate=0.87,
        avg_response_time_ms=1400,
        current_workload=0,
        specializations={
            "python": 0.95,
            "javascript": 0.93,
            "typescript": 0.92,
            "react": 0.90,
            "code_generation": 0.94,
        }
    ),
    "debug": AgentCapability(
        agent_type="debug",
        strengths=["bug_finding", "error_analysis", "troubleshooting", "root_cause"],
        weaknesses=["feature_development", "design"],
        success_rate=0.85,
        avg_response_time_ms=1600,
        current_workload=0,
        specializations={
            "error_analysis": 0.94,
            "bug_fixing": 0.92,
            "troubleshooting": 0.90,
            "stack_trace_analysis": 0.93,
            "debugging": 0.95,
        }
    ),
    "review": AgentCapability(
        agent_type="review",
        strengths=["code_review", "quality_assessment", "feedback", "best_practices"],
        weaknesses=["implementation", "quick_fixes"],
        success_rate=0.88,
        avg_response_time_ms=1300,
        current_workload=0,
        specializations={
            "code_review": 0.96,
            "quality_assessment": 0.93,
            "best_practices": 0.94,
            "security_review": 0.88,
            "performance_review": 0.90,
        }
    ),
    "test": AgentCapability(
        agent_type="test",
        strengths=["test_generation", "coverage", "edge_cases", "test_strategy"],
        weaknesses=["implementation", "design"],
        success_rate=0.86,
        avg_response_time_ms=1500,
        current_workload=0,
        specializations={
            "unit_tests": 0.95,
            "integration_tests": 0.90,
            "test_coverage": 0.93,
            "edge_cases": 0.92,
            "test_strategy": 0.89,
        }
    ),
    "refactor": AgentCapability(
        agent_type="refactor",
        strengths=["code_restructuring", "clean_code", "design_patterns", "optimization"],
        weaknesses=["new_features", "quick_fixes"],
        success_rate=0.84,
        avg_response_time_ms=1700,
        current_workload=0,
        specializations={
            "code_restructuring": 0.94,
            "clean_code": 0.92,
            "design_patterns": 0.90,
            "refactoring": 0.95,
            "code_organization": 0.91,
        }
    ),
    "security": AgentCapability(
        agent_type="security",
        strengths=["vulnerability_detection", "security_analysis", "threat_modeling", "encryption"],
        weaknesses=["ui_design", "user_experience"],
        success_rate=0.90,
        avg_response_time_ms=1800,
        current_workload=0,
        specializations={
            "vulnerability_detection": 0.96,
            "security_analysis": 0.94,
            "threat_modeling": 0.91,
            "encryption": 0.89,
            "security_best_practices": 0.93,
        }
    ),
    "architecture": AgentCapability(
        agent_type="architecture",
        strengths=["system_design", "scalability", "patterns", "high_level_design"],
        weaknesses=["implementation_details", "debugging"],
        success_rate=0.88,
        avg_response_time_ms=1600,
        current_workload=0,
        specializations={
            "system_design": 0.95,
            "scalability": 0.93,
            "design_patterns": 0.94,
            "microservices": 0.90,
            "architecture": 0.96,
        }
    ),
    "math": AgentCapability(
        agent_type="math",
        strengths=["calculations", "equations", "formulas", "mathematical_reasoning"],
        weaknesses=["code_generation", "design"],
        success_rate=0.93,
        avg_response_time_ms=1100,
        current_workload=0,
        specializations={
            "calculations": 0.98,
            "algebra": 0.95,
            "calculus": 0.93,
            "statistics": 0.92,
            "mathematical_reasoning": 0.94,
        }
    ),
    "research": AgentCapability(
        agent_type="research",
        strengths=["information_gathering", "analysis", "comparison", "investigation"],
        weaknesses=["implementation", "quick_answers"],
        success_rate=0.87,
        avg_response_time_ms=2000,
        current_workload=0,
        specializations={
            "information_gathering": 0.94,
            "research": 0.95,
            "comparison": 0.92,
            "investigation": 0.90,
            "analysis": 0.91,
        }
    ),
    "summary": AgentCapability(
        agent_type="summary",
        strengths=["condensing", "key_points", "brevity", "clarity"],
        weaknesses=["detailed_analysis", "implementation"],
        success_rate=0.90,
        avg_response_time_ms=900,
        current_workload=0,
        specializations={
            "summarization": 0.96,
            "key_points": 0.94,
            "condensing": 0.95,
            "brevity": 0.93,
            "clarity": 0.92,
        }
    ),
    "planning": AgentCapability(
        agent_type="planning",
        strengths=["strategy", "roadmaps", "step_by_step", "organization"],
        weaknesses=["implementation", "debugging"],
        success_rate=0.86,
        avg_response_time_ms=1400,
        current_workload=0,
        specializations={
            "strategy": 0.93,
            "roadmaps": 0.92,
            "planning": 0.95,
            "organization": 0.90,
            "step_by_step": 0.94,
        }
    ),
    "optimization": AgentCapability(
        agent_type="optimization",
        strengths=["performance", "efficiency", "bottlenecks", "speed"],
        weaknesses=["feature_development", "design"],
        success_rate=0.85,
        avg_response_time_ms=1700,
        current_workload=0,
        specializations={
            "performance": 0.95,
            "optimization": 0.96,
            "bottlenecks": 0.93,
            "efficiency": 0.94,
            "speed": 0.92,
        }
    ),
    "documentation": AgentCapability(
        agent_type="documentation",
        strengths=["docs_generation", "clarity", "examples", "api_docs"],
        weaknesses=["implementation", "debugging"],
        success_rate=0.88,
        avg_response_time_ms=1200,
        current_workload=0,
        specializations={
            "documentation": 0.96,
            "api_docs": 0.94,
            "examples": 0.92,
            "clarity": 0.93,
            "technical_writing": 0.95,
        }
    ),
    "migration": AgentCapability(
        agent_type="migration",
        strengths=["upgrades", "transitions", "compatibility", "conversion"],
        weaknesses=["new_features", "design"],
        success_rate=0.83,
        avg_response_time_ms=1900,
        current_workload=0,
        specializations={
            "upgrades": 0.92,
            "migration": 0.94,
            "compatibility": 0.90,
            "conversion": 0.91,
            "transitions": 0.89,
        }
    ),
    "api": AgentCapability(
        agent_type="api",
        strengths=["api_design", "endpoints", "rest", "graphql"],
        weaknesses=["ui_design", "frontend"],
        success_rate=0.87,
        avg_response_time_ms=1500,
        current_workload=0,
        specializations={
            "api_design": 0.95,
            "rest": 0.94,
            "graphql": 0.90,
            "endpoints": 0.93,
            "api_documentation": 0.92,
        }
    ),
    "database": AgentCapability(
        agent_type="database",
        strengths=["sql", "queries", "schema_design", "optimization"],
        weaknesses=["frontend", "ui"],
        success_rate=0.86,
        avg_response_time_ms=1600,
        current_workload=0,
        specializations={
            "sql": 0.96,
            "database_design": 0.93,
            "queries": 0.95,
            "schema": 0.92,
            "optimization": 0.90,
        }
    ),
    "devops": AgentCapability(
        agent_type="devops",
        strengths=["deployment", "ci_cd", "docker", "kubernetes"],
        weaknesses=["frontend", "ui_design"],
        success_rate=0.84,
        avg_response_time_ms=1800,
        current_workload=0,
        specializations={
            "deployment": 0.94,
            "ci_cd": 0.93,
            "docker": 0.95,
            "kubernetes": 0.90,
            "devops": 0.92,
        }
    ),
    "accessibility": AgentCapability(
        agent_type="accessibility",
        strengths=["a11y", "wcag", "aria", "screen_readers"],
        weaknesses=["backend", "databases"],
        success_rate=0.87,
        avg_response_time_ms=1400,
        current_workload=0,
        specializations={
            "accessibility": 0.96,
            "wcag": 0.94,
            "aria": 0.93,
            "screen_readers": 0.92,
            "a11y": 0.95,
        }
    ),
    "i18n": AgentCapability(
        agent_type="i18n",
        strengths=["internationalization", "localization", "translation", "rtl"],
        weaknesses=["backend_logic", "databases"],
        success_rate=0.85,
        avg_response_time_ms=1500,
        current_workload=0,
        specializations={
            "internationalization": 0.95,
            "localization": 0.94,
            "translation": 0.92,
            "i18n": 0.96,
            "rtl": 0.90,
        }
    ),
    "regex": AgentCapability(
        agent_type="regex",
        strengths=["pattern_matching", "regular_expressions", "text_processing"],
        weaknesses=["ui_design", "architecture"],
        success_rate=0.89,
        avg_response_time_ms=1100,
        current_workload=0,
        specializations={
            "regex": 0.97,
            "pattern_matching": 0.96,
            "text_processing": 0.93,
            "regular_expressions": 0.95,
            "validation": 0.92,
        }
    ),
    "git": AgentCapability(
        agent_type="git",
        strengths=["version_control", "merge_conflicts", "branching", "git_workflows"],
        weaknesses=["implementation", "design"],
        success_rate=0.88,
        avg_response_time_ms=1200,
        current_workload=0,
        specializations={
            "git": 0.96,
            "merge_conflicts": 0.94,
            "branching": 0.93,
            "version_control": 0.95,
            "git_workflows": 0.92,
        }
    ),
    "css": AgentCapability(
        agent_type="css",
        strengths=["styling", "flexbox", "grid", "responsive_design"],
        weaknesses=["backend", "databases"],
        success_rate=0.86,
        avg_response_time_ms=1300,
        current_workload=0,
        specializations={
            "css": 0.95,
            "flexbox": 0.94,
            "grid": 0.93,
            "responsive_design": 0.92,
            "styling": 0.96,
        }
    ),
}


class AgentCapabilityRegistry:
    """Registry for managing agent capabilities and performance metrics."""
    
    def __init__(self):
        self.capabilities = AGENT_CAPABILITIES.copy()
        logger.info(f"AgentCapabilityRegistry initialized with {len(self.capabilities)} agents")
    
    def get_capability(self, agent_type: str) -> Optional[AgentCapability]:
        """Get capability profile for an agent."""
        return self.capabilities.get(agent_type)
    
    def get_all_capabilities(self) -> Dict[str, AgentCapability]:
        """Get all agent capabilities."""
        return self.capabilities.copy()
    
    def update_success_rate(self, agent_type: str, success: bool):
        """Update agent success rate based on task outcome."""
        capability = self.capabilities.get(agent_type)
        if not capability:
            logger.warning(f"Unknown agent type: {agent_type}")
            return
        
        capability.total_tasks += 1
        if success:
            capability.successful_tasks += 1
        else:
            capability.failed_tasks += 1
        
        # Recalculate success rate
        if capability.total_tasks > 0:
            capability.success_rate = capability.successful_tasks / capability.total_tasks
        
        capability.last_updated = datetime.now()
        
        logger.info(
            f"Updated {agent_type} success rate: {capability.success_rate:.2%} "
            f"({capability.successful_tasks}/{capability.total_tasks})"
        )
    
    def update_response_time(self, agent_type: str, response_time_ms: float):
        """Update agent average response time using exponential moving average."""
        capability = self.capabilities.get(agent_type)
        if not capability:
            logger.warning(f"Unknown agent type: {agent_type}")
            return
        
        # Exponential moving average (alpha = 0.1)
        alpha = 0.1
        capability.avg_response_time_ms = (
            alpha * response_time_ms +
            (1 - alpha) * capability.avg_response_time_ms
        )
        
        capability.last_updated = datetime.now()
        
        logger.debug(f"Updated {agent_type} avg response time: {capability.avg_response_time_ms:.0f}ms")
    
    def increment_workload(self, agent_type: str):
        """Increment agent workload counter."""
        capability = self.capabilities.get(agent_type)
        if capability:
            capability.current_workload += 1
            logger.debug(f"{agent_type} workload: {capability.current_workload}")
    
    def decrement_workload(self, agent_type: str):
        """Decrement agent workload counter."""
        capability = self.capabilities.get(agent_type)
        if capability and capability.current_workload > 0:
            capability.current_workload -= 1
            logger.debug(f"{agent_type} workload: {capability.current_workload}")
    
    def get_agents_by_strength(self, strength: str, min_confidence: float = 0.8) -> List[str]:
        """Get agents that have a specific strength."""
        agents = []
        for agent_type, capability in self.capabilities.items():
            if strength in capability.strengths:
                agents.append(agent_type)
            elif strength in capability.specializations:
                if capability.specializations[strength] >= min_confidence:
                    agents.append(agent_type)
        return agents
    
    def get_best_agent_for_skill(self, skill: str) -> Optional[str]:
        """Get the best agent for a specific skill based on specialization scores."""
        best_agent = None
        best_score = 0.0
        
        for agent_type, capability in self.capabilities.items():
            score = capability.specializations.get(skill, 0.0)
            if score > best_score:
                best_score = score
                best_agent = agent_type
        
        return best_agent if best_score > 0.5 else None
    
    def get_stats(self) -> Dict:
        """Get overall registry statistics."""
        total_tasks = sum(c.total_tasks for c in self.capabilities.values())
        total_successful = sum(c.successful_tasks for c in self.capabilities.values())
        total_failed = sum(c.failed_tasks for c in self.capabilities.values())
        
        avg_success_rate = (
            total_successful / total_tasks if total_tasks > 0 else 0.0
        )
        
        return {
            "total_agents": len(self.capabilities),
            "total_tasks": total_tasks,
            "successful_tasks": total_successful,
            "failed_tasks": total_failed,
            "avg_success_rate": avg_success_rate,
            "agents_with_workload": sum(
                1 for c in self.capabilities.values() if c.current_workload > 0
            ),
        }


# Global instance
agent_capability_registry = AgentCapabilityRegistry()
