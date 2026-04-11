"""
Adaptive Agent Allocator
=========================

Phase 1 of Agent Autonomy Enhancement.

Intelligently selects the best agent for a task based on:
- Agent capabilities and specializations
- Task requirements and complexity
- Agent workload and performance metrics
- Success rates and response times

Author: Resonant Chat Systems Team
Date: December 26, 2025
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .agent_capability_registry import agent_capability_registry, AgentCapability
from .task_analyzer import task_analyzer, TaskProfile, TaskPriority

logger = logging.getLogger(__name__)


@dataclass
class AgentScore:
    """Score for an agent's suitability for a task."""
    agent_type: str
    total_score: float
    success_rate_score: float
    specialization_score: float
    workload_score: float
    response_time_score: float
    priority_bonus: float


class AdaptiveAgentAllocator:
    """Intelligently allocates tasks to agents based on capabilities and performance."""
    
    # Scoring weights (must sum to 1.0)
    WEIGHT_SUCCESS_RATE = 0.35      # 35% - Most important
    WEIGHT_SPECIALIZATION = 0.30    # 30% - Second most important
    WEIGHT_WORKLOAD = 0.20          # 20% - Load balancing
    WEIGHT_RESPONSE_TIME = 0.15     # 15% - Speed matters
    
    def __init__(self):
        self.registry = agent_capability_registry
        self.analyzer = task_analyzer
        logger.info("AdaptiveAgentAllocator initialized")
    
    def select_best_agent(
        self,
        message: str,
        available_agents: Optional[List[str]] = None
    ) -> Tuple[str, AgentScore]:
        """
        Select the best agent for a task.
        
        Args:
            message: The task message
            available_agents: List of agent types to consider (None = all agents)
            
        Returns:
            Tuple of (agent_type, score_details)
        """
        # Analyze task
        task_profile = self.analyzer.analyze_task(message)
        
        # Get available agents
        if available_agents is None:
            available_agents = list(self.registry.get_all_capabilities().keys())
        
        # Score all agents
        scores = self._score_agents(task_profile, available_agents)
        
        # Select best
        if not scores:
            # Fallback to reasoning agent
            logger.warning("No agents scored, falling back to reasoning agent")
            return "reasoning", AgentScore(
                agent_type="reasoning",
                total_score=0.5,
                success_rate_score=0.5,
                specialization_score=0.5,
                workload_score=0.5,
                response_time_score=0.5,
                priority_bonus=0.0,
            )
        
        best_agent_type, best_score = max(scores.items(), key=lambda x: x[1].total_score)
        
        logger.info(
            f"Selected agent: {best_agent_type} "
            f"(score: {best_score.total_score:.3f}, "
            f"success: {best_score.success_rate_score:.3f}, "
            f"spec: {best_score.specialization_score:.3f})"
        )
        
        return best_agent_type, best_score
    
    def _score_agents(
        self,
        task_profile: TaskProfile,
        available_agents: List[str]
    ) -> Dict[str, AgentScore]:
        """Score all available agents for the task."""
        scores = {}
        
        for agent_type in available_agents:
            capability = self.registry.get_capability(agent_type)
            if not capability:
                continue
            
            score = self._score_agent(task_profile, capability)
            scores[agent_type] = score
        
        return scores
    
    def _score_agent(
        self,
        task_profile: TaskProfile,
        capability: AgentCapability
    ) -> AgentScore:
        """Score a single agent for the task."""
        # 1. Success rate score (35% weight)
        success_rate_score = capability.success_rate
        
        # 2. Specialization score (30% weight)
        specialization_score = self._calculate_specialization_score(
            task_profile.required_skills,
            capability
        )
        
        # 3. Workload score (20% weight) - prefer less busy agents
        workload_score = self._calculate_workload_score(capability.current_workload)
        
        # 4. Response time score (15% weight) - prefer faster agents
        response_time_score = self._calculate_response_time_score(
            capability.avg_response_time_ms
        )
        
        # 5. Priority bonus (additive)
        priority_bonus = self._calculate_priority_bonus(
            task_profile.priority,
            capability
        )
        
        # Calculate total weighted score
        total_score = (
            success_rate_score * self.WEIGHT_SUCCESS_RATE +
            specialization_score * self.WEIGHT_SPECIALIZATION +
            workload_score * self.WEIGHT_WORKLOAD +
            response_time_score * self.WEIGHT_RESPONSE_TIME +
            priority_bonus
        )
        
        return AgentScore(
            agent_type=capability.agent_type,
            total_score=total_score,
            success_rate_score=success_rate_score,
            specialization_score=specialization_score,
            workload_score=workload_score,
            response_time_score=response_time_score,
            priority_bonus=priority_bonus,
        )
    
    def _calculate_specialization_score(
        self,
        required_skills: List[str],
        capability: AgentCapability
    ) -> float:
        """Calculate how well agent's specializations match required skills."""
        if not required_skills:
            return 0.5  # Neutral score
        
        # Get max specialization score for each required skill
        skill_scores = []
        for skill in required_skills:
            # Check direct specialization
            if skill in capability.specializations:
                skill_scores.append(capability.specializations[skill])
            # Check if skill is in strengths
            elif skill in capability.strengths:
                skill_scores.append(0.8)  # High but not perfect
            # Check if skill is in weaknesses
            elif skill in capability.weaknesses:
                skill_scores.append(0.2)  # Low penalty
            else:
                skill_scores.append(0.5)  # Neutral
        
        # Average of all skill scores
        return sum(skill_scores) / len(skill_scores) if skill_scores else 0.5
    
    def _calculate_workload_score(self, current_workload: int) -> float:
        """Calculate workload score (prefer less busy agents)."""
        # Score decreases as workload increases
        # 0 workload = 1.0 score
        # 5 workload = 0.5 score
        # 10+ workload = 0.0 score
        
        if current_workload == 0:
            return 1.0
        elif current_workload >= 10:
            return 0.0
        else:
            return 1.0 - (current_workload / 10)
    
    def _calculate_response_time_score(self, avg_response_time_ms: float) -> float:
        """Calculate response time score (prefer faster agents)."""
        # Score decreases as response time increases
        # 500ms = 1.0 score
        # 1500ms = 0.5 score
        # 3000ms+ = 0.0 score
        
        if avg_response_time_ms <= 500:
            return 1.0
        elif avg_response_time_ms >= 3000:
            return 0.0
        else:
            # Linear interpolation
            return 1.0 - ((avg_response_time_ms - 500) / 2500)
    
    def _calculate_priority_bonus(
        self,
        priority: TaskPriority,
        capability: AgentCapability
    ) -> float:
        """Calculate priority bonus (additive to total score)."""
        # High priority tasks get bonus for high success rate agents
        if priority == TaskPriority.URGENT:
            if capability.success_rate > 0.9:
                return 0.1  # 10% bonus
            elif capability.success_rate > 0.85:
                return 0.05  # 5% bonus
        elif priority == TaskPriority.HIGH:
            if capability.success_rate > 0.9:
                return 0.05  # 5% bonus
        
        return 0.0  # No bonus
    
    def get_top_agents(
        self,
        message: str,
        top_n: int = 3,
        available_agents: Optional[List[str]] = None
    ) -> List[Tuple[str, AgentScore]]:
        """Get top N agents for a task."""
        # Analyze task
        task_profile = self.analyzer.analyze_task(message)
        
        # Get available agents
        if available_agents is None:
            available_agents = list(self.registry.get_all_capabilities().keys())
        
        # Score all agents
        scores = self._score_agents(task_profile, available_agents)
        
        # Sort by score
        sorted_agents = sorted(
            scores.items(),
            key=lambda x: x[1].total_score,
            reverse=True
        )
        
        return sorted_agents[:top_n]
    
    def explain_selection(
        self,
        message: str,
        agent_type: str,
        score: AgentScore
    ) -> str:
        """Generate human-readable explanation of agent selection."""
        task_profile = self.analyzer.analyze_task(message)
        capability = self.registry.get_capability(agent_type)
        
        if not capability:
            return f"Selected {agent_type} (no capability data available)"
        
        explanation = [
            f"Selected: {agent_type}",
            f"Total Score: {score.total_score:.3f}",
            "",
            "Score Breakdown:",
            f"  • Success Rate: {score.success_rate_score:.3f} ({capability.success_rate:.1%})",
            f"  • Specialization: {score.specialization_score:.3f}",
            f"  • Workload: {score.workload_score:.3f} ({capability.current_workload} tasks)",
            f"  • Response Time: {score.response_time_score:.3f} ({capability.avg_response_time_ms:.0f}ms)",
        ]
        
        if score.priority_bonus > 0:
            explanation.append(f"  • Priority Bonus: +{score.priority_bonus:.3f}")
        
        explanation.extend([
            "",
            "Task Analysis:",
            f"  • Complexity: {task_profile.complexity_score:.2%}",
            f"  • Skills: {', '.join(task_profile.required_skills[:3])}",
            f"  • Priority: {task_profile.priority.name}",
        ])
        
        return "\n".join(explanation)


# Global instance
adaptive_agent_allocator = AdaptiveAgentAllocator()
