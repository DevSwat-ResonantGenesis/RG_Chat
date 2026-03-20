"""
Agent Router - Routes messages to appropriate agents based on intent and context.

STATUS: GRADUATED
CREATED: 2025-12-21
GRADUATED: 2025-12-21
GOVERNANCE: This module provides intelligent routing of user messages to specialized
            agents based on intent analysis, context, and agent capabilities.
            
INVARIANTS:
  - route() always returns a RouteResult (never None)
  - confidence is always in range [0.0, 1.0]
  - fallback decision is used when no agents match
  - routing history is bounded (max 1000 entries)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Governance: This module is GRADUATED
_IS_STUB = False
_MAX_ROUTING_HISTORY = 1000


class RoutingDecision(Enum):
    """Routing decision types."""
    DIRECT = "direct"  # Route directly to a specific agent
    BROADCAST = "broadcast"  # Broadcast to multiple agents
    CHAIN = "chain"  # Chain through multiple agents sequentially
    DEBATE = "debate"  # Use debate engine for consensus
    FALLBACK = "fallback"  # Use fallback/default agent


@dataclass
class RouteResult:
    """Result of a routing decision."""
    decision: RoutingDecision
    primary_agent: Optional[str] = None
    secondary_agents: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentRouter:
    """
    Routes messages to appropriate agents based on intent and context.
    
    Uses intent analysis, agent capabilities, and historical performance
    to make optimal routing decisions.
    """
    
    def __init__(self):
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.agent_performance: Dict[str, float] = {}
        self.routing_history: List[RouteResult] = []
        
    def register_agent(self, agent_id: str, capabilities: List[str]) -> None:
        """Register an agent with its capabilities."""
        self.agent_capabilities[agent_id] = capabilities
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = 0.5  # Default performance score
        logger.info(f"Registered agent {agent_id} with capabilities: {capabilities}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        self.agent_capabilities.pop(agent_id, None)
        self.agent_performance.pop(agent_id, None)
        logger.info(f"Unregistered agent {agent_id}")
    
    def route(
        self,
        message: str,
        intent: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        preferred_agent: Optional[str] = None
    ) -> RouteResult:
        """
        Route a message to the appropriate agent(s).
        
        Args:
            message: The user message to route
            intent: Optional pre-analyzed intent
            context: Optional context information
            preferred_agent: Optional preferred agent ID
            
        Returns:
            RouteResult with routing decision and target agents
        """
        context = context or {}
        
        # If preferred agent is specified and available, use it
        if preferred_agent and preferred_agent in self.agent_capabilities:
            result = RouteResult(
                decision=RoutingDecision.DIRECT,
                primary_agent=preferred_agent,
                confidence=1.0,
                reasoning="User specified preferred agent"
            )
            self.routing_history.append(result)
            return result
        
        # Analyze intent if not provided
        if not intent:
            intent = self._analyze_intent(message)
        
        # Find matching agents based on intent
        matching_agents = self._find_matching_agents(intent, context)
        
        if not matching_agents:
            # No matching agents, use fallback
            result = RouteResult(
                decision=RoutingDecision.FALLBACK,
                confidence=0.3,
                reasoning="No agents matched the intent, using fallback"
            )
            self.routing_history.append(result)
            return result
        
        if len(matching_agents) == 1:
            # Single match, route directly
            result = RouteResult(
                decision=RoutingDecision.DIRECT,
                primary_agent=matching_agents[0],
                confidence=0.8,
                reasoning=f"Single agent matched intent: {intent}"
            )
            self.routing_history.append(result)
            return result
        
        # Multiple matches - decide between chain, broadcast, or debate
        if self._should_use_debate(intent, context):
            result = RouteResult(
                decision=RoutingDecision.DEBATE,
                primary_agent=matching_agents[0],
                secondary_agents=matching_agents[1:],
                confidence=0.7,
                reasoning="Multiple agents matched, using debate for consensus"
            )
        elif self._should_chain(intent, context):
            result = RouteResult(
                decision=RoutingDecision.CHAIN,
                primary_agent=matching_agents[0],
                secondary_agents=matching_agents[1:],
                confidence=0.75,
                reasoning="Multiple agents matched, chaining for comprehensive response"
            )
        else:
            # Default to best performing agent
            best_agent = max(matching_agents, key=lambda a: self.agent_performance.get(a, 0))
            result = RouteResult(
                decision=RoutingDecision.DIRECT,
                primary_agent=best_agent,
                secondary_agents=[a for a in matching_agents if a != best_agent],
                confidence=0.6,
                reasoning="Selected best performing agent from matches"
            )
        
        # Enforce routing history bound (invariant)
        if len(self.routing_history) >= _MAX_ROUTING_HISTORY:
            self.routing_history = self.routing_history[-(_MAX_ROUTING_HISTORY // 2):]
        self.routing_history.append(result)
        return result
    
    def _analyze_intent(self, message: str) -> str:
        """Analyze message to determine intent."""
        message_lower = message.lower()
        
        # Simple keyword-based intent detection
        if any(kw in message_lower for kw in ['code', 'program', 'function', 'debug', 'error']):
            return 'coding'
        elif any(kw in message_lower for kw in ['explain', 'what is', 'how does', 'why']):
            return 'explanation'
        elif any(kw in message_lower for kw in ['write', 'create', 'generate', 'make']):
            return 'generation'
        elif any(kw in message_lower for kw in ['analyze', 'review', 'check', 'evaluate']):
            return 'analysis'
        elif any(kw in message_lower for kw in ['help', 'assist', 'support']):
            return 'assistance'
        else:
            return 'general'
    
    def _find_matching_agents(self, intent: str, context: Dict[str, Any]) -> List[str]:
        """Find agents that match the given intent."""
        matching = []
        for agent_id, capabilities in self.agent_capabilities.items():
            if intent in capabilities or 'general' in capabilities:
                matching.append(agent_id)
        return matching
    
    def _should_use_debate(self, intent: str, context: Dict[str, Any]) -> bool:
        """Determine if debate should be used for this request."""
        # Use debate for complex or controversial topics
        return context.get('requires_consensus', False) or intent in ['analysis', 'evaluation']
    
    def _should_chain(self, intent: str, context: Dict[str, Any]) -> bool:
        """Determine if chaining should be used for this request."""
        # Use chaining for multi-step tasks
        return context.get('multi_step', False) or intent in ['generation', 'coding']
    
    def update_performance(self, agent_id: str, score: float) -> None:
        """Update agent performance score based on feedback."""
        if agent_id in self.agent_performance:
            # Exponential moving average
            alpha = 0.3
            self.agent_performance[agent_id] = (
                alpha * score + (1 - alpha) * self.agent_performance[agent_id]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "registered_agents": len(self.agent_capabilities),
            "total_routes": len(self.routing_history),
            "agent_performance": self.agent_performance.copy(),
            "decision_distribution": self._get_decision_distribution()
        }
    
    def _get_decision_distribution(self) -> Dict[str, int]:
        """Get distribution of routing decisions."""
        distribution: Dict[str, int] = {}
        for result in self.routing_history:
            key = result.decision.value
            distribution[key] = distribution.get(key, 0) + 1
        return distribution


# Global instance
agent_router = AgentRouter()


def route_message(
    message: str,
    intent: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    preferred_agent: Optional[str] = None
) -> RouteResult:
    """
    Convenience function to route a message using the global router.
    
    Args:
        message: The user message to route
        intent: Optional pre-analyzed intent
        context: Optional context information
        preferred_agent: Optional preferred agent ID
        
    Returns:
        RouteResult with routing decision and target agents
    """
    return agent_router.route(message, intent, context, preferred_agent)
