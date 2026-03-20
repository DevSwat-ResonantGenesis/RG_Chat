"""
Layer 8: Intelligent Multi-LLM Router
======================================

Advanced routing system that selects the optimal LLM provider based on:
1. Task complexity analysis
2. Cost optimization
3. Response quality requirements
4. Provider availability and rate limits
5. User tier and budget constraints

This implements the Layer 8 "Multi-LLM Routing" from the Hash Sphere architecture.

Author: Resonant Genesis Team
Date: December 29, 2025
"""

import os
import re
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for routing decisions."""
    SIMPLE = 1      # Quick answers, simple questions
    MODERATE = 2    # Standard conversations, explanations
    COMPLEX = 3     # Code generation, analysis, reasoning
    EXPERT = 4      # Multi-step reasoning, research, creative


class ProviderTier(Enum):
    """Provider capability tiers."""
    FAST = 1        # Groq - fastest, good for simple tasks
    BALANCED = 2    # Gemini - good balance of speed/quality
    QUALITY = 3     # GPT-4o - high quality, moderate cost
    PREMIUM = 4     # Claude - best reasoning, highest cost


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    tier: ProviderTier
    cost_per_1k_input: float      # Cost per 1000 input tokens
    cost_per_1k_output: float     # Cost per 1000 output tokens
    avg_latency_ms: int           # Average response latency
    max_context_tokens: int       # Maximum context window
    strengths: List[str]          # What this provider is good at
    rate_limit_rpm: int = 60      # Requests per minute
    
    # Runtime tracking
    last_error_time: Optional[datetime] = None
    error_count: int = 0
    success_count: int = 0


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    provider: str
    reason: str
    estimated_cost: float
    complexity: TaskComplexity
    fallback_providers: List[str]
    metadata: Dict = field(default_factory=dict)


class IntelligentRouter:
    """
    Layer 8 Intelligent Multi-LLM Router.
    
    Analyzes tasks and routes to optimal provider based on:
    - Task complexity
    - Cost constraints
    - Quality requirements
    - Provider availability
    """
    
    # Provider configurations with pricing (as of Dec 2024)
    PROVIDERS = {
        "groq": ProviderConfig(
            name="groq",
            tier=ProviderTier.FAST,
            cost_per_1k_input=0.00005,    # $0.05 per 1M tokens
            cost_per_1k_output=0.00008,
            avg_latency_ms=200,
            max_context_tokens=32768,
            strengths=["speed", "simple_qa", "chat", "summarization"],
            rate_limit_rpm=30,
        ),
        "gemini": ProviderConfig(
            name="gemini",
            tier=ProviderTier.BALANCED,
            cost_per_1k_input=0.000125,   # Gemini 1.5 Flash pricing
            cost_per_1k_output=0.000375,
            avg_latency_ms=800,
            max_context_tokens=1000000,   # 1M context window!
            strengths=["long_context", "multimodal", "analysis", "research"],
            rate_limit_rpm=60,
        ),
        "chatgpt": ProviderConfig(
            name="chatgpt",
            tier=ProviderTier.QUALITY,
            cost_per_1k_input=0.0025,     # GPT-4o pricing
            cost_per_1k_output=0.01,
            avg_latency_ms=1500,
            max_context_tokens=128000,
            strengths=["code", "reasoning", "creativity", "instruction_following"],
            rate_limit_rpm=60,
        ),
        "claude": ProviderConfig(
            name="claude",
            tier=ProviderTier.PREMIUM,
            cost_per_1k_input=0.003,      # Claude 3 Haiku pricing
            cost_per_1k_output=0.015,
            avg_latency_ms=2000,
            max_context_tokens=200000,
            strengths=["reasoning", "analysis", "safety", "long_form", "code_review"],
            rate_limit_rpm=50,
        ),
    }
    
    # Task patterns for complexity detection
    COMPLEXITY_PATTERNS = {
        TaskComplexity.SIMPLE: [
            r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure)[\s!?.]*$",
            r"^what (is|are) (the )?(time|date|weather)",
            r"^(how are you|what's up|how are you\?)",
            r"^hello,?\s*(how are you|what's up)",
        ],
        TaskComplexity.MODERATE: [
            r"explain|describe|tell me about|what does .* mean",
            r"summarize|summary|tldr",
            r"translate|convert",
        ],
        TaskComplexity.COMPLEX: [
            r"```|code|function|class|implement|debug|fix",
            r"analyze|compare|evaluate|assess",
            r"step.?by.?step|how (do|can|should) (i|we)",
            r"create|build|develop|design",
        ],
        TaskComplexity.EXPERT: [
            r"research|investigate|deep.?dive",
            r"optimize|refactor|architect",
            r"multi.?step|complex|advanced",
            r"prove|derive|mathematical",
        ],
    }
    
    # Strength-based routing
    TASK_TO_STRENGTH = {
        "code": ["chatgpt", "claude"],
        "reasoning": ["claude", "chatgpt"],
        "speed": ["groq"],
        "long_context": ["gemini"],
        "analysis": ["claude", "gemini"],
        "creative": ["chatgpt", "claude"],
        "simple": ["groq", "gemini"],
        "research": ["gemini", "claude"],
    }
    
    def __init__(self):
        """Initialize the intelligent router."""
        self.request_history: List[Dict] = []
        self.provider_stats: Dict[str, Dict] = {
            name: {"requests": 0, "errors": 0, "total_cost": 0.0, "avg_latency": 0}
            for name in self.PROVIDERS
        }
        
        # Cost optimization settings
        self.daily_budget = float(os.getenv("LLM_DAILY_BUDGET", "10.0"))
        self.cost_today = 0.0
        self.cost_reset_date = datetime.utcnow().date()
    
    def analyze_task(self, message: str, context: Optional[List[Dict]] = None) -> Tuple[TaskComplexity, List[str]]:
        """
        Analyze task to determine complexity and required strengths.
        
        Returns:
            Tuple of (complexity, list of required strengths)
        """
        message_lower = message.lower()
        
        # Check for complexity patterns
        detected_complexity = TaskComplexity.MODERATE  # Default
        
        for complexity, patterns in self.COMPLEXITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    if complexity.value > detected_complexity.value:
                        detected_complexity = complexity
                    break
        
        # Detect required strengths
        strengths = []
        
        # Code detection
        if any(x in message_lower for x in ["code", "function", "class", "```", "debug", "implement"]):
            strengths.append("code")
        
        # Reasoning detection
        if any(x in message_lower for x in ["why", "explain", "reason", "analyze", "compare"]):
            strengths.append("reasoning")
        
        # Long context detection
        context_length = sum(len(str(m.get("content", ""))) for m in (context or []))
        if context_length > 10000:
            strengths.append("long_context")
        
        # Speed requirement (short messages, simple questions)
        if len(message) < 100 and detected_complexity == TaskComplexity.SIMPLE:
            strengths.append("speed")
        
        # Research/analysis
        if any(x in message_lower for x in ["research", "investigate", "analyze", "study"]):
            strengths.append("research")
        
        # Creative tasks
        if any(x in message_lower for x in ["create", "write", "generate", "story", "poem"]):
            strengths.append("creative")
        
        # Default to simple if no specific strengths detected
        if not strengths:
            strengths.append("simple")
        
        logger.info(f"[Layer8] Task analysis: complexity={detected_complexity.name}, strengths={strengths}")
        return detected_complexity, strengths
    
    def estimate_cost(self, provider: str, input_tokens: int, output_tokens: int = 500) -> float:
        """Estimate cost for a request to a provider."""
        config = self.PROVIDERS.get(provider)
        if not config:
            return 0.0
        
        input_cost = (input_tokens / 1000) * config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * config.cost_per_1k_output
        return input_cost + output_cost
    
    def get_available_providers(self, api_keys: Dict[str, bool]) -> List[str]:
        """Get list of available providers based on API keys."""
        available = []
        for name in self.PROVIDERS:
            key_name = "openai" if name == "chatgpt" else ("google" if name == "gemini" else name)
            if api_keys.get(key_name, False):
                available.append(name)
        return available
    
    def is_provider_healthy(self, provider: str) -> bool:
        """Check if provider is healthy (not rate limited or erroring)."""
        config = self.PROVIDERS.get(provider)
        if not config:
            return False
        
        # Check if provider had recent errors
        if config.last_error_time:
            time_since_error = datetime.utcnow() - config.last_error_time
            if time_since_error < timedelta(minutes=1) and config.error_count > 3:
                return False
        
        return True
    
    def record_request(self, provider: str, success: bool, latency_ms: int, cost: float):
        """Record request outcome for learning."""
        config = self.PROVIDERS.get(provider)
        if config:
            if success:
                config.success_count += 1
                config.error_count = max(0, config.error_count - 1)
            else:
                config.error_count += 1
                config.last_error_time = datetime.utcnow()
        
        # Update stats
        stats = self.provider_stats.get(provider, {})
        stats["requests"] = stats.get("requests", 0) + 1
        if not success:
            stats["errors"] = stats.get("errors", 0) + 1
        stats["total_cost"] = stats.get("total_cost", 0.0) + cost
        
        # Update running average latency
        old_avg = stats.get("avg_latency", latency_ms)
        n = stats["requests"]
        stats["avg_latency"] = old_avg + (latency_ms - old_avg) / n
        
        self.provider_stats[provider] = stats
        
        # Update daily cost
        if datetime.utcnow().date() != self.cost_reset_date:
            self.cost_today = 0.0
            self.cost_reset_date = datetime.utcnow().date()
        self.cost_today += cost
    
    def select_provider(
        self,
        message: str,
        context: Optional[List[Dict]] = None,
        available_providers: Optional[List[str]] = None,
        preferred_provider: Optional[str] = None,
        optimize_for: str = "balanced",  # "cost", "quality", "speed", "balanced"
    ) -> RoutingDecision:
        """
        Select optimal provider using Layer 8 intelligent routing.
        
        Args:
            message: User message
            context: Conversation context
            available_providers: List of providers with valid API keys
            preferred_provider: User's preferred provider (if any)
            optimize_for: Optimization strategy
            
        Returns:
            RoutingDecision with selected provider and reasoning
        """
        # Analyze task
        complexity, strengths = self.analyze_task(message, context)
        
        # Get available providers
        if not available_providers:
            available_providers = list(self.PROVIDERS.keys())
        
        # Filter to healthy providers
        healthy_providers = [p for p in available_providers if self.is_provider_healthy(p)]
        if not healthy_providers:
            healthy_providers = available_providers  # Fallback to all if none healthy
        
        # If user has a preference and it's available, use it
        if preferred_provider:
            normalized = preferred_provider.lower()
            provider_map = {"openai": "chatgpt", "gpt": "chatgpt", "google": "gemini"}
            normalized = provider_map.get(normalized, normalized)
            
            if normalized in healthy_providers:
                return RoutingDecision(
                    provider=normalized,
                    reason=f"User preferred provider: {normalized}",
                    estimated_cost=self.estimate_cost(normalized, len(message) // 4),
                    complexity=complexity,
                    fallback_providers=[p for p in healthy_providers if p != normalized],
                )
        
        # Score each provider
        scores: Dict[str, float] = {}
        
        for provider in healthy_providers:
            config = self.PROVIDERS[provider]
            score = 0.0
            
            # Strength matching (0-40 points)
            strength_matches = sum(1 for s in strengths if s in config.strengths)
            score += strength_matches * 10
            
            # Complexity matching (0-30 points)
            if complexity == TaskComplexity.SIMPLE and config.tier == ProviderTier.FAST:
                score += 30
            elif complexity == TaskComplexity.MODERATE and config.tier == ProviderTier.BALANCED:
                score += 30
            elif complexity == TaskComplexity.COMPLEX and config.tier == ProviderTier.QUALITY:
                score += 30
            elif complexity == TaskComplexity.EXPERT and config.tier == ProviderTier.PREMIUM:
                score += 30
            
            # Cost optimization (0-20 points)
            if optimize_for in ["cost", "balanced"]:
                # Lower cost = higher score
                max_cost = max(p.cost_per_1k_output for p in self.PROVIDERS.values())
                cost_score = (1 - config.cost_per_1k_output / max_cost) * 20
                score += cost_score
            
            # Speed optimization (0-20 points)
            if optimize_for in ["speed", "balanced"]:
                max_latency = max(p.avg_latency_ms for p in self.PROVIDERS.values())
                speed_score = (1 - config.avg_latency_ms / max_latency) * 20
                score += speed_score
            
            # Quality optimization (0-20 points)
            if optimize_for in ["quality", "balanced"]:
                quality_score = config.tier.value * 5
                score += quality_score
            
            # Budget check
            estimated_cost = self.estimate_cost(provider, len(message) // 4)
            if self.cost_today + estimated_cost > self.daily_budget:
                score -= 50  # Penalize if over budget
            
            # Success rate bonus
            stats = self.provider_stats.get(provider, {})
            if stats.get("requests", 0) > 10:
                success_rate = 1 - (stats.get("errors", 0) / stats["requests"])
                score += success_rate * 10
            
            scores[provider] = score
        
        # Select best provider
        best_provider = max(scores, key=scores.get)
        fallbacks = sorted(
            [p for p in healthy_providers if p != best_provider],
            key=lambda p: scores.get(p, 0),
            reverse=True
        )
        
        # Build reason
        config = self.PROVIDERS[best_provider]
        reason_parts = [f"Selected {best_provider} (score: {scores[best_provider]:.1f})"]
        reason_parts.append(f"Task: {complexity.name}, Strengths: {strengths}")
        reason_parts.append(f"Provider tier: {config.tier.name}")
        
        logger.info(f"[Layer8] Routing decision: {best_provider} | {' | '.join(reason_parts)}")
        
        return RoutingDecision(
            provider=best_provider,
            reason=" | ".join(reason_parts),
            estimated_cost=self.estimate_cost(best_provider, len(message) // 4),
            complexity=complexity,
            fallback_providers=fallbacks,
            metadata={
                "scores": scores,
                "strengths": strengths,
                "optimize_for": optimize_for,
            }
        )
    
    def get_stats(self) -> Dict:
        """Get routing statistics."""
        return {
            "provider_stats": self.provider_stats,
            "cost_today": self.cost_today,
            "daily_budget": self.daily_budget,
            "providers": {
                name: {
                    "tier": config.tier.name,
                    "healthy": self.is_provider_healthy(name),
                    "success_count": config.success_count,
                    "error_count": config.error_count,
                }
                for name, config in self.PROVIDERS.items()
            }
        }


# Global instance
intelligent_router = IntelligentRouter()


def select_optimal_provider(
    message: str,
    context: Optional[List[Dict]] = None,
    available_providers: Optional[List[str]] = None,
    preferred_provider: Optional[str] = None,
    optimize_for: str = "balanced",
) -> RoutingDecision:
    """Convenience function to select optimal provider."""
    return intelligent_router.select_provider(
        message=message,
        context=context,
        available_providers=available_providers,
        preferred_provider=preferred_provider,
        optimize_for=optimize_for,
    )
