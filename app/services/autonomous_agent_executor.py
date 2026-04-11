"""
Autonomous Agent Executor
==========================

Phase 2 of Agent Autonomy Enhancement - Decentralized Decision-Making.

Wraps agent execution with autonomous decision-making capabilities.
Attempts local decisions first, only consulting LLM when necessary.

Expected Impact: 30-50% reduction in LLM calls.

Author: Resonant Chat Systems Team
Date: December 26, 2025
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .decision_framework import get_decision_framework, Decision
from .agent_knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from agent execution."""
    content: str
    confidence: float
    method: str  # "local_decision", "cached_decision", "llm_consultation"
    llm_calls: int
    decision_type: Optional[str] = None
    agent_type: Optional[str] = None
    response_time_ms: float = 0.0
    decision_details: Optional[Decision] = None
    provider: Optional[str] = None
    router_metadata: Optional[Dict[str, Any]] = None


class AutonomousAgentExecutor:
    """
    Executes agent tasks with autonomous decision-making.
    
    Workflow:
    1. Analyze task to determine if it can be decided locally
    2. If yes, make local decision (no LLM call)
    3. If no, consult LLM
    4. Cache decision for future reuse
    5. Track metrics
    """
    
    def __init__(self, agent_type: str, router=None, use_knowledge_base: bool = True):
        self.agent_type = agent_type
        self.router = router
        self.framework = get_decision_framework(agent_type)
        self.use_knowledge_base = use_knowledge_base
        
        # Knowledge base (Phase 3)
        if use_knowledge_base:
            self.knowledge_base = get_knowledge_base(agent_type)
            self.shared_kb = get_knowledge_base(None)  # Shared knowledge
        else:
            self.knowledge_base = None
            self.shared_kb = None
        
        # Metrics
        self.total_tasks = 0
        self.local_decisions = 0
        self.cached_decisions = 0
        self.knowledge_hits = 0
        self.llm_consultations = 0
        
        logger.info(
            f"AutonomousAgentExecutor initialized for {agent_type} "
            f"(knowledge_base={use_knowledge_base})"
        )
    
    async def execute_task(
        self,
        task: str,
        context: List[Dict[str, Any]],
        preferred_provider: Optional[str] = None
    ) -> AgentResult:
        """
        Execute task with autonomous decision-making.
        
        Priority:
        1. Try local decision (rule-based or cached)
        2. Fallback to LLM if needed
        3. Cache result for future
        """
        start_time = time.time()
        self.total_tasks += 1
        
        # Analyze task to determine decision type
        decision_type = self._analyze_task_type(task)
        
        # Build context dict for decision framework
        decision_context = self._build_decision_context(task, context)
        
        # Phase 3: Check knowledge base first
        if self.use_knowledge_base and self.knowledge_base:
            knowledge_result = self._search_knowledge(task, decision_type)
            if knowledge_result:
                response_time_ms = (time.time() - start_time) * 1000
                self.knowledge_hits += 1
                
                logger.info(
                    f"[{self.agent_type}] Knowledge base hit: {decision_type} "
                    f"(confidence: {knowledge_result.confidence:.2%}, "
                    f"time: {response_time_ms:.0f}ms, LLM calls: 0)"
                )
                
                return AgentResult(
                    content=knowledge_result.answer,
                    confidence=knowledge_result.confidence,
                    method="knowledge_base",
                    llm_calls=0,
                    decision_type=decision_type,
                    agent_type=self.agent_type,
                    response_time_ms=response_time_ms,
                )
        
        # Check if we can decide locally (Phase 2)
        if self.framework.can_decide_locally(decision_type, decision_context):
            # Make local decision (no LLM!)
            decision = self.framework.make_decision(decision_type, decision_context)
            
            if decision.method in ["rule_based", "cached"]:
                # Successfully decided locally
                response_time_ms = (time.time() - start_time) * 1000
                
                if decision.method == "cached":
                    self.cached_decisions += 1
                else:
                    self.local_decisions += 1
                
                logger.info(
                    f"[{self.agent_type}] Local decision: {decision_type} "
                    f"(method: {decision.method}, confidence: {decision.confidence:.2%}, "
                    f"time: {response_time_ms:.0f}ms, LLM calls: 0)"
                )
                
                return AgentResult(
                    content=self._format_decision_result(decision),
                    confidence=decision.confidence,
                    method=decision.method,
                    llm_calls=0,
                    decision_type=decision_type,
                    agent_type=self.agent_type,
                    response_time_ms=response_time_ms,
                    decision_details=decision,
                )
        
        # Can't decide locally - consult LLM
        result = await self._consult_llm(task, context, preferred_provider)
        response_time_ms = (time.time() - start_time) * 1000
        self.llm_consultations += 1
        
        # Cache the LLM result for future (Phase 2)
        context_hash = self.framework._hash_context(decision_context)
        llm_decision = Decision(
            decision_type=decision_type,
            action="llm_response",
            result=result.get("content", ""),
            confidence=0.8,  # Default confidence for LLM responses
            reasoning="LLM consultation",
            method="llm_consultation",
        )
        self.framework._cache_decision(decision_type, context_hash, llm_decision)
        
        # Capture knowledge (Phase 3)
        if self.use_knowledge_base and self.knowledge_base:
            self._capture_knowledge(
                task,
                result.get("content", ""),
                decision_type,
                confidence=0.8
            )
        
        logger.info(
            f"[{self.agent_type}] LLM consultation: {decision_type} "
            f"(time: {response_time_ms:.0f}ms, LLM calls: 1)"
        )
        
        return AgentResult(
            content=result.get("content", ""),
            confidence=0.8,
            method="llm_consultation",
            llm_calls=1,
            decision_type=decision_type,
            agent_type=self.agent_type,
            response_time_ms=response_time_ms,
            decision_details=llm_decision,
            provider=result.get("provider"),
            router_metadata=result.get("router_metadata"),
        )
    
    def _analyze_task_type(self, task: str) -> str:
        """Analyze task to determine decision type."""
        task_lower = task.lower()
        
        # Code-related tasks
        if "format" in task_lower and "code" in task_lower:
            return "format_code"
        elif "validate" in task_lower and ("syntax" in task_lower or "code" in task_lower):
            return "validate_syntax"
        elif "style" in task_lower or "lint" in task_lower:
            return "check_style"
        
        # Math tasks
        elif any(op in task_lower for op in ["calculate", "compute", "solve"]):
            if any(op in task for op in ["+", "-", "*", "/", "**"]):
                return "simple_math"
        
        # Logical tasks - ONLY match explicit logical expressions, not common English words
        # Must have explicit logical operators with clear programming context
        elif (
            (" && " in task or " || " in task or " AND " in task or " OR " in task) or
            (task_lower.startswith("if ") and " then " in task_lower) or
            ("evaluate" in task_lower and any(op in task for op in ["==", "!=", ">=", "<=", ">", "<"]))
        ):
            return "logical_operators"
        
        # Default: complex task requiring LLM
        return "complex_task"
    
    def _build_decision_context(
        self,
        task: str,
        context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build context dict for decision framework."""
        # Extract relevant info from task and context
        decision_context = {
            "task": task,
            "task_length": len(task),
        }
        
        # Try to extract language if code-related
        if "python" in task.lower():
            decision_context["language"] = "python"
        elif "javascript" in task.lower() or "js" in task.lower():
            decision_context["language"] = "javascript"
        elif "typescript" in task.lower() or "ts" in task.lower():
            decision_context["language"] = "typescript"
        
        # Try to extract code if present
        if "```" in task:
            # Extract code block
            parts = task.split("```")
            if len(parts) >= 3:
                code_block = parts[1]
                # Remove language identifier if present
                lines = code_block.split("\n")
                if lines[0].strip() in ["python", "javascript", "typescript", "js", "ts"]:
                    decision_context["language"] = lines[0].strip()
                    decision_context["code"] = "\n".join(lines[1:])
                else:
                    decision_context["code"] = code_block
        
        return decision_context
    
    async def _consult_llm(
        self,
        task: str,
        context: List[Dict[str, Any]],
        preferred_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Consult LLM for decision."""
        if not self.router:
            logger.warning("No router available for LLM consultation")
            return {
                "content": "Error: No LLM router available",
                "provider": "error",
            }
        
        try:
            # Get agent-specific system prompts
            system_prompts = self._get_agent_prompts()
            
            # Combine with context
            messages = system_prompts + context
            
            # Call LLM
            response = await self.router.route_query(
                    message=task,
                    context=messages,
                    preferred_provider=preferred_provider
                )
            
            router_meta = response.get("metadata", {})
            return {
                "content": response.get("response", ""),
                "provider": response.get("provider", "unknown"),
                "router_metadata": router_meta,
            }
        except Exception as e:
            logger.error(f"LLM consultation failed: {e}", exc_info=True)
            return {
                "content": f"Error consulting LLM: {str(e)}",
                "provider": "error",
            }
    
    def _get_agent_prompts(self) -> List[Dict[str, str]]:
        """Get agent-specific system prompts."""
        # Simplified prompts - in production, would use full prompts from agent_engine
        prompts = {
            "code": [
                {
                    "role": "system",
                    "content": "You are a code generation expert. Write clean, efficient code."
                }
            ],
            "reasoning": [
                {
                    "role": "system",
                    "content": "You are a reasoning expert. Analyze problems logically."
                }
            ],
            "debug": [
                {
                    "role": "system",
                    "content": "You are a debugging expert. Find and fix bugs."
                }
            ],
        }
        
        return prompts.get(self.agent_type, [
            {
                "role": "system",
                "content": f"You are a {self.agent_type} expert."
            }
        ])
    
    def _format_decision_result(self, decision: Decision) -> str:
        """Format decision result as response text."""
        if decision.method == "rule_based":
            return f"{decision.reasoning}\n\nResult: {decision.result}"
        elif decision.method == "cached":
            return f"{decision.reasoning}\n\nResult: {decision.result}"
        else:
            return str(decision.result)
    
    def update_decision_success(
        self,
        decision_type: str,
        context: Dict[str, Any],
        success: bool
    ):
        """Update success rate of a decision."""
        context_hash = self.framework._hash_context(context)
        self.framework.update_decision_success(decision_type, context_hash, success)
    
    def _search_knowledge(self, task: str, decision_type: str) -> Optional[Any]:
        """Search knowledge base for relevant entry."""
        # Search agent-specific knowledge first
        results = self.knowledge_base.search(
            query=task,
            agent_type=self.agent_type,
            min_confidence=0.75,
            min_success_rate=0.7,
            top_k=1
        )
        
        if results:
            entry = results[0]
            # Update usage
            self.knowledge_base.update_success(entry.id, success=True)
            return entry
        
        # Search shared knowledge
        if self.shared_kb:
            results = self.shared_kb.search(
                query=task,
                min_confidence=0.8,
                min_success_rate=0.75,
                top_k=1
            )
            
            if results:
                entry = results[0]
                self.shared_kb.update_success(entry.id, success=True)
                return entry
        
        return None
    
    def _capture_knowledge(
        self,
        question: str,
        answer: str,
        topic: str,
        confidence: float = 0.8
    ):
        """Capture knowledge from LLM response."""
        # Extract tags from question
        tags = self._extract_tags(question)
        
        # Add to knowledge base
        self.knowledge_base.add_entry(
            question=question,
            answer=answer,
            topic=topic,
            agent_type=self.agent_type,
            confidence=confidence,
            tags=tags,
        )
        
        logger.debug(f"Captured knowledge: {topic} (tags: {tags})")
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags from text."""
        tags = []
        text_lower = text.lower()
        
        # Programming languages
        languages = ["python", "javascript", "typescript", "java", "c++", "rust", "go"]
        for lang in languages:
            if lang in text_lower:
                tags.append(lang)
        
        # Common topics
        topics = [
            "debugging", "testing", "optimization", "security",
            "database", "api", "frontend", "backend",
            "async", "performance", "error handling"
        ]
        for topic in topics:
            if topic in text_lower:
                tags.append(topic)
        
        return tags[:5]  # Limit to 5 tags
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        local_rate = (
            (self.local_decisions + self.cached_decisions + self.knowledge_hits) / self.total_tasks
            if self.total_tasks > 0 else 0.0
        )
        
        llm_reduction = (
            (self.local_decisions + self.cached_decisions + self.knowledge_hits) / self.total_tasks
            if self.total_tasks > 0 else 0.0
        )
        
        metrics = {
            "agent_type": self.agent_type,
            "total_tasks": self.total_tasks,
            "local_decisions": self.local_decisions,
            "cached_decisions": self.cached_decisions,
            "knowledge_hits": self.knowledge_hits,
            "llm_consultations": self.llm_consultations,
            "local_decision_rate": local_rate,
            "llm_reduction_rate": llm_reduction,
            "cache_stats": self.framework.get_cache_stats(),
        }
        
        # Add knowledge base stats if enabled
        if self.use_knowledge_base and self.knowledge_base:
            metrics["knowledge_stats"] = self.knowledge_base.get_stats()
        
        return metrics
    
    def get_summary(self) -> str:
        """Get human-readable summary of metrics."""
        metrics = self.get_metrics()
        
        return (
            f"Agent: {self.agent_type}\n"
            f"Total Tasks: {metrics['total_tasks']}\n"
            f"Local Decisions: {metrics['local_decisions']} "
            f"({metrics['local_decision_rate']:.1%})\n"
            f"Cached Decisions: {metrics['cached_decisions']}\n"
            f"Knowledge Hits: {metrics['knowledge_hits']}\n"
            f"LLM Consultations: {metrics['llm_consultations']}\n"
            f"LLM Reduction: {metrics['llm_reduction_rate']:.1%}\n"
            f"Cache Entries: {metrics['cache_stats']['total_cached_decisions']}\n"
            f"Cache Success Rate: {metrics['cache_stats']['avg_success_rate']:.1%}"
        )
        
        # Add knowledge stats if available
        if "knowledge_stats" in metrics:
            summary += (
                f"\nKnowledge Entries: {metrics['knowledge_stats']['total_entries']}\n"
                f"Knowledge Success Rate: {metrics['knowledge_stats']['avg_success_rate']:.1%}"
        )


# Global executor registry
_executor_registry: Dict[str, AutonomousAgentExecutor] = {}


def get_autonomous_executor(
    agent_type: str,
    router=None
) -> AutonomousAgentExecutor:
    """Get or create autonomous executor for agent type."""
    if agent_type not in _executor_registry:
        _executor_registry[agent_type] = AutonomousAgentExecutor(agent_type, router)
    
    # Update router if provided
    if router:
        _executor_registry[agent_type].router = router
    
    return _executor_registry[agent_type]


def get_all_executor_metrics() -> List[Dict[str, Any]]:
    """Get metrics from all executors."""
    return [
        executor.get_metrics()
        for executor in _executor_registry.values()
    ]


def get_aggregate_metrics() -> Dict[str, Any]:
    """Get aggregate metrics across all executors."""
    all_metrics = get_all_executor_metrics()
    
    if not all_metrics:
        return {
            "total_executors": 0,
            "total_tasks": 0,
            "total_local_decisions": 0,
            "total_llm_consultations": 0,
            "avg_llm_reduction": 0.0,
        }
    
    total_tasks = sum(m["total_tasks"] for m in all_metrics)
    total_local = sum(m["local_decisions"] + m["cached_decisions"] for m in all_metrics)
    total_llm = sum(m["llm_consultations"] for m in all_metrics)
    
    return {
        "total_executors": len(all_metrics),
        "total_tasks": total_tasks,
        "total_local_decisions": total_local,
        "total_llm_consultations": total_llm,
        "avg_llm_reduction": total_local / total_tasks if total_tasks > 0 else 0.0,
        "executors": all_metrics,
    }
