"""
Agent Chaining System (ACS)
============================

Phase 5.6: User-defined custom agent pipelines.

Features:
- Define custom agent sequences
- Save and reuse pipelines
- Pipeline templates
- Conditional branching
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ChainStep:
    """A single step in an agent chain."""
    agent_type: str
    transform_input: Optional[str] = None  # How to transform previous output
    condition: Optional[str] = None  # Condition to execute this step
    fallback_agent: Optional[str] = None  # Agent to use if this one fails


@dataclass
class AgentChain:
    """A complete agent chain definition."""
    id: str
    name: str
    description: str
    steps: List[ChainStep]
    created_by: str
    created_at: str
    is_template: bool = False


@dataclass
class ChainExecutionResult:
    """Result of chain execution."""
    final_output: str
    steps_executed: int
    step_outputs: Dict[str, str]
    execution_time_ms: float
    success: bool
    error: Optional[str] = None


class AgentChainingEngine:
    """
    Manages custom agent pipelines/chains.
    """
    
    def __init__(self, agent_engine=None):
        self.agent_engine = agent_engine
        self.chains: Dict[str, AgentChain] = {}
        self.user_chains: Dict[str, List[str]] = {}  # user_id -> chain_ids
        
        # Built-in templates
        self._init_templates()
    
    def _init_templates(self):
        """Initialize built-in chain templates."""
        templates = [
            AgentChain(
                id="template_code_quality",
                name="Code Quality Pipeline",
                description="Write code, review it, add tests, then document",
                steps=[
                    ChainStep(agent_type="code"),
                    ChainStep(agent_type="review", transform_input="Review this code:\n{prev}"),
                    ChainStep(agent_type="test", transform_input="Write tests for:\n{prev}"),
                    ChainStep(agent_type="documentation", transform_input="Document this:\n{prev}"),
                ],
                created_by="system",
                created_at=datetime.now().isoformat(),
                is_template=True,
            ),
            AgentChain(
                id="template_research_summary",
                name="Research & Summarize",
                description="Research a topic, explain it simply, then summarize",
                steps=[
                    ChainStep(agent_type="research"),
                    ChainStep(agent_type="explain", transform_input="Explain this simply:\n{prev}"),
                    ChainStep(agent_type="summary", transform_input="Summarize:\n{prev}"),
                ],
                created_by="system",
                created_at=datetime.now().isoformat(),
                is_template=True,
            ),
            AgentChain(
                id="template_secure_code",
                name="Secure Code Pipeline",
                description="Write code, security audit, fix issues",
                steps=[
                    ChainStep(agent_type="code"),
                    ChainStep(agent_type="security", transform_input="Security audit:\n{prev}"),
                    ChainStep(agent_type="code", transform_input="Fix security issues:\n{prev}"),
                    ChainStep(agent_type="test", transform_input="Write security tests:\n{prev}"),
                ],
                created_by="system",
                created_at=datetime.now().isoformat(),
                is_template=True,
            ),
            AgentChain(
                id="template_refactor_safe",
                name="Safe Refactoring",
                description="Review, refactor, test, review again",
                steps=[
                    ChainStep(agent_type="review"),
                    ChainStep(agent_type="refactor", transform_input="Refactor based on review:\n{prev}"),
                    ChainStep(agent_type="test", transform_input="Write tests for refactored code:\n{prev}"),
                    ChainStep(agent_type="review", transform_input="Final review:\n{prev}"),
                ],
                created_by="system",
                created_at=datetime.now().isoformat(),
                is_template=True,
            ),
        ]
        
        for template in templates:
            self.chains[template.id] = template
    
    def set_agent_engine(self, agent_engine):
        """Set the agent engine for spawning agents."""
        self.agent_engine = agent_engine
    
    def create_chain(
        self,
        user_id: str,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
    ) -> AgentChain:
        """Create a new custom chain."""
        import hashlib
        
        chain_id = hashlib.sha256(
            f"{user_id}:{name}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        chain_steps = [
            ChainStep(
                agent_type=step["agent_type"],
                transform_input=step.get("transform_input"),
                condition=step.get("condition"),
                fallback_agent=step.get("fallback_agent"),
            )
            for step in steps
        ]
        
        chain = AgentChain(
            id=chain_id,
            name=name,
            description=description,
            steps=chain_steps,
            created_by=user_id,
            created_at=datetime.now().isoformat(),
            is_template=False,
        )
        
        self.chains[chain_id] = chain
        
        if user_id not in self.user_chains:
            self.user_chains[user_id] = []
        self.user_chains[user_id].append(chain_id)
        
        logger.info(f"🔗 Created chain: {name} ({chain_id})")
        return chain
    
    def get_chain(self, chain_id: str) -> Optional[AgentChain]:
        """Get a chain by ID."""
        return self.chains.get(chain_id)
    
    def list_chains(self, user_id: Optional[str] = None) -> List[AgentChain]:
        """List available chains."""
        if user_id:
            # User's chains + templates
            user_chain_ids = self.user_chains.get(user_id, [])
            return [
                chain for chain in self.chains.values()
                if chain.id in user_chain_ids or chain.is_template
            ]
        else:
            # All templates
            return [chain for chain in self.chains.values() if chain.is_template]
    
    def delete_chain(self, chain_id: str, user_id: str) -> bool:
        """Delete a user's chain."""
        chain = self.chains.get(chain_id)
        if not chain or chain.is_template or chain.created_by != user_id:
            return False
        
        del self.chains[chain_id]
        if user_id in self.user_chains:
            self.user_chains[user_id] = [
                cid for cid in self.user_chains[user_id] if cid != chain_id
            ]
        
        logger.info(f"🗑️ Deleted chain: {chain_id}")
        return True
    
    async def execute_chain(
        self,
        chain_id: str,
        initial_task: str,
        context: List[Dict[str, Any]],
        preferred_provider: Optional[str] = None,
    ) -> ChainExecutionResult:
        """Execute an agent chain."""
        start_time = datetime.now()
        
        chain = self.chains.get(chain_id)
        if not chain:
            return ChainExecutionResult(
                final_output="",
                steps_executed=0,
                step_outputs={},
                execution_time_ms=0,
                success=False,
                error=f"Chain not found: {chain_id}",
            )
        
        if not self.agent_engine:
            return ChainExecutionResult(
                final_output="",
                steps_executed=0,
                step_outputs={},
                execution_time_ms=0,
                success=False,
                error="Agent engine not set",
            )
        
        logger.info(f"🔗 Executing chain: {chain.name} ({len(chain.steps)} steps)")
        
        current_output = initial_task
        step_outputs = {}
        steps_executed = 0
        
        for i, step in enumerate(chain.steps):
            logger.info(f"  Step {i+1}/{len(chain.steps)}: {step.agent_type}")
            
            # Build task for this step
            if step.transform_input and steps_executed > 0:
                task = step.transform_input.replace("{prev}", current_output)
                task = task.replace("{original}", initial_task)
            else:
                task = current_output
            
            # Check condition if specified
            if step.condition:
                if not self._evaluate_condition(step.condition, current_output):
                    logger.info(f"    Skipping (condition not met)")
                    continue
            
            try:
                result = await self.agent_engine.spawn(
                    task=task,
                    context=context,
                    agent_type=step.agent_type,
                    model=preferred_provider,
                )
                
                output = result.get("content", "")
                if not output and step.fallback_agent:
                    # Try fallback agent
                    logger.info(f"    Trying fallback: {step.fallback_agent}")
                    result = await self.agent_engine.spawn(
                        task=task,
                        context=context,
                        agent_type=step.fallback_agent,
                        model=preferred_provider,
                    )
                    output = result.get("content", "")
                
                current_output = output
                step_outputs[f"step_{i+1}_{step.agent_type}"] = output
                steps_executed += 1
                
            except Exception as e:
                logger.error(f"    Step failed: {e}")
                if step.fallback_agent:
                    try:
                        result = await self.agent_engine.spawn(
                            task=task,
                            context=context,
                            agent_type=step.fallback_agent,
                            model=preferred_provider,
                        )
                        current_output = result.get("content", "")
                        step_outputs[f"step_{i+1}_{step.fallback_agent}"] = current_output
                        steps_executed += 1
                    except:
                        pass
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Chain complete: {steps_executed}/{len(chain.steps)} steps")
        
        return ChainExecutionResult(
            final_output=current_output,
            steps_executed=steps_executed,
            step_outputs=step_outputs,
            execution_time_ms=execution_time,
            success=steps_executed > 0,
        )
    
    def _evaluate_condition(self, condition: str, output: str) -> bool:
        """Evaluate a simple condition."""
        condition_lower = condition.lower()
        output_lower = output.lower()
        
        # Simple keyword-based conditions
        if condition.startswith("contains:"):
            keyword = condition[9:].strip().lower()
            return keyword in output_lower
        
        if condition.startswith("not_contains:"):
            keyword = condition[13:].strip().lower()
            return keyword not in output_lower
        
        if condition.startswith("length_gt:"):
            try:
                min_length = int(condition[10:].strip())
                return len(output) > min_length
            except:
                return True
        
        # Default: always true
        return True
    
    def clone_template(self, template_id: str, user_id: str, new_name: str) -> Optional[AgentChain]:
        """Clone a template to create a user chain."""
        template = self.chains.get(template_id)
        if not template or not template.is_template:
            return None
        
        return self.create_chain(
            user_id=user_id,
            name=new_name,
            description=template.description,
            steps=[
                {
                    "agent_type": step.agent_type,
                    "transform_input": step.transform_input,
                    "condition": step.condition,
                    "fallback_agent": step.fallback_agent,
                }
                for step in template.steps
            ],
        )


# Global instance
agent_chaining = AgentChainingEngine()
