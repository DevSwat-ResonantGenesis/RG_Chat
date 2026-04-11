"""
Multi-Agent Team Engine (MATE)
===============================

Implements internal teams that coordinate multiple agents for complex tasks.
Teams run agents in sequence or parallel and merge results.

Teams:
- code_review_team: code → review → test
- security_audit_team: security → review → architecture  
- architecture_team: architecture → review → planning
- learning_team: explain → research → summary
- debug_team: debug → test → review
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TeamResult:
    """Result from a team execution."""
    content: str
    team_name: str
    agents_used: List[str]
    workflow_type: str
    execution_time_ms: float
    agent_outputs: Dict[str, str] = field(default_factory=dict)
    merged: bool = False


@dataclass
class TeamDefinition:
    """Definition of an internal team."""
    name: str
    agents: List[str]
    workflow: str  # 'sequential' or 'parallel_merge'
    description: str
    trigger_keywords: List[str]


# Internal team definitions
INTERNAL_TEAMS: Dict[str, TeamDefinition] = {
    "code_review_team": TeamDefinition(
        name="Code Review Team",
        agents=["code", "review", "test"],
        workflow="sequential",
        description="Full code review pipeline: generate code, review it, then create tests",
        trigger_keywords=["full review", "review my code", "code audit", "complete review", "review and test"]
    ),
    "security_audit_team": TeamDefinition(
        name="Security Audit Team",
        agents=["security", "review", "architecture"],
        workflow="parallel_merge",
        description="Comprehensive security analysis from multiple perspectives",
        trigger_keywords=["security audit", "penetration test", "vulnerability scan", "security review", "audit security"]
    ),
    "architecture_team": TeamDefinition(
        name="Architecture Team",
        agents=["architecture", "review", "planning"],
        workflow="sequential",
        description="System design with review and implementation planning",
        trigger_keywords=["design system", "architect", "system design", "design architecture", "plan architecture"]
    ),
    "learning_team": TeamDefinition(
        name="Learning Team",
        agents=["explain", "research", "summary"],
        workflow="sequential",
        description="Educational content: explain, research deeper, then summarize",
        trigger_keywords=["teach me", "learn about", "tutorial", "educate me", "help me understand"]
    ),
    "debug_team": TeamDefinition(
        name="Debug Team",
        agents=["debug", "test", "review"],
        workflow="sequential",
        description="Thorough debugging: find bugs, create tests, review fixes",
        trigger_keywords=["fix everything", "debug thoroughly", "find all bugs", "complete debug", "fix and test"]
    ),
    "full_stack_team": TeamDefinition(
        name="Full Stack Team",
        agents=["api", "database", "code", "test"],
        workflow="sequential",
        description="End-to-end feature development: API design, database, implementation, tests",
        trigger_keywords=["full stack", "end to end", "complete feature", "build feature", "full implementation"]
    ),
    "refactor_team": TeamDefinition(
        name="Refactor Team",
        agents=["review", "refactor", "test"],
        workflow="sequential",
        description="Safe refactoring: review current code, refactor, then verify with tests",
        trigger_keywords=["safe refactor", "refactor with tests", "clean and test", "refactor safely"]
    ),
    "accessibility_team": TeamDefinition(
        name="Accessibility Team",
        agents=["accessibility", "review", "test"],
        workflow="sequential",
        description="A11y compliance audit: check accessibility, review, create a11y tests",
        trigger_keywords=["accessibility audit", "a11y check", "wcag compliance", "make accessible"]
    ),
    "performance_team": TeamDefinition(
        name="Performance Team",
        agents=["optimization", "review", "test"],
        workflow="sequential",
        description="Performance optimization: analyze, optimize, verify with benchmarks",
        trigger_keywords=["performance audit", "speed optimization", "make faster", "optimize performance"]
    ),
}


class TeamEngine:
    """
    Multi-Agent Team Engine
    
    Coordinates multiple agents working together on complex tasks.
    Supports sequential and parallel workflows.
    """
    
    def __init__(self, agent_engine=None):
        self.agent_engine = agent_engine
        self.teams = INTERNAL_TEAMS
        self.execution_history: List[Dict[str, Any]] = []
    
    def set_agent_engine(self, agent_engine):
        """Set the agent engine for spawning agents."""
        self.agent_engine = agent_engine
    
    # Valid agent types that can be used in teams
    VALID_AGENT_TYPES = [
        "reasoning", "code", "research", "debug", "planning", "math",
        "security", "architecture", "test", "review", "explain",
        "optimization", "documentation", "migration", "api", "database",
        "devops", "refactor", "accessibility", "i18n", "regex", "git", "css",
        "summary",
    ]

    def register_team(
        self,
        team_id: str,
        name: str,
        agents: List[str],
        workflow: str = "sequential",
        description: str = "",
        trigger_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Register a new custom team dynamically.

        Returns a dict with the team's metadata on success.
        Raises ValueError for invalid inputs.
        """
        if not team_id or not name:
            raise ValueError("team_id and name are required")
        if not agents or len(agents) < 2:
            raise ValueError("A team must have at least 2 agents")
        if workflow not in ("sequential", "parallel_merge"):
            raise ValueError("workflow must be 'sequential' or 'parallel_merge'")
        # Validate agent types
        invalid = [a for a in agents if a not in self.VALID_AGENT_TYPES]
        if invalid:
            raise ValueError(
                f"Invalid agent types: {invalid}. "
                f"Valid types: {', '.join(self.VALID_AGENT_TYPES)}"
            )

        team_def = TeamDefinition(
            name=name,
            agents=agents,
            workflow=workflow,
            description=description or f"Custom team: {name}",
            trigger_keywords=trigger_keywords or [],
        )
        self.teams[team_id] = team_def
        logger.info(f"👥 Registered custom team: {name} ({team_id}) agents={agents} workflow={workflow}")
        return {
            "id": team_id,
            "name": name,
            "agents": agents,
            "workflow": workflow,
            "description": team_def.description,
        }

    def should_use_team(self, message: str) -> Optional[str]:
        """Determine if a team should be used based on message content.

        Conservative: only matches very explicit team requests with compound
        keywords (2+ words) in short messages (< 200 chars). Most messages
        should go straight to a single focused LLM agent, not a multi-agent team.
        Users can also explicitly select teams via the UI team picker (teamId).
        """
        message_lower = message.lower()

        if len(message_lower) > 200:
            return None

        for team_id, team_def in self.teams.items():
            for keyword in team_def.trigger_keywords:
                if " " not in keyword:
                    continue
                if keyword in message_lower:
                    logger.info(f"👥 Team triggered: {team_def.name} (keyword: '{keyword}')")
                    return team_id

        return None
    
    def get_team(self, team_id: str) -> Optional[TeamDefinition]:
        """Get team definition by ID."""
        return self.teams.get(team_id)
    
    def list_teams(self) -> List[Dict[str, Any]]:
        """List all available teams."""
        return [
            {
                "id": team_id,
                "name": team_def.name,
                "agents": team_def.agents,
                "workflow": team_def.workflow,
                "description": team_def.description,
                "trigger_keywords": team_def.trigger_keywords,
            }
            for team_id, team_def in self.teams.items()
        ]
    
    async def run_team(
        self,
        team_id: str,
        task: str,
        context: List[Dict[str, Any]],
        preferred_provider: Optional[str] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> TeamResult:
        """Run a team workflow on a task."""
        start_time = datetime.now()
        
        team_def = self.teams.get(team_id)
        if not team_def:
            raise ValueError(f"Unknown team: {team_id}")
        
        if not self.agent_engine:
            raise RuntimeError("Agent engine not set")
        
        logger.info(f"👥 Starting team: {team_def.name} with agents: {team_def.agents}, preferred_provider={preferred_provider}")
        
        if team_def.workflow == "sequential":
            result = await self._run_sequential(team_def, task, context, preferred_provider, images)
        elif team_def.workflow == "parallel_merge":
            result = await self._run_parallel_merge(team_def, task, context, preferred_provider, images)
        else:
            raise ValueError(f"Unknown workflow type: {team_def.workflow}")
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        team_result = TeamResult(
            content=result["content"],
            team_name=team_def.name,
            agents_used=team_def.agents,
            workflow_type=team_def.workflow,
            execution_time_ms=execution_time,
            agent_outputs=result.get("agent_outputs", {}),
            merged=result.get("merged", False),
        )
        
        # Record execution history
        self.execution_history.append({
            "team_id": team_id,
            "team_name": team_def.name,
            "task": task[:100],
            "execution_time_ms": execution_time,
            "timestamp": datetime.now().isoformat(),
        })
        
        logger.info(f"✅ Team {team_def.name} completed in {execution_time:.0f}ms")
        return team_result
    
    async def _run_sequential(
        self,
        team_def: TeamDefinition,
        task: str,
        context: List[Dict[str, Any]],
        preferred_provider: Optional[str] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run agents sequentially, passing output to next agent."""
        current_input = task
        agent_outputs = {}
        accumulated_context = list(context)
        
        for i, agent_type in enumerate(team_def.agents):
            logger.info(f"  🤖 Agent {i+1}/{len(team_def.agents)}: {agent_type} with provider={preferred_provider}")
            
            # Build task for this agent
            if i == 0:
                agent_task = current_input
            else:
                # Include previous agent's output in the task
                prev_agent = team_def.agents[i-1]
                agent_task = (
                    f"Previous agent ({prev_agent}) output:\n{current_input}\n\n"
                    f"Your task: Continue from the previous agent's work. "
                    f"Original request: {task}"
                )
            
            try:
                result = await self.agent_engine.spawn(
                    task=agent_task,
                    context=accumulated_context,
                    agent_type=agent_type,
                    model=preferred_provider,
                    images=images,
                )
                
                current_input = result.get("content", "")
                agent_outputs[agent_type] = current_input
                
                # Add to context for next agent
                accumulated_context.append({
                    "role": "assistant",
                    "content": f"[{agent_type} agent]: {current_input[:500]}"
                })
                
            except Exception as e:
                logger.error(f"Agent {agent_type} failed: {e}")
                agent_outputs[agent_type] = f"Error: {str(e)}"
        
        return {
            "content": current_input,
            "agent_outputs": agent_outputs,
            "merged": False,
        }
    
    async def _run_parallel_merge(
        self,
        team_def: TeamDefinition,
        task: str,
        context: List[Dict[str, Any]],
        preferred_provider: Optional[str] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run agents in parallel and merge their outputs."""
        agent_outputs = {}
        
        # Run all agents in parallel
        async def run_agent(agent_type: str) -> Tuple[str, str]:
            try:
                result = await self.agent_engine.spawn(
                    task=task,
                    context=context,
                    agent_type=agent_type,
                    model=preferred_provider,
                    images=images,
                )
                return agent_type, result.get("content", "")
            except Exception as e:
                logger.error(f"Agent {agent_type} failed: {e}")
                return agent_type, f"Error: {str(e)}"
        
        logger.info(f"  🔄 Running {len(team_def.agents)} agents in parallel...")
        tasks = [run_agent(agent_type) for agent_type in team_def.agents]
        results = await asyncio.gather(*tasks)
        
        for agent_type, output in results:
            agent_outputs[agent_type] = output
        
        # Merge outputs using a synthesis prompt
        logger.info("  🔀 Merging agent outputs...")
        merge_context = [
            {
                "role": "system",
                "content": (
                    "You are a synthesis agent merging expert opinions. "
                    "CRITICAL RULES:\n"
                    "1. Output ONLY the final synthesized answer\n"
                    "2. DO NOT mention 'Based on', 'Here is', or 'To address'\n"
                    "3. DO NOT list steps, actions, or meta-instructions\n"
                    "4. DO NOT reference 'the user message' or 'expert opinions'\n"
                    "5. Write as if YOU are the single expert giving ONE unified answer\n"
                    "6. Be direct and natural - no preamble or explanation of process\n\n"
                    "Merge these insights naturally:"
                )
            }
        ]
        
        for agent_type, output in agent_outputs.items():
            merge_context.append({
                "role": "user",
                "content": f"Expert ({agent_type}): {output}"
            })
        
        try:
            merge_result = await self.agent_engine.spawn(
                task="Synthesize these expert opinions into a single comprehensive response.",
                context=merge_context,
                agent_type="reasoning",
                model=preferred_provider,
                images=images,
            )
            merged_content = merge_result.get("content", "")
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            # Fallback: concatenate outputs
            merged_content = "\n\n".join([
                f"**{agent_type.title()} Analysis:**\n{output}"
                for agent_type, output in agent_outputs.items()
            ])
        
        return {
            "content": merged_content,
            "agent_outputs": agent_outputs,
            "merged": True,
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for teams."""
        if not self.execution_history:
            return {"total_executions": 0, "teams": {}}
        
        stats = {
            "total_executions": len(self.execution_history),
            "teams": {},
        }
        
        for entry in self.execution_history:
            team_id = entry["team_id"]
            if team_id not in stats["teams"]:
                stats["teams"][team_id] = {
                    "name": entry["team_name"],
                    "executions": 0,
                    "total_time_ms": 0,
                    "avg_time_ms": 0,
                }
            
            stats["teams"][team_id]["executions"] += 1
            stats["teams"][team_id]["total_time_ms"] += entry["execution_time_ms"]
        
        # Calculate averages
        for team_id, team_stats in stats["teams"].items():
            if team_stats["executions"] > 0:
                team_stats["avg_time_ms"] = team_stats["total_time_ms"] / team_stats["executions"]
        
        return stats


# Global instance
team_engine = TeamEngine()
