"""Agent facade for internal agents, debate engine, and teams.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/domain/agent/facade.py
Extended with team support, memory, metrics, and specialization (Phase 4).
Extended with voting, confidence, feedback, chaining, sandbox, citations (Phase 5).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from ...services.agent_engine import agent_engine
from ...services.debate_engine import debate_engine
from ...services.team_engine import team_engine
from ...services.agent_memory import agent_memory_store
from ...services.agent_metrics import agent_metrics
from ...services.agent_specialization import agent_specialization
from ...services.dynamic_team_composer import dynamic_team_composer
from ...services.agent_voting import agent_voting
from ...services.agent_confidence import confidence_analyzer
from ...services.user_feedback import user_feedback
from ...services.agent_chaining import agent_chaining
from ...services.context_persistence import context_persistence
from ...services.code_sandbox import code_sandbox
from ...services.cross_validation import cross_validation
from ...services.source_citations import source_citations
from ...services.hallucination_detector import hallucination_detector
from ...domain.provider import get_router_for_internal_use

logger = logging.getLogger(__name__)


def _init_engines():
    """Initialize engines with router."""
    router = get_router_for_internal_use()
    agent_engine.set_router(router)
    debate_engine.set_router(router)
    team_engine.set_agent_engine(agent_engine)
    agent_voting.set_agent_engine(agent_engine)
    agent_chaining.set_agent_engine(agent_engine)
    cross_validation.set_agent_engine(agent_engine)


async def maybe_run_debate(
    *,
    message: str,
    context_messages: List[Dict],
    preferred_provider: Optional[str] = None,
    images: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[str], bool]:
    """Run debate if the debate engine decides it is needed.

    Returns (response_text, debate_used_flag).
    """
    _init_engines()
    
    use_debate = debate_engine.should_use_debate(message)
    response_text = None
    use_debate_flag = False

    if use_debate:
        try:
            result = await debate_engine.run_debate(
                task=message,
                context=context_messages,
                preferred_provider=preferred_provider,
                images=images,
            )
            response_text = result.get("content", "")
            if response_text:
                use_debate_flag = True
        except Exception:
            response_text = None
            use_debate_flag = False

    return response_text, use_debate_flag


async def maybe_spawn_agent(
    *,
    message: str,
    context_messages: List[Dict],
    user_id: Optional[str] = None,
    user_api_keys: Optional[Dict[str, str]] = None,
    preferred_provider: Optional[str] = None,
    forced_agent_type: Optional[str] = None,
    images: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[Dict]]:
    """Spawn an internal agent if AgentEngine decides it is needed.

    Returns (response_text, agent_type, llm_provider, router_metadata) where agent_type
    is None if no agent was used. llm_provider is the actual LLM provider used.
    router_metadata contains model, fallback_chain, was_fallback, usage, etc.
    """
    # Set user API keys on the router BEFORE initializing engines
    if user_api_keys:
        router = get_router_for_internal_use()
        router.set_user_api_keys(user_api_keys)
        logger.info(f"🔑 Set {len(user_api_keys)} user API keys for agent: {list(user_api_keys.keys())}")
    
    _init_engines()
    start_time = datetime.now()
    
    use_agent = forced_agent_type or agent_engine.should_spawn_agent(message)
    if not use_agent:
        return None, None, None, None

    try:
        # Get specialization prompt if available
        specialization_prompt = ""
        if user_id:
            specialization_prompt = agent_specialization.get_specialization_prompt(user_id, use_agent)
        
        # Get relevant memories
        memories = []
        if user_id:
            memories = agent_memory_store.retrieve(use_agent, user_id, message, limit=3)
        
        # Build enhanced context with memories and specialization
        enhanced_context = list(context_messages)
        if specialization_prompt:
            enhanced_context.insert(0, {"role": "system", "content": specialization_prompt})
        if memories:
            memory_context = "Relevant past interactions:\n" + "\n".join([
                f"- Task: {m.task[:100]}... Response: {m.response[:100]}..."
                for m in memories
            ])
            enhanced_context.insert(0, {"role": "system", "content": memory_context})
        
        result = await agent_engine.spawn(
            task=message,
            context=enhanced_context,
            agent_type=use_agent,
            model=preferred_provider,
            images=images,
        )
        response_text = result.get("content", "")
        llm_provider = result.get("provider", None)
        router_metadata = {
            "model": result.get("model"),
            "fallback_chain": result.get("fallback_chain"),
            "was_fallback": result.get("was_fallback", False),
            "preferred_provider": result.get("preferred_provider"),
            "usage": result.get("usage"),
        }
        
        # Record metrics
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        agent_metrics.record(
            agent_type=use_agent,
            execution_time_ms=execution_time,
            token_count=len(response_text.split()),  # Rough estimate
            success=bool(response_text),
            user_id=user_id or "anonymous",
            task_length=len(message),
            response_length=len(response_text),
        )
        
        # Store memory for future use
        if user_id and response_text:
            agent_memory_store.store(
                agent_type=use_agent,
                user_id=user_id,
                task=message,
                response=response_text,
                context=context_messages,
            )
            # Learn from interaction
            agent_specialization.learn_from_interaction(
                user_id=user_id,
                agent_type=use_agent,
                task=message,
                response=response_text,
                context=context_messages,
            )
        
        return response_text, use_agent, llm_provider, router_metadata
    except Exception as e:
        logger.error(f"Agent spawn failed: {e}")
        # Record failure metric
        if user_id:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            agent_metrics.record(
                agent_type=use_agent,
                execution_time_ms=execution_time,
                token_count=0,
                success=False,
                user_id=user_id,
                task_length=len(message),
                response_length=0,
                error_message=str(e),
            )
        return None, None, None, None


async def maybe_run_team(
    *,
    message: str,
    context_messages: List[Dict],
    preferred_provider: Optional[str] = None,
    user_id: Optional[str] = None,
    user_api_keys: Optional[Dict[str, str]] = None,
    images: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[str], Optional[str], bool]:
    """Run a team workflow if appropriate for the task.

    Returns (response_text, team_name, team_used_flag).
    """
    # Set user API keys on the router BEFORE initializing engines
    if user_api_keys:
        router = get_router_for_internal_use()
        router.set_user_api_keys(user_api_keys)
        logger.info(f"🔑 Set {len(user_api_keys)} user API keys for team: {list(user_api_keys.keys())}")
    
    _init_engines()
    
    # Check for predefined team triggers
    team_id = team_engine.should_use_team(message)
    
    # If no predefined team, check for dynamic team composition
    if not team_id:
        should_use_dynamic, dynamic_team = dynamic_team_composer.should_use_dynamic_team(message)
        if should_use_dynamic and dynamic_team:
            logger.info(f"🎯 Dynamic team composed: {dynamic_team.name}")
            try:
                # Run dynamic team using team engine's run methods
                result = await _run_dynamic_team(dynamic_team, message, context_messages, preferred_provider)
                return result, dynamic_team.name, True
            except Exception as e:
                logger.error(f"Dynamic team failed: {e}")
                return None, None, False
    
    if not team_id:
        return None, None, False
    
    try:
        result = await team_engine.run_team(
            team_id=team_id,
            task=message,
            context=context_messages,
            preferred_provider=preferred_provider,
            images=images,
        )
        return result.content, result.team_name, True
    except Exception as e:
        logger.error(f"Team execution failed: {e}")
        return None, None, False


async def _run_dynamic_team(
    team,
    task: str,
    context: List[Dict],
    preferred_provider: Optional[str] = None,
) -> str:
    """Run a dynamically composed team."""
    from ...services.team_engine import TeamDefinition
    
    # Create a temporary team definition
    temp_team = TeamDefinition(
        name=team.name,
        agents=team.agents,
        workflow=team.workflow,
        description=team.rationale,
        trigger_keywords=[],
    )
    
    # Add to team engine temporarily
    team_engine.teams["_dynamic"] = temp_team
    
    try:
        result = await team_engine.run_team(
            team_id="_dynamic",
            task=task,
            context=context,
            preferred_provider=preferred_provider,
        )
        return result.content
    finally:
        # Clean up temporary team
        if "_dynamic" in team_engine.teams:
            del team_engine.teams["_dynamic"]


def get_agent_stats() -> Dict[str, Any]:
    """Get agent performance statistics."""
    return {
        "metrics": agent_metrics.get_all_stats(),
        "memory": agent_memory_store.get_stats(),
        "teams": team_engine.get_execution_stats(),
        "top_agents": agent_metrics.get_top_agents(),
        "slowest_agents": agent_metrics.get_slowest_agents(),
        "errors": agent_metrics.get_error_summary(),
        "feedback": user_feedback.get_all_stats(),
    }


def get_team_list() -> List[Dict[str, Any]]:
    """Get list of available teams."""
    return team_engine.list_teams()


# ============================================
# Phase 5 Features
# ============================================

async def run_voting(
    *,
    task: str,
    context_messages: List[Dict],
    candidate_agents: List[str] = None,
    voter_agents: List[str] = None,
    preferred_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Run agent voting on a task."""
    _init_engines()
    
    result = await agent_voting.run_voting(
        task=task,
        context=context_messages,
        candidate_agents=candidate_agents,
        voter_agents=voter_agents,
        preferred_provider=preferred_provider,
    )
    
    return {
        "content": result.winner.content,
        "winner_agent": result.winner.agent_type,
        "total_votes": result.total_votes,
        "consensus_score": result.consensus_score,
        "voting_summary": result.voting_summary,
        "execution_time_ms": result.execution_time_ms,
    }


def analyze_confidence(response: str, task: str = "") -> Dict[str, Any]:
    """Analyze confidence of a response."""
    result = confidence_analyzer.analyze(response, task)
    return {
        "score": result.score,
        "level": result.level,
        "factors": result.factors,
        "should_escalate": result.should_escalate,
        "explanation": result.explanation,
        "badge": confidence_analyzer.get_confidence_badge(result.score),
    }


def submit_feedback(
    message_id: str,
    user_id: str,
    agent_type: str,
    is_positive: bool,
    task: str = "",
    response: str = "",
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """Submit user feedback for an agent response."""
    entry = user_feedback.submit_feedback(
        message_id=message_id,
        user_id=user_id,
        agent_type=agent_type,
        is_positive=is_positive,
        task=task,
        response=response,
        comment=comment,
    )
    return {
        "id": entry.id,
        "agent_type": entry.agent_type,
        "feedback_type": entry.feedback_type,
        "timestamp": entry.timestamp,
    }


def get_feedback_stats() -> Dict[str, Any]:
    """Get feedback statistics."""
    return {
        "all_stats": user_feedback.get_all_stats(),
        "best_agents": user_feedback.get_best_agents(),
        "needs_improvement": user_feedback.get_agents_needing_improvement(),
    }


async def run_chain(
    chain_id: str,
    task: str,
    context_messages: List[Dict],
    preferred_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute an agent chain."""
    _init_engines()
    
    result = await agent_chaining.execute_chain(
        chain_id=chain_id,
        initial_task=task,
        context=context_messages,
        preferred_provider=preferred_provider,
    )
    
    return {
        "content": result.final_output,
        "steps_executed": result.steps_executed,
        "step_outputs": result.step_outputs,
        "execution_time_ms": result.execution_time_ms,
        "success": result.success,
        "error": result.error,
    }


def get_chain_list(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get available chains."""
    chains = agent_chaining.list_chains(user_id)
    return [
        {
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "steps": [s.agent_type for s in c.steps],
            "is_template": c.is_template,
        }
        for c in chains
    ]


def create_chain(
    user_id: str,
    name: str,
    description: str,
    steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a custom chain."""
    chain = agent_chaining.create_chain(
        user_id=user_id,
        name=name,
        description=description,
        steps=steps,
    )
    return {
        "id": chain.id,
        "name": chain.name,
        "steps": [s.agent_type for s in chain.steps],
    }


async def execute_code(
    code: str,
    language: Optional[str] = None,
    test_input: str = "",
) -> Dict[str, Any]:
    """Execute code in sandbox."""
    result = await code_sandbox.execute(code, language, test_input)
    return {
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "execution_time_ms": result.execution_time_ms,
        "language": result.language,
        "exit_code": result.exit_code,
    }


async def validate_response(
    response: str,
    task: str,
    agent_type: str,
    context_messages: List[Dict] = None,
    preferred_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Cross-validate an agent response."""
    _init_engines()
    
    result = await cross_validation.validate(
        original_response=response,
        task=task,
        primary_agent=agent_type,
        context=context_messages,
        preferred_provider=preferred_provider,
    )
    
    return {
        "is_valid": result.is_valid,
        "confidence_boost": result.confidence_boost,
        "issues": result.issues_found,
        "corrections": result.corrections,
        "hallucinations": result.hallucination_flags,
        "validator": result.validator_agent,
        "summary": result.validation_summary,
    }


def add_citations(response: str, task: str = "", agent_type: str = "") -> Dict[str, Any]:
    """Add citations to a response."""
    result = source_citations.add_citations_to_response(response, task, agent_type)
    return {
        "content": result.content,
        "citations": [
            {"id": c.id, "type": c.source_type, "title": c.title, "url": c.url}
            for c in result.citations
        ],
        "citation_count": result.citation_count,
        "has_verified_sources": result.has_verified_sources,
    }


def detect_hallucinations(response: str, task: str = "") -> Dict[str, Any]:
    """Detect potential hallucinations in a response."""
    result = hallucination_detector.analyze(response, task)
    return {
        "risk_score": result.risk_score,
        "risk_level": result.risk_level,
        "flags": [
            {"type": f.type, "content": f.content, "confidence": f.confidence, "suggestion": f.suggestion}
            for f in result.flags
        ],
        "summary": result.summary,
        "should_warn": result.should_warn_user,
        "warning": hallucination_detector.get_warning_message(result),
    }


def get_project_context(user_id: str, project_name: str) -> Dict[str, Any]:
    """Get or create project context."""
    project = context_persistence.create_or_get_project(user_id, project_name)
    return {
        "project_id": project.project_id,
        "name": project.name,
        "languages": project.languages,
        "frameworks": project.frameworks,
        "session_count": project.session_count,
        "context_prompt": context_persistence.build_context_prompt(project.project_id),
    }


def update_project_context(
    project_id: str,
    languages: List[str] = None,
    frameworks: List[str] = None,
    coding_conventions: Dict[str, str] = None,
):
    """Update project context."""
    context_persistence.update_project_context(
        project_id=project_id,
        languages=languages,
        frameworks=frameworks,
        coding_conventions=coding_conventions,
    )
