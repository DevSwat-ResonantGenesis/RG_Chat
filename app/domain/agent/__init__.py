# Agent domain layer
from .facade import (
    maybe_run_debate,
    maybe_spawn_agent,
    maybe_run_team,
    get_agent_stats,
    get_team_list,
    # Phase 5 features
    run_voting,
    analyze_confidence,
    submit_feedback,
    get_feedback_stats,
    run_chain,
    get_chain_list,
    create_chain,
    execute_code,
    validate_response,
    add_citations,
    detect_hallucinations,
    get_project_context,
    update_project_context,
)

__all__ = [
    "maybe_run_debate",
    "maybe_spawn_agent",
    "maybe_run_team",
    "get_agent_stats",
    "get_team_list",
]
