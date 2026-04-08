"""Master prompt assembler for the Resonant Agent Architect.

Combines all prompt modules into a single system prompt for the ReAct agent loop.
"""

from .identity import IDENTITY_PROMPT
from .system import SYSTEM_PROMPT
from .session_start import SESSION_START_PROMPT
from .modes import MODES_PROMPT
from .lifecycle import LIFECYCLE_PROMPT
from .style import STYLE_PROMPT


def build_architect_system_prompt(
    workspace_summary: str = "",
    memory_facts: str = "",
) -> str:
    """Assemble the full architect system prompt with live context injected.

    Args:
        workspace_summary: Live summary of user's agents (from analyze_workspace).
        memory_facts: Known facts about the user from memory service.

    Returns:
        Complete system prompt string for the ReAct agent loop.
    """
    parts = [
        IDENTITY_PROMPT,
        SYSTEM_PROMPT,
        SESSION_START_PROMPT,
        MODES_PROMPT,
        LIFECYCLE_PROMPT,
        STYLE_PROMPT,
    ]

    # Inject live workspace context
    context_block = "\n<live_context>\n"
    context_block += f"CURRENT WORKSPACE:\n{workspace_summary or 'No agents yet — fresh workspace.'}\n"
    if memory_facts:
        context_block += f"\nKNOWN USER FACTS:\n{memory_facts}\n"
    context_block += "</live_context>"
    parts.append(context_block)

    # Tool usage instructions (specific to ReAct loop)
    parts.append("""<tool_instructions>
You have tools to take action. Use them in this order:

1. FIRST: search_memory — understand who this user is, what they've built before, what failed, what they prefer
2. THEN: analyze_workspace or analyze_agent — understand the current state of their agents
3. THEN: REASON about what you found — what's good, what's wrong, what's missing, what could be better
4. THEN: ACT — modify_agent, create_agent, run_agent — take the action that helps most
5. THEN: store_insight — save what you learned for next time
6. FINALLY: respond_to_user — explain everything in natural, advisory language with specific reasoning

CRITICAL RULES:
- Maximum 8 tool calls per turn. Be efficient.
- ALWAYS end with respond_to_user.
- Never fabricate agent IDs — get them from analyze_workspace or analyze_agent.
- Never dump raw data. Always INTERPRET it and explain what it means for the user.
- When you find a problem, FIX it and explain why. Don't just report it.
- When you create or modify, explain the REASONING behind every choice.
</tool_instructions>""")

    return "\n\n".join(parts)
