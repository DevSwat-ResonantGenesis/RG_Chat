"""
Multi-Agent Debate Layer (MADL)
================================

Patch #41: Creates two internal reasoning agents that debate internally
(never shown to user) and return the best merged answer.

Both agents use FULL LLM reasoning via route_query (not KB fragments).
Debate triggers only for genuinely complex analytical/comparison tasks.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/debate_engine.py
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Debate triggers: analytical/comparison queries that benefit from two perspectives
_DEBATE_PATTERNS = [
    r"\b(?:compare|contrast|versus|vs\.?|pros?\s+(?:and|&)\s+cons?)\b",
    r"\b(?:trade[\s-]?offs?|advantages?\s+(?:and|&)\s+disadvantages?)\b",
    r"\b(?:which\s+is\s+better|should\s+i\s+(?:use|choose|pick))\b",
    r"\b(?:evaluate|assess|critique|analyze\s+(?:the|this|these))\b",
    r"\b(?:debate|argue\s+(?:for|against)|weigh\s+(?:the|up))\b",
    r"\b(?:strengths?\s+(?:and|&)\s+weaknesses?)\b",
    r"\b(?:best\s+approach|best\s+strategy|best\s+way\s+to)\b",
]
_DEBATE_RE = re.compile("|".join(_DEBATE_PATTERNS), re.IGNORECASE)

# Skip debate for these — simple/short/greeting messages
_SKIP_PATTERNS = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|ok|yes|no|sure|cool|bye|good)\b",
    re.IGNORECASE,
)

# Minimum message length to consider debate (avoids overhead for short questions)
_MIN_DEBATE_LENGTH = 40


class DebateEngine:
    """
    Multi-Agent Debate Layer

    Creates two internal reasoning agents that debate internally
    (never shown to user) and return the best merged answer.
    Both agents use full LLM reasoning via route_query.
    """

    def __init__(self, router=None):
        self.router = router

    def set_router(self, router):
        """Set the AI router for making LLM calls."""
        self.router = router

    async def run_debate(
        self,
        task: str,
        context: List[Dict[str, Any]],
        preferred_provider: Optional[str] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run a multi-agent debate and return the best merged answer."""
        try:
            if not self.router:
                logger.warning("MultiAIRouter not available, cannot run debate")
                return {
                    "content": "",
                    "provider": "error",
                    "error": "Router not available"
                }

            logger.info(f"🧠 Starting multi-agent debate for task: {task[:50]}...")

            # Agent A — Analytical (precise, logical, factual)
            logger.info("🤖 Agent A (Analyst) reasoning via LLM...")
            agent_a_context = [
                {
                    "role": "system",
                    "content": (
                        "You are Agent A: a rigorous analyst. "
                        "Think step-by-step about the question, weigh evidence, "
                        "and give a precise, well-reasoned answer. "
                        "Focus on accuracy, facts, and logical structure. "
                        "Be concise — no filler."
                    ),
                }
            ] + context

            agent_a_result = await self.router.route_query(
                message=task,
                context=agent_a_context,
                preferred_provider=preferred_provider or "groq",
                images=images,
            )
            agent_a_content = agent_a_result.get("response", "")
            logger.info(f"✅ Agent A completed: {len(agent_a_content)} chars")

            # Agent B — Creative strategist (divergent, contextual)
            logger.info("🤖 Agent B (Strategist) reasoning via LLM...")
            agent_b_context = [
                {
                    "role": "system",
                    "content": (
                        "You are Agent B: a creative strategist. "
                        "Consider alternative angles, edge cases, and real-world "
                        "implications the user might not have considered. "
                        "Offer practical, actionable insights. "
                        "Be concise — no filler."
                    ),
                }
            ] + context

            agent_b_result = await self.router.route_query(
                message=task,
                context=agent_b_context,
                preferred_provider=preferred_provider or "groq",
                images=images,
            )
            agent_b_content = agent_b_result.get("response", "")
            logger.info(f"✅ Agent B completed: {len(agent_b_content)} chars")

            # Evaluator — synthesises the best merged answer
            logger.info("⚖️ Evaluator synthesising best answer...")
            evaluator_context = [
                {
                    "role": "system",
                    "content": (
                        "You are a synthesis judge. You receive two expert perspectives. "
                        "Produce ONE unified answer that keeps the strongest points from each, "
                        "resolves contradictions, and reads as a single authoritative response. "
                        "Output ONLY the final answer — no labels, no 'Agent A said...'. "
                        "Be direct and natural."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Perspective 1 (Analyst):\n{agent_a_content}"
                },
                {
                    "role": "user",
                    "content": f"Perspective 2 (Strategist):\n{agent_b_content}"
                },
            ]

            evaluator_result = await self.router.route_query(
                message="Synthesise the two perspectives into one authoritative answer.",
                context=evaluator_context,
                preferred_provider=preferred_provider or "groq",
                images=images,
            )

            final_answer = evaluator_result.get("response", "")
            logger.info(f"✅ Debate complete: Final answer {len(final_answer)} chars")

            return {
                "content": final_answer,
                "provider": "debate_engine",
                "agent_a": agent_a_content[:100],
                "agent_b": agent_b_content[:100],
                "debate_used": True
            }

        except Exception as e:
            logger.error(f"Error in debate engine: {e}", exc_info=True)
            return {
                "content": "",
                "provider": "error",
                "error": str(e),
                "debate_used": False
            }

    def should_use_debate(self, message: str) -> bool:
        """Determine if multi-agent debate adds value for this message.

        Triggers for genuinely complex analytical/comparison tasks where
        two LLM perspectives produce a better answer than one.
        Skips greetings, short messages, and simple questions.
        """
        if not message or len(message.strip()) < _MIN_DEBATE_LENGTH:
            return False
        if _SKIP_PATTERNS.match(message.strip()):
            return False
        if _DEBATE_RE.search(message):
            logger.info(f"🧠 Debate triggered for: {message[:60]}...")
            return True
        return False


# Global instance (router will be set later)
debate_engine = DebateEngine()
