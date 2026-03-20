"""
Multi-Agent Debate Layer (MADL)
================================

Patch #41: Creates two internal reasoning agents that debate internally
(never shown to user) and return the best merged answer.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/debate_engine.py
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class DebateEngine:
    """
    Multi-Agent Debate Layer
    
    Creates two internal reasoning agents that debate internally
    and return the best merged answer.
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
            logger.info("🤖 Agent A (Analyst) thinking...")
            agent_a_context = [
                {
                    "role": "system",
                    "content": "You are Agent A: highly analytical, factual, logical. "
                               "Explain your answer briefly, no chain-of-thought. "
                               "Focus on precision and accuracy."
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
            
            # Agent B — Intuitive (creative, divergent, contextual)
            logger.info("🤖 Agent B (Intuitive) thinking...")
            agent_b_context = [
                {
                    "role": "system",
                    "content": "You are Agent B: creative, divergent, contextual. "
                               "Explain your answer briefly, no chain-of-thought. "
                               "Focus on creative solutions and alternative perspectives."
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
            
            # Evaluator — picks best answer
            logger.info("⚖️ Evaluator judging debate...")
            evaluator_context = [
                {
                    "role": "system",
                    "content": "You are the debate judge. "
                               "Do not output chain-of-thought. "
                               "Output only the final merged answer that combines "
                               "the best elements from both agents' responses. "
                               "Be concise and direct."
                },
                {
                    "role": "user",
                    "content": f"Agent A (Analytical): {agent_a_content}"
                },
                {
                    "role": "user",
                    "content": f"Agent B (Intuitive): {agent_b_content}"
                }
            ]
            
            evaluator_result = await self.router.route_query(
                    message="Evaluate which answer is better and give ONLY the final merged answer.",
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
        """Determine if debate should be used based on message content.

        DISABLED: Multi-agent debate without LLM reasoning produces worse answers
        than a single focused LLM call. Agents argue from memory fragments and
        reach consensus on the most common fragment, not the most correct answer.
        Re-enable only if debate agents are given LLM reasoning capability.
        """
        return False


# Global instance (router will be set later)
debate_engine = DebateEngine()
