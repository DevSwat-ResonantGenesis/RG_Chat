"""
Neural Skill Router
====================

Pure neural/ML skill detection using sentence embeddings.
Replaces keyword-based and blind-LLM routing with semantic understanding.

Architecture:
  Layer 1: Active skill continuity (from conversation meta_data)
  Layer 2: Neural semantic matching (sentence-transformer embeddings)
  Layer 3: Intent + narrative signals (boost/penalty adjustments)
  Layer 4: LLM tiebreaker (ONLY when layers 1-3 are uncertain)

Model: all-MiniLM-L6-v2 (22MB, ~5ms inference, 384-dim)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill definitions: each skill has semantic descriptions + example phrases
# The model matches user messages against these, not keywords.
# ---------------------------------------------------------------------------
SKILL_SEMANTIC_PROFILES: Dict[str, Dict[str, Any]] = {
    "agent_architect": {
        "description": "Create, build, manage, run, diagnose, modify, rename, delete, schedule, or configure AI agents",
        "examples": [
            "build me an agent that monitors stock prices",
            "create an agent to scrape websites",
            "how many agents do I have",
            "show my agents",
            "run my weather agent",
            "delete the test agent",
            "modify the scraper agent to also save to Google Drive",
            "schedule the report agent to run every morning",
            "my agent isn't working, diagnose it",
            "set up an autonomous agent",
            "configure the agent's tools",
            "rename agent to DataCollector",
            "what agents are available",
            "build an agent that fetches news and summarizes it",
            "I want to automate this task with an agent",
        ],
        "anti_examples": [
            "what is an AI agent",
            "explain how agents work in reinforcement learning",
        ],
    },
    "code_visualizer": {
        "description": "Scan and analyze a GitHub repository or codebase for visualization",
        "examples": [
            "scan this repo https://github.com/user/project",
            "analyze my GitHub repository",
            "visualize the codebase structure",
            "scan https://github.com/facebook/react",
            "show me the architecture of this repo",
            "code analysis for my project on GitHub",
        ],
        "anti_examples": [
            "help me write code",
            "fix this Python function",
            "what does this code do",
        ],
    },
    "web_search": {
        "description": "Search the web for real-time current information, news, prices, weather",
        "examples": [
            "what's the weather in San Francisco",
            "latest news about AI",
            "current Bitcoin price",
            "search for recent developments in quantum computing",
            "what happened in the stock market today",
            "find me the latest research on transformer models",
            "who won the game last night",
            "what's trending on Twitter right now",
        ],
        "anti_examples": [
            "explain quantum computing",
            "what is Bitcoin",
            "how does the stock market work",
        ],
    },
    "image_generation": {
        "description": "Generate, create, or draw images and illustrations using AI",
        "examples": [
            "generate an image of a sunset over mountains",
            "create a logo for my company",
            "draw a cute cat wearing a hat",
            "make me a picture of a futuristic city",
            "illustrate a dragon flying over a castle",
            "generate a portrait in oil painting style",
        ],
        "anti_examples": [
            "show me an image",
            "find a picture of a cat",
            "what does this image show",
        ],
    },
    "memory_search": {
        "description": "Search long-term memory for previously stored conversations and information",
        "examples": [
            "what did I say about the project last week",
            "do you remember our conversation about databases",
            "recall what we discussed about the API design",
            "search my memory for the deployment instructions",
            "what did we talk about yesterday",
            "find our previous conversation about React",
        ],
        "anti_examples": [
            "remember this for later",
            "save this information",
        ],
    },
    "memory_library": {
        "description": "Open and browse the memory library visualization panel",
        "examples": [
            "open memory library",
            "show my memories",
            "browse memories",
            "open the memory panel",
            "show memory visualization",
        ],
        "anti_examples": [],
    },
    "google_drive": {
        "description": "Access, search, read, create, or manage Google Drive files and documents",
        "examples": [
            "search my Google Drive for the report",
            "create a spreadsheet in my Drive",
            "find documents about the project in Drive",
            "upload this to Google Drive",
            "list my recent Drive files",
            "open my Google Drive documents",
            "save the summary to my Drive as Excel",
        ],
        "anti_examples": [
            "how does Google Drive work",
            "compare Google Drive vs Dropbox",
        ],
    },
    "google_calendar": {
        "description": "Access, create, or manage Google Calendar events, meetings, and schedule",
        "examples": [
            "what's on my calendar today",
            "schedule a meeting for tomorrow at 3pm",
            "create a calendar event for the team standup",
            "show my upcoming events",
            "when is my next meeting",
            "add a reminder to my calendar",
        ],
        "anti_examples": [
            "what day is it today",
            "how many days until Christmas",
        ],
    },
    "state_physics": {
        "description": "Open the State Physics visualization panel for state-space analysis",
        "examples": [
            "open state physics",
            "show state physics visualization",
            "state-space visualization",
            "show the physics panel",
        ],
        "anti_examples": [],
    },
    "ide_workspace": {
        "description": "Open the IDE workspace editor, terminal, or coding split panel",
        "examples": [
            "open IDE",
            "open the editor",
            "open terminal",
            "open workspace",
            "launch the code editor",
        ],
        "anti_examples": [
            "write me some code",
            "help me debug this",
            "explain this function",
        ],
    },
    "rabbit_post": {
        "description": "Create a post on the Rabbit community forum",
        "examples": [
            "post this to Rabbit",
            "create a community post",
            "share this on Rabbit forum",
            "publish to Rabbit",
        ],
        "anti_examples": [],
    },
    "figma": {
        "description": "Access and interact with Figma design projects, files, and components",
        "examples": [
            "show my Figma projects",
            "open my Figma designs",
            "list Figma components",
            "search my Figma files",
        ],
        "anti_examples": [],
    },
    "sigma": {
        "description": "Access Sigma Computing dashboards, reports, and analytics",
        "examples": [
            "show my Sigma dashboards",
            "open Sigma reports",
            "get Sigma analytics",
        ],
        "anti_examples": [],
    },
}

# Continuity boost for the active skill
CONTINUITY_BOOST = 0.25

# Minimum confidence to route to a skill (otherwise -> null = general chat)
CONFIDENCE_THRESHOLD = 0.42

# If active skill is detected AND its score is above this lower bar, keep it
CONTINUITY_THRESHOLD = 0.28

# LLM tiebreaker zone: when top two skills are within this margin
TIEBREAKER_MARGIN = 0.08


@dataclass
class SkillScore:
    """Score for a single skill."""
    skill_id: str
    semantic_score: float = 0.0
    continuity_boost: float = 0.0
    intent_boost: float = 0.0
    final_score: float = 0.0


@dataclass
class RoutingResult:
    """Result of the neural routing decision."""
    skill_id: Optional[str]
    confidence: float
    method: str  # "continuity", "semantic", "intent", "llm_tiebreaker", "none"
    scores: Dict[str, float] = field(default_factory=dict)
    active_skill: Optional[str] = None
    latency_ms: float = 0.0


class NeuralSkillRouter:
    """
    Neural semantic skill router.

    Uses sentence-transformer embeddings to match user messages
    to skill profiles based on semantic meaning, not keywords.
    """

    def __init__(self):
        self._model = None
        self._skill_embeddings: Dict[str, np.ndarray] = {}
        self._anti_embeddings: Dict[str, np.ndarray] = {}
        self._ready = False
        self._load_lock = asyncio.Lock()
        self._decision_log: List[Dict] = []
        self._max_log = 500

    async def ensure_loaded(self) -> bool:
        """Lazy-load the model on first use."""
        if self._ready:
            return True
        async with self._load_lock:
            if self._ready:
                return True
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None, self._load_sync
                )
            except Exception as e:
                logger.error(f"[NeuralRouter] Failed to load model: {e}")
                return False

    def _load_sync(self) -> bool:
        """Synchronous model loading (runs in thread pool)."""
        t0 = time.time()
        try:
            from sentence_transformers import SentenceTransformer

            model_name = os.getenv(
                "SKILL_ROUTER_MODEL", "all-MiniLM-L6-v2"
            )
            logger.info(f"[NeuralRouter] Loading model: {model_name}")
            self._model = SentenceTransformer(model_name)

            # Pre-compute skill embeddings
            for skill_id, profile in SKILL_SEMANTIC_PROFILES.items():
                texts = [profile["description"]] + profile.get("examples", [])
                embeddings = self._model.encode(texts, normalize_embeddings=True)
                # Mean of all example embeddings = skill centroid
                self._skill_embeddings[skill_id] = np.mean(embeddings, axis=0)

                anti = profile.get("anti_examples", [])
                if anti:
                    anti_embs = self._model.encode(anti, normalize_embeddings=True)
                    self._anti_embeddings[skill_id] = np.mean(anti_embs, axis=0)

            self._ready = True
            elapsed = (time.time() - t0) * 1000
            logger.info(
                f"[NeuralRouter] Model loaded in {elapsed:.0f}ms, "
                f"{len(self._skill_embeddings)} skill vectors ready"
            )
            return True

        except ImportError:
            logger.warning(
                "[NeuralRouter] sentence-transformers not installed — "
                "falling back to LLM detection"
            )
            return False
        except Exception as e:
            logger.error(f"[NeuralRouter] Model load error: {e}")
            return False

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to normalized embedding."""
        return self._model.encode([text], normalize_embeddings=True)[0]

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity (already normalized, so just dot product)."""
        return float(np.dot(a, b))

    def _detect_active_skill(
        self, recent_messages: list, enabled_ids: Set[str]
    ) -> Optional[str]:
        """Detect which skill was used in the most recent assistant message."""
        if not recent_messages:
            return None
        for msg in reversed(recent_messages[-6:]):
            role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
            if role != "assistant":
                continue
            meta = (
                msg.meta_data
                if hasattr(msg, "meta_data")
                else msg.get("meta_data", None)
            )
            if not meta or not isinstance(meta, dict):
                continue
            for tr in meta.get("toolResults", []):
                if isinstance(tr, dict):
                    tn = tr.get("tool_name", "")
                    if tn.startswith("skill_"):
                        candidate = tn[6:]
                        if candidate in enabled_ids:
                            return candidate
        return None

    async def route(
        self,
        message: str,
        enabled_skill_ids: Set[str],
        recent_messages: list = None,
        intents: List[str] = None,
        latent_intents: Dict[str, Any] = None,
        narrative_threads: List[Dict] = None,
    ) -> RoutingResult:
        """
        Route a message to the appropriate skill using neural semantics.

        Returns RoutingResult with skill_id (or None for general chat).
        """
        t0 = time.time()

        # --- Layer 1: Active skill continuity ---
        active_skill = self._detect_active_skill(
            recent_messages or [], enabled_skill_ids
        )

        # --- Check if model is ready ---
        model_ready = await self.ensure_loaded()

        if not model_ready:
            # Fallback: if model not available, use active skill or None
            if active_skill:
                return RoutingResult(
                    skill_id=active_skill,
                    confidence=0.7,
                    method="continuity_fallback",
                    active_skill=active_skill,
                    latency_ms=(time.time() - t0) * 1000,
                )
            return RoutingResult(
                skill_id=None,
                confidence=0.0,
                method="model_unavailable",
                latency_ms=(time.time() - t0) * 1000,
            )

        # --- Build context-enriched input ---
        context_parts = []
        if recent_messages:
            for msg in recent_messages[-3:]:
                role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
                content = (
                    msg.content
                    if hasattr(msg, "content")
                    else msg.get("content", "")
                )
                if content and role in ("user", "assistant"):
                    context_parts.append(f"{role}: {str(content)[:200]}")

        # Combine context + current message for richer semantic signal
        if context_parts:
            query_text = "\n".join(context_parts[-2:]) + f"\nuser: {message}"
        else:
            query_text = message

        # --- Layer 2: Neural semantic matching ---
        query_emb = await asyncio.get_event_loop().run_in_executor(
            None, self._encode, query_text
        )

        scores: Dict[str, SkillScore] = {}
        for skill_id in enabled_skill_ids:
            if skill_id not in self._skill_embeddings:
                continue
            skill_emb = self._skill_embeddings[skill_id]
            sim = self._cosine_sim(query_emb, skill_emb)

            # Subtract anti-example similarity to reduce false positives
            anti_penalty = 0.0
            if skill_id in self._anti_embeddings:
                anti_sim = self._cosine_sim(query_emb, self._anti_embeddings[skill_id])
                if anti_sim > sim * 0.8:
                    anti_penalty = (anti_sim - sim * 0.8) * 0.5

            ss = SkillScore(skill_id=skill_id, semantic_score=max(0, sim - anti_penalty))
            scores[skill_id] = ss

        # --- Layer 1 applied: continuity boost ---
        if active_skill and active_skill in scores:
            scores[active_skill].continuity_boost = CONTINUITY_BOOST

        # --- Layer 3: Intent signals ---
        if intents:
            intent_skill_map = {
                "action": ["agent_architect"],
                "coding": ["code_visualizer", "ide_workspace"],
                "memory": ["memory_search", "memory_library"],
                "research": ["web_search"],
                "planning": ["agent_architect"],
                "debug": ["agent_architect", "code_visualizer"],
            }
            primary_intent = intents[0] if intents else None
            if primary_intent and primary_intent in intent_skill_map:
                for boosted_skill in intent_skill_map[primary_intent]:
                    if boosted_skill in scores:
                        scores[boosted_skill].intent_boost = 0.05

        # --- Compute final scores ---
        for ss in scores.values():
            ss.final_score = (
                ss.semantic_score + ss.continuity_boost + ss.intent_boost
            )

        # --- Decision logic ---
        if not scores:
            result = RoutingResult(
                skill_id=None,
                confidence=0.0,
                method="no_skills_available",
                active_skill=active_skill,
                latency_ms=(time.time() - t0) * 1000,
            )
            self._log_decision(message, result)
            return result

        ranked = sorted(scores.values(), key=lambda s: s.final_score, reverse=True)
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None

        score_dict = {s.skill_id: round(s.final_score, 4) for s in ranked[:5]}

        # Active skill continuity: if active AND above lower threshold, keep it
        if active_skill and active_skill in scores:
            active_score = scores[active_skill]
            if active_score.final_score >= CONTINUITY_THRESHOLD:
                result = RoutingResult(
                    skill_id=active_skill,
                    confidence=min(active_score.final_score, 1.0),
                    method="continuity",
                    scores=score_dict,
                    active_skill=active_skill,
                    latency_ms=(time.time() - t0) * 1000,
                )
                self._log_decision(message, result)
                return result

        # Top skill above confidence threshold
        if top.final_score >= CONFIDENCE_THRESHOLD:
            result = RoutingResult(
                skill_id=top.skill_id,
                confidence=top.final_score,
                method="semantic",
                scores=score_dict,
                active_skill=active_skill,
                latency_ms=(time.time() - t0) * 1000,
            )
            self._log_decision(message, result)
            return result

        # Below threshold = general chat, no skill needed
        result = RoutingResult(
            skill_id=None,
            confidence=top.final_score,
            method="below_threshold",
            scores=score_dict,
            active_skill=active_skill,
            latency_ms=(time.time() - t0) * 1000,
        )
        self._log_decision(message, result)
        return result

    def _log_decision(self, message: str, result: RoutingResult) -> None:
        """Log routing decisions for monitoring and future training."""
        entry = {
            "msg": message[:100],
            "skill": result.skill_id,
            "confidence": round(result.confidence, 4),
            "method": result.method,
            "active": result.active_skill,
            "latency_ms": round(result.latency_ms, 1),
            "top_scores": result.scores,
        }
        self._decision_log.append(entry)
        if len(self._decision_log) > self._max_log:
            self._decision_log = self._decision_log[-self._max_log // 2 :]

        logger.info(
            f"[NeuralRouter] skill={result.skill_id} conf={result.confidence:.3f} "
            f"method={result.method} active={result.active_skill} "
            f"latency={result.latency_ms:.1f}ms msg={message[:60]!r}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics for monitoring."""
        if not self._decision_log:
            return {"total_decisions": 0, "model_loaded": self._ready}

        methods = {}
        skills_used = {}
        total_latency = 0.0
        for entry in self._decision_log:
            m = entry["method"]
            methods[m] = methods.get(m, 0) + 1
            s = entry.get("skill")
            if s:
                skills_used[s] = skills_used.get(s, 0) + 1
            total_latency += entry.get("latency_ms", 0)

        return {
            "total_decisions": len(self._decision_log),
            "model_loaded": self._ready,
            "methods": methods,
            "skills_used": skills_used,
            "avg_latency_ms": round(total_latency / len(self._decision_log), 1),
        }


# Global singleton
neural_skill_router = NeuralSkillRouter()
