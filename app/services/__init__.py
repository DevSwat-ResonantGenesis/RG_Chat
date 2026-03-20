# Chat services
from .resonance_hashing import ResonanceHasher
from .rag_engine import RAGEngine, rag_engine
from .memory_merge import merge_and_rank_memories, compute_hybrid_score
from .personality_dna import PersonalityDNA, personality_dna
from .intent_engine import IntentEngine, intent_engine
from .emotional_normalizer import EmotionalContextNormalizer, emotional_normalizer
from .multi_timeline_engine import MultiTimelineEngine, multi_timeline_engine
from .knowledge_graph import SelfEvolvingKnowledgeGraph, knowledge_graph
from .thought_branching import ProbabilisticThoughtBranching, thought_branching
from .evidence_graph import EvidenceGraph, evidence_graph
from .debate_engine import DebateEngine, debate_engine
from .agent_engine import AgentEngine, agent_engine
from .narrative_continuity_engine import NarrativeContinuityEngine, narrative_continuity_engine
from .temporal_thread_engine import TemporalThreadEngine, temporal_thread_engine
from .token_optimizer import ResonanceTokenOptimizer, token_optimizer
from .insight_seed_engine import InsightSeedEngine, insight_seed_engine
from .pmi_layer import PMIManager, pmi_manager
from .deterministic_universe import DeterministicResonanceHasher, UniverseDeriver, deterministic_hasher
from .latent_intent_predictor import LatentIntentPredictor, latent_intent_predictor
from .dual_memory_engine import DualMemoryEngine, dual_memory_engine
from .magnetic_pull import MagneticPullSystem, magnetic_pull_system
from .autonomous_error_correction import AutonomousErrorCorrection, error_correction
from .causal_reasoning import CausalReasoner, causal_reasoner
from .neural_gravity_engine import NeuralGravityEngine, neural_gravity_engine
from .hybrid_memory_ranker import compute_score, rank_memories
from .user_api_keys import UserApiKeyService, user_api_key_service
# Autonomous Services (L3-L5)
from .agent_router import AgentRouter, agent_router, route_message, RoutingDecision
from .response_cache import ResponseCache, response_cache, get_cached_response, cache_response
from .self_improving_agent import SelfImprovingAgent, self_improving_agent, FeedbackType
from .autonomous_planner import AutonomousPlanner, autonomous_planner, create_task_plan

__all__ = [
    # Core services
    "ResonanceHasher",
    "RAGEngine",
    "rag_engine",
    "merge_and_rank_memories",
    "compute_hybrid_score",
    # Context building
    "PersonalityDNA",
    "personality_dna",
    "IntentEngine",
    "intent_engine",
    "EmotionalContextNormalizer",
    "emotional_normalizer",
    # Advanced reasoning
    "MultiTimelineEngine",
    "multi_timeline_engine",
    "SelfEvolvingKnowledgeGraph",
    "knowledge_graph",
    "ProbabilisticThoughtBranching",
    "thought_branching",
    "EvidenceGraph",
    "evidence_graph",
    # Agent system
    "DebateEngine",
    "debate_engine",
    "AgentEngine",
    "agent_engine",
    # Narrative & Temporal
    "NarrativeContinuityEngine",
    "narrative_continuity_engine",
    "TemporalThreadEngine",
    "temporal_thread_engine",
    # Optimization
    "ResonanceTokenOptimizer",
    "token_optimizer",
    # Insight & PMI
    "InsightSeedEngine",
    "insight_seed_engine",
    "PMIManager",
    "pmi_manager",
    # Deterministic Universe
    "DeterministicResonanceHasher",
    "UniverseDeriver",
    "deterministic_hasher",
    # Latent Intent
    "LatentIntentPredictor",
    "latent_intent_predictor",
    # Dual Memory
    "DualMemoryEngine",
    "dual_memory_engine",
    # Magnetic Pull
    "MagneticPullSystem",
    "magnetic_pull_system",
    # Error Correction
    "AutonomousErrorCorrection",
    "error_correction",
    # Causal Reasoning
    "CausalReasoner",
    "causal_reasoner",
    # Neural Gravity
    "NeuralGravityEngine",
    "neural_gravity_engine",
    # Hybrid Memory Ranker
    "compute_score",
    "rank_memories",
    # User API Keys
    "UserApiKeyService",
    "user_api_key_service",
    # Autonomous Services (L3-L5)
    "AgentRouter",
    "agent_router",
    "route_message",
    "RoutingDecision",
    "ResponseCache",
    "response_cache",
    "get_cached_response",
    "cache_response",
    "SelfImprovingAgent",
    "self_improving_agent",
    "FeedbackType",
    "AutonomousPlanner",
    "autonomous_planner",
    "create_task_plan",
]
