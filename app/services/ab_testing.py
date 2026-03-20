"""
A/B Testing Infrastructure for Memory Retrieval
Test different scoring formulas and configurations.

Provides deterministic variant assignment and result logging.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from threading import Lock
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of an A/B test experiment."""
    user_id: str
    variant: str
    query: str
    feedback: Optional[bool]  # True = positive, False = negative, None = no feedback
    latency_ms: float
    memory_count: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class VariantConfig:
    """Configuration for a scoring variant."""
    rag: float = 0.30
    resonance: float = 0.25
    proximity: float = 0.20
    recency: float = 0.15
    anchor: float = 0.10
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class MemoryABTest:
    """
    A/B testing for memory retrieval configurations.
    
    Supports multiple experiments with different variants.
    Uses deterministic assignment based on user_id hash.
    """
    
    def __init__(self, results_path: Optional[str] = None):
        """
        Initialize A/B testing infrastructure.
        
        Args:
            results_path: Optional path to persist experiment results
        """
        self.results_path = results_path
        self._lock = Lock()
        
        # Define experiments and variants
        self.experiments: Dict[str, Dict[str, VariantConfig]] = {
            "scoring_v1": {
                "control": VariantConfig(
                    rag=0.30, resonance=0.25, proximity=0.20, recency=0.15, anchor=0.10
                ),
                "semantic_heavy": VariantConfig(
                    rag=0.45, resonance=0.20, proximity=0.15, recency=0.10, anchor=0.10
                ),
                "recency_heavy": VariantConfig(
                    rag=0.25, resonance=0.20, proximity=0.15, recency=0.30, anchor=0.10
                ),
                "resonance_heavy": VariantConfig(
                    rag=0.25, resonance=0.35, proximity=0.20, recency=0.10, anchor=0.10
                ),
            },
            "memory_limit": {
                "control": {"limit": 20},
                "more_context": {"limit": 30},
                "less_context": {"limit": 10},
            },
        }
        
        # Results storage
        self._results: Dict[str, List[ExperimentResult]] = {}
        
        # Load existing results
        if results_path:
            self._load_results()
    
    def get_variant(self, user_id: str, experiment: str = "scoring_v1") -> str:
        """
        Get deterministic variant assignment for a user.
        
        Args:
            user_id: User ID
            experiment: Experiment name
            
        Returns:
            Variant name (e.g., "control", "semantic_heavy")
        """
        if experiment not in self.experiments:
            return "control"
        
        # Deterministic hash-based assignment
        hash_input = f"{user_id}:{experiment}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        variants = list(self.experiments[experiment].keys())
        variant_idx = hash_val % len(variants)
        
        return variants[variant_idx]
    
    def get_config(self, user_id: str, experiment: str = "scoring_v1") -> Dict[str, Any]:
        """
        Get configuration for user's variant.
        
        Args:
            user_id: User ID
            experiment: Experiment name
            
        Returns:
            Configuration dictionary
        """
        variant = self.get_variant(user_id, experiment)
        config = self.experiments.get(experiment, {}).get(variant)
        
        if config is None:
            return {}
        
        if isinstance(config, VariantConfig):
            return config.to_dict()
        return config
    
    def get_weights(self, user_id: str, experiment: str = "scoring_v1") -> Dict[str, float]:
        """
        Get scoring weights for user's variant.
        
        Args:
            user_id: User ID
            experiment: Experiment name
            
        Returns:
            Dictionary of weight name -> value
        """
        config = self.get_config(user_id, experiment)
        
        # Return only weight-related keys
        weight_keys = ["rag", "resonance", "proximity", "recency", "anchor"]
        return {k: v for k, v in config.items() if k in weight_keys}
    
    def log_result(
        self,
        user_id: str,
        experiment: str,
        query: str,
        latency_ms: float,
        memory_count: int,
        feedback: Optional[bool] = None
    ) -> None:
        """
        Log an experiment result.
        
        Args:
            user_id: User ID
            experiment: Experiment name
            query: Search query
            latency_ms: Query latency in milliseconds
            memory_count: Number of memories returned
            feedback: Optional user feedback (True/False/None)
        """
        variant = self.get_variant(user_id, experiment)
        
        result = ExperimentResult(
            user_id=user_id,
            variant=variant,
            query=query,
            feedback=feedback,
            latency_ms=latency_ms,
            memory_count=memory_count,
        )
        
        with self._lock:
            if experiment not in self._results:
                self._results[experiment] = []
            
            self._results[experiment].append(result)
            
            # Keep only last 10000 results per experiment
            if len(self._results[experiment]) > 10000:
                self._results[experiment] = self._results[experiment][-10000:]
        
        # Persist periodically
        if self.results_path and len(self._results.get(experiment, [])) % 100 == 0:
            self._save_results()
    
    def update_feedback(
        self,
        user_id: str,
        experiment: str,
        query: str,
        feedback: bool
    ) -> None:
        """
        Update feedback for a recent result.
        
        Args:
            user_id: User ID
            experiment: Experiment name
            query: Search query
            feedback: User feedback (True = positive, False = negative)
        """
        with self._lock:
            results = self._results.get(experiment, [])
            
            # Find matching result (most recent first)
            for result in reversed(results):
                if result.user_id == user_id and result.query == query:
                    result.feedback = feedback
                    break
    
    def get_stats(self, experiment: str = "scoring_v1") -> Dict[str, Any]:
        """
        Get experiment statistics.
        
        Args:
            experiment: Experiment name
            
        Returns:
            Statistics per variant
        """
        with self._lock:
            results = self._results.get(experiment, [])
            
            if not results:
                return {"experiment": experiment, "total_results": 0, "variants": {}}
            
            variants = self.experiments.get(experiment, {}).keys()
            stats = {}
            
            for variant in variants:
                variant_results = [r for r in results if r.variant == variant]
                
                if not variant_results:
                    stats[variant] = {
                        "total": 0,
                        "positive": 0,
                        "negative": 0,
                        "no_feedback": 0,
                        "positive_rate": 0.0,
                        "avg_latency_ms": 0.0,
                        "avg_memory_count": 0.0,
                    }
                    continue
                
                positive = len([r for r in variant_results if r.feedback is True])
                negative = len([r for r in variant_results if r.feedback is False])
                no_feedback = len([r for r in variant_results if r.feedback is None])
                total = len(variant_results)
                
                with_feedback = positive + negative
                positive_rate = positive / with_feedback if with_feedback > 0 else 0.0
                
                avg_latency = sum(r.latency_ms for r in variant_results) / total
                avg_memory_count = sum(r.memory_count for r in variant_results) / total
                
                stats[variant] = {
                    "total": total,
                    "positive": positive,
                    "negative": negative,
                    "no_feedback": no_feedback,
                    "positive_rate": round(positive_rate, 4),
                    "positive_rate_percent": f"{positive_rate * 100:.1f}%",
                    "avg_latency_ms": round(avg_latency, 2),
                    "avg_memory_count": round(avg_memory_count, 1),
                }
            
            return {
                "experiment": experiment,
                "total_results": len(results),
                "variants": stats,
            }
    
    def get_winner(self, experiment: str = "scoring_v1", min_samples: int = 100) -> Optional[str]:
        """
        Determine winning variant based on positive feedback rate.
        
        Args:
            experiment: Experiment name
            min_samples: Minimum samples per variant to declare winner
            
        Returns:
            Winning variant name or None if not enough data
        """
        stats = self.get_stats(experiment)
        variants = stats.get("variants", {})
        
        best_variant = None
        best_rate = 0.0
        
        for variant, data in variants.items():
            if data["total"] < min_samples:
                continue
            
            if data["positive_rate"] > best_rate:
                best_rate = data["positive_rate"]
                best_variant = variant
        
        return best_variant
    
    def add_experiment(
        self,
        name: str,
        variants: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Add a new experiment.
        
        Args:
            name: Experiment name
            variants: Dictionary of variant_name -> config
        """
        with self._lock:
            self.experiments[name] = {
                variant_name: VariantConfig(**config) if all(k in config for k in ["rag", "resonance"]) else config
                for variant_name, config in variants.items()
            }
        
        logger.info(f"[ABTest] Added experiment '{name}' with {len(variants)} variants")
    
    def _save_results(self) -> None:
        """Persist results to file."""
        if not self.results_path:
            return
        
        try:
            path = Path(self.results_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                exp: [asdict(r) for r in results]
                for exp, results in self._results.items()
            }
            
            with open(path, 'w') as f:
                json.dump(data, f)
            
            logger.debug(f"[ABTest] Saved results to {path}")
        except Exception as e:
            logger.error(f"Failed to save A/B test results: {e}")
    
    def _load_results(self) -> None:
        """Load results from file."""
        if not self.results_path:
            return
        
        try:
            path = Path(self.results_path)
            if not path.exists():
                return
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            for exp, results in data.items():
                self._results[exp] = [
                    ExperimentResult(**r) for r in results
                ]
            
            total = sum(len(r) for r in self._results.values())
            logger.info(f"[ABTest] Loaded {total} results from {len(self._results)} experiments")
        except Exception as e:
            logger.error(f"Failed to load A/B test results: {e}")
    
    def __repr__(self) -> str:
        return f"MemoryABTest(experiments={list(self.experiments.keys())})"


# Global singleton
ab_tester = MemoryABTest(results_path=None)  # Set from config if needed
