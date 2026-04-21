"""
Neural Agent Classifier
=========================

Trained neural agent classifier — mirrors ToolClassifier architecture.

Uses the SAME sentence-transformer encoder as the tool classifier (all-MiniLM-L6-v2)
+ a separate sklearn MLP for agent-type classification.

Agent types match AGENT_CAPABILITIES in agent_capability_registry.py (23 agents).
Model is stored in PostgreSQL for container-independent persistence.
Active learning accumulates predictions for periodic retraining.
"""
from __future__ import annotations

import asyncio
import logging
import os
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Agent labels — must match agent_capability_registry.py keys
ALL_AGENTS = [
    "reasoning",
    "explain",
    "code",
    "debug",
    "review",
    "test",
    "refactor",
    "security",
    "architecture",
    "math",
    "research",
    "summary",
    "planning",
    "optimization",
    "documentation",
    "migration",
    "api",
    "database",
    "devops",
    "accessibility",
    "i18n",
    "regex",
    "git",
    "css",
]

AGENT_TO_IDX = {a: i for i, a in enumerate(ALL_AGENTS)}
IDX_TO_AGENT = {i: a for i, a in enumerate(ALL_AGENTS)}

_FLUSH_BATCH = 50


@dataclass
class AgentPrediction:
    """Result of the agent classifier."""
    agent_type: str
    confidence: float
    probabilities: Dict[str, float]
    method: str  # "neural", "adaptive_fallback", "keyword_fallback"
    latency_ms: float = 0.0


# ---------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------

async def _load_agent_model_from_db():
    """Load the latest active agent classifier from PostgreSQL."""
    from ..db import async_session
    from sqlalchemy import text
    try:
        async with async_session() as session:
            row = await session.execute(
                text(
                    "SELECT model_blob, stats_json, n_samples, version "
                    "FROM agent_classifier_models "
                    "WHERE is_active = true "
                    "ORDER BY version DESC LIMIT 1"
                )
            )
            result = row.fetchone()
            if result:
                blob, stats, n_samples, version = result
                clf = pickle.loads(blob)
                return clf, stats or {}, n_samples, version
    except Exception as e:
        logger.warning(f"[AgentClassifier] DB load failed: {e}")
    return None, {}, 0, 0


async def _save_agent_model_to_db(classifier, stats: dict, n_samples: int, version: int):
    """Save the trained agent classifier to PostgreSQL."""
    from ..db import async_session
    from sqlalchemy import text
    try:
        blob = pickle.dumps(classifier)
        async with async_session() as session:
            # Deactivate old models
            await session.execute(
                text("UPDATE agent_classifier_models SET is_active = false WHERE is_active = true")
            )
            # Insert new model
            await session.execute(
                text("""
                    INSERT INTO agent_classifier_models
                        (version, model_blob, n_samples, train_accuracy, cv_accuracy, stats_json, is_active)
                    VALUES (:version, :blob, :n_samples, :train_acc, :cv_acc, :stats::jsonb, true)
                """),
                {
                    "version": version,
                    "blob": blob,
                    "n_samples": n_samples,
                    "train_acc": stats.get("train_accuracy", 0),
                    "cv_acc": stats.get("cv_accuracy", 0),
                    "stats": __import__("json").dumps(stats),
                },
            )
            await session.commit()
            logger.info(
                f"[AgentClassifier] Model v{version} saved to DB "
                f"({len(blob)} bytes, {n_samples} samples)"
            )
    except Exception as e:
        logger.error(f"[AgentClassifier] DB save failed: {e}", exc_info=True)


async def _save_agent_active_samples(samples: List[Dict]):
    """Batch-insert active learning samples into PostgreSQL."""
    from ..db import async_session
    from sqlalchemy import text
    try:
        async with async_session() as session:
            for s in samples:
                await session.execute(
                    text("""
                        INSERT INTO agent_active_samples
                            (user_message, predicted_agent, confidence, method, probabilities, user_id)
                        VALUES (:msg, :predicted, :conf, :method, :probs::jsonb, :uid)
                    """),
                    {
                        "msg": s["msg"][:500],
                        "predicted": s.get("predicted"),
                        "conf": s.get("conf", 0),
                        "method": s.get("method", ""),
                        "probs": __import__("json").dumps(s.get("probs", {})),
                        "uid": s.get("user_id"),
                    },
                )
            await session.commit()
            logger.info(f"[AgentClassifier] Flushed {len(samples)} active samples to DB")
    except Exception as e:
        logger.warning(f"[AgentClassifier] Active sample flush failed: {e}")


async def _load_agent_active_samples(min_confidence: float = 0.5) -> List[Tuple[str, str]]:
    """Load high-confidence active learning samples for retraining."""
    from ..db import async_session
    from sqlalchemy import text
    samples = []
    try:
        async with async_session() as session:
            rows = await session.execute(
                text(
                    "SELECT user_message, predicted_agent "
                    "FROM agent_active_samples "
                    "WHERE confidence >= :conf "
                    "ORDER BY created_at DESC "
                    "LIMIT 5000"
                ),
                {"conf": min_confidence},
            )
            for row in rows.fetchall():
                msg, agent = row
                if agent in AGENT_TO_IDX:
                    samples.append((msg, agent))
    except Exception as e:
        logger.warning(f"[AgentClassifier] Active sample load failed: {e}")
    return samples


# ---------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------

class AgentClassifier:
    """
    Neural agent classifier.

    Uses sentence-transformers for encoding + sklearn MLP for classification.
    Shares the same encoder model as ToolClassifier.
    Model + active learning data stored in PostgreSQL.
    """

    def __init__(self):
        self._encoder = None
        self._classifier = None
        self._is_trained = False
        self._load_lock = asyncio.Lock()
        self._pending_samples: List[Dict] = []
        self._model_version = 0
        self._train_stats: Dict[str, Any] = {}

    async def ensure_ready(self) -> bool:
        """Load encoder + classifier, training from seed if needed."""
        if self._is_trained and self._encoder is not None:
            return True
        async with self._load_lock:
            if self._is_trained and self._encoder is not None:
                return True
            try:
                ok = await asyncio.get_event_loop().run_in_executor(
                    None, self._load_encoder
                )
                if not ok:
                    return False

                clf, stats, n_samples, version = await _load_agent_model_from_db()
                if clf is not None:
                    try:
                        n_model_classes = len(clf.classes_)
                    except Exception:
                        n_model_classes = -1
                    if n_model_classes == len(ALL_AGENTS):
                        self._classifier = clf
                        self._train_stats = stats
                        self._model_version = version
                        self._is_trained = True
                        logger.info(
                            f"[AgentClassifier] Loaded model v{version} from DB "
                            f"({n_samples} samples, acc={stats.get('train_accuracy', '?')})"
                        )
                        return True
                    else:
                        logger.warning(
                            f"[AgentClassifier] DB model has {n_model_classes} classes "
                            f"but ALL_AGENTS has {len(ALL_AGENTS)} — retraining..."
                        )

                logger.info("[AgentClassifier] No model in DB, training from seed...")
                await self._train_and_save(source="seed")
                return True

            except Exception as e:
                logger.error(f"[AgentClassifier] Init failed: {e}", exc_info=True)
                return False

    def _load_encoder(self) -> bool:
        """Load the sentence-transformer encoder (synchronous).

        Tries to share the encoder instance with ToolClassifier if already loaded.
        """
        try:
            # Try sharing encoder with ToolClassifier
            from .tool_classifier import tool_classifier
            if tool_classifier._encoder is not None:
                self._encoder = tool_classifier._encoder
                logger.info("[AgentClassifier] Sharing encoder with ToolClassifier")
                return True
        except Exception:
            pass

        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("SKILL_ROUTER_MODEL", "all-MiniLM-L6-v2")
            logger.info(f"[AgentClassifier] Loading encoder: {model_name}")
            self._encoder = SentenceTransformer(model_name)
            return True
        except ImportError:
            logger.warning("[AgentClassifier] sentence-transformers not installed")
            return False
        except Exception as e:
            logger.error(f"[AgentClassifier] Encoder load error: {e}")
            return False

    def _encode(self, message: str) -> np.ndarray:
        """Encode message to embedding."""
        return self._encoder.encode([message], normalize_embeddings=True)[0]

    def _train_on_samples(
        self, samples: List[Tuple[str, str]], source: str = "unknown"
    ) -> Dict[str, Any]:
        """Train the MLP classifier on labeled samples (synchronous)."""
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_score

        logger.info(f"[AgentClassifier] Encoding {len(samples)} samples...")
        X_list, y_list = [], []
        for msg, agent_type in samples:
            emb = self._encode(msg)
            X_list.append(emb)
            y_list.append(AGENT_TO_IDX[agent_type])

        X = np.array(X_list)
        y = np.array(y_list)

        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        )

        cv_mean, cv_std = 0.0, 0.0
        if len(samples) > 30:
            n_cv = min(5, len(samples) // 10)
            try:
                cv_scores = cross_val_score(clf, X, y, cv=n_cv, scoring="accuracy")
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
            except Exception:
                pass

        clf.fit(X, y)
        train_acc = float(clf.score(X, y))
        self._classifier = clf
        self._is_trained = True

        class_dist = Counter(y_list)
        class_stats = {
            IDX_TO_AGENT[k]: v for k, v in sorted(class_dist.items())
        }

        self._train_stats = {
            "n_samples": len(samples),
            "n_classes": len(set(y_list)),
            "train_accuracy": round(train_acc, 4),
            "cv_accuracy": round(cv_mean, 4),
            "cv_std": round(cv_std, 4),
            "class_distribution": class_stats,
            "source": source,
            "timestamp": time.time(),
        }

        logger.info(
            f"[AgentClassifier] Training complete: "
            f"accuracy={train_acc:.3f}, cv={cv_mean:.3f}±{cv_std:.3f}, "
            f"classes={len(set(y_list))}, samples={len(samples)}, source={source}"
        )
        return self._train_stats

    async def _train_and_save(self, source: str = "seed") -> Dict[str, Any]:
        """Train from seed (+ active data) and save to DB."""
        from .agent_training_data import get_agent_training_data
        samples = get_agent_training_data()

        # Include active learning data from DB
        active = await _load_agent_active_samples(min_confidence=0.5)
        if active:
            samples.extend(active)
            logger.info(f"[AgentClassifier] Added {len(active)} active samples from DB")

        stats = await asyncio.get_event_loop().run_in_executor(
            None, self._train_on_samples, samples, source
        )

        self._model_version += 1
        await _save_agent_model_to_db(
            self._classifier, stats, len(samples), self._model_version
        )
        return stats

    async def predict(
        self,
        message: str,
        user_id: str = None,
    ) -> AgentPrediction:
        """
        Predict which agent type to use for this message.

        Returns the best agent with confidence and probabilities.
        Every prediction is logged to DB for continuous learning.
        """
        t0 = time.time()

        ready = await self.ensure_ready()
        if not ready:
            return AgentPrediction(
                agent_type="reasoning",
                confidence=0.5,
                probabilities={},
                method="model_unavailable",
                latency_ms=(time.time() - t0) * 1000,
            )

        # Encode + predict
        emb = await asyncio.get_event_loop().run_in_executor(
            None, self._encode, message
        )

        proba = self._classifier.predict_proba(emb.reshape(1, -1))[0]

        prob_dict: Dict[str, float] = {}
        for idx, prob in enumerate(proba):
            agent = IDX_TO_AGENT.get(idx, "reasoning")
            prob_dict[agent] = round(float(prob), 4)

        # Select best agent
        best_agent = max(prob_dict, key=prob_dict.get)
        best_prob = prob_dict[best_agent]

        latency = (time.time() - t0) * 1000

        result = AgentPrediction(
            agent_type=best_agent,
            confidence=best_prob,
            probabilities=prob_dict,
            method="neural",
            latency_ms=latency,
        )

        # Active learning: queue sample for DB
        self._pending_samples.append({
            "msg": message[:500],
            "predicted": result.agent_type,
            "conf": round(result.confidence, 4),
            "method": result.method,
            "probs": {k: v for k, v in sorted(prob_dict.items(), key=lambda x: -x[1])[:5]},
            "user_id": user_id,
        })
        if len(self._pending_samples) >= _FLUSH_BATCH:
            asyncio.create_task(self._flush_to_db())

        logger.info(
            f"[AgentClassifier] agent={result.agent_type} conf={result.confidence:.3f} "
            f"method={result.method} latency={latency:.1f}ms "
            f"msg={message[:60]!r}"
        )

        return result

    async def _flush_to_db(self) -> None:
        """Flush pending active learning samples to PostgreSQL."""
        if not self._pending_samples:
            return
        batch = self._pending_samples[:]
        self._pending_samples.clear()
        await _save_agent_active_samples(batch)

    async def retrain(self) -> Dict[str, Any]:
        """Retrain classifier using seed + active learning data from DB."""
        await self._flush_to_db()
        stats = await self._train_and_save(source="retrain")
        return stats

    async def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        return {
            "is_trained": self._is_trained,
            "model_version": self._model_version,
            "train_stats": self._train_stats,
            "pending_samples": len(self._pending_samples),
        }


# Global singleton
agent_classifier = AgentClassifier()


async def preload_agent_classifier() -> None:
    """Call at app startup (lifespan) to pre-train/load the classifier."""
    t0 = time.time()
    logger.info("[AgentClassifier] Preloading at startup...")
    ok = await agent_classifier.ensure_ready()
    elapsed = (time.time() - t0) * 1000
    if ok:
        stats = await agent_classifier.get_stats()
        logger.info(
            f"[AgentClassifier] Preload complete in {elapsed:.0f}ms — "
            f"v{stats['model_version']}, "
            f"samples={stats['train_stats'].get('n_samples', 0)}, "
            f"accuracy={stats['train_stats'].get('train_accuracy', 0)}"
        )
    else:
        logger.warning(f"[AgentClassifier] Preload FAILED in {elapsed:.0f}ms")
