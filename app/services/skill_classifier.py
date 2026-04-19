"""
Trained Neural Skill Classifier
=================================

A REAL trained ML classifier for skill routing.
Not keyword matching. Not an LLM prompt. A trained model.

Architecture:
  1. Sentence-transformer encodes (message + context) → 384-dim embedding
  2. Trained classification head (2-layer MLP) maps embedding → skill probabilities
  3. Active skill continuity boost from meta_data.toolResults
  4. Active learning: every prediction is saved to PostgreSQL
  5. Model stored in PostgreSQL — survives container restarts forever
  6. Retraining merges seed data + all accumulated active learning samples

Persistence:
  - Model weights: skill_classifier_models table (LargeBinary blob)
  - Active learning: skill_active_samples table (one row per prediction)
  - No filesystem dependency. Container can be destroyed and rebuilt.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Skill labels (None = general chat)
ALL_SKILLS = [
    None,  # index 0 = general chat / no tool
    "agent_architect",
    "code_visualizer",
    "web_search",
    "image_generation",
    "memory_search",
    "memory_library",
    "google_drive",
    "google_calendar",
    "state_physics",
    "ide_workspace",
    "rabbit_post",
    "figma",
    "sigma",
]

SKILL_TO_IDX = {s: i for i, s in enumerate(ALL_SKILLS)}
IDX_TO_SKILL = {i: s for i, s in enumerate(ALL_SKILLS)}

# Batch size for flushing active learning samples to DB
_FLUSH_BATCH = 50


@dataclass
class ClassifierPrediction:
    """Result of the classifier."""
    skill_id: Optional[str]
    confidence: float
    probabilities: Dict[str, float]
    method: str  # "classifier", "continuity", "fallback"
    active_skill: Optional[str] = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------
# DB helpers (async — run inside the existing SQLAlchemy session)
# ---------------------------------------------------------------

async def _load_model_from_db():
    """Load the latest active classifier from PostgreSQL."""
    from ..db import async_session
    from sqlalchemy import select, text
    try:
        async with async_session() as session:
            row = await session.execute(
                text(
                    "SELECT model_blob, stats_json, n_samples, version "
                    "FROM skill_classifier_models "
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
        logger.warning(f"[SkillClassifier] DB load failed: {e}")
    return None, {}, 0, 0


async def _save_model_to_db(classifier, stats: dict, n_samples: int, version: int):
    """Save the trained classifier to PostgreSQL."""
    from ..db import async_session
    from ..models import SkillClassifierModel
    try:
        blob = pickle.dumps(classifier)
        async with async_session() as session:
            # Deactivate old models
            from sqlalchemy import update
            await session.execute(
                update(SkillClassifierModel)
                .where(SkillClassifierModel.is_active == True)
                .values(is_active=False)
            )
            # Insert new model
            new_model = SkillClassifierModel(
                version=version,
                model_blob=blob,
                n_samples=n_samples,
                train_accuracy=stats.get("train_accuracy", 0),
                cv_accuracy=stats.get("cv_accuracy", 0),
                stats_json=stats,
                is_active=True,
            )
            session.add(new_model)
            await session.commit()
            logger.info(
                f"[SkillClassifier] Model v{version} saved to DB "
                f"({len(blob)} bytes, {n_samples} samples)"
            )
    except Exception as e:
        logger.error(f"[SkillClassifier] DB save failed: {e}", exc_info=True)


async def _save_active_samples(samples: List[Dict]):
    """Batch-insert active learning samples into PostgreSQL."""
    from ..db import async_session
    from ..models import SkillActiveSample
    try:
        async with async_session() as session:
            for s in samples:
                session.add(SkillActiveSample(
                    user_message=s["msg"][:500],
                    predicted_skill=s.get("predicted"),
                    confidence=s.get("conf", 0),
                    method=s.get("method", ""),
                    active_skill=s.get("active"),
                    probabilities=s.get("probs", {}),
                    intents=s.get("intents", []),
                    user_id=s.get("user_id"),
                ))
            await session.commit()
            logger.info(f"[SkillClassifier] Flushed {len(samples)} active samples to DB")
    except Exception as e:
        logger.warning(f"[SkillClassifier] Active sample flush failed: {e}")


async def _load_active_samples_from_db(min_confidence: float = 0.6) -> List[Tuple]:
    """Load high-confidence active learning samples for retraining."""
    from ..db import async_session
    from sqlalchemy import text
    samples = []
    try:
        async with async_session() as session:
            rows = await session.execute(
                text(
                    "SELECT user_message, predicted_skill "
                    "FROM skill_active_samples "
                    "WHERE confidence >= :conf "
                    "ORDER BY created_at DESC "
                    "LIMIT 5000"
                ),
                {"conf": min_confidence},
            )
            for row in rows.fetchall():
                msg, skill = row
                samples.append((msg, [], skill))
    except Exception as e:
        logger.warning(f"[SkillClassifier] Active sample load failed: {e}")
    return samples


async def _count_active_samples() -> int:
    """Count total active learning samples in DB."""
    from ..db import async_session
    from sqlalchemy import text
    try:
        async with async_session() as session:
            result = await session.execute(
                text("SELECT count(*) FROM skill_active_samples")
            )
            return result.scalar() or 0
    except Exception:
        return 0


# ---------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------

class SkillClassifier:
    """
    Trained neural skill classifier.

    Uses sentence-transformers for encoding + sklearn MLP for classification.
    Model + active learning data stored in PostgreSQL — container-independent.
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
                # Load the sentence-transformer encoder (sync, in thread pool)
                ok = await asyncio.get_event_loop().run_in_executor(
                    None, self._load_encoder
                )
                if not ok:
                    return False

                # Try loading trained model from PostgreSQL
                clf, stats, n_samples, version = await _load_model_from_db()
                if clf is not None:
                    self._classifier = clf
                    self._train_stats = stats
                    self._model_version = version
                    self._is_trained = True
                    logger.info(
                        f"[SkillClassifier] Loaded model v{version} from DB "
                        f"({n_samples} samples, acc={stats.get('train_accuracy', '?')})"
                    )
                    return True

                # No model in DB — train from seed and save
                logger.info("[SkillClassifier] No model in DB, training from seed...")
                await self._train_and_save(source="seed")
                return True

            except Exception as e:
                logger.error(f"[SkillClassifier] Init failed: {e}", exc_info=True)
                return False

    def _load_encoder(self) -> bool:
        """Load the sentence-transformer encoder (synchronous)."""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("SKILL_ROUTER_MODEL", "all-MiniLM-L6-v2")
            logger.info(f"[SkillClassifier] Loading encoder: {model_name}")
            self._encoder = SentenceTransformer(model_name)
            return True
        except ImportError:
            logger.warning("[SkillClassifier] sentence-transformers not installed")
            return False
        except Exception as e:
            logger.error(f"[SkillClassifier] Encoder load error: {e}")
            return False

    def _encode_sample(
        self, message: str, context: List[Dict[str, str]]
    ) -> np.ndarray:
        """Encode a (message, context) pair to embedding."""
        parts = []
        if context:
            for msg in context[-3:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    parts.append(f"{role}: {content[:200]}")
        parts.append(f"user: {message}")
        text = "\n".join(parts)
        return self._encoder.encode([text], normalize_embeddings=True)[0]

    def _train_on_samples(
        self, samples: List[Tuple], source: str = "unknown"
    ) -> Dict[str, Any]:
        """Train the MLP classifier on labeled samples (synchronous)."""
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_score

        logger.info(f"[SkillClassifier] Encoding {len(samples)} samples...")
        X_list, y_list = [], []
        for msg, ctx, skill_id in samples:
            emb = self._encode_sample(msg, ctx)
            X_list.append(emb)
            y_list.append(SKILL_TO_IDX.get(skill_id, 0))

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
            (IDX_TO_SKILL.get(k) or "none"): v
            for k, v in sorted(class_dist.items())
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
            f"[SkillClassifier] Training complete: "
            f"accuracy={train_acc:.3f}, cv={cv_mean:.3f}±{cv_std:.3f}, "
            f"classes={len(set(y_list))}, samples={len(samples)}, source={source}"
        )
        return self._train_stats

    async def _train_and_save(self, source: str = "seed") -> Dict[str, Any]:
        """Train from seed (+ active data) and save to DB."""
        from .skill_training_data import get_training_data
        samples = get_training_data()

        # Include active learning data from DB
        active = await _load_active_samples_from_db(min_confidence=0.6)
        if active:
            samples.extend(active)
            logger.info(f"[SkillClassifier] Added {len(active)} active samples from DB")

        stats = await asyncio.get_event_loop().run_in_executor(
            None, self._train_on_samples, samples, source
        )

        self._model_version += 1
        await _save_model_to_db(
            self._classifier, stats, len(samples), self._model_version
        )
        return stats

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

    async def predict(
        self,
        message: str,
        enabled_skill_ids: Set[str],
        recent_messages: list = None,
        intents: List[str] = None,
        user_id: str = None,
    ) -> ClassifierPrediction:
        """
        Predict which skill to route to.

        Uses active skill continuity + trained classifier.
        Every prediction is logged to DB for continuous learning.
        """
        t0 = time.time()

        # --- Layer 1: Active skill continuity ---
        active_skill = self._detect_active_skill(
            recent_messages or [], enabled_skill_ids
        )

        # --- Ensure model ready ---
        ready = await self.ensure_ready()
        if not ready:
            if active_skill:
                return ClassifierPrediction(
                    skill_id=active_skill,
                    confidence=0.7,
                    probabilities={},
                    method="continuity_fallback",
                    active_skill=active_skill,
                    latency_ms=(time.time() - t0) * 1000,
                )
            return ClassifierPrediction(
                skill_id=None,
                confidence=0.0,
                probabilities={},
                method="model_unavailable",
                latency_ms=(time.time() - t0) * 1000,
            )

        # --- Build context ---
        ctx_dicts = []
        if recent_messages:
            for msg in recent_messages[-3:]:
                role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
                content = (
                    msg.content
                    if hasattr(msg, "content")
                    else msg.get("content", "")
                )
                if content:
                    ctx_dicts.append({"role": role, "content": str(content)[:200]})

        # --- Encode + predict ---
        emb = await asyncio.get_event_loop().run_in_executor(
            None, self._encode_sample, message, ctx_dicts
        )

        proba = self._classifier.predict_proba(emb.reshape(1, -1))[0]

        prob_dict: Dict[str, float] = {}
        for idx, prob in enumerate(proba):
            skill = IDX_TO_SKILL.get(idx)
            label = skill if skill else "none"
            if skill is None or skill in enabled_skill_ids:
                prob_dict[label] = round(float(prob), 4)

        best_skill = None
        best_prob = prob_dict.get("none", 0.0)
        for skill_id in enabled_skill_ids:
            sp = prob_dict.get(skill_id, 0.0)
            if sp > best_prob:
                best_prob = sp
                best_skill = skill_id

        # --- Continuity boost ---
        CONTINUITY_BOOST = 0.30
        CONTINUITY_MIN = 0.10

        if active_skill and active_skill in prob_dict:
            active_prob = prob_dict[active_skill]
            if active_prob >= CONTINUITY_MIN:
                boosted = min(active_prob + CONTINUITY_BOOST, 0.99)
                if boosted > best_prob:
                    best_skill = active_skill
                    best_prob = boosted
                    prob_dict[active_skill] = round(boosted, 4)

        latency = (time.time() - t0) * 1000

        result = ClassifierPrediction(
            skill_id=best_skill,
            confidence=best_prob,
            probabilities=prob_dict,
            method="continuity" if (best_skill == active_skill and active_skill is not None) else "classifier",
            active_skill=active_skill,
            latency_ms=latency,
        )

        # --- Active learning: queue sample for DB ---
        self._pending_samples.append({
            "msg": message[:500],
            "predicted": result.skill_id,
            "conf": round(result.confidence, 4),
            "method": result.method,
            "active": result.active_skill,
            "probs": {k: v for k, v in sorted(prob_dict.items(), key=lambda x: -x[1])[:5]},
            "intents": (intents or [])[:3],
            "user_id": user_id,
        })
        if len(self._pending_samples) >= _FLUSH_BATCH:
            asyncio.create_task(self._flush_to_db())

        logger.info(
            f"[SkillClassifier] skill={result.skill_id} conf={result.confidence:.3f} "
            f"method={result.method} active={active_skill} "
            f"latency={latency:.1f}ms msg={message[:60]!r}"
        )

        return result

    async def _flush_to_db(self) -> None:
        """Flush pending active learning samples to PostgreSQL."""
        if not self._pending_samples:
            return
        batch = self._pending_samples[:]
        self._pending_samples.clear()
        await _save_active_samples(batch)

    async def retrain(self) -> Dict[str, Any]:
        """
        Retrain classifier using seed data + all active learning from DB.
        The model gets smarter with every retrain.
        """
        # Flush any pending samples first
        await self._flush_to_db()

        stats = await self._train_and_save(source="retrain")
        return stats

    async def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics including DB counts."""
        active_count = await _count_active_samples()
        return {
            "is_trained": self._is_trained,
            "model_version": self._model_version,
            "train_stats": self._train_stats,
            "pending_samples": len(self._pending_samples),
            "active_samples_in_db": active_count,
        }


# Global singleton
skill_classifier = SkillClassifier()


async def preload_skill_classifier() -> None:
    """
    Call at app startup (lifespan) to pre-train/load the classifier.
    Loads from PostgreSQL — no filesystem dependency.
    """
    t0 = time.time()
    logger.info("[SkillClassifier] Preloading at startup...")
    ok = await skill_classifier.ensure_ready()
    elapsed = (time.time() - t0) * 1000
    if ok:
        stats = await skill_classifier.get_stats()
        logger.info(
            f"[SkillClassifier] Preload complete in {elapsed:.0f}ms — "
            f"v{stats['model_version']}, "
            f"samples={stats['train_stats'].get('n_samples', 0)}, "
            f"accuracy={stats['train_stats'].get('train_accuracy', 0)}, "
            f"active_in_db={stats['active_samples_in_db']}"
        )
    else:
        logger.warning(f"[SkillClassifier] Preload FAILED in {elapsed:.0f}ms")
