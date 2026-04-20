#!/usr/bin/env python3
"""
Standalone training script for the neural tool classifier.

Trains on seed data from tool_training_data.py, saves the model as a pickle
file that can be loaded into PostgreSQL on next deploy.

Usage:
    python3 train_classifier.py

Output:
    tool_classifier_model.pkl — trained model (pickle)
"""
import os
import sys
import time
import pickle
from collections import Counter

import numpy as np

# Add app to path so we can import from it
sys.path.insert(0, os.path.dirname(__file__))

from app.services.tool_classifier import ALL_TOOLS, TOOL_TO_IDX, IDX_TO_TOOL
from app.services.tool_training_data import get_training_data


def main():
    print(f"=== Neural Tool Classifier Training ===")
    print(f"Tools: {len(ALL_TOOLS)} labels ({len(ALL_TOOLS) - 1} tools + None)")
    print()

    # 1. Load training data
    samples = get_training_data()
    print(f"Training samples: {len(samples)}")

    # Validate all labels exist in ALL_TOOLS
    unknown = set()
    for msg, ctx, tool_id in samples:
        if tool_id not in TOOL_TO_IDX:
            unknown.add(tool_id)
    if unknown:
        print(f"ERROR: Unknown tool IDs in training data: {unknown}")
        sys.exit(1)

    # 2. Load sentence-transformer encoder
    from sentence_transformers import SentenceTransformer
    model_name = os.getenv("SKILL_ROUTER_MODEL", "all-MiniLM-L6-v2")
    print(f"Loading encoder: {model_name} ...")
    encoder = SentenceTransformer(model_name)
    print(f"Encoder loaded (dim={encoder.get_sentence_embedding_dimension()})")
    print()

    # 3. Encode all samples
    print("Encoding samples...")
    t0 = time.time()
    X_list, y_list = [], []
    for msg, ctx, tool_id in samples:
        parts = []
        if ctx:
            for m in ctx[-3:]:
                role = m.get("role", "user")
                content = m.get("content", "")
                if content:
                    parts.append(f"{role}: {content[:200]}")
        parts.append(f"user: {msg}")
        text = "\n".join(parts)
        emb = encoder.encode([text], normalize_embeddings=True)[0]
        X_list.append(emb)
        y_list.append(TOOL_TO_IDX.get(tool_id, 0))

    X = np.array(X_list)
    y = np.array(y_list)
    encode_time = time.time() - t0
    print(f"Encoded {len(samples)} samples in {encode_time:.1f}s")
    print(f"Embedding shape: {X.shape}")
    print()

    # 4. Class distribution
    class_dist = Counter(y_list)
    n_classes = len(class_dist)
    print(f"Classes with samples: {n_classes} / {len(ALL_TOOLS)}")
    classes_without_samples = set(range(len(ALL_TOOLS))) - set(class_dist.keys())
    if classes_without_samples:
        missing_tools = [IDX_TO_TOOL.get(i, "?") for i in sorted(classes_without_samples)]
        print(f"WARNING: {len(missing_tools)} tools have NO training samples:")
        for t in missing_tools[:20]:
            print(f"  - {t}")
        if len(missing_tools) > 20:
            print(f"  ... and {len(missing_tools) - 20} more")
    print()

    # 5. Train MLP
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score

    print("Training MLPClassifier(256, 128)...")
    t0 = time.time()

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

    # Cross-validation
    if len(samples) > 30:
        n_cv = min(5, len(samples) // 10)
        try:
            cv_scores = cross_val_score(clf, X, y, cv=n_cv, scoring="accuracy")
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
            print(f"Cross-validation ({n_cv}-fold): {cv_mean:.4f} ± {cv_std:.4f}")
        except Exception as e:
            print(f"Cross-validation failed: {e}")
            cv_mean, cv_std = 0.0, 0.0
    else:
        cv_mean, cv_std = 0.0, 0.0

    clf.fit(X, y)
    train_acc = float(clf.score(X, y))
    train_time = time.time() - t0

    print(f"Training complete in {train_time:.1f}s")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"MLP iterations: {clf.n_iter_}")
    print()

    # 6. Test a few predictions
    print("=== Quick Predictions Test ===")
    test_msgs = [
        "hello how are you",
        "scan this GitHub repo https://github.com/org/repo",
        "search the web for latest AI news",
        "generate an image of a cat",
        "list my agents",
        "create a pull request",
        "send a message on Slack",
        "what's the weather in Tokyo",
        "create a Google Sheet",
        "scrape this LinkedIn page",
        "post a tweet about our launch",
        "what time is it",
    ]
    for msg in test_msgs:
        emb = encoder.encode([f"user: {msg}"], normalize_embeddings=True)[0]
        probs = clf.predict_proba([emb])[0]
        top_idx = np.argsort(probs)[::-1][:3]
        top = [(IDX_TO_TOOL.get(i, "?"), probs[i]) for i in top_idx]
        tool_str = ", ".join(f"{t}={p:.3f}" for t, p in top)
        print(f"  '{msg[:50]}' → {tool_str}")
    print()

    # 7. Save model
    stats = {
        "n_samples": len(samples),
        "n_classes": n_classes,
        "n_tools_total": len(ALL_TOOLS),
        "train_accuracy": round(train_acc, 4),
        "cv_accuracy": round(cv_mean, 4),
        "cv_std": round(cv_std, 4),
        "encode_time_s": round(encode_time, 1),
        "train_time_s": round(train_time, 1),
        "source": "seed",
        "timestamp": time.time(),
    }

    output_path = os.path.join(os.path.dirname(__file__), "tool_classifier_model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump({"classifier": clf, "stats": stats, "version": 1}, f)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model saved: {output_path} ({size_mb:.2f} MB)")
    print(f"Stats: {stats}")
    print()
    print("Done! Load this into PostgreSQL via the /retrain endpoint or on next container boot.")


if __name__ == "__main__":
    main()
