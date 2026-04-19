# RG Chat

> **Part of the [ResonantGenesis](https://dev-swat.com) platform** — Resonant Chat engine with skills, multi-provider streaming, and IDE completions.

[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()
[![License: RG Source Available](https://img.shields.io/badge/License-RG%20Source%20Available-blue.svg)](LICENSE.txt)

## Features
- Multi-provider LLM routing (OpenAI, Anthropic, Groq, Gemini)
- SSE streaming with hallucination detection
- **Neural Skill Classifier** — trained ML model for intelligent skill routing
- Skills system (Agent Architect, Code Visualizer, Web Search, Memory, Google Drive/Calendar, IDE, etc.)
- Debate engine, error correction, multi-provider chunking
- Chat analytics dashboard
- IDE completions endpoint for Resonant IDE
- Provider status monitoring

## Neural Skill Classifier (ML)
Real trained neural network for skill routing — not keywords, not LLM prompts.

**Architecture:**
1. `all-MiniLM-L6-v2` sentence-transformer encodes (message + context) → 384-dim embedding
2. Trained 2-layer MLP (256→128→14 classes) maps embedding → skill probabilities
3. Active skill continuity boost from conversation `meta_data.toolResults`
4. Active learning: every prediction saved to PostgreSQL for continuous improvement

**Persistence (PostgreSQL — container-independent):**
- `skill_classifier_models` — trained model weights (binary blob), versioned
- `skill_active_samples` — every routing decision logged for retraining

**Performance:** ~5ms inference, no API calls, no external dependencies at runtime.

**Files:**
- `app/services/skill_classifier.py` — classifier, DB persistence, active learning
- `app/services/skill_training_data.py` — 250+ curated seed training samples
- `app/services/neural_skill_router.py` — embedding cosine-sim fallback (backup)

## Volume Mounts
- `rg_llm` — Shared LLM client library (read-only)
- `platform_tools` — Shared agent tools (read-only)

## Quick Start
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Deployment
- **Container**: `chat_service` | **Port**: 8000
- **Server path**: `/home/deploy/RG_Chat`

---
**Organization**: [DevSwat-ResonantGenesis](https://github.com/DevSwat-ResonantGenesis) | **Platform**: [dev-swat.com](https://dev-swat.com)
