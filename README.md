# RG Chat

> **Part of the [ResonantGenesis](https://dev-swat.com) platform** — Resonant Chat engine with skills, multi-provider streaming, and IDE completions.

[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()
[![License: RG Source Available](https://img.shields.io/badge/License-RG%20Source%20Available-blue.svg)](LICENSE.txt)

## Features
- Multi-provider LLM routing (OpenAI, Anthropic, Groq, Gemini)
- SSE streaming with hallucination detection
- Skills system (Code Visualizer, web search, etc.)
- Debate engine, error correction, multi-provider chunking
- Chat analytics dashboard
- IDE completions endpoint for Resonant IDE
- Provider status monitoring

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
