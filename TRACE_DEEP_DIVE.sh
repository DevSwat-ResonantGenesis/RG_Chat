#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# RG_Chat — FULL PIPELINE TRACE & DEPENDENCY DEEP DIVE
# Generated: 2026-04-26
# ═══════════════════════════════════════════════════════════════════

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. SERVICE IDENTITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Container:    chat_service
# Port:         8000
# Database:     PostgreSQL (chat_db)
# Framework:    FastAPI + SQLAlchemy (asyncpg)
# Entrypoint:   app/main.py → uvicorn
# Lines of code (approx):
#   resonant_chat.py  ~6,610 lines (main router, SSE streaming)
#   tool_executor.py  ~1,798 lines (82KB — tool dispatch)
#   74 service files   in app/services/ (73 py + 1 tools/)
#   18 tool modules    in app/services/tools/
#   12 router files    in app/routers/

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. STARTUP SEQUENCE (app/main.py lifespan)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Auto-create DB tables (if missing):
#      - hallucination_settings
#      - knowledge_base_entries
#      - tool_classifier_models
#      - tool_active_samples
#      - agent_classifier_models
#      - agent_active_samples
# 2. Pre-train/load ToolClassifier (sentence-transformers + MLPClassifiers)
#      - Loads from DB if counts match, else trains from seed data
#      - 70+ tool definitions, ~1000+ training samples
# 3. Pre-train/load AgentClassifier (same encoder)
#      - Classifies user intent → agent type (reasoning, architecture, code, etc.)
# 4. Start ProviderStatusMonitor loop (background task)
#      - Monitors LLM provider health via WebSocket

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. ROUTERS (8 mounted)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# resonant_chat_router  — /resonant-chat/* (main chat, SSE stream, tools, memory, agents)
# analytics_router      — /analytics/* (usage stats, metrics)
# streaming_router      — /streaming/* (chunked streaming for large responses)
# websocket_router      — /ws/* (WebSocket for real-time chat)
# provider_status_router — /providers/* (LLM provider health WebSocket)
# tools_router          — /tools/* (enable/disable tools, list tools)
# owner_catalog_router  — /owner/* (platform owner catalog management)
# ide_completions_router — /ide/* (IDE autocomplete/inline completions)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. MAIN MESSAGE PIPELINE (/resonant-chat/message/stream)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Request enters → SSE streaming response
#
# STEP 1: AUTH
#   ├── get_crypto_identity(request) or x-user-id header
#   ├── Extract: user_id, org_id, user_role, is_superuser, unlimited_credits
#   └── _sanitize_sensitive_tokens(message)
#
# STEP 2: CHAT DB
#   ├── Find or create ResonantChat row (PostgreSQL)
#   ├── Store user message as ResonantChatMessage
#   └── Compute resonance hash (ResonanceHasher → 3D xyz coords)
#
# STEP 3: CREDIT DEDUCTION
#   └── POST http://billing_service:8000/billing/deduct-credits
#       (via credit_deduction.py — HTTPStatusError 402 = credits exhausted)
#
# STEP 4: TOOL DETECTION (3-layer classification)
#   ├── Layer 1: Check architect pipeline continuity (recent toolResults with present_options)
#   ├── Layer 2: Neural ToolClassifier (sentence-transformers → MLPClassifier)
#   │     └── If prediction == "agent_architect" → use_architect = True
#   └── Layer 3: Regex keyword guard (_is_agent_intent regex)
#         └── Catches: create/build/run/list/manage/delete + agent(s)
#
# STEP 5A: ARCHITECT PATH (if use_architect == True)
#   ├── Emit SSE: {"event": "start", "tool": "agent_architect"}
#   ├── Build conversation_history from last 8 messages
#   ├── Get user API keys (BYOK) from DB
#   ├── POST http://agent_architect:8000/api/message/stream
#   │     (SSE streaming — forwards ALL events live to frontend)
#   ├── Forward event types:
#   │     text       → {"event": "chunk", "content": ...}
#   │     options    → {"event": "options", "options": ...}
#   │     tool_call  → {"event": "step", "step": "tool_call", ...}
#   │     tool_result→ {"event": "step", "step": "tool_result", ...}
#   │     thinking   → {"event": "step", "step": "thinking", ...}
#   │     phase/build_progress/test_step/etc → {"event": "step", ...}
#   │     error      → {"event": "error", "error": ...}
#   │     complete   → extract final text + options
#   └── Store assistant message with meta_data.toolResults
#
# STEP 5B: REGULAR LLM PATH (if use_architect == False)
#   ├── Extract memories:
#   │     └── POST http://memory_service:8000/memory/hash-sphere/extract
#   ├── Build context messages (_build_context_messages):
#   │     ├── System prompt (resonant chat instructions)
#   │     ├── Memory context (RAG results)
#   │     ├── Conversation history
#   │     └── User message
#   ├── maybe_spawn_agent() → agent routing
#   │     ├── AgentClassifier → determines agent type
#   │     ├── route_query() → UnifiedLLMClient → llm_service:8000
#   │     └── Streams LLM response chunks
#   └── Emit SSE: {"event": "chunk", "content": ...}
#
# STEP 6: POST-PROCESSING
#   ├── Strip present_options() artifacts from text
#   ├── Compute resonance hash for assistant message
#   ├── Store assistant message (PostgreSQL)
#   ├── Background: ingest both messages to memory_service
#   │     └── POST http://memory_service:8000/memory/ingest (x2)
#   └── Emit SSE: {"event": "done", "message_id": ..., "hash": ...}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. NON-ARCHITECT TOOL PIPELINE (/resonant-chat/message — sync)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# The sync /message endpoint uses a different flow with tool injection:
#
# STEP 1-4: Same auth, DB, credit, detection as above
# STEP 5: Memory extraction (hash sphere + RAG)
# STEP 6: Context building (system prompt + memories + history)
# STEP 7: Neural tool classification
#   ├── ToolClassifier.predict(message) → tool_id, confidence
#   ├── If confidence > threshold → execute tool
#   │     └── ToolExecutor.execute(tool_id, message, context)
#   ├── Tool output injected into LLM context as system message
#   │     └── GROUNDED RESULTS with anti-hallucination instructions
#   └── STEP 7.95: If delegate_to_pipeline=True → web_search_needed=True
# STEP 8: LLM call (route_query → UnifiedLLMClient → llm_service)
# STEP 9: Agent routing (maybe_spawn_agent / maybe_run_debate)
# STEP 10: Post-processing (hallucination check, output correction, metrics)
# STEP 11: Store response + background memory ingestion

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. TOOL EXECUTOR — 100+ TOOLS IN 12 CATEGORIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# tool_executor.py dispatches to:
#
# CORE TOOLS (dedicated executors):
#   code_visualizer    → http://rg_ast_analysis:8000
#   web_search         → web_tools.py (Tavily/DuckDuckGo)
#   image_generation   → image_generation.py (DALL-E via llm_service)
#   memory_search      → http://memory_service:8000
#   memory_library     → opens panel (frontend-only)
#   state_physics      → http://rg_users_invarients_sim:8091
#   ide_workspace      → http://ide_platform_service:8080
#   rabbit_post        → http://rabbit_api_service:8000 (community posts)
#
# AGENT ARCHITECT (40+ tool IDs → _execute_agent_architect):
#   agent_architect, agents_list, agents_create, agents_start, agents_stop,
#   agents_status, agents_delete, agents_update, agents_sessions,
#   workspace_snapshot, build_agent, run_agent, modify_agent, delete_agent,
#   set_trigger, get_credits_info, create_tool, auto_build_tool, etc.
#   ALL → POST http://agent_architect:8000/api/message/stream (SSE)
#
# MODULAR INTEGRATION TOOLS (→ _execute_integration):
#   Web:      fetch_url, read_webpage, reddit_search, image_search, news_search,
#             youtube_search, deep_research, wikipedia, weather, stock_crypto, etc.
#   Memory:   memory_read, memory_write, memory_stats, hash_sphere_search, etc.
#   CodeViz:  code_visualizer_scan, _functions, _trace, _governance, _graph, _pipeline
#   Physics:  sp_state, sp_reset, sp_nodes, sp_metrics, sp_simulate, sp_galaxy, etc.
#   Rabbit:   create_rabbit_post, list_rabbit_communities, rabbit_vote, etc.
#   Dev:      execute_code, http_request, external_http_request, dev_tool
#   GitHub:   github_create_repo, _list_repos, _list_files, _upload_file, _pull_request, etc.
#   Files:    file_read, file_write, file_edit, multi_edit, file_list, file_delete
#   Email:    send_email, list_emails, draft_email, email_search
#   Google:   google_drive, google_calendar, google_docs
#   Media:    generate_image, generate_audio, generate_video, text_to_speech

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. EXTERNAL SERVICE CONNECTIONS (8 services)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Service               URL (Docker internal)              Circuit Breaker  Used For
# ────────────────────   ──────────────────────────────     ───────────────  ─────────────────
# memory_service         http://memory_service:8000         YES (3/60s)      Memory extraction, ingestion, RAG, hash sphere, anchors
# billing_service        http://billing_service:8000        YES (2/30s)      Credit deduction, balance check, plan limits
# auth_service           http://auth_service:8000           YES (2/30s)      User API keys, integrations
# llm_service            http://llm_service:8000            NO (via rg_llm)  All LLM calls via UnifiedLLMClient
# agent_architect        http://agent_architect:8000        NO               Agent management (SSE streaming)
# rg_ast_analysis        http://rg_ast_analysis:8000        NO               Code analysis, GitHub scanning
# rg_users_invarients    http://rg_users_invarients_sim:8091 NO              State physics simulation
# ide_platform_service   http://ide_platform_service:8080   NO               IDE workspace tools

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. ML CLASSIFIERS (2 neural classifiers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ToolClassifier (tool_classifier.py):
#   - sentence-transformers (all-MiniLM-L6-v2) → 384-dim embeddings
#   - One MLPClassifier per tool group
#   - Trained from tool_training_data.py (~1000+ samples)
#   - Persisted in PostgreSQL (tool_classifier_models table)
#   - Active learning: predictions logged to tool_active_samples
#
# AgentClassifier (agent_classifier.py):
#   - Same encoder → classifies agent type (reasoning, code, architecture, etc.)
#   - Trained from agent_training_data.py
#   - Anti-collision: "agent management" → "reasoning" (not "architecture")
#   - Persisted in agent_classifier_models table

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. INTERNAL SERVICES (30+ modules in app/services/)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# CORE PIPELINE:
#   resonance_hashing.py    — ResonanceHasher (text → 3D xyz coordinates)
#   rag_engine.py           — RAG search across memories
#   memory_merge.py         — Merge + rank memories from multiple sources
#   intent_engine.py        — Extract intents from user messages
#   emotional_normalizer.py — Normalize emotional state
#   personality_dna.py      — User personality profiling
#
# REASONING:
#   knowledge_graph.py      — Self-evolving knowledge graph
#   thought_branching.py    — Probabilistic thought trees
#   evidence_graph.py       — Evidence tracking
#   causal_reasoning.py     — Causal inference
#   reasoning_engine.py     — Structured reasoning
#   debate_engine.py        — Multi-agent debate
#
# AGENTS:
#   agent_engine.py         — Agent spawning + orchestration
#   agent_classifier.py     — Neural agent type classification
#   agent_chaining.py       — Agent-to-agent delegation
#   agent_memory.py         — Per-agent memory
#   agent_voting.py         — Multi-agent voting
#   team_engine.py          — Agent teams
#   dynamic_team_composer.py — Dynamic team assembly
#
# AUTONOMOUS:
#   autonomous_planner.py   — Task planning
#   autonomous_agent_executor.py — Autonomous agent execution
#   autonomous_error_correction.py — Self-healing
#   self_improving_agent.py — Learning from feedback
#
# OUTPUT QUALITY:
#   hallucination_detector.py — Detect hallucinated content
#   output_correction.py    — Post-process corrections
#   source_citations.py     — Add source citations
#   enhanced_metrics.py     — NLP metrics calculation
#
# MEMORY:
#   dual_memory_engine.py   — User + agent dual memory
#   hybrid_memory_ranker.py — Score and rank memories
#   memory_optimizer.py     — Memory compression
#   narrative_continuity_engine.py — Thread continuity
#   temporal_thread_engine.py — Time-based context
#
# OTHER:
#   credit_deduction.py     — Credit management
#   plan_limits.py          — Plan enforcement (GTM)
#   dsid_integration.py     — DSID blockchain integration
#   web_search.py           — Web search pipeline
#   image_generation.py     — DALL-E image generation
#   response_cache.py       — Response caching
#   token_optimizer.py      — Token usage optimization
#   token_tracker.py        — Token usage tracking

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. DATABASE SCHEMA (PostgreSQL — chat_db)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# resonant_chats          — Chat sessions (user_id, org_id, title, status)
# resonant_chat_messages  — Messages (chat_id, role, content, hash, xyz, meta_data)
# hallucination_settings  — Per-user hallucination detection config
# knowledge_base_entries  — User knowledge base entries
# tool_classifier_models  — Persisted ML tool classifier
# tool_active_samples     — Active learning samples for tool classifier
# agent_classifier_models — Persisted ML agent classifier
# agent_active_samples    — Active learning samples for agent classifier

echo "RG_Chat trace complete. See above for full pipeline."
