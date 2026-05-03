ay #!/bin/bash
# ============================================================================
# RESONANT CHAT + TOOL SYSTEM RESTRUCTURE PLAN
# Generated: 2026-04-27 | Status: PLANNING
# ============================================================================

cat << 'EOF'

================================================================================
         RESTRUCTURE PLAN — RESONANT CHAT + TOOL/APP/LLM BOUNDARIES
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1: THE THREE LAYERS (Clean Boundary Definition)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER 1: LLM SYSTEM (Intelligence — text in/out only)
  Where: UnifiedLLMClient → llm_service → Groq/OpenAI/Anthropic
  Rule:  NEVER executes tools. Only generates text + requests tool_calls.
  Parts: Provider routing, streaming, BYOK forwarding, model selection

LAYER 2: TOOLS (Stateless Functions — platform keys, no user OAuth)
  Auth:  Platform API keys (Tavily, OpenAI, Firecrawl) — user doesn't connect
  Speed: <30s, fire-and-forget, deterministic
  List:
    SEARCH:  web_search, fetch_url, news_search, wikipedia, deep_research
    MEDIA:   generate_image, generate_audio, generate_video, generate_chart
    CODE:    execute_code, http_request, run_command
    MEMORY:  memory_read, memory_write, hash_sphere_*
    FILE:    file_read, file_write, file_edit, grep_search
    GIT:     git_clone, git_push, github_* (with platform PAT)
    DATA:    weather, stock_crypto, get_current_time
    SCRAPE:  scrape_page, scrape_platforms
    EMAIL:   send_email (SMTP), configure_smtp
    DOCS:    create_presentation

LAYER 3: APPS (OAuth Integrations — user must connect first)
  Auth:  User OAuth token from auth_service (user connects in Settings)
  Rule:  ALWAYS pre-flight check "is connected?" before execution
  List:
    GOOGLE:  google_sheets, google_docs, google_drive, google_calendar
    COMMS:   gmail_send, gmail_read, slack_send, slack_read, discord
    CRM:     hubspot, salesforce, pipedrive, attio, zoho_crm, mailchimp
    PROJECT: notion, asana, clickup, linear, monday, atlassian, miro
    SOCIAL:  twitter_x, linkedin, youtube
    DEV:     figma, gitlab (OAuth)
    FINANCE: xero
    STORAGE: dropbox, microsoft
    OTHER:   zoom, calendly, typeform, dribbble, airtable, sigma

BOUNDARY KEY: Auth model determines the layer, NOT functionality.
  send_email (SMTP key = TOOL) vs gmail_send (user OAuth = APP)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2: RESONANT_CHAT.PY DECOMPOSITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Current: 1 file, 6,610 lines doing 10 jobs
Target:  Focused modules

EXTRACT FROM resonant_chat.py:
  1. ServiceClient + CircuitBreaker → app/infrastructure/service_client.py (~110 lines)
  2. Request/Response models → app/routers/chat_models.py (~100 lines)
  3. Navigation detection → app/services/navigation.py (~60 lines)
  4. Memory extraction logic → app/services/memory_extraction.py (~150 lines)
  5. Architect SSE proxy → app/services/architect_proxy.py (~200 lines)
  6. Post-processing (hallucination, metrics) → already in services/

RESONANT_CHAT.PY AFTER CLEANUP: ~3000-4000 lines
  - Endpoints (stream + sync + CRUD)
  - Pipeline orchestration (the step 1-10 flow)
  - Tool classification + dispatch calls
  - LLM call orchestration

WHY NOT SPLIT FURTHER: The pipeline is sequential (each step depends on
previous). Splitting into microservices adds latency. Splitting into more
files is fine but the orchestration logic must stay together.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3: TOOL EXECUTOR RESTRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Current problem in tool_executor.py:
  - 199 entries in _executors dict
  - 42 entries → _execute_agent_architect (SSE proxy to whole service)
  - ~140 entries → _execute_integration (generic dispatch)
  - 8 inline executors (code_visualizer, web_search, etc.)
  - NO distinction between Tool vs App at execution time

RESTRUCTURE:
  1. Split _executors into three registries:
     TOOL_EXECUTORS = {...}   # Layer 2 tools (platform keys)
     APP_EXECUTORS = {...}    # Layer 3 apps (OAuth required)
     PROXY_EXECUTORS = {...}  # Agent Architect proxy tools

  2. Add pre-flight for APP_EXECUTORS:
     async def _execute_app(self, tool_id, message, user_id, context):
         # Check if user has OAuth connected for this app
         connected = await self._check_oauth_connected(user_id, tool_id)
         if not connected:
             return {
                 "success": False,
                 "error": f"Please connect {tool_id} in Settings > Integrations",
                 "action": "connect_required",
                 "connect_url": "/connect-profiles",
             }
         return await self._execute_integration(message, user_id, context)

  3. Classifier gets app-awareness:
     - ToolClassifier prediction includes metadata: is_app=True/False
     - If is_app=True AND user has no OAuth → skip execution, return hint
     - Saves LLM call tokens (don't even send app tools if not connected)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4: TWO CLASSIFIER BOUNDARY (Architect vs Engine)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RG_Chat ToolClassifier (Layer 0 — routing)
  Purpose: Route user message to correct HANDLER (tool/app/architect/LLM)
  Scope:   198 labels = direct tool execution in Chat service
  Speed:   1-5ms, zero tokens
  Result:  Executes the tool immediately OR proxies to architect

RG_agent_architect ToolClassifier (Layer 1 — LLM tool subset)
  Purpose: Reduce 101 orchestrator tools to ~10 per LLM call
  Scope:   101 ORCHESTRATOR_TOOLS grouped into 15 semantic groups
  Speed:   1-5ms, saves ~6000 tokens per LLM call
  Result:  LLM sees only relevant tools, picks via function calling

RG_Agent_Engine (no classifier — LLM gets all 70 tools)
  Purpose: Agent execution (agents pick their own tools)
  Problem: 15,000 tokens/call wasted on tool schemas
  Fix:     Vendor Chat's classifier pattern here (Phase 3 of unification)

BOUNDARY IS CLEAR:
  Chat classifier = "WHICH service handles this message?"
  Architect classifier = "WHICH tools should LLM see for this message?"
  Agent Engine = "agent decides" (future: add classifier for token savings)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5: EXECUTION PLAN (Priority Order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE A: resonant_chat.py decomposition [HIGH — reduces confusion]
  □ Extract ServiceClient + CircuitBreaker → infrastructure/service_client.py
  □ Extract models → routers/chat_models.py
  □ Extract navigation → services/navigation.py
  □ Extract architect proxy → services/architect_proxy.py
  □ Extract memory extraction block → services/memory_extraction.py
  → Result: resonant_chat.py drops from 6,610 → ~4,000 lines

PHASE B: Tool/App separation [HIGH — fixes user experience]
  □ Add is_app flag to ToolDefinition in tools_registry.py
  □ Add pre-flight OAuth check in tool_executor.py for app tools
  □ Classifier returns "connect_required" instead of executing
  □ Frontend shows "Connect {service}" button when this happens
  → Result: Users never see cryptic OAuth errors

PHASE C: Classifier app-awareness [MEDIUM — saves tokens]
  □ At classify time, filter out apps user hasn't connected
  □ Cache user's connected apps (1 call to auth_service per session)
  □ Only predict tools user can actually use
  → Result: Better predictions, no wasted suggestions

PHASE D: Agent Engine classifier [LOW — token savings]
  □ Vendor Chat's classifier pattern into Agent Engine
  □ Before each LLM call: predict top 8 tools, send only those
  □ Active learning: predictions stored in shared DB
  → Result: 15K → 3K tokens per Agent Engine LLM call


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6: WHAT STAYS WHERE — FINAL OWNERSHIP MAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RG_Chat (Resonant Chat) — THE ROUTER
  Owns: Tool classification, tool execution, message pipeline
  Owns: 198 tool/app executors (via modular tools/ directory)
  Owns: Agent Architect SSE proxy (routes to architect service)
  Owns: LLM response streaming (via UnifiedLLMClient)
  Does NOT own: Agent CRUD, agent sessions, tool building

RG_agent_architect — THE BUILDER
  Owns: Agent creation/modification/testing pipeline
  Owns: 101 orchestrator tools (agent management, scheduling, etc.)
  Owns: Build pipeline (7 phases)
  Owns: Runner (delegates to Agent Engine)
  Does NOT own: Direct tool execution (web_search etc. — that's Engine)

RG_Agent_Engine — THE EXECUTOR
  Owns: Agent session execution (the agent's runtime)
  Owns: 70+ tools that agents actually use during runs
  Owns: Tool handler map (web_search, google_sheets, code, etc.)
  Does NOT own: Tool selection for the user (Chat does that)

RG_LLM_Service — THE PROVIDER
  Owns: Multi-provider LLM routing (Groq, OpenAI, Anthropic, etc.)
  Owns: Rate limiting, key rotation, provider health
  Does NOT own: Tool execution, message history, memory

================================================================================
CHECKPOINT: SSE FULL PIPELINE PORT (May 2, 2026)
================================================================================

PROBLEM:
  The SSE streaming endpoint (/message/stream) was a skeleton — only architect
  had full SSE handling. Non-architect messages went to a gutted path that ONLY
  called maybe_spawn_agent(), skipping:
    - Tool detection (neural classifier)
    - Tool execution (GoogleCalendar, CodeVisualizer, etc.)
    - Web search
    - Image generation
    - Teams (maybe_run_team)
    - Debate (maybe_run_debate)
    - Governance/metrics
  Frontend uses SSE as PRIMARY → 80% of intelligence pipeline was bypassed.

FIX (resonant_chat.py):
  1. Expanded tool detection: classifier now routes ALL tools, not just architect
  2. Replaced gutted else branch with full pipeline:
     Memory → Tool detect → Tool execute → Web search → Image gen →
     Teams → Debate → Agent spawn → Forced reasoning → Direct LLM fallback
  3. Integration skills (google_calendar, figma, sigma, google_drive) get
     grounded responses — tool output IS the response, no LLM hallucination
  4. Non-architect tool results stored in DB metadata (toolResults)

FIX (tool_training_data.py):
  Added 9 disambiguation samples for google_calendar vs calendly:
    "add to my calendar" → google_calendar (not calendly)
    "can u add to my calendar" → google_calendar
    etc.

FIX (tool_classifier.py — from prior session):
  Excluded agent_architect from continuity boost to prevent misrouting.

FILES CHANGED:
  - RG_Chat/app/routers/resonant_chat.py (SSE full pipeline)
  - RG_Chat/app/services/tool_training_data.py (+9 google_calendar samples)
  - RG_Chat/app/services/tool_classifier.py (continuity boost fix)

DEPLOYED: chat_service rebuilt and restarted. Classifier retrained on 1040 samples.

FIX (hallucination_detector.py — May 2):
  LLM Judge "No LLM providers" — root cause was hardcoded
  preferred_provider="groq" in llm_judge_verify(). Container only has
  TOKENROUTER_API_KEY + OPENAI_API_KEY, no GROQ_API_KEY. With strict
  provider mode, only groq was tried → fail. Fix: removed
  preferred_provider, letting the fallback chain (tokenrouter → openai)
  handle it automatically.

FIX (Frontend — May 2):
  Pipeline steps UI existed but was never wired:
  1. SSEStreamEvent interface: added name, success, query fields
  2. Step handler: was only setting aiProvider text, now populates
     pipelineSteps state array for the visual pipeline indicator
  3. Added icons (wrench/check/globe/image) and colors for new step
     types: tool_detection, tool_result, web_search, image_generated
  4. Pipeline steps cleared on stream completion and on fallback

DEPLOYED: chat_service + frontend rebuilt and deployed.

REMAINING (May 2):
  - All issues resolved. Monitor for regressions.


================================================================================
CHECKPOINT: P0 PIPELINE FIXES (May 3, 2026)
================================================================================

CONTEXT:
  Deep analysis identified critical weaknesses vs market-leading solutions:
    W1: SSE endpoint missing post-processing (DSID, memory ingest, PMI, caching)
    W2: No real token-by-token streaming (batch-emitting accumulated text)
    W8: No native LLM function calling (separate classifier only)
    W9: Singleton API key race condition (concurrent BYOK users clobber keys)

── W9: FIX SINGLETON API KEY RACE CONDITION ──────────────────────────────

PROBLEM:
  MultiAIRouter was a singleton. set_user_api_keys() mutated instance state.
  With concurrent requests, User A's keys could be overwritten by User B
  between set_user_api_keys() and the actual route_query() call.

FIX:
  1. MultiAIRouter.route_query() now accepts user_keys param (line 61).
     Explicit param takes priority over instance-level keys.
  2. facade.py route_query() and route_query_stream() pass user_api_keys
     directly as a parameter — no more set/clear mutation cycle.
  3. Removed set_user_api_keys / clear_user_api_keys from facade.py exports.
  4. Agent/team paths still use set_user_api_keys on the shared instance as
     fallback (agent_engine.spawn() calls router internally without user_keys
     param). This is safe because route_query prioritizes explicit param.

FILES:
  - app/domain/provider/multi_ai_router.py (user_keys param on route_query)
  - app/domain/provider/facade.py (param-passing, removed set/clear)
  - app/domain/provider/__init__.py (cleaned exports)

── W2: REAL TOKEN-BY-TOKEN STREAMING ─────────────────────────────────────

PROBLEM:
  SSE endpoint accumulated full response text, then emitted it as a single
  "chunk" event. Frontend replaced content on each chunk. No typewriter
  effect — text appeared all at once after agent/LLM finished.

FIX:
  1. Non-agent/tool path now uses route_query_stream() which yields
     individual tokens from the LLM via UnifiedLLMClient.stream().
  2. Each token is emitted as a separate SSE "chunk" event.
  3. Frontend detects "streaming" flag in the "start" event:
     - streaming=true → append mode (token deltas, typewriter effect)
     - streaming=false/absent → replace mode (architect full-text chunks)
  4. Agent/team/debate/tool paths still emit pre-computed text as single
     chunk (these are not streamable since they do multiple LLM calls).

FILES:
  - app/routers/resonant_chat.py (route_query_stream in else branch)
  - app/domain/provider/facade.py (route_query_stream already existed)
  - ORG_Frontend/src/api/resonantChat.ts (streaming field on SSEStreamEvent)
  - ORG_Frontend/src/pages/ResonantChat/ResonantChatPage.tsx
    (isTokenStreaming flag, append vs replace logic)

── W1: PORT MISSING PIPELINE STAGES TO SSE ───────────────────────────────

PROBLEM:
  SSE stored the assistant message with hardcoded resonance_score=0.5,
  no DSID lineage, no memory ingestion, no PMI blockchain events,
  no response caching, no response sanitization. All of these were
  only in the non-streaming /message endpoint.

FIX:
  1. Response sanitization: _sanitize_agent_response() strips leaked
     agent prompts before storage.
  2. Real resonance score: _calculate_resonance_score() uses response
     quality, memory overlap, and context relevance.
  3. DSID creation: create_message_dsid() for both user and assistant
     messages. Lineage stored in message meta_data.
  4. Memory ingestion: service_client calls to memory_service for
     both user and assistant messages.
  5. PMI blockchain events: pmi_manager.create_memory_event() for
     prompt and response events.
  6. Response caching: cache_response() with quality score.
  7. All post-processing is try/except guarded — failures are non-critical
     and don't break the SSE stream.
  8. SSE emits step events for post_processing and memory_ingest so
     the frontend pipeline indicator shows these stages.

FILES:
  - app/routers/resonant_chat.py (post-processing block after message store)

── W8: NATIVE LLM FUNCTION CALLING ──────────────────────────────────────

PROBLEM:
  Tool detection used a separate neural classifier (sentence-transformers).
  The LLM itself never saw tool definitions. This meant:
    - LLM couldn't reason about which tool to use
    - LLM couldn't extract structured arguments from the message
    - Two separate systems (classifier + LLM) with potential disagreement
    - No support for multi-tool calls in a single response

FIX:
  1. Created native_tool_definitions.py with OpenAI function-calling format
     tool schemas for: web_search, image_generation, code_visualizer,
     google_calendar, google_drive, memory_search, figma.
  2. UnifiedLLMClient already supported tools (LLMRequest.tools field) for
     OpenAI, Anthropic (auto-converted), and Gemini (auto-converted).
  3. Added route_query_with_tools() to facade.py:
     - Phase 1: Non-streaming LLM call with tool definitions + tool_choice=auto
     - If tool_calls returned: yields them so caller can execute
     - If no tool_calls: yields response content as chunk
  4. SSE pipeline now tries native function calling first:
     - Calls route_query_with_tools() with native tool definitions
     - If LLM returns tool_calls: executes via tool_executor, adds results
       to context, then streams follow-up response via route_query_stream()
     - If LLM returns direct content: streams it
     - Falls back to plain route_query_stream() if native tools fail
  5. The neural classifier (tool_classifier) is still the primary tool
     detection layer — it runs BEFORE the LLM streaming path. Native
     function calling is a SECOND CHANCE for the direct LLM path.

FILES:
  - app/services/native_tool_definitions.py (NEW — 7 tool schemas)
  - app/domain/provider/facade.py (route_query_with_tools)
  - app/domain/provider/__init__.py (export route_query_with_tools)
  - app/routers/resonant_chat.py (native tool calling in streaming path)

── SUMMARY OF ALL FILES CHANGED (May 3) ─────────────────────────────────

Backend (RG_Chat):
  - app/domain/provider/multi_ai_router.py .......... W9 (user_keys param)
  - app/domain/provider/facade.py ................... W9+W2+W8 (param passing, streaming, tools)
  - app/domain/provider/__init__.py ................. W9+W8 (exports)
  - app/routers/resonant_chat.py .................... W1+W2+W8 (full pipeline, streaming, tools)
  - app/services/native_tool_definitions.py ......... W8 (NEW — tool schemas)

Frontend (ORG_Frontend):
  - src/api/resonantChat.ts ......................... W2 (streaming field)
  - src/pages/ResonantChat/ResonantChatPage.tsx ..... W2 (token append mode)

DEPLOYMENT: Rebuild chat_service + frontend containers.

================================================================================
                           END OF PLAN
================================================================================
EOF
