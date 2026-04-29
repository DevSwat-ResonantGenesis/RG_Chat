#!/bin/bash
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
                           END OF PLAN
================================================================================
EOF
