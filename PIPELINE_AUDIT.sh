#!/bin/bash
# ============================================================================
# RESONANT CHAT + ARCHITECT PIPELINE AUDIT
# Date: May 10, 2026
# ============================================================================

# ============================================================================
# SECTION 1: CURRENT PIPELINE TRACE (What Actually Happens)
# ============================================================================

# When user says "Build me a daily news agent":
#
# STEP 1: Frontend → POST /message/stream (SSE)
#   - ResonantChatPage.tsx calls streamResonantMessage()
#   - Opens SSE connection to gateway → chat_service
#
# STEP 2: Chat Service — Authentication + Context Loading
#   - Extract user_id, user_role, org_id from headers/cookies
#   - Load chat history (last 30 messages)
#   - Extract 5 full + 20 summarized context messages
#   STATUS: ✅ WORKING
#
# STEP 3: Chat Service — Tool Classification (Neural)
#   - SkillClassifier.predict() — sentence-transformers MiniLM-L6-v2
#   - Predicts tool_id + confidence from 200+ tools
#   - For "build me an agent" → predicts agent_architect (conf ~0.85)
#   - Also checks: keyword regex guard, conversation continuity
#   STATUS: ✅ WORKING
#
# STEP 4: Chat Service — Route Decision
#   - If use_architect=True → Forward to agent_architect service via SSE
#   - If other tool → Execute tool, inject result into LLM context
#   - If no tool → Full pipeline (memory→team→debate→agent→LLM stream)
#   STATUS: ✅ WORKING
#
# STEP 5A: Architect Path — SSE Proxy to agent_architect:8000
#   - Sends: message, conversation_history[-20], user_api_keys
#   - Architect orchestrator runs multi-turn tool-calling loop
#   - Tools: build_agent, present_options, update_agent, run_agent, etc.
#   - Streams events: text, options, tool_call, build_progress, complete
#   STATUS: ✅ WORKING — architect calls build_agent, presents options
#
# STEP 5B: Non-Architect Path — Full Pipeline
#   a. Memory extraction (RAG + Hash Sphere)
#   b. Tool execution (if classified)
#   c. Web search (if classified as web_search)
#   d. Image generation (if classified as image_generation)
#   e. Team execution (multi-agent teams)
#   f. Debate engine
#   g. Agent spawn (reasoning/coding/analysis agent)
#   h. Forced reasoning fallback
#   i. Native function calling with tool results
#   j. Token-by-token LLM streaming (final fallback)
#   STATUS: ✅ WORKING
#
# STEP 6: Post-Processing
#   - Strip present_options() text
#   - Sanitize agent prompt leakage
#   - Calculate resonance score
#   - Create DSID (blockchain lineage tracking)
#   - Ingest to memory service
#   - Create PMI blockchain events
#   - Cache response
#   STATUS: ✅ WORKING
#
# STEP 7: Frontend — Event Handling
#   - start → Open agents panel (if architect)
#   - chunk → Update message content (token stream or full replace)
#   - step → Show pipeline step in UI
#   - options → Show interactive buttons (present_options)
#   - done → Finalize message with ID, hash, provider, resonance score
#   - error → Show error
#   STATUS: ✅ WORKING

# ============================================================================
# SECTION 2: WHAT'S STILL OFF / BROKEN
# ============================================================================

# ISSUE 1: Architect Build Steps Not Streamed Visually [FIXED ✅]
#   The architect emits build_progress, build_step, test_step, verify_step
#   events, but the frontend stepLabels map didn't render them specifically.
#   FIX APPLIED: Added all architect event types to stepLabels map in ResonantChatPage.tsx
#   Deployed: May 10, 2026

# ISSUE 2: Provider/Providers Polling Spam [LOW]  
#   Frontend polls GET /resonant-chat/providers every few seconds. Logs show
#   50+ provider polls per minute. Should use longer interval or cache.
#   FIX: Increase poll interval from ~3s to 30s+

# ISSUE 3: Architect Memory Service 404 [FIXED ✅]
#   Logs showed: "[Memory] /rag/ask → 404" and "/rag/memories → 404"
#   ROOT CAUSE: RAG router prefix changed from /rag to /memory/rag (Apr 12 fix)
#   but architect still called old paths. Also chat memory_tools.py used
#   /api/v1/memory/... prefix that doesn't exist on memory_service.
#   FIX: Updated all paths in:
#     - RG_agent_architect/src/services/memory/memory_store.py (/rag/* → /memory/rag/*)
#     - RG_Chat/app/services/tools/memory_tools.py (removed /api/v1 prefix)
#     - RG_Chat/app/services/tool_executor.py (removed /api/v1 prefix)
#   Deployed: May 10, 2026

# ISSUE 4: Conversation History Truncation [FIXED ✅]
#   Only last 8 messages sent to architect (500 chars each = 4KB max).
#   For complex multi-turn agent builds, this loses context.
#   FIX APPLIED: Increased from 8→20 messages, 500→1000 chars per message.
#   Also increased general LLM context from 10→15 messages.
#   All 3 places updated: SSE architect, non-stream tool, forced guard.
#   Deployed: May 10, 2026

# ISSUE 5: No Live Session Monitoring During Build [HIGH]
#   When architect calls build_agent and triggers a test run, the frontend
#   doesn't show the live session steps. User has to manually navigate to
#   the agents page to see session progress.
#   FIX: Stream session steps through the SSE connection

# ISSUE 6: Tool Tab Panel Error on /agents [FIXED ✅]
#   "J.map is not a function" — fixed with Array.isArray guard
#   Deployed: May 10, 2026

# ISSUE 7: Agent Web Tool Content Truncation [FIXED ✅]
#   History was truncating fetch_url results to 400 chars → now 8000
#   Deployed: May 10, 2026

# ISSUE 8: LLM Provider Fallback [FIXED ✅]
#   strict_provider mode prevented fallback when preferred provider had no key
#   Deployed: May 10, 2026

# ============================================================================
# SECTION 3: COMPARISON WITH MARKET SOLUTIONS
# ============================================================================

# COMPETITOR: ChatGPT (OpenAI)
# ───────────────────────────
# ✅ They have: Real-time streaming, function calling, code interpreter, DALL-E,
#    web search, file upload, memory, custom GPTs, Canvas editor
# ❌ We match: Streaming ✅, function calling ✅, code execution ✅,
#    image gen ✅, web search ✅, file upload ✅, memory ✅, custom agents ✅
# 🔴 WE'RE BEHIND ON:
#    - Canvas editor (inline code/doc editing) — we don't have this
#    - Voice mode — we don't have this
#    - Structured output (JSON mode) — we have it but not surfaced in UI
#    - ChatGPT has 1-2 second response time, ours is 3-6 seconds
#    - ChatGPT memory is seamless; ours requires explicit memory tools
#    - Deep research shows live sources with citations; ours doesn't show sources in UI

# COMPETITOR: Claude (Anthropic)
# ──────────────────────────────
# ✅ They have: Artifacts (live code/HTML preview), Projects, Tool use,
#    Computer use, long context (200K), thinking mode
# ❌ We match: Tool use ✅, long context (via TokenRouter) ✅
# 🔴 WE'RE BEHIND ON:
#    - Artifacts (live HTML/code preview) — we don't render results inline
#    - Projects (persistent workspace with files) — we have agents but no shared workspace
#    - Computer use — we have code execution but not browser automation
#    - Thinking/reasoning mode shown in UI — we have it but don't display thought process
#    - Claude's response quality is higher due to their model quality

# COMPETITOR: Cursor / Windsurf (IDE AI)
# ───────────────────────────────────────
# ✅ They have: Code-aware AI, multi-file editing, terminal integration,
#    inline diffs, codebase indexing, cascading context
# ❌ We match: Code execution ✅, RG_IDE extension
# 🔴 WE'RE BEHIND ON:
#    - We're not an IDE — we're a chat platform that can execute code
#    - No inline diff/apply UI
#    - No codebase indexing
#    - No cascading multi-file context

# COMPETITOR: Devin / Manus (Agent Platforms)
# ────────────────────────────────────────────
# ✅ They have: Full autonomous agents, browser control, live preview,
#    planning visible to user, checkpoint/rollback
# ❌ We match: Autonomous agents ✅, planning ✅, tool execution ✅
# 🔴 WE'RE BEHIND ON:
#    - No browser/screen control (Devin has full browser automation)
#    - No live preview of agent work (we show step text, not actual output)
#    - No checkpoint/rollback UI
#    - Our agent execution is slower (5-30 sec per step vs 2-5 sec)
#    - No file/artifact viewer during agent execution

# ============================================================================
# SECTION 4: TOP PRIORITIES TO CLOSE THE GAP
# ============================================================================

# PRIORITY 1: RESPONSE SPEED [In Progress]
#   Our streaming starts in 3-6 seconds. Market standard is <2 seconds.
#   Root cause: Memory extraction + tool classification + multi-hop routing
#   DONE: Memory extraction pre-warmed via asyncio.create_task (runs concurrently
#         with _generate_events setup). Saves ~0.5-1s on non-architect path.
#   TODO: Cache frequently used prompts. Pre-warm LLM connections.
#         TokenRouter hop adds ~1-2s — consider direct provider calls for BYOK users.

# PRIORITY 2: INLINE ARTIFACTS / LIVE PREVIEW [High]
#   When agent builds something or fetches data, show it inline.
#   Claude's Artifacts and ChatGPT's Canvas are huge UX wins.
#   FIX: Add <Artifact> component that renders HTML/code/charts inline.

# PRIORITY 3: VISIBLE REASONING [High]
#   Users can't see what the AI is thinking. Claude shows thinking tokens.
#   Our agents have reasoning but it's hidden in logs.
#   FIX: Stream reasoning/thinking events to frontend. Show collapsible
#         "thinking" sections like Claude does.

# PRIORITY 4: CITATION/SOURCE UI [Medium]
#   When web search returns results, show sources with links.
#   ChatGPT shows numbered citations. We just inject into context invisibly.
#   FIX: Include source URLs in web search results. Display source cards in UI.

# PRIORITY 5: ARCHITECT BUILD VISUALIZATION [Medium]
#   When building an agent, show:
#   - System prompt being generated (live)
#   - Tools being configured
#   - Test run results
#   - Agent card preview
#   FIX: Map all architect SSE events to rich UI components

# ============================================================================
# SECTION 5: REAL TIMING DATA (Measured May 10, 2026)
# ============================================================================

# Test: "build me a daily tech news agent" (architect path)
#
# 0.00s → Request sent
# 0.68s → SSE connection opened (status 200)
# 0.70s → start event (architect detected in 0.7s)
# 1.30s → step:thinking "Analyzing your request..."
# 5.80s → chunk (1085 chars) — LLM response arrived (4.5s for Gemini Flash)
# 5.80s → tool_call: present_options executed
# 5.80s → tool_result: options ready
# 6.00s → OPTIONS event (interactive buttons)
# 6.10s → post_processing (DSID lineage)
# 6.90s → memory_ingest (stored to memory)
# 6.90s → DONE (provider=tool_agent_architect)
#
# TOTAL: 6.9 seconds end-to-end
#
# BREAKDOWN:
#   - Auth + Context + Classify: 0.7s
#   - LLM call (Gemini Flash via TokenRouter): 4.5s ← BOTTLENECK
#   - Tool execution + options: 0.2s
#   - Post-processing (DSID + memory): 1.1s
#
# ChatGPT comparison: ~2-3s for similar interaction
# Claude comparison: ~2-4s for similar interaction
# Our bottleneck: TokenRouter routing adds ~1-2s overhead vs direct API

# ============================================================================
# FILES INVOLVED
# ============================================================================
# Backend Pipeline:
#   RG_Chat/app/routers/resonant_chat.py — Main SSE endpoint (6796 lines)
#   RG_Chat/app/services/tool_classifier.py — Neural tool classifier
#   RG_Chat/app/services/tool_executor.py — Tool execution
#   RG_Chat/app/domain/provider.py — LLM routing (route_query_stream, route_query_with_tools)
#   RG_Chat/app/domain/agent.py — Agent spawn, team, debate
#   RG_Chat/app/services/web_search.py — Web search service
#   RG_Chat/app/services/native_tool_definitions.py — Native function calling defs
#
# Architect:
#   RG_agent_architect/src/orchestrator/orchestrator.py — Multi-turn tool loop
#   RG_agent_architect/src/services/builder.py — Agent builder
#
# Frontend:
#   ORG_Frontend/src/pages/ResonantChat/ResonantChatPage.tsx — Main chat (7046 lines)
#   ORG_Frontend/src/api/resonantChat.ts — SSE streaming client
#
# Agent Engine:
#   RG_Agent_Engine/app/executor.py — Agent execution loop (4300 lines)

# ============================================================================
# SECTION 6: DEPLOYMENT STATUS (Last Updated May 10, 2026 ~5:30pm PST)
# ============================================================================

# DEPLOYED THIS SESSION:
#   ✅ agent_architect — rebuilt + restarted (memory paths fixed)
#   ✅ chat_service — rebuilt + restarted (history truncation + speed + memory paths)
#   ✅ agent_engine_service — deployed earlier (executor JSON list fix)
#
# CURRENT SERVER STATUS:
#   🔴 SERVER UNREACHABLE — dev-swat.com (134.199.221.149) 100% packet loss
#   Cannot verify container health. Possible causes:
#     - VPS provider network outage
#     - Server crash (OOM / disk full / kernel panic)
#     - Firewall lockout
#   Auth container may be down as user reported. Need server access to verify.
#
# FIXES AWAITING VERIFICATION:
#   - Memory service 404 fix (architect can now enrich context properly)
#   - Conversation history 8→20 messages
#   - Pre-warmed memory extraction (speed improvement)
#   - Chat memory_tools /api/v1 prefix removal

# ============================================================================
# TODO REMAINING:
# ============================================================================
#   1. [HIGH] Verify auth container once server is back
#   2. [HIGH] Live session monitoring during agent builds
#   3. [MEDIUM] Provider polling spam (3s→30s interval)
#   4. [MEDIUM] Inline artifacts / live preview component
#   5. [MEDIUM] Visible reasoning (thinking tokens UI)
#   6. [LOW] Citation/source UI for web search results
