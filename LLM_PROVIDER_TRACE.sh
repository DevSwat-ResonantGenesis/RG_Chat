#!/bin/bash
# ============================================================================
# LLM PROVIDER + BYOK TRACE — FULL PLATFORM MAP
# Generated: 2026-04-27 | Status: TRACED
# ============================================================================

cat << 'EOF'

================================================================================
         LLM PROVIDER + BYOK TRACE — ACROSS ALL RG SERVICES
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1: THE UNIFIED CLIENT — rg_llm (RG_UnifiedLLMClient)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOCATION: RG_UnifiedLLMClient/src/rg_llm/
FILES:    client.py, keys.py, providers.py, models.py, __init__.py

WHAT IT IS:
  A Python library (not a service!) that provides:
    - UnifiedLLMClient class with .complete() and .stream() methods
    - Direct HTTP calls to provider APIs (OpenAI, Anthropic, Gemini, Groq, etc.)
    - BYOK dual-key resolution: user key first → platform key fallback
    - Provider fallback chain with cooldown tracking
    - Tool calling support (OpenAI, Anthropic, Gemini formats)
    - Streaming support (SSE parsing)

SUPPORTED PROVIDERS (providers.py — single source of truth):
  Tier 1: openai, anthropic, groq, google (Gemini)
  Tier 2: deepseek, mistral, together, perplexity, fireworks, openrouter, cohere, bedrock
  Total: 12 providers

DEFAULT FALLBACK ORDER:
  openai → anthropic → groq → google → deepseek → mistral

HOW IT'S MOUNTED:
  Docker volume mount "rg_llm" maps RG_UnifiedLLMClient/src/rg_llm → /app/rg_llm
  Services import with: from rg_llm import UnifiedLLMClient, LLMRequest


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2: SERVICE-BY-SERVICE LLM USAGE MAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SERVICE              | LLM METHOD           | BYOK? | BYOK SOURCE
---------------------+----------------------+-------+---------------------------
RG_Chat              | ✅ UnifiedLLMClient  | ✅ YES | auth_service internal API
RG_Agent_Engine      | ✅ UnifiedLLMClient  | ✅ YES | auth_service internal API
RG_agent_architect   | ✅ UnifiedLLMClient  | ✅ YES | Passed from RG_Chat
RG_Axtention_IDE     | ✅ UnifiedLLMClient  | ✅ YES | auth_service internal API
RG_Ed_Service        | ✅ UnifiedLLMClient  | ❌ NO  | Platform keys only
RG_LLM_Service       | ❌ OWN PROVIDERS     | ⚠️ PARTIAL | Via request body only
---------------------+----------------------+-------+---------------------------


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3: DETAILED BYOK FLOW PER SERVICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[A] RG_Chat (Resonant Chat) — FULL BYOK ✅
  File: app/services/user_api_keys.py
  Flow:
    1. User sends message → resonant_chat.py
    2. Fetches user keys: GET auth_service/auth/internal/user-api-keys/{user_id}
       (uses AUTH_SERVICE_URL + AUTH_INTERNAL_SERVICE_KEY headers)
    3. Returns Dict[str, str] mapping provider → decrypted API key
    4. Keys passed to MultiAIRouter.set_user_api_keys(keys)
    5. MultiAIRouter calls UnifiedLLMClient.complete(request, user_keys=keys)
    6. UnifiedLLMClient tries: BYOK key first → platform env key fallback
  Providers: All 12 from providers.py

[B] RG_Agent_Engine — FULL BYOK ✅
  File: app/executor.py (_get_user_api_key method)
  Flow:
    1. Agent session starts → executor loads user_id from session
    2. Fetches BYOK keys: GET auth_service/auth/internal/user-api-keys/{user_id}
    3. Keys passed to _llm_client.complete(request, user_keys=keys)
    4. ALSO used for tool-specific keys (Figma, Google Drive, Sigma, Suno, etc.)
       → Same _get_user_api_key(session, "figma") method fetches per-provider
  LLM Providers: All 12 via UnifiedLLMClient
  Tool Providers: figma, google-drive, google-calendar, sigma, suno,
                  replicate, openai (for DALL-E/TTS)

[C] RG_agent_architect — BYOK VIA PASSTHROUGH ✅
  File: src/core/llm_client.py
  Flow:
    1. RG_Chat calls architect: POST /api/message/stream
    2. RG_Chat includes user_api_keys in request body:
       svc_payload["user_api_keys"] = context.get("user_api_keys", {})
    3. Architect's call_llm() receives user_api_keys parameter
    4. Passes to UnifiedLLMClient.complete(request, user_keys=user_api_keys)
  NOTE: Architect does NOT fetch keys itself — relies on Chat forwarding them

[D] RG_Axtention_IDE — FULL BYOK ✅
  File: app/llm_client.py (fetch_user_byok_keys function)
  Flow:
    1. IDE agent loop receives user_id from extension
    2. Fetches: GET auth_service/api-keys/user/{user_id}
    3. Keys passed to UnifiedLLMClient.complete(request, user_keys=keys)
  ⚠️ DIFFERENT ENDPOINT: Uses /api-keys/user/{user_id} not
     /auth/internal/user-api-keys/{user_id} like Chat/Engine
     (Both work, but inconsistent)

[E] RG_Ed_Service — NO BYOK ❌
  File: app/agents/controller.py, app/tools/builtin.py
  Flow:
    1. Creates UnifiedLLMClient() with NO byok_fetcher
    2. Calls _client.complete(LLMRequest(...)) with NO user_keys
    3. Falls back to platform env keys only
  GAP: Ed service runs user file operations + LLM calls but never fetches
       user's own API keys. Users always burn platform credits here.

[F] RG_LLM_Service — SEPARATE IMPLEMENTATION ⚠️
  File: app/routers.py, app/providers/*.py, app/multi_provider/multi_ai_router.py
  Flow:
    1. Receives BYOK keys in request body: request.user_api_keys
    2. Has its OWN provider implementations (OpenAIProvider, AnthropicProvider,
       OllamaProvider) — NOT using UnifiedLLMClient
    3. Has its OWN MultiAIRouter with manual key rotation (groq_key_index, etc.)
    4. Supports: openai, anthropic, groq, google/gemini, ollama
  GAP: This is a LEGACY service with duplicate provider code
       NOT using UnifiedLLMClient at all
  WHO CALLS IT: Nobody in production calls it for LLM anymore
                (All services use UnifiedLLMClient directly)
                Still used for /llm/providers endpoint (provider list)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4: BYOK KEY STORAGE + AUTH SERVICE ENDPOINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STORAGE: RG_Auth → PostgreSQL (user_api_keys table)
  - Keys stored ENCRYPTED (AES via crypto.py)
  - Fields: user_id, provider, encrypted_key, key_prefix, name, is_primary, is_valid

INTERNAL API (service-to-service, no JWT required):
  GET /auth/internal/user-api-keys/{user_id}
    → Returns: { "keys": [{ "provider": "openai", "api_key": "sk-...", "is_primary": true }] }
    → Called by: RG_Chat, RG_Agent_Engine
    → Auth: x-internal-service-key header

ALTERNATE API (used by IDE):
  GET /api-keys/user/{user_id}
    → Same data, different path
    → Called by: RG_Axtention_IDE

USER-FACING API:
  GET /auth/user/api-keys   → Lists user's own keys (masked)
  POST /auth/user/api-keys  → Add a new key
  DELETE /auth/user/api-keys/{id} → Remove a key

FRONTEND:
  Settings > API Keys page → user adds openai, anthropic, groq, google keys
  Settings > Connect Profiles → Google OAuth for Drive/Calendar/Gmail


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5: PROVIDER CONFIG — PLATFORM KEYS (ENV VARS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

These env vars are set in docker-compose.unified.yml and available to all
services that have the rg_llm volume mount:

ENV VAR              | PROVIDER     | USED BY
---------------------+--------------+----------------------------------
OPENAI_API_KEY       | OpenAI       | All services (platform fallback)
ANTHROPIC_API_KEY    | Anthropic    | All services (platform fallback)
GROQ_API_KEY         | Groq         | All services (platform fallback)
GEMINI_API_KEY       | Google       | All services (platform fallback)
GOOGLE_API_KEY       | Google (alt) | All services (platform fallback)
DEEPSEEK_API_KEY     | DeepSeek     | All services (if configured)
MISTRAL_API_KEY      | Mistral      | All services (if configured)
TOGETHER_API_KEY     | Together     | All services (if configured)
PERPLEXITY_API_KEY   | Perplexity   | All services (if configured)
FIREWORKS_API_KEY    | Fireworks    | All services (if configured)
OPENROUTER_API_KEY   | OpenRouter   | All services (if configured)
COHERE_API_KEY       | Cohere       | All services (if configured)

KEY RESOLUTION ORDER (per request):
  1. User's BYOK key for requested provider
  2. Platform env key for requested provider
  3. User's BYOK key for fallback providers (in DEFAULT_FALLBACK_ORDER)
  4. Platform env key for fallback providers


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6: GAPS + ISSUES FOUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GAP 1: RG_Ed_Service has NO BYOK support ❌
  Impact: Users with BYOK keys still use platform credits in Ed Service
  Fix: Add byok_fetcher or pass user_keys from calling service

GAP 2: RG_LLM_Service is a DEAD SERVICE for LLM routing ⚠️
  - Has its own OpenAIProvider, AnthropicProvider, OllamaProvider classes
  - Has its own MultiAIRouter with manual key rotation
  - Nobody calls it for LLM anymore — all services use UnifiedLLMClient directly
  - Only used for /llm/providers endpoint (returns available provider list)
  Impact: 1000+ lines of dead LLM code that's never called
  Fix: Strip down to just the /providers endpoint or remove entirely

GAP 3: IDE uses DIFFERENT auth endpoint for BYOK ⚠️
  - Chat/Engine: GET /auth/internal/user-api-keys/{user_id}
  - IDE: GET /api-keys/user/{user_id}
  Impact: Inconsistent, but both work. Could break if one endpoint changes.
  Fix: Standardize all services to use the same endpoint

GAP 4: RG_agent_architect has NO fallback if Chat doesn't send keys ⚠️
  - Relies entirely on Chat forwarding user_api_keys in request body
  - If Chat fails to include keys, architect uses only platform keys
  - No independent BYOK fetch capability
  Impact: If user's preferred provider is set in their keys, architect
          might use wrong provider when Chat doesn't forward keys
  Fix: Add byok_fetcher to architect, or verify Chat always sends keys

GAP 5: No BYOK caching ⚠️
  - Every LLM call in Chat triggers: HTTP GET → auth_service → decrypt → return
  - That's 5+ auth_service calls per user message (tool classifier, LLM, etc.)
  Impact: Extra latency (5ms per call × 5 calls = 25ms added)
  Fix: Cache BYOK keys per user for 60s in-memory (they rarely change)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7: ARCHITECTURE DIAGRAM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────────────┐
  │                    FRONTEND (User)                       │
  │  Settings > API Keys → POST /auth/user/api-keys         │
  └──────────────────────────┬──────────────────────────────┘
                             │ (stores encrypted in DB)
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │                    RG_Auth Service                       │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │ user_api_keys table (encrypted)                  │    │
  │  │ GET /auth/internal/user-api-keys/{user_id}       │    │
  │  │ → Returns decrypted keys per provider            │    │
  │  └─────────────────────────────────────────────────┘    │
  └──────┬──────────┬──────────┬──────────┬────────────────┘
         │          │          │          │
    ┌────▼───┐ ┌────▼───┐ ┌───▼────┐ ┌───▼────┐
    │RG_Chat │ │RG_Eng  │ │RG_IDE  │ │RG_Ed   │
    │BYOK ✅ │ │BYOK ✅ │ │BYOK ✅ │ │BYOK ❌ │
    └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
        │          │          │          │
        │   passes keys       │          │
        │   in request        │          │
        ▼          ▼          ▼          ▼
  ┌─────────────────────────────────────────────────────────┐
  │              rg_llm (UnifiedLLMClient)                   │
  │  ┌──────────────────────────────────────────────────┐   │
  │  │ 1. Try BYOK key for preferred provider           │   │
  │  │ 2. Try platform key for preferred provider       │   │
  │  │ 3. Try BYOK keys for fallback providers          │   │
  │  │ 4. Try platform keys for fallback providers      │   │
  │  └──────────────────────────────────────────────────┘   │
  └────┬──────┬──────┬──────┬──────┬──────┬────────────────┘
       │      │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼      ▼
    OpenAI Anthropic Groq  Google DeepSeek Mistral ...
    (12 providers total)

  RG_agent_architect:
    Chat → POST /api/message/stream { user_api_keys: {...} }
    → architect uses keys from request body (no independent fetch)

  RG_LLM_Service:
    ⚠️ LEGACY — has own provider implementations, NOT using rg_llm
    Nobody calls it for LLM routing anymore
    Only serves GET /llm/providers (available provider list)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8: ANSWER TO USER'S QUESTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: Are LLM providers unified across the entire RG platform?
A: YES, mostly. 5 of 6 LLM-calling services use the same UnifiedLLMClient
   library (rg_llm) with the same provider list, same fallback chain, and
   same key resolution logic. The exception is RG_LLM_Service which has its
   own duplicate provider code, but it's effectively dead — nobody calls it
   for LLM anymore.

Q: Are user BYOK keys connected to all services?
A: NO — 4 of 6 services fetch BYOK keys. The gaps:
   - RG_Ed_Service: ❌ Never fetches BYOK keys (platform keys only)
   - RG_agent_architect: ⚠️ Only gets keys if Chat forwards them
   - RG_LLM_Service: ⚠️ Legacy, accepts keys in request body but unused

Q: Is the BYOK flow consistent?
A: MOSTLY. Chat and Agent Engine use the same internal endpoint.
   IDE uses a different endpoint path (works but inconsistent).
   No caching — keys re-fetched on every request.

================================================================================
                           END OF TRACE
================================================================================
EOF
